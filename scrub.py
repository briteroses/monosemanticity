import os
import sys
import uuid
from typing import Any, Callable, Iterable, Literal, Optional, Tuple, Sequence, Type, Union, List, cast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import interp.tools.optional as op
import rust_circuit as rc
import torch
from interp.circuit.causal_scrubbing.dataset import color_dataset, Dataset
from interp.circuit.causal_scrubbing.testing_utils import IntDataset
from interp.circuit.causal_scrubbing.experiment import (
    Experiment,
    ExperimentCheck,
    ExperimentEvalSettings,
    ScrubbedExperiment,
)
from interp.circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    CondSampler,
    ExactSampler,
    FuncSampler,
    InterpNode,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id, negative_log_likelyhood
from interp.tools.indexer import SLICER as S
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR
from interp.circuit.testing.notebook import NotebookInTesting
from interp import cui
from interp.ui.very_named_tensor import VeryNamedTensor
from transformers import AutoTokenizer

import msem_utils.models
from create_dataset import create_concept_dataset

from datasets import load_dataset
from pprint import pprint
import random
import re
import functools
import pickle as pkl
import pdb

MODEL_ID = "gelu_12_tied"
SEQ_LEN = 9
MLP_WIDTH = 3072


def sever_neuron_from_mlp(mlp, neuron_id) -> rc.Concat:
    """
    pass in an MLP along with the target neuron to be severed; returns split MLP
    meant to be used in an updater for a transformer with an MLP matcher
    """
    inps = []
    if neuron_id > 0:
        inps.append(rc.Index(mlp, I[:, 0:neuron_id], name=f"{mlp.name}"))
    splitter = [
        rc.Index(mlp, I[i : i + 1, neuron_id : neuron_id + 1], name=f"{mlp.name}_target_neuron_{neuron_id}_at_pos_{i}")
        for i in range(SEQ_LEN)
    ]
    split_sever_neuron = rc.Concat(*splitter, axis=0, name=f"{mlp.name}_target_neuron_{neuron_id}")
    inps.append(split_sever_neuron)
    if neuron_id < MLP_WIDTH - 1:
        inps.append(rc.Index(mlp, I[:, neuron_id + 1 : MLP_WIDTH], name=f"{mlp.name}"))
    return rc.Concat(*inps, axis=1, name=f"{mlp.name}_severed")


def sever_neuron_from_index(c_idx, neuron_id) -> rc.Concat:
    inps = []
    start, stop = c_idx.idx[1].start, c_idx.idx[1].stop
    if neuron_id > start:
        inps.append(rc.Index(c_idx, I[:, 0 : neuron_id - start], name=f"{c_idx.name}"))
    splitter = [
        rc.Index(
            c_idx,
            I[i : i + 1, neuron_id - start : neuron_id - start + 1],
            name=f"{c_idx.name}_target_neuron_{neuron_id}_at_pos_{i}",
        )
        for i in range(SEQ_LEN)
    ]
    split_sever_neuron = rc.Concat(*splitter, axis=0, name=f"{c_idx.name}_target_neuron_{neuron_id}")
    inps.append(split_sever_neuron)
    if neuron_id < stop - 1:
        inps.append(rc.Index(c_idx, I[:, neuron_id - start + 1 : stop - start], name=f"{c_idx.name}"))
    return rc.Concat(*inps, axis=1, name=f"{c_idx.name}_severed")


def sever_and_split_neuron(model: rc.Circuit, layer, neuron_id):
    """
    model should be unsplit, equivalent to the renamed_circuit in the standard load_model_id
    """
    mlp_matcher = rc.IterativeMatcher(f"m{layer}.p_bias").chain(rc.restrict(rc.IterativeMatcher("m.act"), end_depth=3))
    if len(model.get(mlp_matcher)) == 0:  # this layer was already sliced
        mlp_matcher = rc.IterativeMatcher(f"m{layer}.p_bias").chain(
            rc.restrict(rc.IterativeMatcher("m.act_severed"), end_depth=3)
        )
        mlp_slice_matcher = mlp_matcher.chain(
            rc.restrict(
                rc.IterativeMatcher(
                    lambda x: isinstance(x, rc.Index)
                    and x.cast_index().idx[1].start <= neuron_id
                    and x.cast_index().idx[1].stop >= neuron_id
                ),
                end_depth=2,
            )
        )
        sever_transform = functools.partial(sever_neuron_from_index, neuron_id=neuron_id)
        model_severed = model.update(mlp_slice_matcher, sever_transform)
        model_severed = model_severed.update(mlp_matcher, rc.concat_fuse)
        model_severed = model_severed.update(
            mlp_matcher.chain(rc.restrict(rc.IterativeMatcher("m.act"), start_depth=2, end_depth=3)),
            lambda c: rc.push_down_index_once(c) if isinstance(c, rc.Index) else c,
        )
    else:
        sever_transform = functools.partial(sever_neuron_from_mlp, neuron_id=neuron_id)
        model_severed = model.update(mlp_matcher, sever_transform)
    return model_severed


def load_concept_dataset(concept, slice_idx=0, num_samples=10000):
    pkl_file = str(Path(__file__).resolve().parents[0]) + f"/msem_datasets/{concept}.pickle"
    with open(pkl_file, "rb") as handle:
        raw_data = pkl.load(handle)
    ds = Dataset(
        {
            "text": rc.Array(raw_data["text"][slice_idx : slice_idx + num_samples], "text"),
            "annotation": rc.Array(
                raw_data["annotation"][slice_idx : slice_idx + num_samples].to(torch.int64), "annotation"
            ),
        }
    )
    return ds


def sample_concept_val_at_pos(ds, pos):
    return ds.annotation.value[:, pos]


def last_concept_ds(ds):
    last_token_is_concept = torch.max(ds.annotation.value[:, -1:], dim=1).values
    return ds[torch.where(last_token_is_concept == 1)]


def chain_to_position(im: rc.IterativeMatcher, neuron_id: int, i: int):
    return im.chain(rc.restrict(rc.IterativeMatcher(f"m.act_target_neuron_{neuron_id}_at_pos_{i}"), end_depth=2))


def kl_of_logits(input_logits, target_logits):
    log_soft_max = torch.nn.LogSoftmax(dim=2)
    soft_max = torch.nn.Softmax(dim=2)
    kl_divergence = torch.nn.KLDivLoss(reduction="mean")
    kl = kl_divergence(log_soft_max(input_logits), soft_max(target_logits))
    return kl


def scrubbing(neurons: Union[Tuple, List[Tuple]], concept, measure="loss", data_idx=0):
    """
    neurons: one 2-tuple specifying (layer number, neuron number), or a list of such 2-tuples
    layer: the index of mlp layer to target; ranges 0 to 11
    neuron_id: the index of neuron in the specified mlp to target; ranges 0 to MLP_WIDTH-1
    concept: a String for the concept neuron
    measure: options are "loss", "KL", "MLP" (only for single neurons)
    """

    # --- slice out the neurons from the circuit and create the metric circuit
    neuron_sliced_circuit = msem_utils.models.get_model(model_id=MODEL_ID, seq_len=SEQ_LEN)
    concept_ds = load_concept_dataset(concept, slice_idx=10000 * data_idx)

    if isinstance(neurons, Tuple):
        neurons = [neurons]

    for neuron in neurons:
        layer, neuron_id = neuron
        neuron_sliced_circuit = sever_and_split_neuron(neuron_sliced_circuit, layer, neuron_id)

    toks_int_var = rc.Array(torch.zeros(SEQ_LEN + 1, dtype=torch.int64), "text")
    input_toks = toks_int_var.index(I[:-1], name="input_text")
    true_toks = toks_int_var.index(I[1:], name="true_text")

    circuit_for_metric = neuron_sliced_circuit.update(
        rc.IterativeMatcher("tokens"),
        transform=lambda _: input_toks,
    )

    metric = None
    if measure == "loss":
        metric = rc.Module(
            negative_log_likelyhood.spec, **{"ll.input": circuit_for_metric, "ll.label": true_toks}, name="loss"
        )
    elif measure == "KL":
        metric = circuit_for_metric
    elif measure == "MLP":
        assert len(neurons) == 1
        layer, neuron_id = neurons[0]
        metric = circuit_for_metric.get_unique(
            rc.IterativeMatcher("final.input").chain(rc.restrict(rc.IterativeMatcher(f"m{layer}"), end_depth=3))
        )
    assert metric

    # --- create correspondences for causal scrubbing
    neuron_matchers = [
        rc.IterativeMatcher(f"m{layer}.p_bias").chain(
            rc.restrict(rc.IterativeMatcher(f"m.act_target_neuron_{neuron_id}"), end_depth=4)
        )
        for (layer, neuron_id) in neurons
    ]
    all_neurons = rc.IterativeMatcher.any(*neuron_matchers)

    out_base = InterpNode(cond_sampler=ExactSampler(), name="out", other_inputs_sampler=ExactSampler())
    corr_base = Correspondence()
    corr_base.add(out_base, corr_root_matcher)

    out_not = InterpNode(cond_sampler=ExactSampler(), name="out", other_inputs_sampler=ExactSampler())  # type: ignore
    neuron_not = out_not.make_descendant(UncondSampler(), name="neuron")
    corr_not = Correspondence()
    corr_not.add(out_not, corr_root_matcher)
    corr_not.add(neuron_not, all_neurons)

    out_scrub = InterpNode(cond_sampler=ExactSampler(), name="out", other_inputs_sampler=ExactSampler())
    interp_nodes_by_pos = []
    for i in range(SEQ_LEN):
        sample_concept = functools.partial(sample_concept_val_at_pos, pos=i)
        interp_nodes_by_pos.append(out_scrub.make_descendant(FuncSampler(sample_concept), name=f"neuron_pos_{i}"))
    corr_scrub = Correspondence()
    corr_scrub.add(out_scrub, corr_root_matcher)
    for i in range(SEQ_LEN):
        neuron_position_matchers = [
            chain_to_position(neuron_matchers[j], neurons[j][1], i) for j in range(len(neurons))
        ]
        all_neuron_positions = rc.IterativeMatcher.any(*neuron_position_matchers)
        corr_scrub.add(interp_nodes_by_pos[i], all_neuron_positions)

    # --- run causal scrubbing and print results
    device = "cpu" if NotebookInTesting.currently_in_notebook_test else "cuda:0"
    eval_settings = ExperimentEvalSettings(
        device_dtype=device, optim_settings=rc.OptimizationSettings(scheduling_naive=True)
    )
    res_base = (
        Experiment(metric, concept_ds, corr_base, num_examples=1000).scrub().evaluate(eval_settings).to(torch.float64)
    )
    res_not = (
        Experiment(metric, concept_ds, corr_not, num_examples=1000).scrub().evaluate(eval_settings).to(torch.float64)
    )
    res_scrub = (
        Experiment(metric, concept_ds, corr_scrub, num_examples=1000).scrub().evaluate(eval_settings).to(torch.float64)
    )

    if measure == "KL":
        base_kl = kl_of_logits(res_base, res_base)
        not_kl = kl_of_logits(res_not, res_base)
        scrub_kl = kl_of_logits(res_scrub, res_base)
        print(f"provided neurons for concept {concept}:")
        pprint(neurons)
        print(f"neuron set original KL divergence: {base_kl}")
        print(f"neuron set random-ablation KL divergence: {not_kl}")
        print(f"neuron set scrubbed KL divergence: {scrub_kl}")
        pct_recovered = (1 - scrub_kl / not_kl) * 100.0
        pct_recovered = int(1e1 * pct_recovered) / 1e1
        print(f"percent KL recovered: {pct_recovered}%")

        return not_kl.item(), scrub_kl.item(), pct_recovered

    if measure == "loss":
        print(f"provided neurons for concept {concept}:")
        pprint(neurons)
        print(f"neuron set original loss: {res_base.mean()}")
        print(f"neuron set random-ablation loss: {res_not.mean()}")
        print(f"neuron set scrubbed loss: {res_scrub.mean()}")
        sanity_check = "IS" if res_not.mean() > res_base.mean() else "IS NOT"
        print(f"sanity check: random-ablation loss {sanity_check} greater than original loss")
        pct_recovered = (res_not.mean() - res_scrub.mean()) / (res_not.mean() - res_base.mean()) * 100.0
        pct_recovered = int(1e1 * pct_recovered) / 1e1
        print(f"percent loss recovered: {pct_recovered}%")

        # return res_base.mean(), res_not.mean(), res_scrub.mean(), pct_recovered
        return (
            (res_not.mean() - res_base.mean()).item(),
            (res_scrub.mean() - res_base.mean()).item(),
            pct_recovered,
            res_not.var() + res_base.var(),
            res_scrub.var() + res_base.var(),
        )


def repeat_scrubbing(neurons, concept, repeats=10):
    base_losses = []
    base_losses_errors = []
    scrub_losses = []
    scrub_losses_errors = []
    pct_losses = []
    for n in neurons:
        neuron_base_loss = 0
        neuron_scrub_loss = 0
        neuron_base_var = 0
        neuron_scrub_var = 0
        for idx in range(repeats):
            base_loss, scrub_loss, _, base_var, scrub_var = scrubbing(n, concept, measure="loss", data_idx=idx)
            neuron_base_loss += base_loss
            neuron_scrub_loss += scrub_loss
            neuron_base_var += base_var
            neuron_scrub_var += scrub_var
        not_minus_base = neuron_base_loss / len(neurons)
        base_losses.append(not_minus_base)
        base_losses_errors.append(2 * math.sqrt(neuron_base_var))
        scrub_minus_base = neuron_scrub_loss / len(neurons)
        scrub_losses.append(scrub_minus_base)
        scrub_losses_errors.append(2 * math.sqrt(neuron_base_var))
        pct_loss = (not_minus_base - scrub_minus_base) / not_minus_base * 100.0
        pct_loss = int(1e1 * pct_loss) / 1e1
        pct_losses.append(str(pct_loss) + "%")

    # TODO: UNFINISHED ! left hanging in the middle of the variance formulation

    both_loss = base_losses + scrub_losses
    both_error = base_losses_errors + scrub_losses_errors
    x_neurons = [str(n) for n in neurons] + [str(n) for n in neurons]
    loss_category = ["random ablation loss gap"] * len(neurons) + ["scrub loss gap"] * len(neurons)
    loss_df = pd.DataFrame({"neurons": x_neurons, "losses": both_loss, "category": loss_category, "error": both_error})

    sns.set(style="darkgrid")
    loss_plot = sns.barplot(
        x="neurons",
        y="losses",
        hue="category",
        data=loss_df,
        palette=["steelblue", "purple"],
    )
    x_coords_for_error = [p.get_x() + 0.5 * p.get_width() for p in loss_plot.patches]
    y_coords_for_error = [p.get_height() for p in loss_plot.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=loss_df["error"], fmt="none", c="k")
    loss_plot.axhline(0, linestyle="--")
    loss_plot.bar_label(loss_plot.containers[1], pct_losses, padding=2)
    plt.xticks(rotation=45)
    plt.xlabel("(layer, index) of MLP neuron")
    plt.ylabel("Loss Difference from Baseline")
    plt.tight_layout()
    plt.savefig(f"plots/{concept}_losses.png", dpi=300)
    plt.clf()


def plotter(neurons, concept, for_notebook=False):

    base_losses = []
    scrub_losses = []
    pct_losses = []
    for n in neurons:
        base, scrub, pct, _, _ = scrubbing(n, concept, measure="loss")
        base_losses.append(base)
        scrub_losses.append(scrub)
        pct_losses.append(str(pct) + "%")

    both_loss = base_losses + scrub_losses
    x_neurons = [str(n) for n in neurons] + [str(n) for n in neurons]
    loss_category = ["random ablation loss gap"] * len(neurons) + ["scrub loss gap"] * len(neurons)
    loss_df = pd.DataFrame({"neurons": x_neurons, "losses": both_loss, "category": loss_category})

    sns.set(style="darkgrid")
    loss_plot = sns.barplot(
        x="neurons",
        y="losses",
        hue="category",
        data=loss_df,
        palette=["steelblue", "purple"],
    )
    loss_plot.axhline(0, linestyle="--")
    loss_plot.bar_label(loss_plot.containers[1], pct_losses, padding=2)
    plt.xticks(rotation=45)
    plt.xlabel("(layer, index) of MLP neuron")
    plt.ylabel("Loss Difference from Baseline")
    plt.tight_layout()
    if for_notebook:
        plt.show()
    else:
        plt.savefig(f"plots/{concept}_losses.png", dpi=300)
    plt.clf()

    base_KLs = []
    scrub_KLs = []
    pct_KLs = []
    for n in neurons:
        base, scrub, pct = scrubbing(n, concept, measure="KL")
        base_KLs.append(base)
        scrub_KLs.append(scrub)
        pct_KLs.append(str(pct) + "%")

    x_neurons = [str(n) for n in neurons] + [str(n) for n in neurons]
    both_KL = base_KLs + scrub_KLs
    KL_category = ["random ablation KL div"] * len(neurons) + ["scrub KL div"] * len(neurons)
    KL_df = pd.DataFrame({"neurons": x_neurons, "KLs": both_KL, "category": KL_category})

    sns.set(style="darkgrid")
    KL_plot = sns.barplot(
        x="neurons",
        y="KLs",
        hue="category",
        data=KL_df,
        palette=["steelblue", "purple"],
    )
    KL_plot.axhline(0, linestyle="--")
    KL_plot.bar_label(KL_plot.containers[1], pct_KLs, padding=2)
    plt.xticks(rotation=45)
    plt.xlabel("(layer, index) of MLP neuron")
    plt.ylabel("KL Divergence of Softmax")
    plt.tight_layout()
    if for_notebook:
        plt.show()
    else:
        plt.savefig(f"plots/{concept}_KLs.png", dpi=300)
    plt.clf()


if __name__ == "__main__":
    neurons = [
        (6, 1811),
        (10, 2214),
        (8, 545),
        (0, 1558),
        (2, 2940),
        (11, 2443),
        (0, 1176),
        (0, 627),
        (8, 770),
        (0, 2131),
    ]
    concept = "prepositions"
    plotter(neurons, concept)
    # scrubbing((6, 1811), "prepositions", measure="MLP")
