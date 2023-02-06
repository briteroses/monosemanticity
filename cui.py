import os
import sys
from typing import Any, Callable, Iterable, Literal, Optional, Tuple, Sequence, Type, Union, List, cast
from pathlib import Path

import interp.tools.optional as op
import numpy as np
import rust_circuit as rc
import torch
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.indexer import TORCH_INDEXER as i
from interp import cui
from interp.ui.very_named_tensor import VeryNamedTensor
import remix_utils
from transformers import AutoTokenizer

from datasets import load_dataset
from pprint import pprint
import random
import re

import msem_utils.models

MODEL_ID = "gelu_12_tied"
SEQ_LEN = 40
NUM_EXAMPLES = 100
GRAB_RANDOM_SLICE = 184  # random.randint(0, 3000)
SLICE_WIDTH = 3  # 10
MLP_WIDTH = 3072

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenized_sample():
    openwebtext = load_dataset("openwebtext", split="train")
    # openwebtext = openwebtext.shuffle()
    random_sample = random.randint(
        0, 8013769 - NUM_EXAMPLES
    )  # shuffle function seems to give the same shuffle every time
    sample_of_openwebtext = openwebtext["text"][random_sample : random_sample + NUM_EXAMPLES]
    del openwebtext

    tokenized_sample_of_openwebtext = tokenizer(
        sample_of_openwebtext, return_tensors="pt", padding=True, truncation=True
    )["input_ids"].squeeze()
    tokenized_sample_of_openwebtext = tokenized_sample_of_openwebtext[:, :SEQ_LEN]

    return tokenized_sample_of_openwebtext


def evaluate_on_dataset(c: rc.Circuit, tokens: torch.Tensor) -> torch.Tensor:
    """Run the circuit on all elements of tokens. Assumes the 'tokens' module exists in the circuit."""
    arr = rc.Array(tokens, name="tokens")
    var = rc.DiscreteVar(arr)
    c2 = c.update(rc.IterativeMatcher("tokens"), lambda _: var)
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([var.probs_and_group]))
    evaluations = transform.sample(c2).evaluate()
    """interleave with shape (1, 2, 1) zero tensor to correspond to the skip newlines"""
    evaluations = torch.concat(
        [
            val
            for pair in zip(
                torch.chunk(evaluations, NUM_EXAMPLES, dim=0), [torch.tensor([0, 0])[None, :, None]] * NUM_EXAMPLES
            )
            for val in pair
        ][:-1],
        dim=1,
    )
    return evaluations


def show_neuron_patterns(circs, seqs):
    remix_utils.await_without_await(lambda: cui.init(port=6789))
    neuron_firings = torch.concat(
        [
            torch.concat(
                [evaluate_on_dataset(circ, seqs).squeeze(2).unsqueeze(1) for circ in circs[layer]], dim=1
            ).unsqueeze(1)
            for layer in range(len(circs))
        ],
        dim=1,
    )
    text_concat = []
    for i in range(NUM_EXAMPLES):
        text_concat.extend(tokenizer.batch_decode(seqs[i]))
        if i < NUM_EXAMPLES - 1:
            text_concat.extend(["\n", "\n"])
    vnts = []
    for i in range(12):
        for j in range(SLICE_WIDTH):
            neuron = circs[i][j]
            neuron_firings = evaluate_on_dataset(neuron, seqs).squeeze()
            vnts.append(
                VeryNamedTensor(
                    neuron_firings,
                    dim_names=["text"],
                    dim_types=["seq"],
                    dim_idx_names=[text_concat],
                    title=f"pattern, layer {i}, neuron {GRAB_RANDOM_SLICE+j}",
                )
            )
    # vnt = VeryNamedTensor(
    #         neuron_firings[0],
    #         dim_names="layer neuron text".split(),
    #         dim_types="facet facet seq".split(),
    #         dim_idx_names=[list(range(len(circs))), list(range(GRAB_RANDOM_SLICE, GRAB_RANDOM_SLICE+SLICE_WIDTH)), text_concat],
    #         title="neuron patterns",
    #     )
    remix_utils.await_without_await(lambda: cui.show_tensors(*vnts))


def excerpt_nodes():
    neuron_sliced_circuit = msem_utils.models.get_neuron_sliced_model(model_id=MODEL_ID, seq_len=SEQ_LEN)
    mlp_neurons = [
        [
            rc.substitute_all_modules(
                neuron_sliced_circuit.get_unique(
                    rc.IterativeMatcher("final.input")
                    .chain(rc.restrict(rc.IterativeMatcher(f"m{layer}.p_bias"), end_depth=2))
                    .chain(rc.restrict(rc.IterativeMatcher("m.act_split_neurons"), end_depth=3))
                    .chain(rc.restrict(rc.IterativeMatcher(f"m.act_neuron_{neuron_id}"), end_depth=2))
                )
            )
            for neuron_id in range(GRAB_RANDOM_SLICE, GRAB_RANDOM_SLICE + SLICE_WIDTH)
        ]
        for layer in range(12)
    ]
    return mlp_neurons


if __name__ == "__main__":
    mlp_neurons = excerpt_nodes()
    tokenized_sample_of_openwebtext = tokenized_sample()
    show_neuron_patterns(mlp_neurons, tokenized_sample_of_openwebtext)
