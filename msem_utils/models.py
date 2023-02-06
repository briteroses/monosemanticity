import os
import sys
from typing import Any, Callable, Iterable, Literal, Optional, Tuple, Sequence, Type, Union, List, cast
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import interp.tools.optional as op
import numpy as np
import rust_circuit as rc
import torch
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.indexer import TORCH_INDEXER as I
from interp import cui
from interp.ui.very_named_tensor import VeryNamedTensor
import remix_utils
from transformers import AutoTokenizer

from datasets import load_dataset
from pprint import pprint
import random
import re


def get_model(model_id="gelu_12_tied", seq_len=40):
    circ_dict, tokenizer, model_info = load_model_id(model_id)
    unbound_circuit = circ_dict["t.bind_w"]

    tokens_arr = rc.Array(torch.zeros(seq_len).to(torch.long), name="tokens")
    # We use this to index into the tok_embeds to get the proper embeddings
    token_embeds = rc.GeneralFunction.gen_index(circ_dict["t.w.tok_embeds"], tokens_arr, 0, name="tok_embeds")
    bound_circuit = model_info.bind_to_input(unbound_circuit, token_embeds, circ_dict["t.w.pos_embeds"])

    transformed_circuit = bound_circuit.update(
        "t.bind_w",
        lambda c: configure_transformer(
            c,
            To.ATTN_HEAD_MLP_NORM,
            split_by_head_config="full",
            use_pull_up_head_split=True,
            use_flatten_res=True,
        ),
    )
    transformed_circuit = rc.conform_all_modules(transformed_circuit)

    subbed_circuit = transformed_circuit.cast_module().substitute()
    subbed_circuit = subbed_circuit.rename("logits")

    def module_but_norm(circuit: rc.Circuit):
        if isinstance(circuit, rc.Module):
            if "norm" in circuit.name or "ln" in circuit.name or "final" in circuit.name:
                return False
            else:
                return True
        return False

    for i in range(100):
        subbed_circuit = subbed_circuit.update(module_but_norm, lambda c: c.cast_module().substitute())

    renamed_circuit = subbed_circuit.update(rc.Regex(r"[am]\d(.h\d)?$"), lambda c: c.rename(c.name + ".inner"))
    renamed_circuit = renamed_circuit.update("t.inp_tok_pos", lambda c: c.rename("embeds"))

    for l in range(model_info.params.num_layers):
        # b0 -> a1.input, ... b11 -> final.input
        next = "final" if l == model_info.params.num_layers - 1 else f"a{l+1}"
        renamed_circuit = renamed_circuit.update(f"b{l}", lambda c: c.rename(f"{next}.input"))

        # b0.m -> m0, etc.
        renamed_circuit = renamed_circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
        renamed_circuit = renamed_circuit.update(f"b{l}.m.p_bias", lambda c: c.rename(f"m{l}.p_bias"))
        renamed_circuit = renamed_circuit.update(f"b{l}.a", lambda c: c.rename(f"a{l}"))
        renamed_circuit = renamed_circuit.update(f"b{l}.a.p_bias", lambda c: c.rename(f"a{l}.p_bias"))

        for h in range(model_info.params.num_layers):
            # b0.a.h0 -> a0.h0, etc.
            renamed_circuit = renamed_circuit.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))

    return renamed_circuit


def split_mlp_neurons_to_concat(c: rc.Circuit, mlp_width=3072) -> rc.Concat:
    inps = [rc.Index(c, I[:, i : i + 1], name=f"{c.name}_neuron_{i}") for i in range(mlp_width)]
    return rc.Concat(*inps, axis=1, name=f"{c.name}_split_neurons")


def get_neuron_sliced_model(model_id="gelu_12_tied", seq_len=40):
    renamed_circuit = get_model(model_id, seq_len)

    mlp_matcher = (
        rc.IterativeMatcher("final.input")
        .chain(rc.restrict(rc.Regex(r"^m\d\d?\.p_bias$"), end_depth=2))
        .chain(rc.restrict(rc.IterativeMatcher("m.act"), end_depth=3))
    )
    neuron_sliced_circuit = renamed_circuit.update(mlp_matcher, split_mlp_neurons_to_concat)

    return neuron_sliced_circuit
