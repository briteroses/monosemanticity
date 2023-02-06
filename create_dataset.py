from pprint import pprint
import random
import glob
import re
import functools
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import pickle as pkl

import rust_circuit as rc
from interp.circuit.causal_scrubbing.dataset import color_dataset, Dataset

SEQ_LEN = 10
ROOT_DIR = Path(__file__).resolve().parents[0]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def get_subset(dataset_size=10000):
    dataset = load_dataset("openwebtext", split="train")
    dataset = dataset.shuffle()

    dataset_subset = dataset["text"][:dataset_size]
    del dataset

    return dataset_subset


def load_whitelist(concept):
    path_to_json = str(ROOT_DIR / "whitelists")
    json_files = glob.glob(path_to_json + f"/{concept}.json")
    try:
        whitelist = json.load(open(json_files[0], "r"))
    except:
        raise ValueError("whitelist doesn't exist! make the whitelist first in create_whitelist.py")
    return whitelist[concept]


def process_for_concept(dataset, concept):
    whitelist = load_whitelist(concept)
    tokenized_subset = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True)["input_ids"].squeeze()
    tokenized_subset = tokenized_subset[:, :SEQ_LEN]
    annotation = torch.stack([tokenized_subset == tok for tok in whitelist], dim=0).sum(dim=0).bool()
    return tokenized_subset, annotation


def create_concept_dataset(concept, dataset_size=100000):
    dataset_subset = get_subset(dataset_size)
    preprocessed, annotation = process_for_concept(dataset_subset, concept)
    to_pkl = {
        "text": preprocessed,
        "annotation": annotation,
    }
    pkl_file = f"msem_datasets/{concept}.pickle"
    with open(pkl_file, "wb") as handle:
        pkl.dump(to_pkl, handle)
    return to_pkl


if __name__ == "__main__":
    concept = "prepositions"
    pprint(create_concept_dataset(concept))
