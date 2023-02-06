from pprint import pprint
import random
import re
import functools
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def create_whitelist(concept, concept_words):
    """
    concept: String label for the semantic concept,
    concept_words: List of words pertaining to concept
    """
    token_ids = tokenizer(concept_words)["input_ids"]
    token_ids = [item for sublist in token_ids for item in sublist]
    whitelist = token_ids
    to_json = {
        concept: whitelist,
        "concept_words": concept_words,
    }
    json_object = json.dumps(to_json)
    json_file = f"whitelists/{concept}.json"
    with open(json_file, "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    # concept = "months_and_seasons"
    # concept_words = [
    #     "January",
    #     "February",
    #     "March",
    #     "April",
    #     "May",
    #     "June",
    #     "July",
    #     "August",
    #     "September",
    #     "October",
    #     "November",
    #     "December",
    #     "winter",
    #     "spring",  # polysemy :(
    #     "summer",
    #     "autumn",
    # ]
    concept = "Obama"
    concept_words = ["Obama"]
    # concept_words = """
    # about
    # above
    # across
    # after
    # against
    # among
    # around
    # at
    # before
    # behind
    # below
    # beside
    # between
    # by
    # down
    # during
    # for
    # from
    # in
    # inside
    # into
    # near
    # of
    # off
    # on
    # out
    # over
    # through
    # to
    # toward
    # under
    # up
    # with
    # """.strip().split()
    concept_words.extend([" " + s for s in concept_words])
    create_whitelist(concept, concept_words)
