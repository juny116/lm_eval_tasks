from typing import Dict, List
from math_verify import parse, verify

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": doc["answer"],
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    gold = parse(doc["solution"])
    answer = parse(results[0])
    if verify(gold, answer):
        retval = 1
    results = {
        "exact_match": retval,
    }
    return results

