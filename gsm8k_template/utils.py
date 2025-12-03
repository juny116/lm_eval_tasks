from typing import Dict, List
from math_verify import parse, verify

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        parsed = parse(doc["answer"])[0]
        parsed = str(parsed)
        out_doc = {
            "problem": doc["question"],
            "solution": doc["answer"],
            "answer": parsed,
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    gold = parse(doc["answer"])
    answer = parse(results[0])
    if verify(gold, answer):
        retval = 1
    results = {
        "exact_match": retval,
    }
    return results

