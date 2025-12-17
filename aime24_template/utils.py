import re
from typing import Dict, List
from math_verify import parse, verify

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    # to match aime24 format
    gold = parse("\\boxed{"+str(doc["Answer"])+"}")
    answer = parse(results[0])
    print(f"Predicted answer: {answer}, Gold answer: {gold}")
    if verify(gold, answer):
        retval = 1
        print("-----Answer is correct.-----")
    results = {
        "exact_match": retval,
    }
    return results

