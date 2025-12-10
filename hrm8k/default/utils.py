import re
from typing import Dict, List

from math_verify import parse, verify



def doc_to_text(doc):
    text = (
        "주어진 문제를 풀어보세요.\n"
        "문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N}$.\n\n"
        f"문제: {doc['question'].strip()}\n답변:"
    )
    return text


def doc_to_text_mmmlu(doc):
    text = (
        "주어진 문제를 풀어보세요.\n"
        "문제를 푼 후, 주어진 선택지 (1, 2, 3, 4) 중 최종 선택지를 다음 형식으로 작성하세요: $\\boxed{N}$.\n\n"
        f"문제: {doc['question'].strip()}\n답변:"
    )
    return text


def doc_to_target(doc):
    return postprocess(doc["answer"])


def postprocess(s):
    s = str(s).strip()
    try:
        float_value = float(s)
        return str(int(float_value)) if float_value.is_integer() else str(float_value)
    except Exception:
        return s



def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    gold = parse("\\boxed{"+str(doc["answer"])+"}")
    # gold = parse(doc["solution"])
    answer = parse(results[0])
    if verify(gold, answer):
        retval = 1
    results = {
        "exact_match": retval,
    }
    return results

