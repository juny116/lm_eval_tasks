import re
from typing import Union

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def extract_code_blocks(text: str) -> str:
    """
    Robustly extract Python code from various response formats.
    Priority:
    1. Code within ```python ... ``` blocks
    2. Code within ``` ... ``` blocks (generic)
    3. Direct function definition (def statement)
    4. First occurrence of 'def ' in response
    5. Entire response if starts with valid code
    """
    text = text.strip()
    
    # Priority 1: Extract from ```python ... ``` blocks
    python_blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    if python_blocks:
        return python_blocks[0].strip()
    
    # Priority 2: Extract from generic ``` ... ``` blocks
    generic_blocks = re.findall(r"```\n(.*?)\n```", text, re.DOTALL)
    if generic_blocks:
        return generic_blocks[0].strip()
    
    # Priority 3: Check if response is directly valid Python code starting with 'def'
    if text.startswith("def "):
        return text
    
    # Priority 4: Find first 'def ' and extract from there
    def_match = re.search(r"def\s+\w+\s*\(", text)
    if def_match:
        code = text[def_match.start():]
        # Remove any trailing explanation or markdown after function definition
        code = re.split(r"```|^[A-Za-z]|^$", code)[0].strip()
        return code
    
    # Fallback: Return entire response
    return text


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Extract and build predictions for MBPP."""
    return [[extract_code_blocks(r) for r in resp] for resp in resps]
