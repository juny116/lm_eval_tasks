import evaluate as hf_evaluate
import re


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


def build_predictions_instruct_robust(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """
    Robustly extract code from model responses in various formats.
    Handles: code blocks, direct def statements, and mixed content.
    """
    results = []
    for resp, doc in zip(resps, docs):
        codes = []
        for r in resp:
            code = _extract_code(r)
            # codes.append(doc["prompt"] + code)
            codes.append(doc["prompt"] + code)
        results.append(codes)
    return results


def _extract_code(response: str) -> str:
    """
    Extract Python code from various response formats.
    Priority:
    1. Code within ```python ... ``` blocks
    2. Direct function definition (def statement)
    3. First occurrence of 'def ' in response
    4. Entire response if starts with valid code
    """
    response = response.strip()
    
    # Priority 1: Extract from ```python ... ``` blocks
    python_blocks = re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)
    if python_blocks:
        # Use the first code block, remove trailing backticks
        code = python_blocks[0].strip()
        return code
    
    # Priority 2: Check if response is directly valid Python code
    if response.startswith("def "):
        return response
    
    # Priority 3: Find first 'def ' and extract from there
    def_match = re.search(r"def\s+\w+\s*\(", response)
    if def_match:
        code = response[def_match.start():]
        # Remove any trailing explanation or markdown after function definition
        code = re.split(r"```|^[A-Za-z]|^$", code)[0].strip()
        return code
    
    # Priority 4: Try to extract code block without python marker
    code_blocks = re.findall(r"```\n(.*?)\n```", response, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()
        return code
    
    # Fallback: Return entire response (might work for simple cases)
    return response
