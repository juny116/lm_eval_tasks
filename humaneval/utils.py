import re
import evaluate as hf_evaluate


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


def _remove_think_tags(text: str) -> str:
    """<think>...</think> 태그 제거 및 코드 블록 추출"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # 마크다운 코드 블록 추출: ```python ... ```
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Fallback: 코드 블록이 없으면 "def "부터 시작하는 부분만 반환
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            # "def "부터 문장 끝까지 추출
            rest = '\n'.join(lines[i:])
            # 이상한 문자 제거 (공백, 영문, 숫자, 기호만 유지)
            rest = re.sub(r'[^\w\s()[\]{}:,.\-+*/<>=!&|^@#%$\\;\'"\n]', '', rest)
            return rest.strip()
    
    return text


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + _remove_think_tags(r) for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (
                _remove_think_tags(r) if _remove_think_tags(r).find("```") == -1 
                else _remove_think_tags(r)[: _remove_think_tags(r).find("```")]
            )
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
