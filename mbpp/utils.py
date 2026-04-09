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
    # <think> 태그 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Pattern to match ```...``` blocks
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    # (+ ```) as we add the opening "```python" to the gen_prefix
    matches = re.findall(pattern, r"```" + text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Fallback: 마크다운이 없으면, 함수 정의부터 시작하는 가장 가능성 높은 부분 추출
    # "def " 또는 "class " 또는 "import "로 시작하는 라인 찾기
    lines = text.split('\n')
    code_start_idx = -1
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ')):
            code_start_idx = i
            break
    
    if code_start_idx >= 0:
        # 코드 시작부터 끝까지 추출하되, 이상한 문자 제거
        code_lines = lines[code_start_idx:]
        code_text = '\n'.join(code_lines).strip()
        
        # 코드 블록 이후의 이상한 텍스트 제거 (예: "dlfjst")
        # 유효한 파이썬 코드로 보이는 부분까지만 유지
        lines = code_text.split('\n')
        result_lines = []
        for line in lines:
            # 빈 줄이나 주석은 포함
            if not line.strip() or line.strip().startswith('#'):
                result_lines.append(line)
            # 파이썬 코드처럼 보이는 줄
            elif any(line.strip().startswith(kw) for kw in ['def ', 'class ', 'if ', 'for ', 'while ', 'return ', 'import ', 'from ', 'try:', 'except', 'finally:', 'with ', 'raise ', 'assert ']):
                result_lines.append(line)
            # 들여쓰기가 있으면 코드의 일부일 가능성 높음
            elif line[0] in ' \t':
                result_lines.append(line)
            # 그 외는 첫 번째 이상한 문자로 중단
            elif line.strip() and not any(ord(c) < 32 or ord(c) > 126 for c in line):
                # 영문자와 숫자, 특수기호만 있으면 계속
                if any(c.isalpha() or c.isdigit() for c in line):
                    # 마지막 유효한 코드 줄일 가능성 있음
                    last_valid = line.rstrip()
                    # 변수명이나 괄호 같은 것이 있으면 포함
                    if any(c in '()[]{}:,.' for c in last_valid) or last_valid.replace('_', '').replace('-', '').isalnum():
                        result_lines.append(line)
                        break
                    else:
                        break
            else:
                break
        
        return '\n'.join(result_lines).strip()
    
    return ""


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
