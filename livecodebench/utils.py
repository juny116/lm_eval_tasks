import re
import json
import subprocess
import tempfile
import sys
import os
import datasets



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:

    # Remove private_test_cases column if it exists
    if "private_test_cases" in dataset.column_names:
        dataset = dataset.remove_columns(["private_test_cases"])
    print(dataset.column_names)    
    return dataset

def pass_at_k(references, predictions):
    """
    stdin/stdout 기반 스크립트 평가 버전 pass@k.

    여기서:
      - references: 하나의 문제에 대한 테스트 케이스 리스트
        각 원소는 {"input": str, "output": str} 형태 (extract_test_cases에서 생성)
      - predictions: 하나의 문제에 대한 코드 샘플들
        - str: 단일 코드
        - list[str]: 여러 코드 샘플
        - list[list[str]]: lm-eval이 [samples_for_problem] 형태로 줄 수도 있으니 이 경우도 처리

    로직:
      - 각 code에 대해 모든 test case를 순차 실행
      - 각 test case에서:
        - code를 임시 .py 파일로 저장
        - stdin에 tc["input"]을 넣어 실행
        - stdout을 strip하여 tc["output"]과 비교
      - 어떤 code라도 "모든 test case를 통과"하면 pass@1 = 1.0, 아니면 0.0

    반환:
      {"pass_at_k": score}  # score ∈ {0.0, 1.0}
    """

    # ---------- predictions 정리 ----------
    if isinstance(predictions, str):
        code_samples = [predictions]
    elif isinstance(predictions, list):
        # lm-eval이 [[code1, code2, ...]] 형태로 넘겨줄 수도 있음
        if predictions and isinstance(predictions[0], list):
            # 첫 번째 요소가 리스트이면, 그걸 실제 샘플 리스트로 사용
            code_samples = predictions[0]
        else:
            code_samples = predictions
    else:
        code_samples = []

    if not code_samples or not references:
        return {"pass_at_k": 0.0}

    # ---------- 실제 평가 ----------
    def run_script_on_test(code: str, inp: str, expected: str) -> bool:
        """
        code: 전체 파이썬 스크립트
        inp: stdin에 들어갈 문자열
        expected: 기대 stdout (문자열)
        """
        # 임시 파일에 코드 작성
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                input=inp,
                text=True,
                capture_output=True,
                timeout=5.0,  # 너무 오래 돌지 않게 타임아웃 (필요시 조정)
            )
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            return False
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)

        if result.returncode != 0:
            return False

        # stdout 비교 (양쪽 strip)
        actual = result.stdout.strip()
        exp = str(expected).strip()
        ok = (actual == exp)

        return ok

    any_code_passes_all = False

    for idx, code in enumerate(code_samples):
        all_ok = True

        for t_idx, tc in enumerate(references):
            inp = tc.get("input", "")
            out = tc.get("output", "")

            ok = run_script_on_test(code, inp, out)
            if not ok:
                all_ok = False
                break

        if all_ok:
            any_code_passes_all = True
            break

    score = 1.0 if any_code_passes_all else 0.0

    return {"pass_at_k": score}


def doc_to_text(doc):
    """
    Convert LiveCodeBench problem document to prompt text.

    Handles two problem formats:
    1. With starter_code: Problem provides incomplete function to complete
    2. Without starter_code: Problem requires reading stdin and writing stdout

    Args:
        doc (dict): Problem document from dataset
            Keys:
            - question_content (str): Problem description/statement
            - starter_code (str): Optional incomplete code to complete
            - question_title (str): Problem title

    Returns:
        str: Formatted prompt for the model

    Prompt Structure:
    - For WITH starter_code:
      Shows the incomplete starter code and asks to complete it
      Model should write code to fill in the gaps

    - For WITHOUT starter_code:
      Shows stdin format with example inputs/outputs
      Model should write code that reads from stdin and writes to stdout
    """
    prompt = "### Question:\n"
    prompt += doc.get("question_content", "") + "\n\n"

    starter_code = doc.get("starter_code", "").strip()

    if starter_code:
        # Questions with starter code
        prompt += (
            "### Format: You will use the following starter code to write the "
            "solution to the problem and enclose your code within delimiters.\n"
        )
        prompt += "```python\n"
        prompt += starter_code + "\n"
        prompt += "```\n\n"
    else:
        # Questions without starter code
        prompt += (
            "### Format: Read the inputs from stdin solve the problem and write "
            "the answer to stdout (do not directly test on the sample inputs). "
            "Enclose your code within delimiters as follows. Ensure that when "
            "the python program runs, it reads the inputs, runs the algorithm "
            "and writes output to STDOUT.\n"
        )

    prompt += "### Answer: (use the provided format with backticks)"
    return prompt


def build_predictions(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """
    Extract Python code from model responses.

    This function processes raw model responses and extracts executable Python code.
    The extracted code will later be executed against test cases.

    Args:
        resps (list[list[str]]): Model responses
            Structure: List[List[str]]
            - Outer list: one entry per problem
            - Inner list: multiple response samples for same problem
            - Each response is raw text from model (may contain explanations, markdown, etc.)

        docs (list[dict]): Problem documents (for context if needed)
            Used to handle different problem types appropriately

    Returns:
        list[list[str]]: Extracted code samples
            Same structure as input but with only executable code
            Example output: [["def solution(n):\\n    return n*3", "..."], ...]
    """
    results = []

    for resp, doc in zip(resps, docs):
        codes = []
        for r in resp:
            code = _extract_code(r, doc)
            codes.append(code)
        results.append(codes)

    return results


def _extract_code(response: str, doc: dict) -> str:
    """
    Extract executable Python code from model response using robust regex patterns.

    EXTRACTION STRATEGY (Priority Order):
    ====================================
    1. PRIMARY: LiveCodeBench official regex pattern
       Pattern: (?<=```python\\n)((?:\\n|.)+?)(?=\\n```)
       Matches code between ```python and ``` markers (lookbehind/lookahead)
       Most reliable for well-formatted responses

    2. FALLBACK 1: Generic Python code blocks
       Pattern: ```python\\n(.*?)\\n```
       Matches standard markdown code blocks
       For models that don't follow exact format

    3. FALLBACK 2: Entire response
       Returns entire response as-is
       Assumes model generated code without markdown formatting

    Args:
        response (str): Raw model response text
            May contain:
            - ```python ... ``` code blocks
            - Explanations and reasoning
            - Multiple code blocks (extracts first)

        doc (dict): Problem document (available for context)

    Returns:
        str: Extracted Python code ready for execution

    Examples:
        Input: "The solution is:\\n```python\\nprint(5*3)\\n```"
        Output: "print(5*3)"

        Input: "```python\\ndef solve(n):\\n    return n*2\\n```"
        Output: "def solve(n):\\n    return n*2"

        Input: "def main():\\n    print(input())"
        Output: "def main():\\n    print(input())"
    """
    response = response.strip()
    
    # <think>...</think> 태그 제거
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # LiveCodeBench regex pattern - most specific
    # Matches code between ```python and ``` using lookbehind/lookahead
    pattern = r"(?<=```python\n)((?:\n|.)+?)(?=\n```)"
    matches = re.findall(pattern, response)

    if matches:
        code = matches[0].strip()
        return code

    # Fallback: Try to find any code block
    # Matches standard markdown code blocks
    code_blocks = re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Fallback 2: markdown 없이 코드만 있는 경우
    # "def "나 "class "로 시작하는 부분 찾기
    lines = response.split('\n')
    code_start_idx = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith(('def ', 'class ', 'import ')):
            code_start_idx = i
            break
    
    if code_start_idx >= 0:
        code_lines = lines[code_start_idx:]
        result = []
        for line in code_lines:
            # 공백, 영문, 숫자, 파이썬 기호만 유지
            cleaned = re.sub(r'[^\w\s()[\]{}:,.\-+*/<>=!&|^@#%$\\;\'"\n]', '', line)
            if cleaned.strip():
                result.append(cleaned)
        return '\n'.join(result).strip()

    # Fallback: Return entire response as code
    # Assumes response is already code without markdown formatting
    return response


def extract_test_cases(doc: dict) -> list:
    """
    Extract test cases from LiveCodeBench problem document.
    """
    public_tests = doc.get("public_test_cases", "[]")

    try:
        # JSON string -> list[dict]
        if isinstance(public_tests, str):
            test_cases = json.loads(public_tests)
        else:
            test_cases = public_tests

        # 각 test case는 최소한 "input", "output"을 가진 dict라고 가정
        normalized = []
        for tc in test_cases:
            if isinstance(tc, dict):
                test_input = tc.get("input", "")
                test_output = tc.get("output", "")
                normalized.append(
                    {"input": str(test_input), "output": str(test_output)}
                )

        return normalized

    except Exception as e:
        return []


def process_results(doc: dict, resps: list[list[str]]) -> list[dict]:
    """
    Process results for LiveCodeBench evaluation.

    이 함수는 pass_at_k에 들어갈 references를 준비한다.

    Args:
        doc (dict): Problem document
        resps (list[list[str]]): Model responses (여기서는 실제 사용 X)

    Returns:
        list[dict]: 테스트 케이스 리스트
            [{"input": str, "output": str}, ...]
    """
    test_cases = extract_test_cases(doc)
    return test_cases