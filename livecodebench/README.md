# LiveCodeBench Evaluation

LiveCodeBench는 코드 생성 모델 평가를 위한 벤치마크입니다.

## 데이터셋

- **Path**: `lighteval/code_generation_lite`
- **데이터**: 프로그래밍 문제와 해결책

## 프롬프트 형식

### Starter code 있는 경우:
```
### Question:
{문제 설명}

### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
{starter_code}
```

### Answer: (use the provided format with backticks)
```

### Starter code 없는 경우:
```
### Question:
{문제 설명}

### Format: Read the inputs from stdin solve the problem and write the answer to stdout...
```python
# YOUR CODE HERE
```

### Answer: (use the provided format with backticks)
```

## 코드 추출

정규식: `(?<=```python\n)((?:\n|.)+?)(?=\n```)`

## 평가 메트릭

- **pass@k**: k개 생성 중 적어도 하나가 통과하는 비율
- 기본값: k=[1]

## 설정

- Temperature: 0.0 (결정론적)
- Max tokens: 2048
- Do sample: false
