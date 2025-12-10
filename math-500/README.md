# MATH-500

This task uses the HuggingFaceH4/MATH-500 dataset, which contains 500 challenging math problems from the MATH dataset.

## Dataset Information
- **Dataset**: HuggingFaceH4/MATH-500
- **Total Problems**: 500
- **Source**: MATH dataset subset (https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

## Task Format
The task requires models to solve math problems step by step and put the final answer inside `\boxed{}`.

Example:
```
Solve the following math problem step by step. Put your answer inside \boxed{}.

[Problem here]

Remember to put your answer inside \boxed{}.
```

## Evaluation
- **Metric**: exact_match
- **Normalization**: Mathematical expression normalization using `math_verify`
- **Temperature**: 0.0 (deterministic)
