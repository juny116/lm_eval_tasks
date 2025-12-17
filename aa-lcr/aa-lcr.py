# lm_eval/tasks/aa_lcr/aa_lcr.py
from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download
from lm_eval.api.task import Task, Instance
from lm_eval.api.registry import register_task

logger = logging.getLogger(__name__)


def _mean(values):
    """평균 계산"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _read_extracted_text_file(filename: str) -> str:
    # 파일들은 HF dataset repo의 extracted_text/ 아래에 있음
    local_path = hf_hub_download(
        repo_id="ArtificialAnalysis/AA-LCR",
        repo_type="dataset",
        filename=f"extracted_text/{filename}",
    )
    with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _build_prompt(question: str, filenames: str) -> str:
    docs = []
    for fn in [x.strip() for x in filenames.split(";") if x.strip()]:
        docs.append(_read_extracted_text_file(fn))

    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i+1}:\n{doc}\nEND DOCUMENT {i+1}"
        for i, doc in enumerate(docs)
    )

    prompt = (
        "BEGIN INPUT DOCUMENTS\n\n"
        f"{documents_text}\n\n"
        "END INPUT DOCUMENTS\n\n"
        "Answer the following question using the input documents provided above.\n\n"
        "START QUESTION\n\n"
        f"{question}\n"
        "END QUESTION\n"
    )
    return prompt


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _judge_with_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    official_answer: str,
    candidate_answer: str,
    timeout: int = 120,
) -> bool:
    """OpenAI-compatible endpoint를 사용한 답안 평가"""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI package not found. Install: pip install openai")
        return False

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    judge_prompt = (
        "Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.\n"
        "For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.\n\n"
        f"The question, for reference only: {question}\n\n"
        f"The OFFICIAL ANSWER: {official_answer}\n\n"
        f"CANDIDATE ANSWER TO ASSESS: {candidate_answer}\n\n"
        "Reply only with CORRECT or INCORRECT."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        out = resp.choices[0].message.content.strip().upper()
        return "CORRECT" in out
    except Exception as e:
        logger.error(f"Judge API error: {e}")
        return False


@register_task("aa_lcr")
class AALCR(Task):
    """
    AA-LCR: Artificial Analysis Long Context Reasoning
    
    Evaluates long context performance through testing reasoning capabilities 
    across multiple long documents (~100k tokens each).
    
    - 100 hard text-based questions
    - 7 document categories: Company Reports, Industry Reports, Government 
      Consultations, Academia, Legal, Marketing Materials, Survey Reports
    - ~100k tokens per question (cl100k_base tokenizer)
    - Requires minimum 128K context window
    - ~3M total unique input tokens, ~230 documents
    - Evaluated using pass@1 metric with Qwen3 235B as equality checker
    """
    
    VERSION = 0
    DATASET_PATH = "ArtificialAnalysis/AA-LCR"
    DATASET_NAME = "default"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        """문서와 질문으로부터 프롬프트 생성"""
        return _build_prompt(doc["question"], doc["data_source_filenames"])

    def doc_to_target(self, doc):
        """공식 정답 반환"""
        return doc["answer"]

    def construct_requests(self, doc, ctx, **kwargs):
        """생성 기반 요청 구성"""
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments={
                    "context": ctx,
                    "until": ["\n\n", "\nEND", "\n###"],
                    "max_gen_toks": 512,
                    "temperature": 0.0,
                },
            )
        ]

    def process_results(self, doc, results):
        pred = results[0].strip()
        official_answer = doc["answer"].strip()

        # Judge 설정 확인 (task_args 또는 direct attributes)
        use_judge = False
        judge_config = {}
        
        # task_args에서 읽기 시도
        if hasattr(self, "task_args") and self.task_args:
            use_judge = self.task_args.get("use_judge", "0") == "1"
            judge_config = {
                "base_url": self.task_args.get("judge_base_url", "http://localhost:8000/v1"),
                "api_key": self.task_args.get("judge_api_key", "EMPTY"),
                "model": self.task_args.get("judge_model", "gpt-3.5-turbo"),
            }
        
        # Exact match (기본값)
        def _normalize(s: str) -> str:
            return " ".join(s.strip().lower().split())
        
        is_correct = _normalize(pred) == _normalize(official_answer)
        
        # Judge 사용 (optional)
        if use_judge and all(judge_config.values()):
            logger.info(f"Using judge for evaluation")
            is_correct = _judge_with_openai_compatible(
                base_url=judge_config["base_url"],
                api_key=judge_config["api_key"],
                model=judge_config["model"],
                question=doc["question"],
                official_answer=official_answer,
                candidate_answer=pred,
            )

        return {"pass@1": 1.0 if is_correct else 0.0}

    def aggregation(self):
        return {"pass@1": _mean}

    def higher_is_better(self):
        return {"pass@1": True}
