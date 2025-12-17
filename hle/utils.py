from typing import Dict, List
import re
import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter text-only questions and prepare prompts based on question type."""
    def _process_doc(doc: dict) -> dict:
        # Filter: only include text-only questions (no images)
        if doc.get('image') is not None and doc.get('image') != '':
            return None
        
        question = doc.get('question', '')
        answer = doc.get('answer', '')
        answer_type = doc.get('answer_type', 'exactMatch')
        
        if answer_type == 'multipleChoice':
            # For multiple choice, prepare MMLU-Pro style prompt
            prompt = f"Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: X' (where X is the choice letter).\n\n{question}"
            target = answer
        else:  # exactMatch
            # For exact match, prepare text with boxed format
            prompt = f"Answer the following question. Put your final answer inside \\boxed{{}}.\n\n{question}\n\nRemember to put your answer inside \\boxed{{}}."
            target = answer
        
        return {
            'prompt': prompt,
            'target': target,
            'answer_type': answer_type,
            'original_answer': answer,
        }
    
    filtered_dataset = []
    for doc in dataset:
        processed = _process_doc(doc)
        if processed is not None:
            filtered_dataset.append(processed)
    
    return datasets.Dataset.from_dict({
        'prompt': [d['prompt'] for d in filtered_dataset],
        'target': [d['target'] for d in filtered_dataset],
        'answer_type': [d['answer_type'] for d in filtered_dataset],
        'original_answer': [d['original_answer'] for d in filtered_dataset],
    })


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """Evaluate based on question type."""
    answer_type = doc.get('answer_type', 'exactMatch')
    prediction = results[0].strip()
    target = doc['target'].strip()
    
    if answer_type == 'multipleChoice':
        # Extract answer letter (A, B, C, D, E, etc.)
        # Similar to MMLU-Pro - look for Answer: followed by letter
        match = re.search(r'[Aa]nswer\s*:\s*([A-E])', prediction)
        if match:
            pred_answer = match.group(1).upper()
        else:
            # Fallback: look for any standalone letter
            match = re.search(r'\b([A-E])\b', prediction)
            pred_answer = match.group(1).upper() if match else ''
        
        target_answer = target.strip().upper() if target else ''
        retval = 1 if pred_answer == target_answer else 0
    else:  # exactMatch
        # Extract from \boxed{} format if present
        boxed_match = re.search(r'\\boxed\{(.+?)\}', prediction)
        if boxed_match:
            pred_text = boxed_match.group(1)
        else:
            pred_text = prediction
        
        # Normalize and compare text
        def normalize(text):
            text = text.lower().strip()
            text = re.sub(r'\s+', ' ', text)
            # Keep alphanumeric and basic punctuation for answers like "Sale Law"
            return text
        
        norm_pred = normalize(pred_text)
        norm_target = normalize(target)
        retval = 1 if norm_pred == norm_target else 0
    
    return {
        "exact_match": retval,
    }
