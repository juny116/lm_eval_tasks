import re
from typing import Dict, List, Optional


def parse(response: str) -> Optional[str]:
    """
    Parse the model response to extract the answer choice (A, B, C, or D).
    
    Handles various formats:
    - " A" (with leading/trailing spaces)
    - "A. Svetambara" (with explanation after)
    - "Answer: A"
    - "(A)" 
    - etc.
    """
    if not response or not isinstance(response, str):
        return None
    
    # Remove leading and trailing whitespace
    response = response.strip()
    
    if not response:
        return None
    
    # First, try to find A, B, C, or D with various patterns
    # Pattern 1: Look for A/B/C/D as the first character (after removing spaces)
    first_char = response[0].upper()
    if first_char in ['A', 'B', 'C', 'D']:
        return first_char
    
    # Pattern 2: Look for answer patterns like "Answer: A" or "(A)" or "A."
    match = re.search(r'[:\s\(\[]([A-D])[.\)\]\s]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Look for any standalone A, B, C, D (case-insensitive)
    match = re.search(r'\b([A-D])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def verify(answer: Optional[str], gold: str) -> bool:
    """
    Verify if the parsed answer matches the gold standard answer.
    """
    if answer is None:
        return False
    return answer.upper() == gold.upper()


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    gold = ['A', 'B', 'C', 'D'][doc["answer"]]
    answer = parse(results[0])
    print(f"Predicted answer: {answer}, Gold answer: {gold}")
    if verify(answer, gold):
        retval = 1
    results = {
        "exact_match": retval,
    }
    return results

