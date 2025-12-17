from typing import Dict, List
import re


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """Process results for denoising tasks (lyrics and proverbs)."""
    def normalize(text):
        # Handle if text is a list
        if isinstance(text, list):
            text = text[0] if text else ""
        # Remove quotes
        text = re.sub(r'["\']', '', text)
        # Remove trailing punctuation
        text = re.sub(r'[,.]$', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing spaces
        text = text.strip()
        return text

    gold = normalize(doc["answer"])
    pred = normalize(results[0])

    retval = 1 if gold == pred else 0
    return {
        "exact_match": retval,
    }
