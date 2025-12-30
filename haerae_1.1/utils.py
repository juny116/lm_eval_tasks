from typing import Dict, List
import re
import string


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """Process results for denoising tasks (lyrics and proverbs)."""
    def normalize(text):
        # Handle if text is a list
        if isinstance(text, list):
            text = text[0] if text else ""
        # Remove leading/trailing whitespace
        text = text.strip()
        # Remove quotes
        text = re.sub(r'["\']', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        # Strip again after cleanup
        text = text.strip()
        return text

    gold = normalize(doc["answer"])
    pred = normalize(results[0])

    retval = 1 if gold == pred else 0
    return {
        "exact_match": retval,
    }
