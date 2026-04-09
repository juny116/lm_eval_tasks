def doc_to_text(doc):
    """Format ARC question with dynamic number of choices."""
    prompt = "Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n"
    prompt += doc['question'] + "\n"
    for text, label in zip(doc['choices']['text'], doc['choices']['label']):
        prompt += f"{label}) {text}\n"
    return prompt