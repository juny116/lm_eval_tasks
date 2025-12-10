def doc_to_text(doc):
    prompt = "[|system|][|endofturn|][|user|]\nAnswer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n{{question.strip()}}"
    for t, l in zip(doc['choices']['text'], doc['choices']['label']):
        prompt += f"{l}) {t}\n"
    prompt += "[|assistant|]<thought>\n"
    return prompt