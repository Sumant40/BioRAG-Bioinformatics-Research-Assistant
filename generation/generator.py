# generation/generator.py
from groq import Groq
from generation.prompt_templates import (
    BIOMEDICAL_SYSTEM_PROMPT,
    QUERY_TEMPLATE,
    build_context_block,
)

class BioGenerator:

    def __init__(self, config):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.LLM_MODEL

    def generate(self, query: str, hits: list[dict]) -> dict:
        context = build_context_block(hits)
        user_message = QUERY_TEMPLATE.format(context=context, query=query)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": BIOMEDICAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": [h["metadata"] for h in hits],
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        }