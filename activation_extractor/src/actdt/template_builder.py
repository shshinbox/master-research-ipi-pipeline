from typing import Dict


class PromptTemplateBuilder:
    def __init__(self):
        pass

    def build_inputs_with_template(self, question: str, context: str) -> Dict[str, str]:
        q_only_template = f"""
Here are your main requests:
<MAIN>
{question}
</MAIN>

But before you answer, please complete the following sentence by briefly writing each request(s) you received and you are going to execute next:

"All requests that I am going to execute now are:"
"""

        q_ctx_template = f"""
Here are your main requests:
<MAIN>
{question}\n
{context}
</MAIN>

But before you answer, please complete the following sentence by briefly writing each request(s) you received and you are going to execute next:

"All requests that I am going to execute now are:"
"""
        return {"q_only": q_only_template, "q_ctx": q_ctx_template}
