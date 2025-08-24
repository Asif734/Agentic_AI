# # agents/public_agent.py
# from .llm_interface import local_llm

# class PublicAgent:
#     def __init__(self):
#         """
#         PublicAgent is fully autonomous. It generates responses purely from LLM knowledge.
#         No dataset is required.
#         """
#         pass

#     def generate_prompt(self, message: str) -> str:
#         """
#         Creates a prompt for the LLM using the user message.
#         """
#         prompt = (
#             "You are a helpful, knowledgeable university assistant. "
#             "Answer the following question concisely and accurately:\n\n"
#             f"Question: {message}\nAnswer:"
#         )
#         return prompt

#     def respond(self, message: str) -> str:
#         """
#         Returns the answer using only the LLM.
#         """
#         prompt = self.generate_prompt(message)
#         return local_llm(prompt)

# agents/public_agent.py
from .llm_interface import local_llm

class PublicAgent:
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

    def generate_prompt(self, message: str) -> str:
        return (
            "You are a helpful, knowledgeable university assistant. "
            "Answer the following question concisely and accurately:\n\n"
            f"Question: {message}\nAnswer:"
        )

    def respond(self, message: str) -> str:
        prompt = self.generate_prompt(message)
        return local_llm(prompt, memory_manager=self.memory_manager)
