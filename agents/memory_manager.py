# agents/memory_manager.py
class MemoryManager:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.buffer = []

    def add_message(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        if len(self.buffer) > self.max_history:
            self.buffer.pop(0)  # keep only recent messages

    def get_summary(self) -> str:
        """
        Return conversation history as a single string suitable for LLM prompts.
        """
        summary_lines = [f"{m['role']}: {m['content']}" for m in self.buffer]
        return "\n".join(summary_lines)

    def get_contexted_prompt(self, user_prompt: str) -> str:
        """
        Combine memory summary + new user input into one prompt.
        """
        history = self.get_summary()
        if history:
            return f"{history}\nUser: {user_prompt}\nAssistant:"
        else:
            return f"User: {user_prompt}\nAssistant:"
