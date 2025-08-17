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
        Summarize conversation buffer for context.
        You can use an LLM summarization call here, or a simple string join.
        """
        summary_lines = [f"{m['role']}: {m['content']}" for m in self.buffer]
        return "\n".join(summary_lines)
