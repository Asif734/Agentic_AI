# agents/llm_interface.py
import ollama

def local_llm(prompt: str, memory_manager=None) -> str:
    """
    Sends the prompt to the Ollama LLM.
    If memory_manager is provided, include conversation context.
    """
    try:
        if memory_manager:
            prompt = memory_manager.get_contexted_prompt(prompt)

        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"[LLM Error] {str(e)}"
