# llm_interface.py
import ollama

def local_llm(prompt: str) -> str:
    """
    Sends the prompt to the Qwen-3 model via Ollama Python SDK.
    """
    try:
        # Ollama API call
        response = ollama.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # ✅ Correct way to extract the content
        return response['message']['content']
    
    except Exception as e:
        return f"[LLM Error] {str(e)}"
