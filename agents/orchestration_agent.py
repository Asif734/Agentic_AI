import re
from .public_agent import PublicAgent
from .private_agent import PrivateAgent     
from .mental_health_agent import MentalHealthAgent
from utils.logging import log_interaction
from .memory_manager import MemoryManager



class OrchestrationAgent:
    def __init__(self, llm, public_agent, private_agent, mental_health_agent, memory_manager):
        self.llm = llm
        self.public_agent = public_agent
        self.private_agent = private_agent
        self.mental_health_agent = mental_health_agent
        self.memory_manager = memory_manager

    def classify_message(self, message: str) -> str:
        prompt = f"""
        You are a router. Classify the user's message into one of these categories:
        1. PUBLIC: General, casual, or academic questions with no sensitive data.
        2. PRIVATE: Personal data like grades, ID, financial details, or login info.
        3. MENTAL_HEALTH: Messages expressing stress, depression, anxiety, or emotional distress.

        User message: "{message}"
        Answer with only one label: PUBLIC, PRIVATE, or MENTAL_HEALTH.
        """
        
        # Call your function directly
        classification = self.llm(prompt).strip().upper()
    
        # Safety check
        if classification not in ["PUBLIC", "PRIVATE", "MENTAL_HEALTH"]:
            classification = "PUBLIC"
        
        return classification

    def handle_message(self, message: str) -> str:
        category = self.classify_message(message)

        # Store in memory
        self.memory_manager.add_message("User", f"[{category}] {message}")

        # Route to correct agent
        if category == "PUBLIC":
            response = self.public_agent.respond(message)
        elif category == "PRIVATE":
            response = self.private_agent.respond(message)
        elif category == "MENTAL_HEALTH":
            response = self.mental_health_agent.respond(message)
        else:
            response = "Sorry, I couldn’t classify your request."

        # Add assistant response to memory
        self.memory_manager.add_message("Assistant", response)

        return response

