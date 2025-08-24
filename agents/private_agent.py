# agents/private_agent.py
import json

class PrivateAgent:
    def __init__(self, json_file=r"C:\Users\Asif\VSCODE\Agentic_AI\data\private_student_data.json"):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def generate_prompt(self, message: str, student_id="student_123") -> str:
        """
        Create a prompt for the LLM using private student info.
        """
        student = next((s for s in self.data if s["student_id"] == student_id), None)
        if not student:
            return "[PrivateAgent] Student not found."

        prompt = f"Use the following private student info to answer safely:\nName: {student['name']}\nGrades: {student['grades']}\nSchedule: {student['schedule']}\nQuestion: {message}\nAnswer:"
        return prompt
