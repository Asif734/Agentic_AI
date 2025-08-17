# utils/logging.py
import json
from datetime import datetime
import os

LOG_FILE = "chat_interactions.json"

def log_interaction(agent: str, message: str, response: str):
    """
    Logs interactions between user and agents to a JSON file.
    
    :param agent: Name of the agent responding
    :param message: User message
    :param response: Agent response
    """
    # Prepare log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "user_message": message,
        "agent_response": response
    }

    # Check if file exists; if not, create it
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([log_entry], f, indent=4)
    else:
        # Append new entry to existing file
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=4)
