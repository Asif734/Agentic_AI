# utils/session_state.py
import streamlit as st

def get_or_create_state(key: str, default_value):
    """
    Get a value from Streamlit session state, or create it with a default.
    
    :param key: Name of the state variable
    :param default_value: Value to set if key doesn't exist
    :return: The value from session state
    """
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]

def add_message(role: str, content: str):
    """
    Adds a message to the message_history in session state.
    Initializes message_history if it doesn't exist.
    
    :param role: 'User' or 'Assistant'
    :param content: Message content
    """
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    st.session_state.message_history.append({"role": role, "content": content})

def clear_messages():
    """
    Clears all messages from session state.
    """
    st.session_state.message_history = []

def get_messages():
    """
    Returns the current message history.
    """
    return st.session_state.get("message_history", [])
