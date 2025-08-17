# import streamlit as st
# from agents.orchestration_agent import OrchestrationAgent
# from agents.public_agent import PublicAgent
# from agents.private_agent import PrivateAgent
# from agents.mental_health_agent import MentalHealthAgent


# def main():
#     st.set_pag_config(page_title="Secure University CHatbot", layout="wide")
#     st.title("Secure University Chatboot")
#     st.markdown ("Your secure university chatbot for all your needs.")


#     if "orchestration_agent" not in st.session_state:
#         st.session_state.orchestration_agent = OrchestrationAgent(
#             public_agent=PublicAgent(),
#             private_agent=PrivateAgent(),
#             mental_health_agent=MentalHealthAgent()
#         )

#     if "message_history" not in st.session_state:
#         st.session_state.message_history = []
#         st.session_state.message_history.append(
#             {"role": "Assistant", "content": "Hi! How can I assist you today?"}
#         )


#         for message in st.session_state.message_history:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"] )


#         if user_input:=st.chat_input("Type your message here..."):
#             st.session_state.message_history.append({"role": "User", "content": user_input})
#             with st.chat_message("User"):
#                 st.markdown(user_input)

#             response = st.session_state.orchestration_agent.handle_message(user_input)
#             st.session_state.message_history.append({"role": "Assistant", "content": response})

#             with st.chat_message("Assistant"):
#                 st.markdown(response)   


# if __name__ == "__main__":
#     main()



import streamlit as st
from agents.orchestration_agent import OrchestrationAgent
from agents.public_agent import PublicAgent
from agents.private_agent import PrivateAgent
from agents.mental_health_agent import MentalHealthAgent
from agents.llm_interface import local_llm
from agents.memory_manager import MemoryManager


def main():
    st.set_page_config(page_title="Secure University Chatbot", layout="wide")
    st.title("Secure University Chatbot")
    st.markdown("Your secure university chatbot with local LLM integration.")

    if "orchestration_agent" not in st.session_state:
        st.session_state.orchestration_agent = OrchestrationAgent(
            llm=local_llm,
            public_agent=PublicAgent(),
            private_agent=PrivateAgent(),
            mental_health_agent=MentalHealthAgent(),
            memory_manager=MemoryManager(max_history=10)
        )

    if "message_history" not in st.session_state:
        st.session_state.message_history = [
            {"role": "Assistant", "content": "Hi! How can I assist you today?"}
        ]

    # Display all messages
    for message in st.session_state.message_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        st.session_state.message_history.append({"role": "User", "content": user_input})
        with st.chat_message("User"):
            st.markdown(user_input)

        response = st.session_state.orchestration_agent.handle_message(user_input)
        st.session_state.message_history.append({"role": "Assistant", "content": response})
        with st.chat_message("Assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
