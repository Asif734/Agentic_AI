#main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from agents.public_agent_rag import PublicAgentRAG
import shutil
import os

app = FastAPI(title="University Public RAG Agent")

# Initialize the RAG agent
rag_agent = PublicAgentRAG(
    index_path="data/public_index.faiss",
    meta_path="data/public_meta.pkl",
)

# ---------------- Add text ----------------
@app.post("/add-text")
async def add_text(text: str = Form(...), source: str = Form("manual")):
    """
    Add plain text to the RAG database.
    """
    try:
        rag_agent.add_text(text, source)
        return JSONResponse({"status": "success", "message": f"Text added from source '{source}'."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# ---------------- Add PDF ----------------
@app.post("/add-pdf")
async def add_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and add its content to the RAG database.
    """
    try:
        temp_path = f"temp_uploads/{file.filename}"
        os.makedirs("temp_uploads", exist_ok=True)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        rag_agent.add_pdf(temp_path)
        os.remove(temp_path)
        return JSONResponse({"status": "success", "message": f"PDF '{file.filename}' added successfully."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# ---------------- Query ----------------
@app.post("/query")
async def query_rag(query: str = Form(...)):
    """
    Query the RAG agent and get an answer.
    """
    try:
        response = rag_agent.respond(query)
        return JSONResponse({"status": "success", "answer": response})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})




# import streamlit as st
# from agents.orchestration_agent import OrchestrationAgent
# from agents.public_agent import PublicAgent
# from agents.private_agent import PrivateAgent
# from agents.mental_health_agent import MentalHealthAgent 
# from agents.llm_interface import local_llm
# from agents.memory_manager import MemoryManager


# def main():
#     st.set_page_config(page_title="Secure University Chatbot", layout="wide")
#     st.title("Secure University Chatbot")
#     st.markdown("Your secure university chatbot with local LLM integration.")

#     if "orchestration_agent" not in st.session_state:
#         st.session_state.orchestration_agent = OrchestrationAgent(
#             llm=local_llm,
#             public_agent=PublicAgent(),
#             private_agent=PrivateAgent(),
#             mental_health_agent=MentalHealthAgent(),
#             memory_manager=MemoryManager(max_history=10)
#         )

#     if "message_history" not in st.session_state:
#         st.session_state.message_history = [
#             {"role": "Assistant", "content": "Hi! How can I assist you today?"}
#         ]

#     # Display all messages
#     for message in st.session_state.message_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat input
#     if user_input := st.chat_input("Type your message here..."):
#         st.session_state.message_history.append({"role": "User", "content": user_input})
#         with st.chat_message("User"):
#             st.markdown(user_input)

#         response = st.session_state.orchestration_agent.handle_message(user_input)
#         st.session_state.message_history.append({"role": "Assistant", "content": response})
#         with st.chat_message("Assistant"):
#             st.markdown(response)

# if __name__ == "__main__":
#     main()
