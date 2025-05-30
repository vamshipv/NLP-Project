import streamlit as st
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from baseline.generator.generator import Generator
from baseline.retriever.retriever import Retriever

# Set page config
st.set_page_config(page_title="Cat Chat", layout="centered")

# Custom CSS for full white background and chat bubbles
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: white !important;
        }
        .chat-message {
            display: flex;
            margin-bottom: 10px;
        }
        .chat-message.user {
            justify-content: flex-end;
        }
        .chat-message.bot {
            justify-content: flex-start;
        }
        .message-content {
            max-width: 70%;
            padding: 10px 15px;
            border-radius: 15px;
            font-size: 16px;
            line-height: 1.4;
        }
        .user .message-content {
            background-color: #DCF8C6;
            color: black;
            border-top-right-radius: 0;
        }
        .bot .message-content {
            background-color: #F1F0F0;
            color: black;
            border-top-left-radius: 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize retriever and generator
retriever = Retriever()
gen = Generator()
group_id = "Team Dave"

# Load or create index
document_file = "winnie_the_pooh.txt"
base_name = os.path.splitext(os.path.basename(document_file))[0]
index_file = f"{base_name}_faiss.index"
subtext_file = f"{base_name}_subtexts.json"

if os.path.exists(index_file) and os.path.exists(subtext_file):
    retriever.load(index_file, subtext_file)
else:
    retriever.addDocuments(document_file)
    retriever.save(index_file, subtext_file)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("Type your question here...", key="input")
    submitted = st.form_submit_button("Send")

if submitted and user_query:
    try:
        results = retriever.query(user_query)
        appendlist = [res.replace("\n", " ").strip() for res in results]
        context = "\n\n".join(results)
        answer = gen.generate_answer(appendlist, context, user_query, group_id)

        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("bot", answer))
    except Exception as e:
        st.session_state.chat_history.append(("bot", f"Error: {e}"))

# Display chat history
for sender, message in st.session_state.chat_history:
    role_class = "user" if sender == "user" else "bot"
    st.markdown(f"""
        <div class="chat-message {role_class}">
            <div class="message-content">{message}</div>
        </div>
    """, unsafe_allow_html=True)
