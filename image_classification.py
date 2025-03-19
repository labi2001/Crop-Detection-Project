import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import ConversationChain
import google
from google.api_core.exceptions import ResourceExhausted
import time
import json
import os
from gtts import gTTS
import base64

# Set page layout
st.set_page_config(layout="wide")

# Load the model
model = load_model("crop_disease_model.h5")


# Define class labels
class_labels = {
    0: "Common Rust (Corn)",
    1: "Gray Leaf Spot (Corn)",
    2: "Northern Leaf Blight (Corn)",
    3: "Healthy (Corn)",
    4: "Early Blight (Potato)",
    5: "Late Blight (Potato)",
    6: "Healthy (Potato)",
    7: "Brown Spot (Rice)",
    8: "Leaf Blast (Rice)",
    9: "Neck Blast (Rice)",
    10: "Healthy (Rice)",
    11: "Brown Rust (Wheat)",
    12: "Yellow Rust (Wheat)",
    13: "Healthy (Wheat)",
    14: "Red Rot (Sugarcane)",
    15: "Bacterial Blight (Sugarcane)",
    16: "Healthy (Sugarcane)",
}

# Set up Google Gemini API
gemini_key = "AIzaSyB-Yaivm_Qq2cyvOSRj8O0CJrKUWQ7mZWo"
genai.configure(api_key=gemini_key)

# CSS configuration
st.markdown(
    """
    <style>
        /* Import Nunito with bold weight */
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
        
        /* Full-page background */
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            background-color: #6A8042;
        }
        div[data-testid="stAppViewContainer"] {
            background-color: #6A8042;
            padding: 20px;
            height: 100%;
            font-family: 'Nunito', sans-serif;
            color: #FFFADD;
        }
        div[data-testid="stSidebar"] {
            background-color: #6A8042;
            font-family: 'Nunito', sans-serif;
            color: #FFFADD;
        }
        /* Title styling */
        h1 {
            font-family: 'Nunito', sans-serif;
            font-size: 50px;
            color: #FFFADD;
            text-align: center;
            margin-bottom: 10px;
            font-weight: 700;
        }
        /* Subtitle styling */
        .subtitle {
            font-family: 'Nunito', sans-serif;
            font-size: 30px;                   
            color: #FFFADD;
            font-weight: 500;
            text-align: center;
        }

        /* Override red color for active tab */
        button[data-testid="stTab"][aria-selected="true"] {
            background-color: #6A8042 !important;  
            color: #FFFADD !important;           
            border-bottom: none !important; 
            }     

        /* Justify text in AI Assistant chat messages */
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
            text-align: justify;
        }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# Prompt
System_Prompt = """
You are an expert agricultural assistant. You must only answer questions related to these crop diseases:
Corn: Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy
Potato: Early Blight, Late Blight, Healthy
Rice: Brown Spot, Leaf Blast, Neck Blast, Healthy
Wheat: Brown Rust, Yellow Rust, Healthy
Sugarcane: Red Rot, Bacterial Blight, Healthy
If the user asks about any other diseases, politely inform them that you specialize only in the diseases listed above.
"""

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", google_api_key=gemini_key, temperature=0.7
)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
chatbot = ConversationChain(llm=llm, memory=memory)

# Directory setup
History_Dir = "chat_histories"
os.makedirs(History_Dir, exist_ok=True)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "current_session" not in st.session_state:
    st.session_state.current_session = f"Chat {len(os.listdir(History_Dir)) + 1}"
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False


# History functions
def get_saved_chats():
    return sorted(
        [
            f.replace(".json", "")
            for f in os.listdir(History_Dir)
            if f.endswith(".json")
        ],
        reverse=True,
    )


def load_chat_history(session_name):
    file_path = os.path.join(History_Dir, f"{session_name}.json")
    return json.load(open(file_path, "r")) if os.path.exists(file_path) else []


def save_chat_history(session_name, history):
    with open(os.path.join(History_Dir, f"{session_name}.json"), "w") as file:
        json.dump(history, file)


def text_to_speech(text, file_name="response.mp3"):
    tts = gTTS(text=text, lang="en")
    file_path = os.path.join("chat_audio", file_name)
    os.makedirs("chat_audio", exist_ok=True)
    tts.save(file_path)
    return file_path


# Streamlit app UI
st.title(" 🌿 Farmer's AI Assistant")
st.markdown(
    '<p class="subtitle">Ask me about crop diseases, treatments, and prevention!</p>',
    unsafe_allow_html=True,
)

# Tabs
tabs = st.tabs(["Image Classification", "AI Assistant", "History", "Settings"])

# Image Classification tab
with tabs[0]:
    option = st.selectbox("Choose input method", ["Upload Image", "Take a Picture"])
    image_source = None
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image_source = Image.open(uploaded_file)
    elif option == "Take a Picture":
        captured_image = st.camera_input("Take a picture")
        if captured_image is not None:
            image_source = Image.open(captured_image)

    if image_source is not None:
        image = image_source.convert("RGB")
        st.image(image, use_container_width=True)
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        predicted_disease = class_labels.get(predicted_class, "Unknown Disease")
        st.write(f"### Predicted Disease: {predicted_disease}")
        st.write(f"Confidence: {confidence:.2f}%")

# AI Assistant tab
with tabs[1]:
    # Loop through chat history and display messages
    for message in st.session_state.history:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])
        # Convert assistant's response to speech if TTS is enabled
        if message["role"] == "assistant" and st.session_state.tts_enabled:
            audio_file = text_to_speech(
                message["content"], f"audio_{len(st.session_state.history)}.mp3"
            )
            st.audio(open(audio_file, "rb").read(), format="audio/mp3")

# History tab
with tabs[2]:
    saved_chats = get_saved_chats()
    selected_chat = st.selectbox("Load a past chat", saved_chats)
    if st.button("Load Chat"):
        st.session_state.history = load_chat_history(selected_chat)
        st.session_state.current_session = selected_chat
        st.rerun()
    if st.button("New Chat"):
        if st.session_state.history:
            save_chat_history(
                st.session_state.current_session, st.session_state.history
            )
        st.session_state.history = []
        st.session_state.current_session = f"Chat {len(get_saved_chats()) + 1}"
        st.rerun()
    chat_text = "\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.history
        ]
    )
    st.download_button(
        "Download Chat",
        chat_text,
        f"{st.session_state.current_session}.txt",
        mime="text/plain",
    )
    if st.button("Clear All Chats"):
        for file in os.listdir(History_Dir):
            os.remove(os.path.join(History_Dir, file))
        st.session_state.history = []
        st.rerun()

# Settings tab
with tabs[3]:
    st.session_state.tts_enabled = st.checkbox(
        "Enable Text-to-Speech", value=st.session_state.tts_enabled
    )
st.markdown("</div>", unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    chat_history = "\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.history
        ]
    )
    full_prompt = f"{System_Prompt}\n\n{chat_history}\n\nUser: {user_input}"

    # Display loading message for assistant response
    with st.chat_message("assistant"):
        search_msg = st.empty()
        search_msg.markdown("Searching... Please wait")

    try:
        response = chatbot.predict(input=full_prompt)
    except ResourceExhausted:
        time.sleep(3)
        response = "High traffic, please try again."
    except Exception as e:
        response = f"Error: {str(e)}"

    # Store assistant response in chat history
    st.session_state.history.append({"role": "assistant", "content": response})

    # Save chat history
    save_chat_history(st.session_state.current_session, st.session_state.history)
    st.rerun()
