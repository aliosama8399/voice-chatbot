import streamlit as st
import asyncio
from pymongo import MongoClient
from datetime import datetime
import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from gtts import gTTS

# Load environment variables
load_dotenv()

# Initialize MongoDB client
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client['chat_history_db']
collection = db['user_chat']

# Sidebar for ChatOpenAI parameters
st.sidebar.header("ChatOpenAI Configuration")
api_key = st.sidebar.text_input("API Key", value=os.getenv("OPENAI_API_KEY"))
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=2048, value=150)

# Initialize OpenAI API with the provided key and parameters
chat = ChatOpenAI(api_key=api_key, model="gpt-4", temperature=temperature, max_tokens=max_tokens)
parser = StrOutputParser()

# Function to read and chunk the file content
def read_and_chunk_file(file):
    content = file.read().decode("utf-8")
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
    return chunks

# Function to initialize a new session if it does not exist
def initialize_session(session_id):
    if not collection.find_one({"session_id": session_id}):
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "chat_history": [],
            "file_chunks": []
        }
        collection.insert_one(session_data)

# Function to update chat history in MongoDB for a specific session
def update_chat_history(session_id, user_input, response):
    collection.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "chat_history": {
                    "timestamp": datetime.now(),
                    "user_input": user_input,
                    "response": response
                }
            }
        }
    )

# Function to retrieve chat history for a specific session
def get_chat_history(session_id):
    session = collection.find_one({"session_id": session_id})
    if session and "chat_history" in session:
        return "\n".join([f"User: {entry['user_input']}\nBot: {entry['response']}" 
                          for entry in session["chat_history"]])
    return ""

# Function to handle chat queries with MongoDB persistence
async def handle_chat_query(session_id, user_input):
    # Initialize session if it doesn't already exist
    initialize_session(session_id)
    
    # Retrieve the existing chat history for context
    chat_history = get_chat_history(session_id)
    
    # Retrieve the file chunks from MongoDB
    session = collection.find_one({"session_id": session_id})
    file_chunks = session.get("file_chunks", [])
    
    # Use the file chunks as the template
    template = "\n".join(file_chunks)
    
    # Create the prompt using the template
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create a chain to process the input, using OpenAI's model and the prompt template
    chain = LLMChain(
        llm=chat,
        prompt=prompt,
        output_parser=parser
    )
    
    # Invoke the chain to get a response
    response = await chain.acall({"input": user_input})
    
    # Update MongoDB with the latest chat entry
    update_chat_history(session_id, user_input, response['text'])
    
    return response['text']

# Streamlit app
st.title("File Upload and Chatbot")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "upload"

# File upload page
if st.session_state.page == "upload":
    st.header("Upload a File")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "json"])
    if uploaded_file is not None:
        session_id = str(uuid.uuid4())
        chunks = read_and_chunk_file(uploaded_file)
        collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "file_chunks": chunks
                }
            },
            upsert=True
        )
        st.session_state.page = "chat"
        st.session_state.session_id = session_id
        st.success("File uploaded and processed successfully. You can now start chatting.")

# Chatbot page
if st.session_state.page == "chat":
    st.header("Chat with the Bot")
    
    # Audio recording
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        # Save the audio to a file
        with open("user_audio.wav", "wb") as f:
            f.write(audio_bytes)

        # Speech to text
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile("user_audio.wav") as source:
                audio_data = recognizer.record(source)
                user_input = recognizer.recognize_google(audio_data)
            
            if user_input:
                session_id = st.session_state.session_id
                response = asyncio.run(handle_chat_query(session_id, user_input))

                # Save chat history to MongoDB
                update_chat_history(session_id, user_input, response)

                # Text to speech
                tts = gTTS(text=response, lang='en')
                tts.save("response.mp3")
                audio_file = open("response.mp3", "rb").read()
                st.audio(audio_file, format="audio/mp3")

                st.text_area("Bot:", value=response, height=200)
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")