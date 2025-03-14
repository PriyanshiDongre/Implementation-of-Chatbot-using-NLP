import os
import json
import datetime
import sqlite3
import pickle
import uuid
import random
import streamlit as st
import nltk
import ssl
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from googletrans import Translator
from gtts import gTTS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Fix SSL issue for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents.json
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Train Model
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags, patterns = [], []
if isinstance(intents, dict) and "intents" in intents:
    for intent in intents["intents"]:
        if "patterns" in intent and "tag" in intent:
            for pattern in intent["patterns"]:
                tags.append(intent["tag"])
                patterns.append(pattern)

if not patterns:
    raise ValueError("Error: No patterns found! Check 'intents.json' format.")

X_counts = vectorizer.fit_transform(patterns)
X_tfidf = transformer.fit_transform(X_counts)
clf.fit(X_tfidf, tags)

# Setup Database for Learning
conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS learned_responses (id INTEGER PRIMARY KEY AUTOINCREMENT, user_input TEXT UNIQUE, chatbot_response TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS chat_log (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, user_input TEXT, chatbot_response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# Initialize Tools
translator = Translator()
analyzer = SentimentIntensityAnalyzer()

# Store conversation in database
def store_conversation(session_id, user_input, response):
    cursor.execute("INSERT INTO chat_log (session_id, user_input, chatbot_response) VALUES (?, ?, ?)",
                   (session_id, user_input, response))
    conn.commit()

# Retrieve past responses
def get_past_response(user_input):
    cursor.execute("SELECT chatbot_response FROM learned_responses WHERE user_input = ?", (user_input,))
    result = cursor.fetchone()
    return result[0] if result else None

# Multilingual Translation
def translate_text(text, src_lang="auto", dest_lang="en"):
    return translator.translate(text, src=src_lang, dest=dest_lang).text

# Sentiment Analysis
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    return "positive" if score["compound"] >= 0.05 else "negative" if score["compound"] <= -0.05 else "neutral"

# Text-to-Speech (TTS) (Only on Demand)
def speak_response(response, lang="en"):
    tts = gTTS(text=response, lang=lang, slow=False)
    tts.save("response.mp3")
    os.system("start response.mp3")  # Windows

# Self-Learning: Train on New Data
def train_on_new_data():
    cursor.execute("SELECT user_input, chatbot_response FROM learned_responses")
    new_data = cursor.fetchall()
    if new_data:
        new_patterns = [item[0] for item in new_data]
        new_tags = [item[1] for item in new_data]
        X_new_counts = vectorizer.fit_transform(new_patterns)
        X_new_tfidf = transformer.fit_transform(X_new_counts)
        clf.fit(X_new_tfidf, new_tags)

# Convert Voice to Text
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand. Please try again."
        except sr.RequestError:
            return "Sorry, there was an issue with the speech recognition service."

# Modify Chatbot Function for Accuracy & ChatGPT-Style Chat
def chatbot(input_text):
    session_id = str(uuid.uuid4())  # Create unique session ID for each conversation

    # Detect language and translate to English if needed
    detected_lang = translator.detect(input_text).lang
    if detected_lang != "en":
        input_text_translated = translate_text(input_text, src_lang=detected_lang, dest_lang="en")
    else:
        input_text_translated = input_text

    # Check if response exists in past conversations
    past_response = get_past_response(input_text_translated)
    if past_response:
        return past_response  # Return stored response if available

    # Use trained ML model to predict intent
    try:
        input_text_vectorized = transformer.transform(vectorizer.transform([input_text_translated]))
        tag = clf.predict(input_text_vectorized)[0]
    except:
        return "I didn't understand that. Can you try again?"

    # Get response from `intents.json`
    response = None
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            break  # Stop searching after finding the correct tag

    # If no response is found, return a default message
    if not response:
        return "Sorry, I don't have an answer for that."

    # Store chat in conversation history
    store_conversation(session_id, input_text, response)

    return response  # Return final chatbot response



# Main Function with Menu Options
def main():
    st.title("Welcome to Chatbot")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        # Create Input Box and Voice Input Button at the Bottom
        input_container = st.empty()  # Keeps the input box at the bottom

        col1, col2 = st.columns([4, 1])  # Layout: Text input + Voice button

        with col1:
            user_input = input_container.text_input("Type your message here...", value="", key="text_input")

        with col2:
            if st.button("🎤 Voice Input"):
                user_input = voice_input()  # Get voice input only when button is clicked

        if user_input:  # Only process input if it's new
            response = chatbot(user_input)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", response))

            # Clear input field after submission
            st.session_state.user_input = ""

        # Display chat history
        for index, (role, text) in enumerate(st.session_state.chat_history):
            with st.chat_message(role):
                st.write(text)
                if role == "bot":
                    if st.button("🔊 Speak", key=f"tts_{index}"):
                        speak_response(text)

    elif choice == "Conversation History":
        st.header("📜 Conversation History")
        with st.expander("Click to view past conversations"):
            cursor.execute("SELECT user_input, chatbot_response, timestamp FROM chat_log ORDER BY timestamp DESC LIMIT 10")
            chat_logs = cursor.fetchall()
            for log in chat_logs:
                st.text(f"🗣️ {log[0]}")
                st.text(f"🤖 {log[1]}")
                st.text(f"📅 {log[2]}")
                st.markdown("---")

    elif choice == "About":
        st.subheader("About This Chatbot")
        st.write("This chatbot uses NLP and machine learning to provide intelligent responses.")
        st.write("- Stores and retrieves past conversations")
        st.write("- Supports multiple languages with translation")
        st.write("- Sentiment analysis for emotional responses")
        st.write("- Text-to-speech and voice input features")

if __name__ == "__main__":
    main()
