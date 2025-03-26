import os
import json
import re
import nltk
import google.generativeai as genai
import firebase_admin
import streamlit as st
import pandas as pd
import torch
from nltk.corpus import stopwords
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Load Firebase credentials from Streamlit Secrets
firebase_credentials = st.secrets["FIREBASE_CREDENTIALS"]

# Convert string to JSON
cred_dict = json.loads(firebase_credentials)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Firebase (Prevent Duplicate Initialization)
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.discard("no")
stop_words.discard("not")

@st.cache_data
def load_data():
    df = pd.read_csv("labeled_data.csv")
    df = df.rename(columns={"class": "label", "tweet": "text"})
    return df

df = load_data()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_text"] = df["text"].apply(clean_text)

# Load Pre-trained BERT Model
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def classify_text_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    labels = {0: "🚨 Hate Speech", 1: "⚠️ Offensive Language", 2: "✅ Neutral"}
    return labels.get(label, "⚠️ Unclassified")

def classify_text_gemini(text):
    response = genai.GenerativeModel("gemini-2.0-pro-exp").generate_content(f"Classify this message: {text}")
    return response.text

def track_violation(username):
    doc_ref = db.collection("violations").document(username)
    doc = doc_ref.get()
    violations = doc.to_dict().get("count", 0) + 1 if doc.exists else 1
    doc_ref.set({"count": violations})
    return violations

def teachable_moment(text):
    response = genai.GenerativeModel("gemini-2.0-pro-exp").generate_content(f"Provide a teachable response to this message: {text}")
    return response.text

def apply_cooling_off(username):
    db.collection("cooling_off").document(username).set({"status": "suspended"})
    return f"User {username} has been temporarily suspended."

def community_review(username, text):
    db.collection("community_review").document(username).set({"message": text, "status": "pending"})
    return "Message submitted for community moderation."

def resolve_conflict(text):
    response = genai.GenerativeModel("gemini-2.0-pro-exp").generate_content(f"Provide a de-escalation suggestion for this message: {text}")
    return response.text

def analyze_behavior(username):
    violations = db.collection("violations").document(username).get()
    return violations.to_dict().get("count", 0) if violations.exists else 0

# Streamlit UI
st.title("🛡️ AI-Powered Community Moderation System")

st.subheader("🔍 Detect Harmful Content")
user_text = st.text_area("Enter a message to analyze:")

if st.button("Analyze Text"):
    if user_text:
        cleaned_text = clean_text(user_text)
        bert_prediction = classify_text_bert(cleaned_text)
        gemini_prediction = classify_text_gemini(cleaned_text)
        st.write(f"**BERT Model Prediction:** {bert_prediction}")
        st.write(f"**Gemini AI Prediction:** {gemini_prediction}")
        st.write(f"**Teachable Moment:** {teachable_moment(cleaned_text)}")
    else:
        st.warning("Please enter a message.")

st.subheader("📊 Track User Behavior")
behavior_user = st.text_input("Enter username to track:")

if st.button("Check Violations"):
    if behavior_user:
        count = track_violation(behavior_user)
        st.info(f"User {behavior_user} has been flagged {count} times.")
    else:
        st.warning("Please enter a username.")

st.subheader("⏳ Apply Cooling-Off Period")
cool_off_user = st.text_input("Enter username for suspension:")
if st.button("Apply Cooling-Off"):
    if cool_off_user:
        st.success(apply_cooling_off(cool_off_user))
    else:
        st.warning("Please enter a username.")

st.subheader("🗳️ Community Moderation Panel")
moderation_user = st.text_input("Enter username for review:")
moderation_text = st.text_area("Enter message for community review:")
if st.button("Submit for Review"):
    if moderation_user and moderation_text:
        st.info(community_review(moderation_user, moderation_text))
    else:
        st.warning("Please enter a username and message.")

st.subheader("🤖 Conflict Resolution Bot")
dispute_text = st.text_area("Enter message to de-escalate:")
if st.button("Resolve Conflict"):
    if dispute_text:
        st.write(f"**Resolution Suggestion:** {resolve_conflict(dispute_text)}")
    else:
        st.warning("Please enter a message.")

st.subheader("📈 Behavioral Insights")
insights_user = st.text_input("Enter username to analyze:")
if st.button("Analyze Behavior"):
    if insights_user:
        violations = analyze_behavior(insights_user)
        st.info(f"User {insights_user} has {violations} recorded violations.")
    else:
        st.warning("Please enter a username.")

st.write("🔹 **AI-powered moderation for a healthier online community.**")
