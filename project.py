import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download("stopwords")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("labeled_data.csv")  # Ensure dataset is in the same directory
    df = df.rename(columns={"class": "label", "tweet": "text"})
    return df

df = load_data()

# Data Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Apply Cleaning
df["cleaned_text"] = df["text"].apply(clean_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["label"], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Model Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("🛡️ AI-Powered Community Moderation System")
st.write(f"**Model Accuracy:** {accuracy:.2%}")

# Detect Harmful Content
st.subheader("🔍 Detect Harmful Content")
user_text = st.text_area("Enter a message to analyze:")
if st.button("Analyze Text"):
    if user_text:
        text_cleaned = clean_text(user_text)
        text_vectorized = vectorizer.transform([text_cleaned])
        prediction = model.predict(text_vectorized)[0]
        labels = {0: "🚨 Hate Speech", 1: "⚠️ Offensive Language", 2: "✅ Neutral"}
        st.write(f"**Detected:** {labels[prediction]}")
    else:
        st.warning("Please enter a message.")

# Teachable Moments
st.subheader("💡 Teachable Moments")
user_input = st.text_input("Enter a message for suggestion:")
if st.button("Suggest Alternative"):
    replacements = {
        "stupid": "misinformed",
        "idiot": "uninformed",
        "dumb": "lacking perspective"
    }
    words = user_input.split()
    for i, word in enumerate(words):
        if word.lower() in replacements:
            words[i] = replacements[word.lower()]
    st.write("**Suggested Alternative:**", " ".join(words))

# Cooling-Off Periods
st.subheader("⏳ Apply Cooling-Off Period")
user = st.text_input("Enter username for suspension:")
cool_off_users = {}
if st.button("Apply Cooling-Off"):
    if user:
        if user in cool_off_users:
            st.warning(f"User {user} is already suspended.")
        else:
            cool_off_users[user] = "suspended"
            st.success(f"User {user} has been temporarily suspended.")
    else:
        st.warning("Please enter a username.")

# Community Moderation Panel
st.subheader("🗳️ Community Moderation Panel")
moderation_user = st.text_input("Enter username for review:")
moderation_text = st.text_area("Enter message for community review:")
community_votes = {}

if st.button("Submit for Review"):
    if moderation_user and moderation_text:
        if moderation_user not in community_votes:
            community_votes[moderation_user] = []
        community_votes[moderation_user].append(moderation_text)
        
        if len(community_votes[moderation_user]) >= 3:
            st.warning(f"Community voted to remove {moderation_user}'s message.")
        else:
            st.info(f"Message submitted for community review. Votes: {len(community_votes[moderation_user])}/3")
    else:
        st.warning("Please enter a username and message.")

# Behavioral Insights
st.subheader("📊 Track User Behavior")
behavior_user = st.text_input("Enter username to track:")
user_behavior = {}

if st.button("Check Violations"):
    if behavior_user:
        if behavior_user not in user_behavior:
            user_behavior[behavior_user] = 0
        user_behavior[behavior_user] += 1
        st.info(f"User {behavior_user} has been flagged {user_behavior[behavior_user]} times.")
    else:
        st.warning("Please enter a username.")

st.write("**🔹 AI-powered moderation for a healthier online community.**")


st.code("""
pip install streamlit pandas numpy scikit-learn nltk
streamlit run project.py
""")
