import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (Run only once)
nltk.download("stopwords")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("labeled_data.csv")
    df = df.rename(columns={"class": "label", "tweet": "text"})
    return df

df = load_data()

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

df["cleaned_text"] = df["text"].apply(clean_text)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["label"], test_size=0.2, random_state=42)

# Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization Function
def tokenize_data(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

train_dataset = train_dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(tokenize_data, batched=True)

# Load Pretrained BERT Model
@st.cache_resource
def load_bert():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    return model

model = load_bert()

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train Model (Run only once, then save & load)
@st.cache_resource
def train_model():
    trainer.train()
    trainer.save_model("./bert_model")
    return model

model = train_model()

# Load Trained Model for Inference
@st.cache_resource
def load_trained_model():
    return BertForSequenceClassification.from_pretrained("./bert_model")

model = load_trained_model()

# Define Function for Predictions
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()
    labels = {0: "🚨 Hate Speech", 1: "⚠️ Offensive Language", 2: "✅ Neutral"}
    return labels[prediction]

# Streamlit UI
st.title("🛡️ AI-Powered Moderation System (BERT)")
st.write(f"**Model:** Fine-Tuned BERT (`bert-base-uncased`)")

# 🔍 **Detect Harmful Content**
st.subheader("🔍 Detect Harmful Content")
user_text = st.text_area("Enter a message to analyze:")
if st.button("Analyze Text"):
    if user_text:
        st.write(f"**Detected:** {predict_label(user_text)}")
    else:
        st.warning("Please enter a message.")

# 💡 **Teachable Moments**
st.subheader("💡 Teachable Moments")
user_input = st.text_input("Enter a message for suggestion:")
if st.button("Suggest Alternative"):
    replacements = {"stupid": "misinformed", "idiot": "uninformed", "dumb": "lacking perspective"}
    words = user_input.split()
    words = [replacements.get(word.lower(), word) for word in words]
    st.write("**Suggested Alternative:**", " ".join(words))

# ⏳ **Cooling-Off Periods**
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

# 📊 **Track User Behavior**
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

st.write("**🔹 AI-powered moderation with BERT for high accuracy!**")

st.code("""
pip install streamlit pandas numpy transformers torch datasets scikit-learn nltk
streamlit run project.py
""")
