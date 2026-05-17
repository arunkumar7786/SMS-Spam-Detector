import streamlit as st
import pandas as pd
import os
import urllib.request
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Page config
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

# Data constants
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
TSV_PATH = os.path.join(DATA_DIR, "SMSSpamCollection")

def download_data():
    """Downloads and extracts the SMS Spam dataset if not already present."""
    if os.path.exists(TSV_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

@st.cache_resource(show_spinner="Training models...")
def load_models():
    """Loads data, trains Logistic Regression and KNN models using CountVectorizer."""
    download_data()
    
    # Load dataset
    df = pd.read_csv(TSV_PATH, sep="\t", header=None, names=["label", "message"])
    df["Class"] = (df["label"] == "spam").astype(int)

    # Split data into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["Class"], test_size=0.2, random_state=42
    )
    
    # Text to numeric features using CountVectorizer
    vectorizer = CountVectorizer(max_features=3000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression Model
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_vec, y_train)
    log_pred = log_model.predict(X_test_vec)
    
    log_metrics = {
        "Accuracy": accuracy_score(y_test, log_pred),
        "Precision": precision_score(y_test, log_pred),
        "Recall": recall_score(y_test, log_pred),
        "F1 Score": f1_score(y_test, log_pred)
    }

    # Train KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_vec, y_train)
    knn_pred = knn_model.predict(X_test_vec)

    knn_metrics = {
        "Accuracy": accuracy_score(y_test, knn_pred),
        "Precision": precision_score(y_test, knn_pred),
        "Recall": recall_score(y_test, knn_pred),
        "F1 Score": f1_score(y_test, knn_pred)
    }

    return vectorizer, log_model, knn_model, log_metrics, knn_metrics, df

@st.cache_data(show_spinner=False)
def get_samples(_df, n_each=10, seed=7):
    spam = _df[_df["label"] == "spam"].sample(n_each, random_state=seed)
    ham  = _df[_df["label"] == "ham"].sample(n_each, random_state=seed)
    combined = pd.concat([spam, ham]).sample(frac=1, random_state=seed)

    options = [("— Type your own message —", "")]
    for _, row in combined.iterrows():
        tag = "[Spam]" if row["label"] == "spam" else "[Ham]"
        preview = row["message"][:60] + ("..." if len(row["message"]) > 60 else "")
        options.append((f"{tag} {preview}", row["message"]))
    return options

# Initialize models
vectorizer, log_model, knn_model, log_metrics, knn_metrics, df = load_models()
samples = get_samples(df)
sample_labels = [s[0] for s in samples]
sample_messages = [s[1] for s in samples]

if "message_text" not in st.session_state:
    st.session_state.message_text = ""

def on_sample_change():
    idx = sample_labels.index(st.session_state.sample_select)
    st.session_state.message_text = sample_messages[idx]

# App UI
st.title("SMS Spam Classification")
st.write("A simple, professional tool to classify text messages as spam or legitimate (ham).")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_choice = st.sidebar.radio("Select Model", ["Logistic Regression", "K-Nearest Neighbors (KNN)"])

# Main Input Section
st.subheader("Input Message")

st.selectbox(
    "Try a sample from the dataset",
    options=sample_labels,
    index=0,
    key="sample_select",
    on_change=on_sample_change,
)

user_input = st.text_area("Enter the SMS text below to classify it:", value=st.session_state.message_text, height=150)

# Classification
if st.button("Classify Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Transform input
        vec_input = vectorizer.transform([user_input])
        
        # Predict based on selected model
        if model_choice == "Logistic Regression":
            prediction = log_model.predict(vec_input)[0]
            prob = log_model.predict_proba(vec_input)[0]
        else:
            prediction = knn_model.predict(vec_input)[0]
            prob = knn_model.predict_proba(vec_input)[0]
            
        confidence = prob[prediction] * 100
        
        # Display Result
        st.subheader("Result")
        if prediction == 1:
            st.error(f"Classification: **SPAM** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"Classification: **NOT SPAM (HAM)** (Confidence: {confidence:.2f}%)")

# Metrics Section
st.markdown("---")
st.subheader("Model Performance")
st.write(f"Metrics evaluated on a 20% test split of the UCI SMS Spam Collection dataset using **{model_choice}**.")

# Display metrics based on selected model
metrics = log_metrics if model_choice == "Logistic Regression" else knn_metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
col2.metric("Precision", f"{metrics['Precision']:.2%}")
col3.metric("Recall", f"{metrics['Recall']:.2%}")
col4.metric("F1 Score", f"{metrics['F1 Score']:.2%}")
