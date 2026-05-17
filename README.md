# SMS Spam Classifier

A professional and lightweight Streamlit web application designed to classify SMS messages as either spam or legitimate (ham). The application uses traditional machine learning algorithms to provide real-time, interpretable classification results.

## Features

- **Dual Model Support:** Toggle between Logistic Regression and K-Nearest Neighbors (KNN) algorithms on the fly to see how different models classify the same text.
- **Real-time Prediction:** Type or paste any text message to instantly receive a classification verdict and confidence percentage.
- **Dataset Sampling:** Built-in dropdown to test the models against a random selection of real spam and ham messages from the dataset.
- **Performance Metrics Dashboard:** View evaluation metrics (Accuracy, Precision, Recall, F1 Score) calculated on a 20% holdout test set, specific to the currently selected model.
- **Automated Data Management:** The app automatically downloads, extracts, and caches the UCI SMS Spam Collection dataset upon first launch.

## Technologies Used

- **Python 3.12**
- **Streamlit:** For the interactive web interface and UI components.
- **Scikit-Learn:** For text vectorization (`CountVectorizer`), model training (`LogisticRegression`, `KNeighborsClassifier`), and performance evaluation.
- **Pandas:** For dataset loading and manipulation.

## Installation and Usage

1. **Clone the repository or download the project files.**
2. **Install the required dependencies.** 
   It is recommended to use a virtual environment.
   ```bash
   pip install streamlit pandas scikit-learn
   ```
3. **Run the application.**
   Navigate to the directory containing `simple_app.py` and run:
   ```bash
   streamlit run simple_app.py
   ```
4. **Access the web app.**
   Open your browser and navigate to the local URL provided in your terminal (usually `http://localhost:8501`).

## How It Works

1. **Data Preprocessing:** The text data is transformed into a numerical format using `CountVectorizer`, which counts the frequency of the top 3,000 words across the dataset, filtering out common English stop words.
2. **Model Training:** 
   - **Logistic Regression:** Learns mathematical weights for specific words (e.g., assigning high importance to words like "FREE" or "urgent").
   - **KNN:** Maps messages into a high-dimensional space and classifies new messages based on the "votes" of the 5 closest messages in the training set.
3. **Classification:** User input is vectorized and passed to the selected model. The app displays the resulting classification along with the model's confidence probability.

## Dataset

The application utilizes the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from the UCI Machine Learning Repository. It contains a set of SMS tagged messages that have been collected for SMS Spam research, consisting of 5,574 messages tagged as either `ham` (legitimate) or `spam`.

*Note: The dataset is automatically downloaded to a local `/data` directory when the app is run for the first time.*
