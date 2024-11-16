import pandas as pd
import logging
import re
from nltk.corpus import stopwords
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# NLTK stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

dataset_paths = {
    "labels": "data/500_Reddit_users_posts_labels.csv",
    "attempt": "data/suicidal_attempt.csv",
    "behavior": "data/suicidal_behavior.csv",
    "ideation": "data/suicidal_ideation.csv",
    "indicator": "data/suicidal_indicator.csv",
    "batch1": "data/Redditors_and_posts_batch_1.xlsx",
    "batch2": "data/Redditors_and_posts_batch_2.xlsx",
    "batch3": "data/Redditors_and_posts_batch_3.xlsx",
    "batch4": "data/Redditors_and_posts_batch_4.xlsx",
}

# Preprocessing functions
def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip().lower()

def preprocess_text_column(df, column_name):
    """Preprocess text column: clean, tokenize, and remove stopwords."""
    if column_name in df.columns:
        df[column_name] = df[column_name].fillna("")  # Replace NaN with empty string
        df[column_name] = df[column_name].apply(clean_text)  # Clean text
        df[column_name] = df[column_name].apply(lambda x: " ".join(
            word for word in x.split() if word not in stop_words))  # Remove stopwords
    return df

def load_and_prepare_data():
    data_frames = []
    for name, path in dataset_paths.items():
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".xlsx"):
                df = pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file type for {path}")

            # Add source column and handle timestamps if available
            df['source'] = name
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Preprocess text fields (customize the text column name as needed)
            text_column = 'text'  # Replace with the actual column name for text in your dataset
            if text_column in df.columns:
                df = preprocess_text_column(df, text_column)

            data_frames.append(df)
            logging.info(f"Successfully loaded and preprocessed {name}")
        except Exception as e:
            logging.error(f"Error loading {name}: {e}")

    # Combine all datasets
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Standardize column names
    combined_data.columns = [col.strip().lower().replace(" ", "_") for col in combined_data.columns]

    # Drop rows with empty text fields if they exist
    if 'text' in combined_data.columns:
        combined_data = combined_data[combined_data['text'] != ""]

    return combined_data

# Save processed data for downstream use
if __name__ == "__main__":
    combined_data = load_and_prepare_data()
    combined_data.to_csv("cleaned_combined_data.csv", index=False)
    #combined_data = load_and_prepare_data()
    #print(combined_data.head())  # Check the first few rows of the combined dataset
    logging.info("Combined and preprocessed dataset saved as cleaned_combined_data.csv.")

