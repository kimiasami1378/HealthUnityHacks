import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


dataset_paths = {
    "labels": "/mnt/data/500_Reddit_users_posts_labels.csv",
    "attempt": "/mnt/data/suicidal_attempt.csv",
    "behavior": "/mnt/data/suicidal_behavior.csv",
    "ideation": "/mnt/data/suicidal_ideation.csv",
    "indicator": "/mnt/data/suicidal_indicator.csv",
    "batch1": "/mnt/data/Redditors_and_posts_batch_1.xlsx",
    "batch2": "/mnt/data/Redditors_and_posts_batch_2.xlsx",
    "batch3": "/mnt/data/Redditors_and_posts_batch_3.xlsx",
    "batch4": "/mnt/data/Redditors_and_posts_batch_4.xlsx",
}

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

            data_frames.append(df)
            logging.info(f"Successfully loaded {name}")
        except Exception as e:
            logging.error(f"Error loading {name}: {e}")

    # Combine all datasets
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Standardize column names
    combined_data.columns = [col.strip().lower().replace(" ", "_") for col in combined_data.columns]

    return combined_data
