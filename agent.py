import json
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from openCHA.orchestrator import Orchestrator
from openCHA.tasks.task import BaseTask
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_journal_entry
from sentiment_analysis import preprocess_for_insights
import pandas as pd

# Setup logging for debugging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the dataset for retrieval
dataset_path = "/mnt/data/cleaned_combined_data.csv"
if os.path.exists(dataset_path):
    combined_data = pd.read_csv(dataset_path)
else:
    raise FileNotFoundError("Dataset not found. Ensure the dataset is preprocessed and available.")

# Initialize SentenceTransformer for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for dataset
logging.info("Generating embeddings for the dataset...")
combined_data['embedding'] = combined_data['cleaned_text'].apply(lambda x: embedding_model.encode(str(x)))

# Function for retrieving similar entries
def retrieve_similar_entries(query, top_k=5):
    query_embedding = embedding_model.encode(query)
    similarities = combined_data['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    combined_data['similarity'] = similarities
    return combined_data.nlargest(top_k, 'similarity')[['cleaned_text', 'similarity']].to_dict(orient='records')

# Daily Preprocessing Task
class DailyPreprocessingTask(BaseTask):
    def __init__(self, datapipe=None):
        super().__init__(
            name="daily_preprocessing",
            chat_name="DailyPreprocessing",
            description="Preprocess journal entries to extract key insights.",
            inputs=["Raw journal entry text."],
            outputs=["Preprocessed journal entry with lemmatization, tokens, and named entities."],
            datapipe=datapipe,
            output_type=False
        )

    def _execute(self, inputs):
        text = inputs[0]
        preprocessed_data = preprocess_journal_entry(text)
        return json.dumps(preprocessed_data)

    def explain(self):
        return "Preprocesses journal entries to extract tokens, lemmatized text, and named entities."

# Retrieval Task
class RetrievalTask(BaseTask):
    def __init__(self, datapipe=None):
        super().__init__(
            name="retrieval_task",
            chat_name="Retrieval",
            description="Retrieve relevant past journal entries based on query.",
            inputs=["Query text."],
            outputs=["Top-k similar past journal entries."],
            datapipe=datapipe,
            output_type=False
        )

    def _execute(self, inputs):
        query = inputs[0]
        similar_entries = retrieve_similar_entries(query)
        return json.dumps(similar_entries)

    def explain(self):
        return "Retrieves top-k similar past journal entries based on contextual similarity."

# Sentiment Analysis Task
class SentimentAnalysisTask(BaseTask):
    def __init__(self, datapipe=None):
        super().__init__(
            name="sentiment_analysis",
            chat_name="SentimentAnalysis",
            description="Analyze sentiment, emotions, and key phrases of journal entries.",
            inputs=["Preprocessed journal entry text."],
            outputs=["Sentiment analysis and key insights for emotions and topics."],
            datapipe=datapipe,
            output_type=False
        )

    def _execute(self, inputs):
        text = inputs[0]
        sentiment_data = preprocess_for_insights(text)
        return json.dumps(sentiment_data)

    def explain(self):
        return "Analyzes sentiment, emotions, and key phrases for deeper journal entry insights."

# Future Mental Health Prediction Task
class FutureMentalHealthPredictionTask(BaseTask):
    def __init__(self, datapipe=None):
        super().__init__(
            name="future_mental_health_prediction",
            chat_name="FuturePrediction",
            description="Predict future sentiment scores based on past trends using regression.",
            inputs=["Historical journal data with timestamps and sentiment scores."],
            outputs=["Predicted future mental health state."],
            datapipe=datapipe,
            output_type=False
        )

    def _execute(self, inputs):
        historical_data = json.loads(inputs[0])
        
        # Prepare data
        timestamps = [datetime.fromisoformat(entry['timestamp']).timestamp() for entry in historical_data]
        sentiment_scores = [entry['sentiment_score'] for entry in historical_data]

        # Train Linear Regression Model
        X = np.array(timestamps).reshape(-1, 1)
        y = np.array(sentiment_scores)
        model = LinearRegression()
        model.fit(X, y)

        # Predict future mental health state for the next 7 days
        future_timestamps = [(datetime.now() + timedelta(days=i)).timestamp() for i in range(7)]
        future_predictions = model.predict(np.array(future_timestamps).reshape(-1, 1))

        # Prepare results
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        predictions = [{"date": date, "predicted_sentiment_score": round(score, 2)} for date, score in zip(future_dates, future_predictions)]

        return json.dumps(predictions)

    def explain(self):
        return "Predicts future mental health state based on past trends using regression."

# Recommendation Task
class RecommendationTask(BaseTask):
    def __init__(self, datapipe=None):
        super().__init__(
            name="generate_recommendations",
            chat_name="Recommendations",
            description="Generate personalized recommendations based on journal insights and retrievals.",
            inputs=["Sentiment analysis and retrieved context."],
            outputs=["Actionable recommendations based on insights and historical data."],
            datapipe=datapipe,
            output_type=False
        )

    def _execute(self, inputs):
        sentiment_analysis = json.loads(inputs[0])
        retrieved_context = json.loads(inputs[1])
        recommendations = []

        # Example: Provide a recommendation based on negative sentiment
        if sentiment_analysis["overall_sentiment"]["label"] == "NEGATIVE":
            recommendations.append({
                "recommendation": "Practice mindfulness meditation for 10 minutes daily.",
                "justification": "Recurring negative sentiment detected in your entries."
            })

        if any("work" in entry["cleaned_text"] for entry in retrieved_context):
            recommendations.append({
                "recommendation": "Take regular breaks during work to reduce stress.",
                "justification": "Historical data shows frequent mentions of 'work' related stress."
            })

        return json.dumps(recommendations)

    def explain(self):
        return "Generates actionable recommendations from sentiment and historical context."

# Orchestrator Setup
orchestrator = Orchestrator.initialize(
    planner_llm="gpt-4",
    planner_name="tree_of_thought",
    datapipe_name="memory",
    response_generator_name="base_generator",
    available_tasks=[
        DailyPreprocessingTask,
        RetrievalTask,
        SentimentAnalysisTask,
        FutureMentalHealthPredictionTask,
        RecommendationTask,
    ],
)

# Example Journal Entry
journal_entry = """
Today was a challenging day. Morning meetings were intense, and deadlines piled up quickly. 
I felt overwhelmed by the workload and ended up arguing with a colleague. 
Later, I went for a walk, which helped ease my frustration. Tomorrow, I’ll try to organize my tasks better.
"""

# Task Execution
preprocessed_data = orchestrator.execute_task("daily_preprocessing", [journal_entry])
retrieved_context = orchestrator.execute_task("retrieval_task", [json.loads(preprocessed_data)['cleaned_text']])
sentiment_analysis = orchestrator.execute_task("sentiment_analysis", [json.loads(preprocessed_data)['cleaned_text']])
future_predictions = orchestrator.execute_task("future_mental_health_prediction", [json.dumps([
    {"timestamp": "2024-11-14T09:30:00", "sentiment_score": 0.75},
    {"timestamp": "2024-11-15T14:45:00", "sentiment_score": 0.60},
    {"timestamp": "2024-11-16T12:20:00", "sentiment_score": 0.55},
])])
recommendations = orchestrator.execute_task("generate_recommendations", [sentiment_analysis, retrieved_context])

# Outputs
print("Preprocessed Data:", preprocessed_data)
print("Retrieved Context:", retrieved_context)
print("Sentiment Analysis:", sentiment_analysis)
print("Future Predictions:", future_predictions)
print("Recommendations:", recommendations)
