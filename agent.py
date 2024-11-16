import json
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_journal_entry
from sentiment_analysis import preprocess_for_insights
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

# Import necessary classes from openCHA
from openCHA.orchestrator.orchestrator import Orchestrator as BaseOrchestrator
from openCHA.tasks.task import BaseTask

# Setup logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the dataset
dataset_path = "/workspaces/HealthUnityHacks/CHA/cleaned_combined_data.csv"
if os.path.exists(dataset_path):
    combined_data = pd.read_csv(dataset_path, low_memory=False)
else:
    raise FileNotFoundError("Dataset not found. Ensure the dataset is preprocessed and available.")

# Ensure the 'post' column exists and handle missing values
if 'post' not in combined_data.columns:
    raise KeyError("'post' column not found in the dataset. Please check the dataset structure.")
combined_data['post'] = combined_data['post'].fillna("")

# Initialize SentenceTransformer for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the 'post' column
logging.info("Generating embeddings for the dataset...")
combined_data['embedding'] = combined_data['post'].apply(lambda x: embedding_model.encode(str(x)).tolist())

# Define a valid DataPipe class
class ValidDataPipe:
    storage: Dict[str, Any] = {}

    def store(self, key: str, value: Any):
        self.storage[key] = value

    def retrieve(self, key: str) -> Any:
        return self.storage.get(key, None)

# Instantiate a valid datapipe instance
datapipe = ValidDataPipe()

# Function for retrieving similar entries
def retrieve_similar_entries(query, top_k=5):
    query_embedding = embedding_model.encode(query)
    combined_data['similarity'] = combined_data['embedding'].apply(
        lambda x: cosine_similarity([query_embedding], [np.array(x)])[0][0]
    )
    return combined_data.nlargest(top_k, 'similarity')[['post', 'similarity']].to_dict(orient='records')

# Define your TaskType enum with only custom tasks
class TaskType(str, Enum):
    DAILY_PREPROCESSING = "daily_preprocessing"
    RETRIEVAL_TASK = "retrieval_task"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    FUTURE_MENTAL_HEALTH_PREDICTION = "future_mental_health_prediction"
    GENERATE_RECOMMENDATIONS = "generate_recommendations"

# Define your custom tasks without overriding __init__
class DailyPreprocessingTask(BaseTask):
    name: str = "daily_preprocessing"
    chat_name: str = "DailyPreprocessing"
    description: str = "Preprocess journal entries to extract key insights."
    inputs: List[str] = ["Raw journal entry text."]
    outputs: List[str] = ["Preprocessed journal entry with lemmatization, tokens, and named entities."]
    datapipe: Optional[Any] = None  # Will be set during initialization
    output_type: bool = False

    def _execute(self, inputs):
        text = inputs[0]
        preprocessed_data = preprocess_journal_entry(text)
        return json.dumps(preprocessed_data)

    def explain(self):
        return "Preprocesses journal entries to extract tokens, lemmatized text, and named entities."

class RetrievalTask(BaseTask):
    name: str = "retrieval_task"
    chat_name: str = "Retrieval"
    description: str = "Retrieve relevant past journal entries based on query."
    inputs: List[str] = ["Query text."]
    outputs: List[str] = ["Top-k similar past journal entries."]
    datapipe: Optional[Any] = None
    output_type: bool = False

    def _execute(self, inputs):
        query = inputs[0]
        similar_entries = retrieve_similar_entries(query)
        return json.dumps(similar_entries)

    def explain(self):
        return "Retrieves top-k similar past journal entries based on contextual similarity."

class SentimentAnalysisTask(BaseTask):
    name: str = "sentiment_analysis"
    chat_name: str = "SentimentAnalysis"
    description: str = "Analyze sentiment, emotions, and key phrases of journal entries."
    inputs: List[str] = ["Preprocessed journal entry text."]
    outputs: List[str] = ["Sentiment analysis and key insights for emotions and topics."]
    datapipe: Optional[Any] = None
    output_type: bool = False

    def _execute(self, inputs):
        text = inputs[0]
        sentiment_data = preprocess_for_insights(text)
        return json.dumps(sentiment_data)

    def explain(self):
        return "Analyzes sentiment, emotions, and key phrases for deeper journal entry insights."

class FutureMentalHealthPredictionTask(BaseTask):
    name: str = "future_mental_health_prediction"
    chat_name: str = "FuturePrediction"
    description: str = "Predict future sentiment scores based on past trends using regression."
    inputs: List[str] = ["Historical journal data with timestamps and sentiment scores."]
    outputs: List[str] = ["Predicted future mental health state."]
    datapipe: Optional[Any] = None
    output_type: bool = False

    def _execute(self, inputs):
        historical_data = json.loads(inputs[0])
        timestamps = [datetime.fromisoformat(entry['timestamp']).timestamp() for entry in historical_data]
        sentiment_scores = [entry['sentiment_score'] for entry in historical_data]

        X = np.array(timestamps).reshape(-1, 1)
        y = np.array(sentiment_scores)
        model = LinearRegression()
        model.fit(X, y)

        future_timestamps = [(datetime.now() + timedelta(days=i)).timestamp() for i in range(7)]
        future_predictions = model.predict(np.array(future_timestamps).reshape(-1, 1))
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        predictions = [{"date": date, "predicted_sentiment_score": round(score, 2)} for date, score in zip(future_dates, future_predictions)]

        return json.dumps(predictions)

    def explain(self):
        return "Predicts future mental health state based on past trends using regression."

class RecommendationTask(BaseTask):
    name: str = "generate_recommendations"
    chat_name: str = "Recommendations"
    description: str = "Generate personalized recommendations based on journal insights and retrievals."
    inputs: List[str] = ["Sentiment analysis", "Retrieved context."]
    outputs: List[str] = ["Actionable recommendations based on insights and historical data."]
    datapipe: Optional[Any] = None
    output_type: bool = False

    def _execute(self, inputs):
        sentiment_analysis = json.loads(inputs[0])
        retrieved_context = json.loads(inputs[1])
        recommendations = []

        if sentiment_analysis.get("overall_sentiment", {}).get("label", "") == "NEGATIVE":
            recommendations.append({
                "recommendation": "Practice mindfulness meditation for 10 minutes daily.",
                "justification": "Recurring negative sentiment detected in your entries."
            })

        if any("work" in entry["post"] for entry in retrieved_context):
            recommendations.append({
                "recommendation": "Take regular breaks during work to reduce stress.",
                "justification": "Historical data shows frequent mentions of 'work' related stress."
            })

        return json.dumps(recommendations)

    def explain(self):
        return "Generates actionable recommendations from sentiment and historical context."

# Define TASK_TO_CLASS with only your custom tasks
TASK_TO_CLASS = {
    TaskType.DAILY_PREPROCESSING: DailyPreprocessingTask,
    TaskType.RETRIEVAL_TASK: RetrievalTask,
    TaskType.SENTIMENT_ANALYSIS: SentimentAnalysisTask,
    TaskType.FUTURE_MENTAL_HEALTH_PREDICTION: FutureMentalHealthPredictionTask,
    TaskType.GENERATE_RECOMMENDATIONS: RecommendationTask,
}

# Define the initialize_task function
def initialize_task(task_name: str, datapipe=None, **kwargs) -> BaseTask:
    try:
        task_type = TaskType(task_name)
    except ValueError:
        raise ValueError(f"Unknown task: {task_name}")
    task_class = TASK_TO_CLASS.get(task_type)
    if task_class is None:
        raise ValueError(f"Task class not found for task: {task_name}")
    return task_class(datapipe=datapipe, **kwargs)

from openCHA.orchestrator.orchestrator import Orchestrator as BaseOrchestrator

class CustomOrchestrator(BaseOrchestrator):
    @classmethod
    def initialize(cls, **kwargs):
        # Extract and remove 'datapipe' from kwargs
        datapipe = kwargs.pop('datapipe', None)
        # Extract and remove 'available_tasks' from kwargs
        available_tasks = kwargs.pop('available_tasks', [])
        # Initialize your custom tasks
        tasks = {}
        for task_name in available_tasks:
            tasks[task_name] = initialize_task(task_name, datapipe=datapipe)
        # Remove 'tasks' from kwargs if it exists to avoid duplication
        kwargs.pop('tasks', None)
        # Call the base class initializer, passing 'tasks' explicitly
        return super().initialize(tasks=tasks, **kwargs)

# Initialize the Orchestrator
orchestrator = CustomOrchestrator.initialize(
    planner_llm="gpt-4",
    planner_name="tree_of_thought",
    datapipe=datapipe,  # Pass your instantiated datapipe
    response_generator_name="base_generator",
    available_tasks=[task.value for task in TaskType],
)


# Example Journal Entry
journal_entry = """
Today was a challenging day. Morning meetings were intense, and deadlines piled up quickly. 
I felt overwhelmed by the workload and ended up arguing with a colleague. 
Later, I went for a walk, which helped ease my frustration. Tomorrow, I’ll try to organize my tasks better.
"""

# Task Execution
preprocessed_data = orchestrator.execute_task(TaskType.DAILY_PREPROCESSING.value, [journal_entry])
retrieved_context = orchestrator.execute_task(TaskType.RETRIEVAL_TASK.value, [journal_entry])
sentiment_analysis = orchestrator.execute_task(TaskType.SENTIMENT_ANALYSIS.value, [journal_entry])
future_predictions = orchestrator.execute_task(
    TaskType.FUTURE_MENTAL_HEALTH_PREDICTION.value, [json.dumps([
        {"timestamp": "2024-11-14T09:30:00", "sentiment_score": 0.75},
        {"timestamp": "2024-11-15T14:45:00", "sentiment_score": 0.60},
        {"timestamp": "2024-11-16T12:20:00", "sentiment_score": 0.55},
    ])]
)
recommendations = orchestrator.execute_task(
    TaskType.GENERATE_RECOMMENDATIONS.value, [sentiment_analysis, retrieved_context]
)

# Outputs
print("Preprocessed Data:", preprocessed_data)
print("Retrieved Context:", retrieved_context)
print("Sentiment Analysis:", sentiment_analysis)
print("Future Predictions:", future_predictions)
print("Recommendations:", recommendations)

