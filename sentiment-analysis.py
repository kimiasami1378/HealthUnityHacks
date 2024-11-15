from transformers import pipeline
import nltk
from nltk.corpus import wordnet
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load spaCy model for NER (ensure it's defined globally)
nlp = spacy.load("en_core_web_sm")

# Sample emotion lexicon (simplified for demo purposes)
emotion_lexicon = {
    'happy': 'joy',
    'sad': 'sadness',
    'angry': 'anger',
    'fearful': 'fear',
    # Add more words with associated emotions as needed
}

def perform_sentiment_analysis(text):
    sentiment_result = sentiment_pipeline(text)
    return sentiment_result[0]  # Return label and score

def tag_emotions(text):
    words = text.split()
    emotion_tags = {word: emotion_lexicon[word] for word in words if word in emotion_lexicon}
    return emotion_tags

def assign_intensity_scores(emotion_tags):
    intensity_scores = {}
    for word, emotion in emotion_tags.items():
        # Example scoring: Assign a base score of 1.0 and adjust based on word importance
        intensity_scores[word] = 1.0  # Customize scoring logic as needed
    return intensity_scores

def extract_key_phrases(text):
    doc = nlp(text)
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    return key_phrases

def topic_modeling(text, num_topics=2):
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    topics = lda.transform(dtm)
    topic_words = [vectorizer.get_feature_names_out()[index] for index in topics[0].argsort()[-5:]]
    return topic_words

def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

import re

def simple_sentence_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text)

def segment_entry_by_sentiment(text):
    sentences = simple_sentence_tokenize(text)
    positive_sentences = []
    negative_sentences = []

    for sentence in sentences:
        sentiment = perform_sentiment_analysis(sentence)
        if sentiment['label'] == 'POSITIVE':
            positive_sentences.append((sentence, sentiment['score']))
        else:
            negative_sentences.append((sentence, sentiment['score']))

    return {
        "positive_segments": positive_sentences,
        "negative_segments": negative_sentences
    }

def preprocess_for_insights(text):
    # Step 1: Sentiment Analysis
    sentiment_result = perform_sentiment_analysis(text)
    
    # Step 2: Emotion Tagging and Intensity Scoring
    emotion_tags = tag_emotions(text)
    intensity_scores = assign_intensity_scores(emotion_tags)
    
    # Step 3: Keyphrase and Topic Extraction
    key_phrases = extract_key_phrases(text)
    topics = topic_modeling(text)
    
    # Step 4: Named Entity Recognition
    named_entities = extract_named_entities(text)
    
    # Step 5: Sentiment Segmentation
    sentiment_segments = segment_entry_by_sentiment(text)
    
    # Compile all preprocessing outputs
    preprocessed_data = {
        "overall_sentiment": sentiment_result,
        "emotion_tags": emotion_tags,
        "intensity_scores": intensity_scores,
        "key_phrases": key_phrases,
        "topics": topics,
        "named_entities": named_entities,
        "sentiment_segments": sentiment_segments
    }
    
    return preprocessed_data

# Example usage
journal_entry = "Today was a mixed bag of emotions. The morning started off peaceful; I enjoyed my coffee while watching the sunrise, a rare moment of calm. But as the day unfolded, work stress built up—endless meetings and tight deadlines left me feeling overwhelmed. I had a brief argument with a colleague, which left me feeling frustrated and questioning my approach. By evening, I took a walk to clear my mind, which helped ease some of the tension. I’m grateful for that small reset. Tomorrow, I hope to manage my time better and approach conversations with more patience and understanding"

from preprocess import preprocess_journal_entry

pp = preprocess_journal_entry(journal_entry)

print(pp['lemmatized_text'])
#preprocessed_result = preprocess_for_insights(journal_entry)



# Print the output
#print("Preprocessed Data:", preprocessed_result)