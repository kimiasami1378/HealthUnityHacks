import re
import nltk
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import wordnet, stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load spaCy model for Named Entity Recognition (NER) and key phrase extraction
nlp = spacy.load("en_core_web_sm")

# Simplified emotion lexicon for tagging emotions
emotion_lexicon = {
    'happy': 'joy',
    'sad': 'sadness',
    'angry': 'anger',
    'fearful': 'fear',
    'anxious': 'anxiety',
    'grateful': 'gratitude',
    # Add more terms as needed
}

# Function: Perform Sentiment Analysis
def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis using the Hugging Face pipeline.
    Returns sentiment label and confidence score.
    """
    sentiment_result = sentiment_pipeline(text)
    return sentiment_result[0]  # Returns a dictionary with 'label' and 'score'

# Function: Tag Emotions
def tag_emotions(text):
    """
    Tag words in the text with associated emotions based on the emotion lexicon.
    """
    words = text.split()
    emotion_tags = {word: emotion_lexicon[word] for word in words if word in emotion_lexicon}
    return emotion_tags

# Function: Assign Intensity Scores to Emotions
def assign_intensity_scores(emotion_tags):
    """
    Assign intensity scores to emotions. 
    Scores can be customized based on contextual importance.
    """
    intensity_scores = {word: 1.0 for word in emotion_tags}  # Example base scoring
    return intensity_scores

# Function: Extract Key Phrases
def extract_key_phrases(text):
    """
    Extract key phrases using spaCy noun chunks.
    """
    doc = nlp(text)
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    return key_phrases

# Function: Perform Topic Modeling
def topic_modeling(text, num_topics=2):
    """
    Perform topic modeling using Latent Dirichlet Allocation (LDA).
    Returns the top words for each topic.
    """
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    topics = lda.transform(dtm)
    topic_words = [vectorizer.get_feature_names_out()[index] for index in topics[0].argsort()[-5:]]
    return topic_words

# Function: Extract Named Entities
def extract_named_entities(text):
    """
    Extract named entities from text using spaCy's NER.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function: Sentence Tokenization
def simple_sentence_tokenize(text):
    """
    Tokenize text into sentences based on punctuation.
    """
    return re.split(r'(?<=[.!?])\s+', text)

# Function: Segment Entry by Sentiment
def segment_entry_by_sentiment(text):
    """
    Segment text into positive and negative sentences based on sentiment analysis.
    """
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

# Function: Preprocess Text for Insights
def preprocess_for_insights(text):
    """
    Preprocess text for insights by performing sentiment analysis,
    emotion tagging, topic modeling, keyphrase extraction, and segmentation.
    """
    # Step 1: Sentiment Analysis
    sentiment_result = perform_sentiment_analysis(text)
    
    # Step 2: Emotion Tagging and Intensity Scoring
    emotion_tags = tag_emotions(text)
    intensity_scores = assign_intensity_scores(emotion_tags)
    
    # Step 3: Key Phrase and Topic Extraction
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
if __name__ == "__main__":
    journal_entry = """Today was a mixed bag of emotions. The morning started off peaceful; I enjoyed my coffee while watching the sunrise, a rare moment of calm. But as the day unfolded, work stress built up—endless meetings and tight deadlines left me feeling overwhelmed. I had a brief argument with a colleague, which left me feeling frustrated and questioning my approach. By evening, I took a walk to clear my mind, which helped ease some of the tension. I’m grateful for that small reset. Tomorrow, I hope to manage my time better and approach conversations with more patience and understanding."""
    
    preprocessed_result = preprocess_for_insights(journal_entry)
    
    # Print the output
    print("Preprocessed Data:", preprocessed_result)
