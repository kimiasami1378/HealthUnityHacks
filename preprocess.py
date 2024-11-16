import re
import spacy
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('stopwords')

# Load spaCy model for lemmatization and NER
nlp = spacy.load("en_core_web_sm")

# Set of stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Function to clean and normalize text
def clean_text(text):
    """
    Cleans the input text by removing punctuation, normalizing case,
    and reducing multiple spaces to a single space.
    """
    text = re.sub(r'\s+', ' ', text.strip())  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Function to tokenize and remove stopwords
def tokenize_and_remove_stopwords(text):
    """
    Tokenizes text and removes stopwords.
    """
    tokens = text.split()  # Basic tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

# Function to lemmatize text using spaCy
def lemmatize_text(tokens):
    """
    Lemmatizes a list of tokens using spaCy.
    """
    doc = nlp(' '.join(tokens))  # Combine tokens into a single string for processing
    lemmatized = [token.lemma_ for token in doc]
    return lemmatized

# Function to extract named entities
def extract_named_entities(text):
    """
    Extracts named entities from the text using spaCy's NER capabilities.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Full preprocessing pipeline function
def preprocess_journal_entry(text):
    """
    Processes a journal entry through the full preprocessing pipeline:
    cleaning, tokenizing, removing stopwords, lemmatizing, and extracting named entities.
    """
    # Step 1: Clean the text
    cleaned_text = clean_text(text)
    
    # Step 2: Tokenize and remove stopwords
    tokens = tokenize_and_remove_stopwords(cleaned_text)
    
    # Step 3: Lemmatize the tokens
    lemmatized_text = lemmatize_text(tokens)
    
    # Step 4: Extract named entities
    named_entities = extract_named_entities(cleaned_text)
    
    # Return the processed components
    return {
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "lemmatized_text": lemmatized_text,
        "named_entities": named_entities
    }

# Example usage
if __name__ == "__main__":
    journal_entry = "Today I felt really anxious about work and my upcoming project deadlines. In the evening, I went for a walk to clear my mind."
    preprocessed_result = preprocess_journal_entry(journal_entry)

    # Print the preprocessed result
    print("Cleaned Text:", preprocessed_result['cleaned_text'])
    print("Tokens:", preprocessed_result['tokens'])
    print("Lemmatized Text:", preprocessed_result['lemmatized_text'])
    print("Named Entities:", preprocessed_result['named_entities'])

