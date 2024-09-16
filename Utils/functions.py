import re
import inflect
import unicodedata
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import joblib 
import spacy  
from tqdm import tqdm

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the inflect engine
p = inflect.engine()

# Dictionary mapping currency symbols to words
currency_map = {
        '\u0024': 'dollar',    # $
        '\u00A3': 'pound',     # £
        '\u00A5': 'yen',       # ¥
        '\u20AC': 'euro',      # €
        '\u20B9': 'rupee',     # ₹
        '\u20A1': 'colon',     # ₡
        '\u20A2': 'cruzeiro',  # ₢
        '\u20A3': 'french_franc', # ₣
        '\u20A4': 'lira',      # ₤
        '\u20A5': 'mill',      # ₥
        '\u20A6': 'naira',     # ₦
        '\u20A7': 'peseta',    # ₧
        '\u20A8': 'rupee',     # ₨
        '\u20A9': 'won',       # ₩
        '\u20AA': 'shekel',    # ₪
        '\u20AB': 'dong',      # ₫
        '\u20AC': 'euro',      # €
        '\u20AD': 'kip',       # ₭
        '\u20AE': 'tugrik',    # ₮
        '\u20AF': 'drachma',   # ₯
        '\u20B0': 'german_penny', # ₰
        '\u20B1': 'peso',      # ₱
        '\u20B2': 'guarani',   # ₲
        '\u20B3': 'austral',   # ₳
        '\u20B4': 'hryvnia',   # ₴
        '\u20B5': 'cedi',      # ₵
    }

# Stop words and punctuation for processing
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def convert_numbers_to_words(text):
    """Convert all numbers in the text to words."""
    return re.sub(r'\b\d+\b', lambda x: p.number_to_words(x.group()), text)

def clean_article(article):
    """
    Cleans the given article by:
    - Replacing currency symbols with their corresponding words
    - Converting numbers to words
    - Removing URLs, special characters, and extra spaces
    - Tokenizing, removing stopwords and punctuation, and lemmatizing
    """
    # Replace currency symbols with their corresponding words
    for symbol, word in currency_map.items():
        article = article.replace(symbol, word)

    # Convert numbers to words
    try:
        article = convert_numbers_to_words(article)
    except inflect.NumOutOfRangeError:
        article = re.sub(r'\d+', ' large_number ', article)

    # Remove URLs
    article = re.sub(r'https?://[^\s]+', 'url', article)

    # Remove special characters (except standard punctuation)
    article = re.sub(r'[^\w\s.,;\'"-]', '', article)

    # Remove extra spaces, tabs, newlines, and carriage returns
    article = re.sub(r'\s+', ' ', article).strip()

    # Remove accented characters
    article = unicodedata.normalize('NFKD', article).encode('ASCII', 'ignore').decode('utf-8')

    # Tokenization
    tokens = word_tokenize(article)

    # Remove stopwords and punctuation, then lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(token.lower())
        for token in tokens
        if token.lower() not in stop_words and token not in punctuation and len(token) > 1
    ]

    # Rejoin tokens into a cleaned article
    cleaned_article = ' '.join(cleaned_tokens)

    return cleaned_article

nlp = spacy.load('en_core_web_sm')

def extract_names(text):
    """Extracts person names from text using spaCy's NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def extract_clubs(text):
    """Extracts organizations (e.g., clubs) from text using spaCy's NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

def extract_fee(text):
    """Extracts monetary amounts from text using spaCy's NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

# Test usage
if __name__ == "__main__":
    # Sample article text
    article_text = "Lionel Messi signed a deal with Paris Saint-Germain for $41 million."

    
    # Extract named entities
    names = extract_names(article_text)
    clubs = extract_clubs(article_text)
    fees = extract_fee(article_text)

    # Clean the article
    cleaned_text = clean_article(article_text)


    print(f"cleaned_text: {cleaned_text}")

    print(f"Names: {names}")
    print(f"Clubs: {clubs}")
    print(f"Fees: {fees}")