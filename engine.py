import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import Counter
import subprocess

def setup():
    # Download the "punkt" tokenizer models from nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # Is used for synonym expansion
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('wordnet2022')
    # Workaround to fix WordNet lookup error
    try:
        # Copy wordnet2022 to wordnet
        subprocess.run(
            ["cp", "-rf", "/usr/share/nltk_data/corpora/wordnet2022", "/usr/share/nltk_data/corpora/wordnet"],
            check=True
        )
        print("WordNet fix applied successfully.")
    except Exception as e:
        print(f"Error applying WordNet fix: {e}")
        
# Get synonyms
def get_synonyms(word):
    """
    Returns a set of synonyms for a given word using WordNet.
    Args:
        word (str): The input word.
    Returns:
        set: A set of synonyms.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms


def preprocess(text, word_counts, frequency_threshold=1):
    """
    Preprocesses the given text: removes stopwords, corrects spelling,
    expands synonyms, removes low-frequency words and applies stemming.
    Args:
        text (str): The input text.
        word_counts (Counter): A dictionary of word frequencies across all documents.
        frequency_threshold (int): The minimum frequency required to keep a word.
    Returns:
        str: The preprocessed text.
    """
    # Step 1: Handle misspeling
    text = str(TextBlob(text).correct())

    # Step 2: Tokenize the text in lowercase
    words = word_tokenize(text.lower())

    # Step 3: Remove stopwords and apply low-frequency filter
    filtered_words = [
        word for word in words if word not in set(stopwords.words('english')) and
                      word_counts[word] > frequency_threshold
        ]

    # Step 4: Get synonyms for each word
    expanded_words = []
    for word in filtered_words:
        expanded_words.append(word)
        expanded_words.extend(get_synonyms(word))

    # Step 5: Apply stemming
    stemmed_words = [SnowballStemmer("english").stem(word) for word in expanded_words]

    return ' '.join(stemmed_words)

def get_normalized_tokens(text):
    """
    Get normalized tokens in the given text.
    Args:
        text (str): The input text.
    Returns:
        str: The normalized tokens.
    """
    # To lowercase and remove punctuation from tokens
    return word_tokenize(re.sub(r'[^a-zA-Z\s]', '', text.lower()))

def get_word_counts(descriptions: pd.Series) -> Counter:
    """
    Calculates the frequency of each word across all descriptions in the given DataFrame column.
    Args:
        descriptions (pd.Series): A Pandas Series containing text descriptions from each restaurant.

    Returns:
        Counter: A Counter object with each unique word as the key and its frequency across all descriptions as the value.
    """
    # Initialize the Counter
    word_counts = Counter()
    
    # Update counter iterating over all records' descriptions
    for description in descriptions:
        # Remove punctuation from tokens
        tokens = get_normalized_tokens(description)
        # Update with tokens of the current description
        word_counts.update(tokens)

    return word_counts