# NLTK analysis
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
# pandas and numpy
import pandas as pd
import numpy as np
# Collections types
from collections import Counter, defaultdict
# Regex
import re

import subprocess

# Fun after import to download all necessary packages
def setup():
    # Download the "punkt" tokenizer models from nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # For synonym expansion
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('wordnet2022')
    # Fix WordNet lookup error
    try:
        # Copy wordnet2022 to wordnet
        subprocess.run(
            ["cp", "-rf", "/usr/share/nltk_data/corpora/wordnet2022", "/usr/share/nltk_data/corpora/wordnet"],
            check=True
        )
        print("WordNet fix applied successfully.")
    except Exception as e:
        print(f"Error applying WordNet fix: {e}")


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


def preprocess(text):
    """
    Preprocesses the given text: removes stopwords, corrects spelling,
    expands synonyms and applies stemming.
    Args:
        text (str): The input text.
    Returns:
        str: The preprocessed text.
    """
    # Step 1: Handle misspeling
    text = str(TextBlob(text).correct())

    # Step 2: Tokenize the text in lowercase
    tokens = word_tokenize(text.lower())

    # Step 3: Remove stopwords
    filtered_words = [word for word in tokens if word not in set(stopwords.words('english'))]

    # Step 4: Get synonyms for each word
    expanded_words = []
    for word in filtered_words:
        expanded_words.append(word)
        expanded_words.extend(get_synonyms(word))

    # Step 5: Apply stemming
    stemmed_words = [SnowballStemmer("english").stem(word) for word in expanded_words]

    return ' '.join(set(stemmed_words))


def get_normalized_tokens(text):
    """
    Get normalized tokens in the given text.
    Args:
        text (str): The input text.
    Returns:
        list: A list of normalized tokens (words) from the input text.
    """
    # To lowercase and remove punctuation from tokens
    return word_tokenize(re.sub(r'[^a-zA-Z\s]', '', text.lower()))


#=======================================================================================================================================
#_________________________________________________________2. Search Engine______________________________________________________________

def create_index(restaurants):
    """
    Creates a vocabulary and inverted index from a collection of restaurant descriptions.
    
    Args:
        restaurants (pd.DataFrame): A DataFrame containing restaurant data, with a column named 'description'
                                    that holds the text descriptions of each restaurant.
    
    Returns:
        tuple: 
            - vocabulary (dict): A dictionary mapping each unique word to a unique term ID (int).
            - inverted_index (defaultdict): A dictionary where each key is a term ID and the corresponding value
                                            is a list of document IDs where the term appears. Format: 
            {
                "term_id_1": [document_1, document_2, document_4],
                "term_id_2": [document_1, document_3, document_5],
                ...
            }
    """
    vocabulary = {}
    # Dictionary where each key defaults to an empty list
    inverted_index = defaultdict(list)
    term_id = 0
    # Loop over each document (restaurant description) by document_id
    for document_id, description in zip(restaurants.index, restaurants['preprocessed_description']):
        tokens = get_normalized_tokens(description)

        for word in set(tokens):
            # If the word is not in the vocabulary, add it with a new term_id
            if word not in vocabulary:
                vocabulary[word] = term_id
                term_id += 1

            # Get the term_id of the word
            word_id = vocabulary[word]
            # Add document_id to the inverted index for the given word_id
            inverted_index[word_id].append(document_id)

    return vocabulary, inverted_index


def execute(query, vocabulary, inverted_index, restaurants):
    """
    Executes a conjunctive query on the restaurant descriptions, returning only those restaurants
    whose descriptions contain all query terms.

    Args:
        query (str): The user input query string containing one or more search terms.
        vocabulary (dict): A dictionary mapping each unique word in the dataset to a unique term ID (int).
        inverted_index (defaultdict): A dictionary where each key is a term ID, and the corresponding value
                                      is a list of document IDs where that term appears.
        restaurants (pd.DataFrame): A DataFrame containing restaurant data, including columns like 'restaurantName',
                                    'address', 'description', and 'website'.

    Returns:
        pd.DataFrame: A DataFrame containing information about restaurants that match the query.
                      The columns include 'restaurantName', 'address', 'description', and 'website'.
                      If no matches are found, an empty DataFrame is returned with the specified columns.
    """
    
    # Step 1: Normalize query
    query_tokens = get_normalized_tokens(query)

    # Step 2: Map query terms to term IDs
    term_ids = []
    for term in query_tokens:
        if term in vocabulary: term_ids.append(vocabulary[term])
    # print(term_ids)

    # Step 3: Retrieve document lists from the inverted index
    if not term_ids:
        print("No matching terms found in vocabulary.")
        return pd.DataFrame(columns=['restaurantName', 'address', 'description', 'website'])

    # Get the list of document_ids for each term_id
    document_ids = [set(inverted_index[term_id]) for term_id in term_ids if term_id in inverted_index]
    # print(document_ids)

    if not document_ids:
        print("No documents found containing all query terms.")
        return pd.DataFrame(columns=['restaurantName', 'address', 'description', 'website'])

    # Find the intersection of document lists
    intersected_ids = set.intersection(*document_ids) if document_ids else set()
    # print(intersected_ids)

    # Step 4: Retrieve restaurant information for matching document IDs
    results = restaurants.loc[sorted(list(intersected_ids)), ['restaurantName', 'address', 'description', 'website']]

    return results



#=======================================================================================================================================
#______________________________2.2 Ranked Search Engine with TF-IDF and Cosine Similarity_______________________________________________

def calculate_idf(vocabulary, restaurants):
    """
    Calculate the IDF (Inverse Document Frequency) for each term in the vocabulary.
    
    Args:
        vocabulary (dict): A dictionary where keys are terms and values are term IDs.
        restaurants (pd.DataFrame): A DataFrame with a 'preprocessed_description' column containing the cleaned text of each document.
    
    Returns:
        dict: A dictionary with terms as keys and their IDF values as values.
    """
    # Total number of documents
    N = len(restaurants)  

    # DOCUMENT FREQUENCY
    document_frequency = defaultdict(int)  
    for description in restaurants['preprocessed_description']:
        unique_tokens = set(get_normalized_tokens(description))
        for term in unique_tokens:
            if term in vocabulary:
                document_frequency[term] += 1
    
    # INVERSE DOCUMENT FREQUENCY
    idf = {}
    for term in vocabulary.items():
        df_t = document_frequency[term]
        # idf[term] = np.log(N / df_t) if df_t > 0 else 0
        idf[term] = np.log(N / df_t)

    return idf


def calculate_tf_idf_scores(restaurants, idf):
    """
    Calculate TF-IDF scores for each term in each document and store in a dictionary.
    
    Args:
        restaurants (pd.DataFrame): DataFrame with preprocessed descriptions.
        idf (dict): A dictionary with terms as keys and their IDF values as values.
    
    Returns:
        tf_idf_scores (dict): Dictionary of TF-IDF scores for each term in each document.
        Format: {doc_id: {term: tf-idf score}}
    """
    tf_idf_scores = {}
    
    # Iterate over each document to calculate TF-IDF
    for doc_id, description in restaurants['preprocessed_description'].iteritems():
        # Calculate term frequency (TF)
        term_counts = Counter(get_normalized_tokens(description))  
        doc_tf_idf = {}
        
        # Calculate TF-IDF for each term in the document
        for term, tf in term_counts.items():
            # Check if term has an IDF score
            if term in idf:
                # TF * IDF
                tf_idf_score = tf * idf[term]
                doc_tf_idf[term] = tf_idf_score
                
        tf_idf_scores[doc_id] = doc_tf_idf
    
    return tf_idf_scores

def create_tfidf_inverted_index(tf_idf_scores):
    """
    Creates an inverted index from the TF-IDF scores dictionary.
    
    Args:
        tf_idf_scores (dict): Dictionary with TF-IDF scores for each term in each document. Format: {doc_id: {term: tf-idf_score}}
    
    Returns:
        dict: An inverted index where each term is mapped to a list of (document_id, TF-IDF score) tuples.
        Format: {
            "term_id_1": [(document1, tfIdf_{term,document1}), (document2, tfIdf_{term,document2}), ...],
            "term_id_2": [(document1, tfIdf_{term,document1}), (document3, tfIdf_{term,document3}), ...],
            ...
        }
    """
    tfidf_inverted_index = defaultdict(list)
    
    # Iterate over each document and its TF-IDF scores
    for doc_id, term_scores in tf_idf_scores.items():
        # For each term and its tf-idf score in the document
        for term, tfidf_score in term_scores.items():
            # Append (document_id, tfidf_score) tuple to the term's list in the inverted index
            tfidf_inverted_index[term].append((doc_id, tfidf_score))
    
    return tfidf_inverted_index


def calculate_query_vector(query, idf):
    """
    Calculate the TF-IDF vector for the query based on term frequencies and IDF values.
    
    Args:
        query (str): The input query string.
        idf (dict): A dictionary of IDF values for each term in the vocabulary.
    
    Returns:
        dict: A dictionary representing the TF-IDF vector of the query.
    """
    # Calculate term frequency (TF) of the query
    term_counts = Counter(get_normalized_tokens(query))

    query_vector = {}
    for term, tf in term_counts.items():
        # Check if term has an IDF score
        if term in idf:
            # TF * IDF
            query_vector[term] = tf * idf[term]
    return query_vector


def cosine_similarity_score(query_vector, document_vector):
    """
    Calculate the cosine similarity between the query and document vectors.
    
    Args:
        query_vector (dict): TF-IDF vector for the query.
        document_vector (dict): TF-IDF vector for a document.
    
    Returns:
        float: Cosine similarity score.
    """
    # Calculate the dot product between the query and document vectors
    dot_product = sum(query_vector[term] * document_vector.get(term, 0) for term in query_vector)
    
    # Calculate the magnitudes of the query and document vectors
    query_magnitude = np.sqrt(sum(weight ** 2 for weight in query_vector.values()))
    doc_magnitude = np.sqrt(sum(weight ** 2 for weight in document_vector.values()))
    
    # Calculate cosine similarity
    return dot_product / (query_magnitude * doc_magnitude) if query_magnitude and doc_magnitude else 0.0


def rank_documents(query, restaurants, tfidf_inverted_index, vocabulary, idf, k=10):
    """
    Rank documents based on cosine similarity to the query.
    
    Args:
        query (str): The input query string.
        restaurants (pd.DataFrame): DataFrame containing restaurant data.
        tfidf_inverted_index (dict): Inverted index with TF-IDF scores for each term in each document.
        vocabulary (dict): Mapping of terms to term IDs.
        idf (dict): IDF values for each term in the vocabulary.
        k (int): Number of top documents to return.
    
    Returns:
        pd.DataFrame: Top-k documents ranked by cosine similarity.
    """
    # Step 1: Calculate TF-IDF vector for the query
    query_vector = calculate_query_vector(query, vocabulary, idf)
    
    # Step 2: Retrieve relevant documents for each term in the query
    matching_docs = defaultdict(dict)
    for term_id in query_vector.keys():
        if term_id in tfidf_inverted_index:
            for doc_id, tfidf_score in tfidf_inverted_index[term_id]:
                matching_docs[doc_id][term_id] = tfidf_score

    # Step 3: Calculate cosine similarity for each candidate document
    scores = []
    for doc_id, doc_vector in matching_docs.items():
        similarity_score = cosine_similarity_score(query_vector, doc_vector)
        scores.append((doc_id, similarity_score))

    # Step 4: Sort documents by similarity score in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    # Step 5: Retrieve top-k documents
    top_k_docs = [doc_id for doc_id, _ in sorted_scores]
    result_df = restaurants.loc[top_k_docs, ['restaurantName', 'address', 'description', 'website']]
    result_df['similarity_score'] = [score for _, score in sorted_scores]
    
    return result_df.reset_index(drop=True)





    
    
    
    
