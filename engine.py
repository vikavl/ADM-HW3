# NLTK preprocessing
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk.data

# pandas and numpy
import pandas as pd
import numpy as np
from math import sqrt

# Collections types
from collections import defaultdict
# Regex
import re
from tqdm import tqdm

# For Sorting in heap
import heapq

# For downloading WordNet
import subprocess

# Fun after import to download all necessary packages
def setup():
    # Download the "punkt" tokenizer models from nltk
    nltk.download('punkt')
    nltk.data.load('tokenizers/punkt/english.pickle')
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


def preprocess(text):
    """
    Preprocesses the given text: removes stopwords, corrects spelling,
    expands synonyms and applies stemming.
    Args:
        text (str): The input text.
    Returns:
        str: The preprocessed text.
    """
    # Step 1: Tokenize the text in lowercase
    tokens = word_tokenize(text.lower())

    # Step 2: Remove stopwords
    filtered_words = [word for word in tokens if word not in set(stopwords.words('english'))]

    # Step 3: Apply stemming
    stemmed_words = [SnowballStemmer("english").stem(word) for word in filtered_words]

    return ' '.join(set(stemmed_words))


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
        tokens = word_tokenize(description)

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
        restaurants (pd.DataFrame): A DataFrame containing restaurant data.

    Returns:
        pd.DataFrame: A DataFrame containing information about restaurants that match the query.
                      If no matches are found, an empty DataFrame is returned with the specified columns.
    """
    
    # Step 1: Normalize query
    normalized_query = preprocess(query)
    query_tokens = word_tokenize(normalized_query)
    #print(query_tokens)

    # Step 2: Map query terms to term IDs
    term_ids = []
    for term in query_tokens:
        if term in vocabulary: 
            term_ids.append(vocabulary[term])
    #print(term_ids)

    # Step 3: Get the list of document_ids for each term_id
    document_ids = [set(inverted_index[term_id]) for term_id in term_ids if term_id in inverted_index]
    #print(document_ids)

    # Step 4: Find the intersection of document lists
    intersected_ids = set.intersection(*document_ids) if document_ids else set()
    #print(intersected_ids)
    
    # Step 5: Retrieve restaurant information for matching document IDs
    if not intersected_ids:
        # Marker of no results found
        return "No documents found with all terms"
    else:
        # There are some sorted results
        return restaurants.loc[sorted(list(intersected_ids))]



#=======================================================================================================================================
#______________________________2.2 Ranked Search Engine with TF-IDF and Cosine Similarity_______________________________________________

def calculate_tfidf_score(text: str, term, term_id, inverted_index, total_documents):
    # Calculate Term Frequency
    TF_dt = text.count(term)
    # Calculate Document Frequency
    dft = len(list(inverted_index[term_id]))
    # Calculate Inverse Document Frequency (handle division on 0 exception)
    IDF_t = (1.0 + np.log10(total_documents / dft)) if dft > 0 else 1.0
    # TF-IDF weighting
    return TF_dt * IDF_t

def calculate_tfidf_inverted_index(restaurants, inverted_index, vocabulary, total_documents):
    """
    Calculate the TF-IDF (Term Frequency-Inverse Document Frequency) scores for each term in each document
    and store these scores in an inverted index format.

    Args:
        restaurants (pd.DataFrame): DataFrame containing restaurant data with columns:
                                    - 'index': Unique document IDs.
                                    - 'preprocessed_description': Tokenized or preprocessed text of each restaurant description.
        inverted_index (dict): Dictionary where each key is a term ID and the corresponding value is a list of document IDs
                               where the term appears. Format: {term_id: [doc_id_1, doc_id_2, ...]}.
        vocabulary (dict): Dictionary mapping terms (words) to unique term IDs. Format: {term: term_id}.
        total_documents (int): Total number of documents in the collection, used to calculate the IDF.

    Returns:
        defaultdict: A dictionary where each key is a term id and the value is a list of tuples. 
                     Each tuple consists of a document ID and the corresponding TF-IDF score for that term
                     in the document. Format: {term_id: [(doc_id_1, tfidf_score), (doc_id_2, tfidf_score), ...]}.

    Process:
        1. For each document, calculate the term frequency (TF) for each term in the document.
        2. For each term, retrieve its document frequency (DF) from the inverted index.
        3. Calculate the Inverse Document Frequency (IDF) for each term using the formula:
               IDF_t = log10(total_documents / DF_t)
           where DF_t is the document frequency of term t.
        4. Calculate the TF-IDF score by multiplying the term frequency with the IDF for each term in each document:
               TF-IDF = TF * IDF
        5. Store each term's TF-IDF score along with the document ID in an inverted index format for easy retrieval.

    """
    tfidf_inverted_index = defaultdict(list)
    
    # Outer loop for documents
    for doc_id, doc_descr in tqdm(
        zip(restaurants['index'], restaurants['preprocessed_description']),
        total = total_documents,
        desc = "Processing documents"
    ):
        # Inner loop for terms
        for term, term_id in tqdm(vocabulary.items(), desc=f"Calculating TF-IDF for terms in Doc ID: {doc_id}", leave=False):
            tfidf_score = calculate_tfidf_score(doc_descr, term, term_id, inverted_index, total_documents)
            tfidf_inverted_index[term_id].append((doc_id, tfidf_score))

    return tfidf_inverted_index

def cosine_similarity_score(X, Y):
    """
    Compute the cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors in a 
    multidimensional space. It is commonly used to compare document similarity 
    based on their vector representations.

    Args:
        X (list or np.array): The first vector.
        Y (list or np.array): The second vector.

    Returns:
        float: The cosine similarity score between the two vectors, ranging 
               from -1 (completely dissimilar) to 1 (identical). Returns 0 if 
               either vector has zero magnitude.

    Raises:
        ValueError: If the two vectors have different lengths.

    Formula:
        Cosine Similarity = (X · Y) / (||X|| * ||Y||)
        where:
          - X · Y is the dot product of X and Y.
          - ||X|| and ||Y|| are the magnitudes (Euclidean norms) of vectors X and Y.
    """
    if len(X) != len(Y):
        raise ValueError("Vectors must have the same length.")
    dot_product = sum(x * y for x, y in zip(X, Y))
    magnitude_x = sqrt(sum(x**2 for x in X))
    magnitude_y = sqrt(sum(y**2 for y in Y))
    return dot_product / (magnitude_x * magnitude_y) if magnitude_x and magnitude_y else 0.0

def calculate_query_vector(query, inverted_index, vocabulary, total_documents):
    """
    Compute the TF-IDF vector for the given query.

    This function preprocesses the query, tokenizes it, and calculates 
    the TF-IDF scores for the terms present in the query. It outputs both 
    the query vector (a list of TF-IDF scores) and the corresponding term IDs.

    Args:
        query (str): The user's search query as plain text.
        inverted_index (dict): The inverted index mapping term IDs to document IDs.
        vocabulary (dict): The mapping of terms to unique term IDs.
        total_documents (int): The total number of documents in the dataset.

    Returns:
        tuple:
            - query_vector (list): The TF-IDF scores for terms in the query.
            - term_ids (list): The term IDs corresponding to the terms in the query.

    Process:
        1. Normalize the query using preprocessing steps.
        2. Tokenize the query into unique terms.
        3. For each term in the query:
           - Calculate its TF-IDF score using the `calculate_tfidf_score` function.
           - Include the term ID if the term exists in the vocabulary.
        4. Return the query vector and term IDs.

    Note:
        - Only terms present in the vocabulary are included in the query vector.
        - Query terms that do not match any vocabulary terms are ignored.
    """
    normalized_query = preprocess(query)
    query_tokens = word_tokenize(normalized_query)

    # Compute unique terms and their TF-IDF scores efficiently
    tfidf_dict_query = {
        term: calculate_tfidf_score(normalized_query, term, vocabulary[term], inverted_index, total_documents)
        for term in set(query_tokens)
        if term in vocabulary
    }    
    query_vector = list(tfidf_dict_query.values())
    term_ids = [vocabulary[term] for term in tfidf_dict_query.keys()]
    return query_vector, term_ids

def rank_matching_restaurants(query, inverted_index, vocabulary, tfidf_inverted_index, restaurants, total_documents, k=10, top_k=True):
    """
    Rank documents based on cosine similarity to the query and return the top-k results.

    This function retrieves relevant documents based on TF-IDF cosine similarity, 
    which measures the similarity between the query vector and document vectors. 
    It returns a DataFrame of the top-k restaurants, ranked by similarity.

    Args:
        query (str): The user's search query as plain text.
        inverted_index (dict): The inverted index mapping term IDs to document IDs.
        vocabulary (dict): The mapping of terms to unique term IDs.
        tfidf_inverted_index (defaultdict): Precomputed TF-IDF scores for terms in documents.
        restaurants (pd.DataFrame): The DataFrame containing restaurant information.
        total_documents (int): Total number of documents in the dataset.
        k (int): Number of top results to return (default: 10).
        top_k (bool): Whether to return only the top-k results (default: True).

    Returns:
        pd.DataFrame: A DataFrame containing the relevant restaurants, including:
            - 'index': Restaurant index.
            - 'restaurantName': Name of the restaurant.
            - 'address': Address of the restaurant.
            - 'description': Description of the restaurant.
            - 'website': Website of the restaurant.
            - 'cosine_similarity': The cosine similarity score between the query and the restaurant description.

    Process:
        1. Normalize and preprocess the query to calculate its TF-IDF vector.
        2. Retrieve the term IDs for query terms and compute the query vector.
        3. Collect TF-IDF vectors for relevant documents based on the query terms.
        4. Compute cosine similarity between the query vector and each document vector.
        5. Rank the documents by cosine similarity in descending order.
        6. Merge the cosine similarity scores with the restaurant DataFrame.
        7. Return the top-k results if `top_k=True`; otherwise, return all ranked results.

    Notes:
        - Documents with zero vectors (no matching terms) are skipped.
        - If no documents match, an empty DataFrame is returned with the specified columns.

    Example Usage:
        results = rank_matching_restaurants(
            query="Italian food with terrace",
            inverted_index=my_inverted_index,
            vocabulary=my_vocabulary,
            tfidf_inverted_index=my_tfidf_inverted_index,
            restaurants=my_restaurants_df,
            total_documents=len(my_restaurants_df),
            k=5
        )
    """
    total_documents = len(restaurants)

    # Normilize query and cast to vector + get query's terms ids
    query_vector, term_ids = calculate_query_vector(query, inverted_index, vocabulary, total_documents)

    # Collect TF-IDF vectors for the documents
    tfidf_vectors = defaultdict(lambda: defaultdict(float))
    for term_id in term_ids:
        for doc_id, score in tfidf_inverted_index.get(term_id, []):
            tfidf_vectors[doc_id][term_id] = score

    # Create DataFrame with document vectors based on term ids
    tfidf_vectors_df = pd.DataFrame.from_dict(tfidf_vectors, orient='index', columns=term_ids).fillna(0)

    # Compute cosine similarity for each document
    results = [
        {'index': index, 'cosine_similarity': cosine_similarity_score(query_vector, row)}
        for index, row in tfidf_vectors_df.iterrows()
        if row.sum() > 0  # Skip documents with zero vectors
    ]

    # Define a DataFrame with cosine_similarity
    cosine_similarity_df = pd.DataFrame(results)

    if not cosine_similarity_df.empty:
        # Retive columns names for final dataframe
        final_columns = list(restaurants.columns)
        final_columns.append('cosine_similarity')
        # Merge cosine_similarity score with restaurants Dataframe
        merged_df = cosine_similarity_df.merge(restaurants, left_on='index', right_index=True)
        # Sort by cosine_similarity
        merged_df = merged_df.sort_values(by='cosine_similarity', ascending=False)
        if top_k:
            # Return top k
            return merged_df[final_columns].head(k)
        else:
            # Return all results
            return merged_df[final_columns]
    else:
        print("No documents found containing all query terms.")


#=======================================================================================================================================
#____________________________________________________3. Define a New Score!_____________________________________________________________

def map_range_to_values(range_tuple, all_values):
    """
    Map a tuple with range edges to a list of all values within the range.
    """
    # Get the start and end values from the tuple
    start, end = range_tuple

    # Find the indices of the range edges in the list of all values
    start_index = all_values.index(start)
    end_index = all_values.index(end)

    # Return the slice of the list that falls within the range
    return all_values[start_index:end_index + 1]

def calculate_price_range_score(row, selected_price_range):
    """
    Calculate the price range score for a single restaurant row.

    Args:
        row (pd.Series): A single row of the dataframe representing a restaurant.
        selected_price_range (tuple): Tuple specifying the selected price range (e.g., ('€', '€€€')).

    Returns:
        float: The score based on the price range, with higher scores for more affordable options
               within the selected range. Returns 0 if the price range does not match.
    """
    # Define price weights: Higher scores could be given to more affordable options based on the user’s choice.
    price_range_weights = {'€': 1.0, '€€': 0.75, '€€€': 0.5, '€€€€': 0.25}

    # Get row price range
    restaurant_price_range = row['priceRange']

    # Map a tuple with range edges to a list of all values within the range
    valid_price_ranges = map_range_to_values(selected_price_range, list(price_range_weights.keys()))

    # If the restaurant's price range is within the valid range, return its weight
    if restaurant_price_range in valid_price_ranges:
        return price_range_weights[restaurant_price_range]
    return 0

def calculate_new_score(row, selected_metrics):
    """
    Calculate the overall new score for a restaurant based on multiple criteria.

    The scoring function combines multiple metrics (e.g., cosine similarity, price range match, 
    facilities match, and cuisine type match) using weighted scores.

    Args:
        row (pd.Series): A row of the dataset representing a restaurant record.
        selected_metrics (dict): A dictionary containing user-selected metrics, including:
            - "priceRange" (tuple): Selected price range (e.g., ('€', '€€€')).
            - "facilitiesServices" (list): Selected facilities (e.g., ['Terrace', 'WiFi']).
            - "cuisineType" (list): Selected cuisine types (e.g., ['Italian', 'French']).

    Returns:
        float: The combined score for the restaurant, calculated as a weighted sum of individual metric scores.

    Scoring Details:
        - **Cosine Similarity**: Weighted based on its contribution to description match (default: 0.4).
        - **Price Range**: Weighted based on the affordability match (default: 0.3).
        - **Facilities Match**: Weighted based on the proportion of selected facilities present (default: 0.2).
        - **Cuisine Match**: Weighted based on the proportion of selected cuisines present (default: 0.1).
    """
    # Weights for Overall Scoring: My personal choice
    weights = {
        "cosine_similarity": 0.4,
        "priceRange": 0.3,
        "facilitiesServices": 0.2,
        "cuisineType": 0.1
    }

    # Break down metrics
    selected_price_range =selected_metrics["priceRange"]
    selected_facilities = selected_metrics["facilitiesServices"]
    selected_cuisine_type = selected_metrics["cuisineType"]

    # Init score
    score = 0

    # Set weight to prioritize field in overall scoring
    score += row["cosine_similarity"] * weights['cosine_similarity']

    # Add check on empty values
    if selected_price_range:
        # Calculate price range match score
        # Set weight to prioritize field in overall scoring
        score += calculate_price_range_score(row, selected_price_range) * weights['priceRange']

    if selected_facilities:
        # Calculate facility match score: Give more points for matching facilities/services (e.g., “Terrace,” “Air conditioning”).
        # I decided to calculate the final score for facilities/servises as: 
        # [number of matching]/[total numner of facilities/services]
        facilities = [facility for facility in re.findall(r"'([^']*)'", row['facilitiesServices'])]
        if facilities and selected_facilities:
            facility_matches = len(set(facilities) & set(selected_facilities))
            facility_score = facility_matches / len(facilities) if facilities else 0
            # Set weight to prioritize field in overall scoring
            score += facility_score * weights['facilitiesServices']

    if selected_cuisine_type:
        # Calculate cuisine type match score
        # [number of matching]/[total numner of cuisine types in row]
        cuisine_types = [c.strip() for c in row['cuisineType'].split(",")]
        cuisine_matches = sum(1 for cuisine in cuisine_types if cuisine in selected_cuisine_type)
        if cuisine_matches:
            # Set weight to prioritize field in overall scoring
            score += cuisine_matches / len(cuisine_types) * weights['cuisineType']

    return score

# Apply the scoring function to the dataset
def calculate_new_scores(query, selected_metrics, inverted_index, vocabulary, tfidf_inverted_index, restaurants, k=10):
    """
    Compute the final new scores for all restaurants in the dataset based on a combination of metrics.

    The function retrieves relevant restaurants using the conjunctive search engine, calculates 
    TF-IDF cosine similarity for ranking, and then applies the custom scoring logic.

    Args:
        query (str): The user's search query.
        selected_metrics (dict): User-selected metrics for scoring, including:
            - "priceRange" (tuple): Selected price range (e.g., ('€', '€€€')).
            - "facilitiesServices" (list): Selected facilities (e.g., ['Terrace', 'WiFi']).
            - "cuisineType" (list): Selected cuisine types (e.g., ['Italian', 'French']).
        inverted_index (dict): The inverted index created from the dataset.
        vocabulary (dict): The vocabulary mapping terms to unique term IDs.
        tfidf_inverted_index (defaultdict): The precomputed TF-IDF inverted index for the dataset.
        restaurants (pd.DataFrame): The full dataset containing restaurant information.

    Returns:
        pd.DataFrame: A dataframe containing the relevant restaurants with their new scores.

    Process:
        1. Retrieve matching restaurants using the **conjunctive search engine** (Step 2.1).
        2. Calculate **cosine similarity** for retrieved documents (Step 2.2 Ranked Search).
        3. Apply custom scoring for price range, facilities, and cuisine match.
        4. Combine individual scores into an overall score and return the updated dataframe.

    Output Columns:
        - 'index': Restaurant index.
        - 'restaurantName': Name of the restaurant.
        - 'address': Address of the restaurant.
        - 'description': Description of the restaurant.
        - 'website': Website of the restaurant.
        - 'cosine_similarity': TF-IDF cosine similarity score.
        - 'new_score': The final combined score based on the custom scoring logic.
    """
    # Step 1: Retrive documents using 2.1 - Conjunctive Search Engine
    filtered_restaurants = execute(query, vocabulary, inverted_index, restaurants)
    # Step 2: Calculate TF-IDF scores for retrieved documents in Step 1

    # Calculate total number of documents
    total_documents = len(restaurants)
    results = rank_matching_restaurants(query, inverted_index, vocabulary, tfidf_inverted_index, filtered_restaurants, total_documents, top_k=False)

    # Step 3: Initialize a heap to get the top-k documents
    heap = []

    # Step 4: Calculate new scores and maintain the heap
    for _, row in results.iterrows():
        # Calculate the new score for the restaurant
        new_score = calculate_new_score(row, selected_metrics)

        # Push a tuple (new_score, index) into the heap
        heapq.heappush(heap, (new_score, row['index']))

        # Keep only the top-k items in the heap
        if len(heap) > k:
            heapq.heappop(heap)
    
    # Step 5: Extract the top-k items from the heap
    top_k_results = heapq.nlargest(k, heap, key=lambda x: x[0])  # Sort the heap by the new_score
    
    # Step 6: Retrieve the full rows for the top-k indices
    # Extract scores
    top_k_indices = [row[1] for row in top_k_results]
    # Extract docs ids
    top_k_df = results[results['index'].isin(top_k_indices)].copy()
    
    # Step 7: Add the "new_score" column
    top_k_df['new_score'] = [row[0] for row in top_k_results]

    # Step 8: Return the top-k DataFrame
    final_columns = ['index', 'restaurantName', 'address', 'description', 'website', 'cosine_similarity', 'new_score']
    return top_k_df[final_columns]

