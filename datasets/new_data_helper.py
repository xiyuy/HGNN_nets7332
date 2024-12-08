import os
import pandas as pd
import hypernetx as hnx
import networkx as nx
import numpy as np
import ast

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


    ### Function to create the weight array
def create_author_dict_values(author_ids, unique_author_ids):
    """
    Create a dictionary with all unique author IDs initialized to 0,
    then update the values for IDs present in the list `author_ids`.
    Return the values of the dictionary as a list.

    Parameters:
        author_ids (list): List of author IDs for the current row.
        unique_author_ids (set): Set of all unique author IDs across the dataset.

    Returns:
        list: Values of the dictionary as a list (representing author ID weights).
    """
    # Initialize the dictionary with all unique author IDs set to 0
    author_dict = {author_id: 0 for author_id in unique_author_ids}

    if not isinstance(author_ids, list) or len(author_ids) == 0:
        return list(author_dict.values())  # Return all zeros if author_ids is invalid

    # Update values for IDs present in the author_ids list
    for i, author_id in enumerate(author_ids):
        if author_id in author_dict:
            # First and last author gets value 2
            if i == 0 or i == len(author_ids) - 1:
                author_dict[author_id] = 2
            else:
                author_dict[author_id] = 1

    # Return the values of the dictionary as a list
    return list(author_dict.values())

def extract_author_ids(authors):
    try:
        # Safely evaluate the string as a Python object if it's stored as a string
        if isinstance(authors, str):
            authors = ast.literal_eval(authors)
        # Extract 'id' values from the list of dictionaries
        return [author['id'] for author in authors if 'id' in author]
    except (ValueError, SyntaxError, TypeError):
        return []
    
# Function to check if a row's author_ids is a subset of new_aid
def is_subset(row_list, valid_list):
    try:
        return set(row_list).issubset(valid_list)
    except TypeError:
        return False  # Return False if the row_list is not iterable
    
# Function to create the list of 2s and 1s
def generate_2s_and_1s(author_ids):
    """Generate a list with the first and last value as 2, and the rest as 1s."""
    n = len(author_ids)
    if n == 0:
        return []  # Handle cases where there are no authors
    return [2] + [1] * (n - 2) + [2] if n > 1 else [2]

def pre_process_hyp_papers(hyp_papers, hyp_authors):
    # Define a function to extract author IDs
    # Extract relevant columns: 'id' for paper IDs and 'authors' for author information
    papers = hyp_papers[['id', 'authors', 'n_citation']]
    papers = papers[papers['authors'].notna() & papers['n_citation'].notna() & (papers['n_citation'] != 'n_citation')]
    papers['author_ids'] = papers['authors'].apply(extract_author_ids)

    # Log the citation numbers
    papers['n_citation'] = np.log(papers['n_citation'].astype(float) + 1) # add one to all citations because we want to put a weight of 1 for uncited paper and thus keep the hyperedge
    
    # Create a mapping of unique author IDs to array indices
    unique_author_ids = list(set(author_id for author_list in papers['author_ids'] for author_id in author_list if author_id is not None))
    author_id_to_index = {author_id: index for index, author_id in enumerate(unique_author_ids)}

    # Apply filter on authors
    hyp_authors_no_null = hyp_authors[hyp_authors['tags'].notna() & hyp_authors['department'].notna()]

    new_aid = hyp_authors_no_null[hyp_authors_no_null['id'].isin(unique_author_ids)].dropna().id.unique().tolist() # Only including authors who appear in the papers
    new_aid = set(new_aid)  # Convert to set for efficient subset checking


    # Apply filter to extract papers by these authors only
    paper_author_df = papers[papers['author_ids'].apply(lambda x: is_subset(x, new_aid))]
    print("Paper author dataframe dimensions: ", paper_author_df.shape)

    ### Create the author weight matrix

    # Add a new column with the generated list
    paper_author_df['author_weights'] = paper_author_df['author_ids'].apply(generate_2s_and_1s)

    # Filter the dataframe to exclude rows where 'author_ids' has only one element
    paper_author_df = paper_author_df[paper_author_df['author_ids'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False)]
    
    # # Display the resulting DataFrame
    # display(paper_author_df.head())

    # Recreate the unique paper IDs mapping
    unique_paper_ids = list(paper_author_df['id'])  # Ensure we have a list of unique paper IDs
    paper_id_to_index = {paper_id: index for index, paper_id in enumerate(unique_paper_ids)}
    print("Number of papers: ", len(unique_paper_ids))

    # Recreate the unique author IDs mapping
    unique_author_ids = list(set(author_id for author_list in paper_author_df['author_ids'] for author_id in author_list if author_id is not None))
    author_id_to_index = {author_id: index for index, author_id in enumerate(unique_author_ids)}
    print("Number of authors: ", len(unique_author_ids))

    # Apply the function to create the new column with lists of values
    paper_author_df['author_weight_array'] = paper_author_df['author_ids'].apply(lambda x: create_author_dict_values(x, unique_author_ids))

    # Convert the 'author_weights' column to a 2D NumPy array (matrix)
    author_weights_matrix_R = np.vstack(paper_author_df['author_weight_array'].values)


    # Display the shape of the matrix for confirmation
    print("R matrix shape:", author_weights_matrix_R.shape)

    ### Create the incidence matrix
    
    # Replace all values > 0 with 1
    binary_matrix = np.where(author_weights_matrix_R > 0, 1, 0)

    # Find the transpose of the binary matrix
    paper_author_matrix_H = binary_matrix.T

    print("H matrix shape: ", paper_author_matrix_H.shape)

    filtered_authors = hyp_authors_no_null[hyp_authors_no_null['id'].isin(unique_author_ids)]

    return paper_author_df, filtered_authors, paper_author_matrix_H, author_weights_matrix_R


def process_tags_column(authors_df):
    """
    Process the 'tags' column in the authors_df to create a dictionary of unique 't' values
    and generate an array with updated weights.

    Parameters:
        authors_df (pd.DataFrame): DataFrame with a 'tags' column containing a list of dictionaries.

    Returns:
        authors_df (pd.DataFrame): Updated DataFrame with a new 'tags_array' column.
        unique_t_dict (dict): Dictionary of unique 't' values as keys with their respective indices.
    """
    # Step 1: Convert the 'tags' column from strings to proper Python objects
    authors_df['tags'] = authors_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Step 2: Initialize an empty set to collect unique 't' values
    unique_t_set = set()

    # Step 3: Iterate through the 'tags' column to extract all unique 't' values
    for row in authors_df['tags']:
        if isinstance(row, list):  # Ensure the row is a list
            for tag in row:  # Iterate through each dictionary in the list
                if isinstance(tag, dict) and 't' in tag:  # Check if the key 't' exists
                    unique_t_set.add(tag['t'])  # Add the value of 't' to the set

    # Step 4: Create a dictionary with all unique 't' values as keys and 0 as the initial value
    unique_t_dict = {t: 0 for t in unique_t_set}

    # Display the resulting dictionary
    # print("Unique 't' Dictionary:")
    # print(unique_t_dict)

    unique_t_list = list(unique_t_dict.keys())  # To maintain order
    t_index_map = {t: idx for idx, t in enumerate(unique_t_list)}  # Map 't' to indices for the array

    # Step 2: Process each row in 'tags' to generate the corresponding array
    def create_tags_array(tags):
        tag_array = np.zeros(len(unique_t_dict), dtype=int)
        if isinstance(tags, list):  # Ensure the input is a list
            for tag in tags:
                t = tag.get('t')  # Get the 't' value
                w = tag.get('w', 0)  # Get the 'w' value (default 0)
                if t in t_index_map:
                    tag_array[t_index_map[t]] += w  # Add the weight to the respective index
        return tag_array

    # Apply the function to create the tags_array
    authors_df['features'] = authors_df['tags'].apply(create_tags_array)

    return authors_df, unique_t_dict


def load_feature_construct_H_and_R(data_dir, test_perc=0.2, seed=42):
    np.random.seed(seed)

    df, authors_df, H, R = pre_process_hyp_papers(data_dir)
    E_weights = df['n_citation'].values + 1

    # create feature matrix and class labels
    new_authors_df, unique_t_dict = process_tags_column(authors_df)

    # Version 1: X is the original node features
    X = np.vstack(new_authors_df['features'].values)

    # Version 2: X is the placeholder features (all ones)
    # X = np.ones((new_authors_df.shape[0], 1))

    # Version 3: X is the PCA-transformed features
    # # Standardize the data
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # # Create a PCA object with the desired number of components
    # pca = PCA(n_components=1024)

    # # Fit the PCA model to the data
    # pca.fit(X_scaled)

    # # Transform the data to the lower-dimensional space
    # X_pca = pca.transform(X_scaled)

    new_authors_df['Class_Label'] = 0  # All department except the three below will be in class 0
    new_authors_df.loc[new_authors_df.department == 'Compter Science', 'Class_Label'] = 1
    new_authors_df.loc[new_authors_df.department == 'Engineering', 'Class_Label'] = 2
    new_authors_df.loc[new_authors_df.department == 'Mathematics', 'Class_Label'] = 3
    print("Class labels:")
    print(new_authors_df.Class_Label.value_counts())
    Y = np.vstack(new_authors_df['Class_Label'].values)

    # create train-test split
    num_nodes = X.shape[0]
    test_size = int(num_nodes*test_perc)
    train_size = int(num_nodes - test_size)
    idx = np.concatenate((np.repeat(0, test_size),
                          np.repeat(1, train_size)))
    np.random.shuffle(idx)
    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]

    return H, R, E_weights, X, Y, idx_train, idx_test
