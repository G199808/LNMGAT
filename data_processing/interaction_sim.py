import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def interaction_sim():
    """Calculate and save interaction and cosine similarity matrices for drugs and targets."""
    # Read Excel file
    path = '../data/nr.xlsx'
    sheet = 'A_test'
    file_path = path
    df = pd.read_excel(file_path, sheet_name=sheet, index_col=0)

    # Generate the interaction association matrix for B (targets)
    B_interaction_matrix = np.dot(df.T, df)

    # Generate the interaction association matrix for A (drugs)
    A_interaction_matrix = np.dot(df, df.T)

    # Set the diagonal elements of B and A interaction matrices to 1
    np.fill_diagonal(B_interaction_matrix, 1)
    np.fill_diagonal(A_interaction_matrix, 1)

    # Compute cosine similarity matrices
    B_cosine_similarity = cosine_similarity(B_interaction_matrix)
    A_cosine_similarity = cosine_similarity(A_interaction_matrix)

    # Convert matrices to DataFrame format
    B_interaction_matrix_df = pd.DataFrame(B_interaction_matrix, index=df.columns, columns=df.columns)
    A_interaction_matrix_df = pd.DataFrame(A_interaction_matrix, index=df.index, columns=df.index)

    B_cosine_similarity_df = pd.DataFrame(B_cosine_similarity, index=df.columns, columns=df.columns)
    A_cosine_similarity_df = pd.DataFrame(A_cosine_similarity, index=df.index, columns=df.index)

    # Save cosine similarity matrices to CSV files
    # B_interaction_matrix_df.to_csv('../temp_data/B_interaction_matrix.csv', float_format='%.6f')
    # A_interaction_matrix_df.to_csv('../temp_data/A_interaction_matrix.csv', float_format='%.6f')
    B_cosine_similarity_df.to_csv('../temp_data/target_cosine_similarity.csv', float_format='%.6f')
    A_cosine_similarity_df.to_csv('../temp_data/drug_cosine_similarity.csv', float_format='%.6f')

    # Print status messages
    # print("B interaction association matrix has been saved as B_interaction_matrix.csv")
    # print("A interaction association matrix has been saved as A_interaction_matrix.csv")
    print("Target cosine similarity matrix has been saved as target_cosine_similarity.csv")
    print("Drug cosine similarity matrix has been saved as drug_cosine_similarity.csv")


# Run the function
interaction_sim()
