import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


def calculate_fingerprint_similarity(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


def main():
    try:
        # Read Excel file
        df = pd.read_excel('./data/davis_drug.xlsx', sheet_name='Sheet1')

        # Extract drug IDs and SMILES strings
        drug_ids = df['compound_id'].tolist()
        smiles_list = df['compound_iso_smiles'].tolist()

        # Initialize similarity matrix
        num_drugs = len(drug_ids)
        similarity_matrix = np.zeros((num_drugs, num_drugs))

        # Compute similarity matrix
        for i in range(num_drugs):
            for j in range(i, num_drugs):
                if i == j:
                    similarity = 1.0  # Diagonal is 1
                else:
                    similarity = calculate_fingerprint_similarity(smiles_list[i], smiles_list[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        # Round to 5 decimal places
        similarity_matrix = np.round(similarity_matrix, 5)

        # Create DataFrame
        similarity_df = pd.DataFrame(similarity_matrix, index=drug_ids, columns=drug_ids)

        # Save as CSV file
        similarity_df.to_csv('drug_similarity_matrix_rdkit.csv')

        print("Drug similarity matrix calculation completed and saved as drug_similarity_matrix_rdkit.csv")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure that:")
        print("1. The Excel file path is correct")
        print("2. The Excel file contains 'compound_id' and 'compound_iso_smiles' columns")
        print("3. pandas, numpy, and rdkit libraries are installed")


if __name__ == '__main__':
    # Check required libraries
    try:
        import pandas as pd
        import numpy as np
        from rdkit import Chem

        print("All required libraries are installed")
        main()
    except ImportError as e:
        print(f"Missing required library: {str(e)}")
        print("Please install with:")
        print("pip install pandas numpy rdkit")
