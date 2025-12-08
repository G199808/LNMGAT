import pandas as pd
import numpy as np
from data_processing.data_load import load_data


def get_pair_dic():
    """Generate feature dictionary for drug-target pairs."""
    drug_s1, target_s1, drug_s2, target_s2, A, pair_index = load_data('../data/nr.xlsx')  # Load data
    S_drug = 0.5 * drug_s1 + 0.5 * drug_s2
    S_target = 0.5 * target_s1 + 0.5 * target_s2
    num_drugs, num_targets = A.shape
    print(A.shape)
    features = []
    for i in range(num_drugs):
        for j in range(num_targets):
            # Combine drug and target similarity features
            feature = np.concatenate([S_drug[i], S_target[j]])
            features.append(feature)
    pair_dic = {}
    for key, value in zip(pair_index, features):
        pair_dic[key] = value
    print(pair_dic)
    pd.DataFrame.from_dict(pair_dic, orient='index').to_csv("../run/pair_dic_nr.csv", index=True, float_format='%.6f')


def get_train_data():
    """Generate training dataset with top and bottom 5% samples."""
    path1 = '../run/pair_dic_nr.csv'
    path2 = '../run/LapRLS pre.xlsx'
    # Read files
    pair_data = pd.read_csv(path1)
    pair_data.set_index(pair_data.columns[0], inplace=True)
    data_pre = pd.read_excel(path2, sheet_name='nr')
    # Calculate the index positions for top 5% and bottom 5%
    n = len(data_pre)
    top_5_percent_index = int(n * 0.05)
    bottom_5_percent_index = int(n * 0.95)
    positive_index = list(data_pre[:top_5_percent_index]['index'])
    negative_index = list(data_pre[bottom_5_percent_index:]['index'])
    positive_value = pair_data.loc[positive_index].values.tolist()
    negative_value = pair_data.loc[negative_index].values.tolist()

    train_index = positive_index + negative_index
    train_value = positive_value + negative_value

    train_df = pd.DataFrame(train_value, index=train_index)
    labels = [1] * len(positive_index) + [0] * len(negative_index)

    # Insert label column as the first column
    train_df.insert(0, 'label', labels)
    print(train_df)
    train_df.to_csv('../run/train_nr.csv', index=True)


def get_test_data():
    """Generate test dataset using ML_test_index."""
    path1 = '../data/nr.xlsx'
    path2 = '../run/pair_dic_nr.csv'
    xls = pd.ExcelFile(path1)
    test_index = list(xls.parse("ML_test_index")['ML_test'])
    pair_data = pd.read_csv(path2)
    pair_data.set_index(pair_data.columns[0], inplace=True)

    test_value = pair_data.loc[test_index].values.tolist()
    test_df = pd.DataFrame(test_value, index=test_index)
    test_df.to_csv('../run/test_nr.csv', index=True, float_format='%.6f')


def get_testAll_data():
    """Generate full test dataset using ML_testAll_index."""
    path1 = '../data/nr.xlsx'
    path2 = '../run/pair_dic_nr.csv'
    xls = pd.ExcelFile(path1)
    test_index = list(xls.parse("ML_testAll_index")['ML_testAll'])
    pair_data = pd.read_csv(path2)
    pair_data.set_index(pair_data.columns[0], inplace=True)

    test_value = pair_data.loc[test_index].values.tolist()
    test_df = pd.DataFrame(test_value, index=test_index)
    test_df.to_csv('../run/testAll_nr.csv', index=True, float_format='%.6f')


# Uncomment to run functions
# get_pair_dic()
# get_train_data()
# get_test_data()
# get_testAll_data()
