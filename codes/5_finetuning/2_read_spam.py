import pandas as pd 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_balanced_dataset

data_file_path = "data/sms_spam_collection/SMSSpamCollection.tsv"

df = pd.read_csv( 
    data_file_path, sep="\t", header=None, names=["Label", "Text"] 
)
balanced_df = create_balanced_dataset(df) 
print(balanced_df["Label"].value_counts())

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

def random_split(df, train_frac, validation_frac): 
    df = df.sample( 
        frac=1, random_state=123 
    ).reset_index(drop=True) 
    train_end = int(len(df) * train_frac) 
    validation_end = train_end + int(len(df) * validation_frac) 
    train_df = df[:train_end] 
    validation_df = df[train_end:validation_end] 
    test_df = df[validation_end:] 
    return train_df, validation_df, test_df 

train_df, validation_df, test_df = random_split( 
    balanced_df, 0.7, 0.1)

train_df.to_csv("data/spam/train.csv", index=None) 
validation_df.to_csv("data/spam/validation.csv", index=None) 
test_df.to_csv("data/spam/test.csv", index=None)
