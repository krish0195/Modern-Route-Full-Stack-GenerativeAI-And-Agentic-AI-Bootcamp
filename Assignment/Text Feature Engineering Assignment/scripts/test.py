import pandas as pd
from gensim.utils import simple_preprocess
import os


def tokenize_text(df, column_name):
    """
    Tokenize each row of a text column into a list of words
    """
    return df[column_name].apply(lambda x: simple_preprocess(str(x)))


def main():
    # Load data
    BASE_DIR = os.path.dirname(__file__)
    file_path = os.path.join(BASE_DIR, "data", "cleaned_reviews.csv")

    # Load data
    df = pd.read_csv(file_path)


    print("✅ Loaded Data:")
    print(df.head())

    # -----------------------------
    # TOKENIZATION
    # -----------------------------
    df["sentence_tokens"] = tokenize_text(df, "preprocessed_text")

    # -----------------------------
    # SAMPLE OUTPUT
    # -----------------------------
    print("\n✅ Sample Output:")
    for i in range(min(3, len(df))):
        print(f"\nReview {i+1}:")
        print("Original:", df["preprocessed_text"].iloc[i])
        print("Tokens :", df["sentence_tokens"].iloc[i])
        print("-" * 50)


if __name__ == "__main__":
    main()