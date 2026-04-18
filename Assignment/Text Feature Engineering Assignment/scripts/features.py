import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from gensim.utils import simple_preprocess


def simple_preprocess_text(text):
    """
    Converts text to lowercase, removes punctuation, 
    and returns a list of tokens.
    """
    if isinstance(text, str):
        return simple_preprocess(text)
    return []


def one_hot_encoding(texts):
    """One-hot encoding at document level"""
    vocab = list(set(" ".join(texts).split()))
    print(f"Vocabulary size: {len(vocab)}")
    vectors = []
    print("texts:", texts)
    for text in texts:
        words = text.split()
        vector = [1 if word in words else 0 for word in vocab]
        vectors.append(vector)

    return vectors, vocab


def bag_of_words(texts):
    """Bag of words using CountVectorizer"""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def tfidf_features(texts):
    """TF-IDF features using TfidfVectorizer"""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


if __name__ == "__main__":
    # Load data
    BASE_DIR = os.path.dirname(__file__)
    file_path = os.path.join(BASE_DIR, "data", "cleaned_reviews.csv")
    df = pd.read_csv(file_path)

    print("✅ Loaded Data:")
    print(df.head())

    # Apply preprocessing
    if "preprocessed_text" in df.columns:
        df["sentence_tokens"] = df["preprocessed_text"].apply(simple_preprocess_text)
    
    texts = df["preprocessed_text"].astype(str)

    # 1. ONE-HOT ENCODING
    print("\n" + "="*50)
    print("1. ONE-HOT ENCODING")
    print("="*50)
    ohe_vectors, vocab = one_hot_encoding(texts)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample vector shape: {len(ohe_vectors[0])}")

    # 2. BAG OF WORDS
    print("\n" + "="*50)
    print("2. BAG OF WORDS")
    print("="*50)
    bow_vectors, bow_vectorizer = bag_of_words(texts)
    print(f"BOW matrix shape: {bow_vectors.shape}")
    print(f"Vocabulary size: {len(bow_vectorizer.get_feature_names_out())}")

    # 3. TF-IDF
    print("\n" + "="*50)
    print("3. TF-IDF")
    print("="*50)
    tfidf_vectors, tfidf_vectorizer = tfidf_features(texts)
    print(f"TF-IDF matrix shape: {tfidf_vectors.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    # Show sample output
    print("\n" + "="*50)
    print("Sample Output")
    print("="*50)
    for i in range(min(3, len(df))):
        print(f"\nReview {i+1}:")
        print(f"Original: {df['preprocessed_text'].iloc[i][:100]}...")
        print(f"Tokens: {df['sentence_tokens'].iloc[i]}")