"""
Text Feature Engineering Comparison Script
Modular implementation for comparing One-Hot Encoding, Bag of Words, and TF-IDF
"""
import pandas as pd
import numpy as np
import time
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ============================================================================
# MODULE 1: DATA LOADING
# ============================================================================
def load_data(num_samples=None):
    """Load preprocessed text data from CSV."""
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "data", "cleaned_reviews.csv")
    
    df = pd.read_csv(file_path)
    texts = df["preprocessed_text"].astype(str)
    
    if num_samples:
        texts = texts.head(num_samples)
    
    return texts


# ============================================================================
# MODULE 2: ONE-HOT ENCODING
# ============================================================================
def one_hot_encoding(texts):
    """Create One-Hot encoded vectors."""
    vocab = list(set(" ".join(texts).split()))
    vectors = []
    
    for text in texts:
        words = text.split()
        vector = [1 if word in words else 0 for word in vocab]
        vectors.append(vector)
    
    return np.array(vectors), vocab


def print_one_hot(vectors, vocab):
    """Print One-Hot Encoding results."""
    print("\n" + "="*70)
    print("1. ONE-HOT ENCODING")
    print("="*70)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Output shape: {vectors.shape}")
    print(f"Memory usage: {vectors.nbytes / 1024:.2f} KB")
    print(f"Sparsity: {(vectors == 0).sum() / vectors.size * 100:.2f}% zeros")
    print(f"Sample (doc 1, first 10): {vectors[0][:10]}")


# ============================================================================
# MODULE 3: BAG OF WORDS
# ============================================================================
def bag_of_words(texts):
    """Create Bag of Words vectors using CountVectorizer."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    return matrix, features


def print_bow(matrix, features):
    """Print Bag of Words results."""
    print("\n" + "="*70)
    print("2. BAG OF WORDS (CountVectorizer)")
    print("="*70)
    print(f"Vocabulary size: {len(features)}")
    print(f"Output shape: {matrix.shape}")
    print(f"Memory usage: {matrix.data.nbytes / 1024:.2f} KB")
    sparsity = 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
    print(f"Sparsity: {sparsity * 100:.2f}%")
    print(f"Non-zero values (doc 1): {matrix[0].nnz}")


# ============================================================================
# MODULE 4: TF-IDF
# ============================================================================
def tfidf_features(texts):
    """Create TF-IDF vectors using TfidfVectorizer."""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    return matrix, features


def print_tfidf(matrix, features):
    """Print TF-IDF results."""
    print("\n" + "="*70)
    print("3. TF-IDF (Term Frequency-Inverse Document Frequency)")
    print("="*70)
    print(f"Vocabulary size: {len(features)}")
    print(f"Output shape: {matrix.shape}")
    print(f"Memory usage: {matrix.data.nbytes / 1024:.2f} KB")
    sparsity = 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
    print(f"Sparsity: {sparsity * 100:.2f}%")
    print(f"Non-zero values (doc 1): {matrix[0].nnz}")


# ============================================================================
# MODULE 5: COMPARISON
# ============================================================================
def print_comparison():
    """Print comparison table of all techniques."""
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    data = {
        "Technique": ["One-Hot", "Bag of Words", "TF-IDF"],
        "Data Type": ["Binary", "Integer", "Float"],
        "Word Freq": ["No", "Yes", "Yes (Weighted)"],
        "Memory": ["High", "Low", "Low"],
        "Speed": ["Fast", "Very Fast", "Fast"],
        "Best Use": ["Binary", "Classification", "Search/Ranking"]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))


# ============================================================================
# MODULE 6: MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("TEXT FEATURE ENGINEERING COMPARISON")
    print("="*70)
    
    # Load data
    texts = load_data(num_samples=10)
    print(f"\n✅ Loaded {len(texts)} documents")
    print(f"✅ Sample: {texts.iloc[0][:60]}...")
    
    # Apply techniques
    ohe_vectors, vocab = one_hot_encoding(texts)
    bow_matrix, bow_features = bag_of_words(texts)
    tfidf_matrix, tfidf_features = tfidf_features(texts)
    
    # Print results
    print_one_hot(ohe_vectors, vocab)
    print_bow(bow_matrix, bow_features)
    print_tfidf(tfidf_matrix, tfidf_features)
    print_comparison()
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
