import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import time

# Load data
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "data", "cleaned_reviews.csv")
df = pd.read_csv(file_path)

texts = df["review_text"].astype(str).head(10)  # Use first 10 for comparison

print("="*80)
print("COMPARISON ANALYSIS: Feature Engineering Techniques")
print("="*80)

# ============================================================================
# 1. ONE-HOT ENCODING (Document Level)
# ============================================================================
print("\n1. ONE-HOT ENCODING (Document Level)")
print("-" * 80)

print("Description:")
print("  • Binary representation: each word gets a binary vector")
print("  • 1 if word appears in document, 0 otherwise")
print("  • Vocabulary size = Vector dimension")

vocab = list(set(" ".join(texts).split()))
print(f"\nVocabulary size: {len(vocab)}")
print(f"Sample vocabulary (first 10): {vocab[:10]}")

# Create one-hot vectors
ohe_vectors = []
for text in texts:
    words = text.split()
    vector = [1 if word in words else 0 for word in vocab]
    ohe_vectors.append(vector)

ohe_vectors = np.array(ohe_vectors)
print(f"\nOutput shape: {ohe_vectors.shape}")
print(f"Output type: Dense numpy array")
print(f"Sparsity: {(ohe_vectors == 0).sum() / ohe_vectors.size * 100:.2f}% zeros")
print(f"Memory usage: {ohe_vectors.nbytes / 1024:.2f} KB")

print("\nSample vector (first document, first 10 dimensions):")
print(ohe_vectors[0][:10])

print("\nAdvantages:")
print("  ✓ Simple and interpretable")
print("  ✓ No word order dependency")
print("  ✓ Works well for small vocabularies")

print("\nDisadvantages:")
print("  ✗ High dimensionality with large vocabularies")
print("  ✗ Loss of word frequency information")
print("  ✗ Very sparse vectors")
print("  ✗ Memory intensive")

# ============================================================================
# 2. BAG OF WORDS (CountVectorizer)
# ============================================================================
print("\n" + "="*80)
print("2. BAG OF WORDS (CountVectorizer)")
print("-" * 80)

print("Description:")
print("  • Counts term frequency in each document")
print("  • Each dimension = count of that word in document")
print("  • Preserves word frequency information")

vectorizer_bow = CountVectorizer()
bow_matrix = vectorizer_bow.fit_transform(texts)
bow_feature_names = vectorizer_bow.get_feature_names_out()

print(f"\nVocabulary size: {len(bow_feature_names)}")
print(f"Sample vocabulary (first 10): {list(bow_feature_names[:10])}")

print(f"\nOutput shape: {bow_matrix.shape}")
print(f"Output type: Sparse CSR matrix")
print(f"Sparsity: {1 - (bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])) * 100:.2f}% sparse")
print(f"Memory usage: {bow_matrix.data.nbytes / 1024:.2f} KB (sparse storage)")

print("\nSample vector (first document):")
print(f"  Non-zero values: {bow_matrix[0].nnz}")
non_zero_indices = bow_matrix[0].nonzero()[1]
non_zero_values = bow_matrix[0].data
if len(non_zero_indices) > 0:
    sample_dict = {bow_feature_names[idx]: int(bow_matrix[0, idx]) for idx in non_zero_indices}
    print(f"  Non-zero elements: {sample_dict}")

print("\nAdvantages:")
print("  ✓ Preserves word frequency information")
print("  ✓ Sparse matrix representation (memory efficient)")
print("  ✓ Works well for text classification")
print("  ✓ Simple and fast computation")

print("\nDisadvantages:")
print("  ✗ Still high dimensional")
print("  ✗ Ignores word importance (all words weighted equally)")
print("  ✗ Order of words not considered")
print("  ✗ Context ignored")

# ============================================================================
# 3. TF-IDF (Term Frequency-Inverse Document Frequency)
# ============================================================================
print("\n" + "="*80)
print("3. TF-IDF (Term Frequency-Inverse Document Frequency)")
print("-" * 80)

print("Description:")
print("  • Weighted term frequency based on importance")
print("  • TF-IDF = Term Frequency × Inverse Document Frequency")
print("  • Highlights rare and important words")
print("  • Reduces weight of common words")

vectorizer_tfidf = TfidfVectorizer()
tfidf_matrix = vectorizer_tfidf.fit_transform(texts)
tfidf_feature_names = vectorizer_tfidf.get_feature_names_out()

print(f"\nVocabulary size: {len(tfidf_feature_names)}")
print(f"Sample vocabulary (first 10): {list(tfidf_feature_names[:10])}")

print(f"\nOutput shape: {tfidf_matrix.shape}")
print(f"Output type: Sparse CSR matrix with float values")
print(f"Sparsity: {1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}% sparse")
print(f"Memory usage: {tfidf_matrix.data.nbytes / 1024:.2f} KB (sparse storage)")

print("\nSample vector (first document):")
print(f"  Non-zero values: {tfidf_matrix[0].nnz}")
top_indices = np.argsort(tfidf_matrix[0].toarray()[0])[-5:][::-1]
print(f"  Top 5 TF-IDF values:")
for idx in top_indices:
    if tfidf_matrix[0, idx] > 0:
        print(f"    {tfidf_feature_names[idx]}: {tfidf_matrix[0, idx]:.4f}")

print("\nAdvantages:")
print("  ✓ Identifies important words automatically")
print("  ✓ Better for text classification and search")
print("  ✓ Sparse representation (memory efficient)")
print("  ✓ Works well on large datasets")
print("  ✓ Reduces noise from common words")

print("\nDisadvantages:")
print("  ✗ Still loses word order and context")
print("  ✗ More complex than BOW")
print("  ✗ Requires IDF computations")

# ============================================================================
# SUMMARY COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY COMPARISON TABLE")
print("="*80)

comparison_data = {
    "Technique": ["One-Hot Encoding", "Bag of Words", "TF-IDF"],
    "Vector Type": ["Binary", "Integer Counts", "Float Weights"],
    "Preserves Frequency": ["No", "Yes", "Yes (weighted)"],
    "Vocabulary": [f"{len(vocab)}", f"{len(bow_feature_names)}", f"{len(tfidf_feature_names)}"],
    "Sparsity": ["Very High", "High", "High"],
    "Memory Usage": ["High", "Low", "Low"],
    "Speed": ["Fast", "Very Fast", "Fast"],
    "Best For": ["Binary features", "Text classification", "Search/ranking"],
    "Handles Stopwords": ["No", "No", "Yes (weights low)"]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# DETAILED OUTPUT COMPARISON
# ============================================================================
print("\n" + "="*80)
print("DETAILED OUTPUT COMPARISON (First Document)")
print("="*80)

print("\nOne-Hot Encoding Vector (first 10 dimensions):")
print(f"  {ohe_vectors[0][:10]}")
print(f"  Shape: {ohe_vectors[0].shape}")
print(f"  Mean value: {np.mean(ohe_vectors[0]):.4f}")
print(f"  Std deviation: {np.std(ohe_vectors[0]):.4f}")

print("\nBag of Words Vector (non-zero elements):")
bow_dense = bow_matrix[0].toarray()[0]
print(f"  Non-zero count: {np.count_nonzero(bow_dense)}")
print(f"  Max term count: {np.max(bow_dense)}")
print(f"  Total term count: {np.sum(bow_dense)}")
print(f"  Mean value (non-zero): {np.mean(bow_dense[bow_dense > 0]):.4f}")

print("\nTF-IDF Vector (non-zero elements):")
tfidf_dense = tfidf_matrix[0].toarray()[0]
print(f"  Non-zero count: {np.count_nonzero(tfidf_dense)}")
print(f"  Max TF-IDF score: {np.max(tfidf_dense):.4f}")
print(f"  Min TF-IDF score (non-zero): {np.min(tfidf_dense[tfidf_dense > 0]):.4f}")
print(f"  Mean TF-IDF value (non-zero): {np.mean(tfidf_dense[tfidf_dense > 0]):.4f}")

# ============================================================================
# USE CASE RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("USE CASE RECOMMENDATIONS")
print("="*80)

print("\nOne-Hot Encoding:")
print("  → Small datasets with limited vocabulary")
print("  → When word presence/absence is important")
print("  → Simple feature representation")

print("\nBag of Words:")
print("  → Text classification tasks")
print("  → Sentiment analysis")
print("  → Spam detection")
print("  → Topic modeling (LDA, NMF)")

print("\nTF-IDF:")
print("  → Information retrieval and search engines")
print("  → Document similarity and clustering")
print("  → Text classification with large vocabularies")
print("  → Feature extraction for machine learning")
print("  → Ranking and relevance scoring")

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE METRICS")
print("="*80)

# Time comparison
print("\nComputation Speed (for 10 documents):")

start = time.time()
for text in texts:
    words = text.split()
    vector = [1 if word in words else 0 for word in vocab]
ohe_time = time.time() - start
print(f"  One-Hot Encoding: {ohe_time*1000:.4f} ms")

start = time.time()
vectorizer_bow.fit_transform(texts)
bow_time = time.time() - start
print(f"  Bag of Words: {bow_time*1000:.4f} ms")

start = time.time()
vectorizer_tfidf.fit_transform(texts)
tfidf_time = time.time() - start
print(f"  TF-IDF: {tfidf_time*1000:.4f} ms")

print("\n" + "="*80)
print("END OF COMPARISON")
print("="*80)
