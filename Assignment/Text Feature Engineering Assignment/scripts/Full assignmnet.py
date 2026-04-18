"""
Text Feature Engineering - Modular Implementation
All tasks in simple, modular functions
"""
import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================================
# TASK 1: PREPROCESSING
# ============================================================================
def load_data():
    """Load cleaned reviews with preprocessed_text."""
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "data", "cleaned_reviews.csv")
    df = pd.read_csv(file_path)
    return df


def create_sentiment_labels(texts):
    """Create sentiment labels based on review content."""
    positive_words = ['amazing', 'excellent', 'great', 'perfect', 'love', 
                      'good', 'wonderful', 'fantastic', 'best', 'awesome']
    negative_words = ['bad', 'terrible', 'worst', 'awful', 'waste', 
                      'poor', 'horrible', 'disappointing', 'hate', 'useless']
    
    labels = []
    for text in texts:
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            labels.append(1)  # Positive
        else:
            labels.append(0)  # Negative
    
    return labels


def get_preprocessed_texts(df):
    """Get preprocessed text column (already cleaned)."""
    texts = df["preprocessed_text"].astype(str).tolist()
    labels = create_sentiment_labels(texts)
    return texts, labels


# ============================================================================
# TASK 2: VOCABULARY CREATION
# ============================================================================
def build_vocabulary(texts):
    """Build vocabulary and analyze word frequencies."""
    # Split all texts into words
    all_words = []
    for text in texts:
        words = text.split()
        all_words.extend(words)
    
    # Count word frequencies
    word_freq = Counter(all_words)
    vocab = sorted(word_freq.keys())
    
    return vocab, word_freq


def print_vocabulary_analysis(vocab, word_freq):
    """Print vocabulary size and top frequent words."""
    print("\n" + "="*70)
    print("TASK 2: VOCABULARY CREATION & ANALYSIS")
    print("="*70)
    
    print(f"\nVocabulary size: {len(vocab)} unique words")
    print(f"Total words: {sum(word_freq.values())}")
    
    # Top 15 most frequent words
    top_words = word_freq.most_common(15)
    print("\nTop 15 Most Frequent Words:")
    print("-" * 70)
    for word, freq in top_words:
        print(f"  {word:<20} : {freq:>5} times")


# ============================================================================
# TASK 3: FEATURE ENGINEERING
# ============================================================================
def one_hot_encoding(texts, vocab):
    """Create One-Hot Encoded vectors."""
    vectors = []
    for text in texts:
        words = set(text.split())
        vector = [1 if word in words else 0 for word in vocab]
        vectors.append(vector)
    return np.array(vectors)


def bag_of_words(texts):
    """Create Bag of Words using CountVectorizer."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def tfidf_features(texts):
    """Create TF-IDF using TfidfVectorizer."""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def print_feature_engineering(ohe_vectors, bow_matrix, tfidf_matrix):
    """Print feature engineering results."""
    print("\n" + "="*70)
    print("TASK 3: FEATURE ENGINEERING")
    print("="*70)
    
    print("\n1. ONE-HOT ENCODING:")
    print(f"   Shape: {ohe_vectors.shape}")
    print(f"   Data type: Binary (0/1)")
    print(f"   Memory: {ohe_vectors.nbytes / 1024:.2f} KB")
    
    print("\n2. BAG OF WORDS (CountVectorizer):")
    print(f"   Shape: {bow_matrix.shape}")
    print(f"   Data type: Integer counts")
    print(f"   Memory: {bow_matrix.data.nbytes / 1024:.2f} KB")
    
    print("\n3. TF-IDF (TfidfVectorizer):")
    print(f"   Shape: {tfidf_matrix.shape}")
    print(f"   Data type: Float weights")
    print(f"   Memory: {tfidf_matrix.data.nbytes / 1024:.2f} KB")


# ============================================================================
# TASK 4: COMPARISON ANALYSIS
# ============================================================================
def print_comparison():
    """Compare the three feature engineering techniques."""
    print("\n" + "="*70)
    print("TASK 4: COMPARISON ANALYSIS")
    print("="*70)
    
    comparison_data = {
        "Aspect": [
            "Representation",
            "Word Frequency",
            "Word Importance",
            "Data Type",
            "Sparsity",
            "Best For"
        ],
        "One-Hot": [
            "Presence/Absence",
            "No",
            "No",
            "Binary (0/1)",
            "Very High",
            "Simple features"
        ],
        "Bag of Words": [
            "Word Counts",
            "Yes",
            "No",
            "Integer",
            "High",
            "Classification"
        ],
        "TF-IDF": [
            "Weighted Importance",
            "Yes",
            "Yes",
            "Float",
            "High",
            "Search/Ranking"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))


# ============================================================================
# TASK 5: SPARSE MATRIX ANALYSIS
# ============================================================================
def calculate_sparsity(matrix):
    """Calculate sparsity of a sparse matrix."""
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero = matrix.nnz
    sparsity = ((total_elements - non_zero) / total_elements) * 100
    return sparsity


def print_sparse_analysis(ohe_vectors, bow_matrix, tfidf_matrix):
    """Analyze and print sparse matrix information."""
    print("\n" + "="*70)
    print("TASK 5: SPARSE MATRIX ANALYSIS")
    print("="*70)
    
    # One-Hot Encoding
    total_ohe = ohe_vectors.shape[0] * ohe_vectors.shape[1]
    non_zero_ohe = np.count_nonzero(ohe_vectors)
    sparsity_ohe = ((total_ohe - non_zero_ohe) / total_ohe) * 100
    
    print("\n1. ONE-HOT ENCODING:")
    print(f"   Total elements: {total_ohe:,}")
    print(f"   Non-zero values: {non_zero_ohe:,}")
    print(f"   Sparsity: {sparsity_ohe:.2f}%")
    
    # Bag of Words
    sparsity_bow = calculate_sparsity(bow_matrix)
    print("\n2. BAG OF WORDS:")
    print(f"   Total elements: {bow_matrix.shape[0] * bow_matrix.shape[1]:,}")
    print(f"   Non-zero values: {bow_matrix.nnz:,}")
    print(f"   Sparsity: {sparsity_bow:.2f}%")
    
    # TF-IDF
    sparsity_tfidf = calculate_sparsity(tfidf_matrix)
    print("\n3. TF-IDF:")
    print(f"   Total elements: {tfidf_matrix.shape[0] * tfidf_matrix.shape[1]:,}")
    print(f"   Non-zero values: {tfidf_matrix.nnz:,}")
    print(f"   Sparsity: {sparsity_tfidf:.2f}%")
    
    print("\n" + "-"*70)
    print("WHY SPARSE MATRICES ARE INEFFICIENT FOR LARGE-SCALE SYSTEMS:")
    print("-"*70)
    print("""
    1. DENSE vs SPARSE STORAGE:
       • Dense: Stores every element (including millions of zeros)
       • Sparse: Only stores non-zero values + indices
       
    2. COMPUTATION:
       • Dense: Processes zeros, wasting CPU cycles
       • Sparse: Skips zero operations (1000x faster)
       
    3. MEMORY SCALING:
       • 1M docs × 1M vocabulary (dense): 1TB memory
       • Same dataset (sparse 99%): 1GB memory (1000x reduction!)
       
    4. PRACTICAL IMPACT:
       • Makes NLP at scale possible
       • Without sparse matrices, most real applications fail
       • Critical for production systems
    """)


# ============================================================================
# TASK 6: REAL-WORLD QUESTIONS
# ============================================================================
def print_real_world_questions():
    """Answer real-world questions about feature engineering."""
    print("\n" + "="*70)
    print("TASK 6: REAL-WORLD QUESTIONS & ANSWERS")
    print("="*70)
    
    print("""
    Q1: Why does Bag of Words fail in understanding semantic meaning?
    ─────────────────────────────────────────────────────────────────
    
    Answer: BoW treats each word independently and ignores:
    
    • SYNONYMS: "good", "excellent", "great" are treated as DIFFERENT words
      Example:
        Review 1: "This product is good"      (BoW: [good=1])
        Review 2: "This product is excellent" (BoW: [excellent=1])
        → BoW doesn't recognize they mean the same thing!
    
    • NEGATION: "NOT good" vs "good" both count "good"
      Example:
        Review 1: "This phone is NOT good"  → [not=1, good=1]
        Review 2: "This phone is good"      → [good=1]
        → Opposite meanings but BoW only sees "good" frequency!
    
    • CONTEXT: Word order is completely lost
      Example:
        "I love this product, it's bad quality" (sarcasm)
        "I hate this product" (negative)
        → BoW can't distinguish the sentiment!
    
    IMPACT: BoW achieves only ~80-85% accuracy for sentiment analysis.
            Modern embeddings achieve 95%+ accuracy.
    
    
    Q2: When to use Bag of Words and TF-IDF in industry?
    ─────────────────────────────────────────────────────────────────
    
    WHEN TO USE BAG OF WORDS:
    ✓ Fast real-time processing required
    ✓ Simple text classification (spam/not spam)
    ✓ When word frequency is the key signal
    ✓ Small datasets with limited vocabulary
    ✓ Baseline models
    
    Example: Email spam filter - high frequency of "click here" → spam
    
    
    WHEN TO USE TF-IDF:
    ✓ Information retrieval and search engines
    ✓ Document similarity and clustering
    ✓ Text classification with context
    ✓ Large vocabularies (100K+ words)
    ✓ Sentiment analysis
    ✓ Industry standard for feature extraction
    
    Example: Google Search - TF-IDF identifies important keywords
    
    
    Q3: Limitations of TF-IDF in real applications?
    ─────────────────────────────────────────────────────────────────
    
    MAJOR LIMITATIONS:
    
    1. NO SEMANTIC UNDERSTANDING
       • "good" and "great" still treated as different
       • Cannot understand word relationships
    
    2. NO WORD ORDER
       • "I hate good movies" (negative)
       • "Good movies I hate" (negative)
       • But BoW/TF-IDF treats them identically
    
    3. NEGATION PROBLEM
       • "NOT bad" (positive) = "bad" (negative) in TF-IDF
       • Leads to classification errors
    
    4. LOSES CONTEXT
       • "Apple fruit" vs "Apple phone"
       • Same word, different meanings (polysemy)
       • TF-IDF can't distinguish
    
    5. SCALABILITY ISSUES
       • Still requires huge matrices for large vocabularies
       • Becomes slow with millions of features
    
    6. SARCASM & IRONY
       • "Great job, you broke everything!"
       • TF-IDF: "great" = positive
       • Actual: negative (sarcasm)
    
    SOLUTION IN INDUSTRY: Use Word Embeddings (Word2Vec, GloVe) or 
    Transformers (BERT, GPT) for semantic understanding.
    """)


# ============================================================================
# TASK 7: SENTIMENT CLASSIFICATION
# ============================================================================
def prepare_data(texts, labels):
    """Split data into train and test."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer_class, clf_class, method_name):
    """Train a classifier and evaluate performance."""
    # Create features
    vectorizer = vectorizer_class()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train classifier
    clf = clf_class()
    clf.fit(X_train_vec, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_sentiment_classification(X_train, X_test, y_train, y_test):
    """Train and compare sentiment classifiers."""
    print("\n" + "="*70)
    print("TASK 7: SENTIMENT CLASSIFICATION")
    print("="*70)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Positive reviews: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"Negative reviews: {len(y_test)-sum(y_test)} ({(len(y_test)-sum(y_test))/len(y_test)*100:.1f}%)")
    
    # Train multiple models
    print("\n" + "-"*70)
    print("Training Models...")
    print("-"*70)
    
    results = []
    
    # Logistic Regression + Bag of Words
    result = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        CountVectorizer, LogisticRegression,
        "Logistic Regression + Bag of Words"
    )
    results.append(result)
    print(f"✓ {result['method']}")
    
    # Logistic Regression + TF-IDF
    result = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        TfidfVectorizer, LogisticRegression,
        "Logistic Regression + TF-IDF"
    )
    results.append(result)
    print(f"✓ {result['method']}")
    
    # Naive Bayes + Bag of Words
    result = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        CountVectorizer, MultinomialNB,
        "Naive Bayes + Bag of Words"
    )
    results.append(result)
    print(f"✓ {result['method']}")
    
    # Naive Bayes + TF-IDF
    result = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        TfidfVectorizer, MultinomialNB,
        "Naive Bayes + TF-IDF"
    )
    results.append(result)
    print(f"✓ {result['method']}")
    
    # Print results
    print("\n" + "-"*70)
    print("PERFORMANCE COMPARISON")
    print("-"*70)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['f1'].idxmax()
    best_model = results_df.loc[best_idx, 'method']
    best_f1 = results_df.loc[best_idx, 'f1']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   F1-Score: {best_f1:.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main function - Execute all tasks."""
    print("\n" + "="*70)
    print("TEXT FEATURE ENGINEERING - COMPLETE PIPELINE")
    print("="*70)
    
    # Load data
    df = load_data()
    texts, labels = get_preprocessed_texts(df)
    print(f"\n✓ Loaded {len(texts)} documents")
    print(f"✓ Using 'preprocessed_text' column")
    print(f"✓ Sample: {texts[0][:60]}...")
    
    # Task 2: Vocabulary
    vocab, word_freq = build_vocabulary(texts)
    print_vocabulary_analysis(vocab, word_freq)
    
    # Task 3: Feature Engineering
    ohe_vectors = one_hot_encoding(texts, vocab)
    bow_matrix, bow_vec = bag_of_words(texts)
    tfidf_matrix, tfidf_vec = tfidf_features(texts)
    print_feature_engineering(ohe_vectors, bow_matrix, tfidf_matrix)
    
    # Task 4: Comparison
    print_comparison()
    
    # Task 5: Sparse Matrix Analysis
    print_sparse_analysis(ohe_vectors, bow_matrix, tfidf_matrix)
    
    # Task 6: Real-world Questions
    print_real_world_questions()
    
    # Task 7: Sentiment Classification
    X_train, X_test, y_train, y_test = prepare_data(texts, labels)
    print_sentiment_classification(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("✅ ALL TASKS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
