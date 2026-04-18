import re
import nltk
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Clean Text (Basic)
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Sentiment Rule-Based
# -----------------------------
positive_keywords = [
    'amazing', 'excellent', 'great', 'love', 'perfect',
    'fantastic', 'wonderful', 'outstanding', 'brilliant',
    'superb', 'exceptional', 'highly', 'satisfied', 'impressed'
]

negative_keywords = [
    'terrible', 'awful', 'poor', 'bad', 'waste', 'horrible',
    'disappointed', 'disappointing', 'defective', 'cheap', 'worst'
]

def categorize_sentiment(text):
    text = str(text).lower()
    
    positive_count = sum(1 for word in positive_keywords if word in text)
    negative_count = sum(1 for word in negative_keywords if word in text)

    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return 0
    else:
        return 1 if len(text) > 50 else 0

# -----------------------------
# Full Preprocessing Pipeline
# -----------------------------
def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens), tokens


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # File path
    BASE_DIR = os.path.dirname(__file__)
    file_path = os.path.join(BASE_DIR, "data", "reviews.csv")

    # Load data
    df = pd.read_csv(file_path)

    print("✅ Loaded Data:")
    print(df.head())

    # Apply processing
    print("\n🔄 Processing...")

    df["clean_text"] = df["review_text"].apply(clean_text)
    df["preprocessed_text"] = df["review_text"].apply(
        lambda x: preprocess_text(x)[0]
    )
    df["sentence_tokens"] = df["review_text"].apply(categorize_sentiment)

    # Show sample
    print("\n✅ Sample Output:")
    for i in range(3):
        print(f"\nReview {i+1}:")
        print("Original:", df["review_text"].iloc[i])
        print("sentence_tokens:", df["sentence_tokens"].iloc[i])


    # Save file
    output_path = os.path.join(BASE_DIR, "data", "cleaned_reviews.csv")
    df.to_csv(output_path, index=False)

    print(f"\n✅ Saved at: {output_path}")