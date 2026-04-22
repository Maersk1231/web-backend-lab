import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import word2vec

# Read data from files
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# Verify the number of reviews that were read (100,000 in total)
print(f"Read {train['review'].size} labeled train reviews, {test['review'].size} labeled test reviews, " 
      f"and {unlabeled_train['review'].size} unlabeled reviews\n")

# Function to convert a document to a sequence of words, optionally removing stop words
def review_to_wordlist(review, remove_stopwords=False):
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return words

# Download the punkt tokenizer for sentence splitting
print("Downloading punkt tokenizer...")
nltk.download('punkt_tab')

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt_tab/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    # Return the list of sentences (each sentence is a list of words)
    return sentences

# Initialize an empty list of sentences
sentences = []

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

# Check how many sentences we have in total
print(f"Total number of sentences: {len(sentences)}")

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, 
            vector_size=num_features, min_count=min_word_count, 
            window=context, sample=downsampling)

# If you don't plan to train the model any further, calling init_sims will make it more memory-efficient
model.init_sims(replace=True)

# Save the model for later use
model_name = "300features_40minwords_10context"
model.save(model_name)

print(f"Model saved as {model_name}")
print("Word2Vec training completed!")