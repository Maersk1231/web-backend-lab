import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Read data from files
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

# Function to convert a document to a sequence of words, removing stop words
def review_to_wordlist(review, remove_stopwords=True):
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    # 4. Remove stop words if specified
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return words

# Load the trained Word2Vec model
print("Loading Word2Vec model...")
model = Word2Vec.load("300features_40minwords_10context")

# Get the vocabulary and vector size
vocab = model.wv.key_to_index
vector_size = model.vector_size
print(f"Vocabulary size: {len(vocab)}")
print(f"Vector size: {vector_size}")

# Method 1: Vector Averaging
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    # Initialize a feature vector, filled with zeros
    featureVec = np.zeros((num_features,), dtype="float32")
    # Count the number of words that are in the model's vocabulary
    nwords = 0
    # Loop over each word in the review and, if it's in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in model.wv:
            nwords += 1
            featureVec = np.add(featureVec, model.wv[word])
    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate the average feature vector for each one
    # Initialize a counter
    counter = 0
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # Loop through the reviews
    for review in reviews:
        # Print a status message every 1000th review
        if counter % 1000 == 0:
            print(f"Processing review {counter} of {len(reviews)}")
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        # Increment the counter
        counter += 1
    return reviewFeatureVecs

# Process the training and test sets
print("Processing training reviews...")
clean_train_reviews = [review_to_wordlist(review, remove_stopwords=True) for review in train["review"]]
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, vector_size)

print("Processing test reviews...")
clean_test_reviews = [review_to_wordlist(review, remove_stopwords=True) for review in test["review"]]
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, vector_size)

# Train a Random Forest classifier with the vector averaging method
print("Training Random Forest with vector averaging...")
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest = forest.fit(trainDataVecs, train["sentiment"])

# Predict on the test set
print("Predicting test set with vector averaging...")
predicted = forest.predict(testDataVecs)

# Save the results
ooutput = pd.DataFrame({"id": test["id"], "sentiment": predicted})
ooutput.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
print("Vector averaging results saved to Word2Vec_AverageVectors.csv")

# Method 2: Clustering
print("\n--- Clustering Method ---")
# Set the number of clusters - we'll use 1000 clusters
num_clusters = 1000

print(f"Training K-Means with {num_clusters} clusters...")
# Get word vectors from the model
word_vectors = model.wv.vectors
# Train K-Means
kmeans_clustering = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
idx = kmeans_clustering.fit_predict(word_vectors)

# Create a dictionary to map words to clusters
word_centroid_map = dict(zip(model.wv.index_to_key, idx))

# Function to convert a review to a bag-of-centroids
def create_bag_of_centroids(wordlist, word_centroid_map, num_clusters):
    # The number of clusters is equal to the highest cluster index in the word / centroid map
    bag_of_centroids = np.zeros(num_clusters, dtype="float32")
    # Loop over each word in the review
    for word in wordlist:
        if word in word_centroid_map:
            # Find which cluster the word belongs to, and increment that cluster count by one
            cluster_id = word_centroid_map[word]
            bag_of_centroids[cluster_id] += 1
    # Return the "bag of centroids"
    return bag_of_centroids

# Create bags of centroids for training set
print("Creating bags of centroids for training set...")
train_centroids = np.zeros((len(clean_train_reviews), num_clusters), dtype="float32")
for i, review in enumerate(clean_train_reviews):
    if i % 1000 == 0:
        print(f"Processing review {i} of {len(clean_train_reviews)}")
    train_centroids[i] = create_bag_of_centroids(review, word_centroid_map, num_clusters)

# Create bags of centroids for test set
print("Creating bags of centroids for test set...")
test_centroids = np.zeros((len(clean_test_reviews), num_clusters), dtype="float32")
for i, review in enumerate(clean_test_reviews):
    if i % 1000 == 0:
        print(f"Processing review {i} of {len(clean_test_reviews)}")
    test_centroids[i] = create_bag_of_centroids(review, word_centroid_map, num_clusters)

# Train a Random Forest classifier with the clustering method
print("Training Random Forest with clustering...")
forest_clustering = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clustering = forest_clustering.fit(train_centroids, train["sentiment"])

# Predict on the test set
print("Predicting test set with clustering...")
predicted_clustering = forest_clustering.predict(test_centroids)

# Save the results
ooutput_clustering = pd.DataFrame({"id": test["id"], "sentiment": predicted_clustering})
ooutput_clustering.to_csv("Word2Vec_BagOfCentroids.csv", index=False, quoting=3)
print("Clustering results saved to Word2Vec_BagOfCentroids.csv")

print("\nAll methods completed!")