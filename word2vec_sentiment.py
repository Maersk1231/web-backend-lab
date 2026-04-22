import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import gensim

# 加载训练好的Word2Vec模型
model_name = "300features_40minwords_10context"
model = gensim.models.Word2Vec.load(model_name)

# 文本清洗函数
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

# 方法1：向量平均法
def makeFeatureVec(words, model, num_features):
    # 初始化一个空向量
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    # 获取模型的词汇表
    index2word_set = set(model.wv.index_to_key)
    # 遍历评论中的每个词
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model.wv[word])
    # 求平均值
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # 初始化特征矩阵
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # 遍历每个评论
    for review in reviews:
        if counter % 1000 == 0:
            print(f"处理评论 {counter}/{len(reviews)}")
        # 清洗评论并生成特征向量
        reviewFeatureVecs[counter] = makeFeatureVec(review_to_wordlist(review, remove_stopwords=True), model, num_features)
        counter += 1
    return reviewFeatureVecs

# 方法2：聚类法
def create_bag_of_centroids(wordlist, word_centroid_map):
    # 词聚类的数量
    num_centroids = max(word_centroid_map.values()) + 1
    # 初始化重心袋
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    # 遍历评论中的每个词
    for word in wordlist:
        if word in word_centroid_map:
            centroid = word_centroid_map[word]
            bag_of_centroids[centroid] += 1
    return bag_of_centroids

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 设置参数
    num_features = 300
    
    # 方法1：使用向量平均法
    print("\n方法1: 向量平均法")
    print("生成训练集特征...")
    trainDataVecs = getAvgFeatureVecs(train["review"], model, num_features)
    print("生成测试集特征...")
    testDataVecs = getAvgFeatureVecs(test["review"], model, num_features)
    
    # 训练随机森林
    print("\n训练随机森林分类器...")
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest = forest.fit(trainDataVecs, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    result1 = forest.predict(testDataVecs)
    
    # 生成提交文件
    output1 = pd.DataFrame({"id": test["id"], "sentiment": result1})
    output1.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print("向量平均法提交文件已生成: Word2Vec_AverageVectors.csv")
    
    # 方法2：使用聚类法
    print("\n方法2: 聚类法")
    # 准备K-均值聚类
    word_vectors = model.wv.vectors
    num_clusters = int(word_vectors.shape[0] / 5)  # 每个聚类约5个词
    print(f"词汇量: {word_vectors.shape[0]}, 聚类数: {num_clusters}")
    
    # 运行K-均值聚类
    print("运行K-均值聚类...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    idx = kmeans.fit_predict(word_vectors)
    
    # 创建词到聚类的映射
    word_centroid_map = {}
    for i, word in enumerate(model.wv.index_to_key):
        word_centroid_map[word] = idx[i]
    
    # 生成训练集和测试集的重心袋
    print("生成训练集重心袋...")
    train_centroids = np.zeros((train.shape[0], num_clusters), dtype="float32")
    for i, review in enumerate(train["review"]):
        if i % 1000 == 0:
            print(f"处理训练评论 {i}/{train.shape[0]}")
        train_centroids[i] = create_bag_of_centroids(review_to_wordlist(review, remove_stopwords=True), word_centroid_map)
    
    print("生成测试集重心袋...")
    test_centroids = np.zeros((test.shape[0], num_clusters), dtype="float32")
    for i, review in enumerate(test["review"]):
        if i % 1000 == 0:
            print(f"处理测试评论 {i}/{test.shape[0]}")
        test_centroids[i] = create_bag_of_centroids(review_to_wordlist(review, remove_stopwords=True), word_centroid_map)
    
    # 训练随机森林
    print("\n训练随机森林分类器...")
    forest2 = RandomForestClassifier(n_estimators=100, random_state=42)
    forest2 = forest2.fit(train_centroids, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    result2 = forest2.predict(test_centroids)
    
    # 生成提交文件
    output2 = pd.DataFrame({"id": test["id"], "sentiment": result2})
    output2.to_csv("Word2Vec_BagOfCentroids.csv", index=False, quoting=3)
    print("聚类法提交文件已生成: Word2Vec_BagOfCentroids.csv")
    
    print("\nWord2Vec情感分析任务完成！")

if __name__ == "__main__":
    main()
