import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import gensim

# 文本清洗函数
def review_to_wordlist(review, remove_stopwords=False):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
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

# 向量平均法
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model.wv[word])
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000 == 0:
            print(f"处理评论 {counter}/{len(reviews)}")
        reviewFeatureVecs[counter] = makeFeatureVec(review_to_wordlist(review, remove_stopwords=True), model, num_features)
        counter += 1
    return reviewFeatureVecs

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 方法1：TF-IDF + 逻辑回归（最可能提高分数的方法）
    print("\n方法1: TF-IDF + 逻辑回归")
    # 使用更优化的参数
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word', 
        tokenizer=None, 
        preprocessor=None, 
        stop_words='english', 
        max_features=15000,  # 增加特征数量
        ngram_range=(1, 2),  # 包含二元组
        min_df=3,  # 最小文档频率
        max_df=0.8  # 最大文档频率
    )
    
    print("生成TF-IDF特征...")
    train_tfidf = tfidf_vectorizer.fit_transform(train["review"])
    test_tfidf = tfidf_vectorizer.transform(test["review"])
    
    # 训练逻辑回归模型，使用更优参数
    print("训练逻辑回归分类器...")
    log_reg = LogisticRegression(
        max_iter=2000, 
        C=0.8,  # 正则化参数
        solver='liblinear', 
        penalty='l2'  # L2正则化
    )
    log_reg.fit(train_tfidf, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    result_tfidf = log_reg.predict(test_tfidf)
    
    # 生成提交文件
    output_tfidf = pd.DataFrame({"id": test["id"], "sentiment": result_tfidf})
    output_tfidf.to_csv("TFIDF_LogisticRegression_Improved.csv", index=False, quoting=3)
    print("改进的TF-IDF+逻辑回归提交文件已生成: TFIDF_LogisticRegression_Improved.csv")
    
    # 方法2：Word2Vec + 随机森林（优化参数）
    print("\n方法2: Word2Vec + 随机森林")
    # 加载Word2Vec模型
    model_name = "300features_40minwords_10context"
    model = gensim.models.Word2Vec.load(model_name)
    num_features = 300
    
    print("生成Word2Vec特征...")
    train_w2v = getAvgFeatureVecs(train["review"], model, num_features)
    test_w2v = getAvgFeatureVecs(test["review"], model, num_features)
    
    # 训练优化的随机森林
    print("训练随机森林分类器...")
    forest = RandomForestClassifier(
        n_estimators=200,  # 增加树的数量
        max_depth=50,  # 增加树的深度
        min_samples_split=5,  # 最小分裂样本数
        min_samples_leaf=2,  # 最小叶节点样本数
        random_state=42
    )
    forest.fit(train_w2v, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    result_w2v = forest.predict(test_w2v)
    
    # 生成提交文件
    output_w2v = pd.DataFrame({"id": test["id"], "sentiment": result_w2v})
    output_w2v.to_csv("Word2Vec_RandomForest_Improved.csv", index=False, quoting=3)
    print("改进的Word2Vec+随机森林提交文件已生成: Word2Vec_RandomForest_Improved.csv")
    
    print("\n情感分析模型改进任务完成！")

if __name__ == "__main__":
    main()
