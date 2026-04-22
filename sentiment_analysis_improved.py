import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
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

# 加载Word2Vec模型
def load_word2vec_model():
    model_name = "300features_40minwords_10context"
    return gensim.models.Word2Vec.load(model_name)

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
    
    # 方法1：改进的词袋模型（TF-IDF）
    print("\n方法1: 改进的词袋模型（TF-IDF）")
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words='english', max_features=10000, ngram_range=(1, 2))
    
    print("生成TF-IDF特征...")
    train_tfidf = tfidf_vectorizer.fit_transform(train["review"])
    test_tfidf = tfidf_vectorizer.transform(test["review"])
    
    # 训练逻辑回归模型
    print("训练逻辑回归分类器...")
    log_reg = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
    log_reg.fit(train_tfidf, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    result_tfidf = log_reg.predict(test_tfidf)
    
    # 生成提交文件
    output_tfidf = pd.DataFrame({"id": test["id"], "sentiment": result_tfidf})
    output_tfidf.to_csv("TFIDF_LogisticRegression.csv", index=False, quoting=3)
    print("TF-IDF+逻辑回归提交文件已生成: TFIDF_LogisticRegression.csv")
    
    # 方法2：Word2Vec + 梯度提升树
    print("\n方法2: Word2Vec + 梯度提升树")
    model = load_word2vec_model()
    num_features = 300
    
    print("生成Word2Vec特征...")
    train_w2v = getAvgFeatureVecs(train["review"], model, num_features)
    test_w2v = getAvgFeatureVecs(test["review"], model, num_features)
    
    # 标准化特征
    scaler = StandardScaler()
    train_w2v_scaled = scaler.fit_transform(train_w2v)
    test_w2v_scaled = scaler.transform(test_w2v)
    
    # 训练梯度提升树
    print("训练梯度提升树分类器...")
    gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    gbm.fit(train_w2v_scaled, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    result_gbm = gbm.predict(test_w2v_scaled)
    
    # 生成提交文件
    output_gbm = pd.DataFrame({"id": test["id"], "sentiment": result_gbm})
    output_gbm.to_csv("Word2Vec_GradientBoosting.csv", index=False, quoting=3)
    print("Word2Vec+梯度提升树提交文件已生成: Word2Vec_GradientBoosting.csv")
    
    # 方法3：特征融合（TF-IDF + Word2Vec）
    print("\n方法3: 特征融合（TF-IDF + Word2Vec）")
    # 将TF-IDF特征转换为密集矩阵
    train_tfidf_dense = train_tfidf.toarray()
    test_tfidf_dense = test_tfidf.toarray()
    
    # 融合特征
    train_combined = np.hstack([train_tfidf_dense, train_w2v_scaled])
    test_combined = np.hstack([test_tfidf_dense, test_w2v_scaled])
    
    print(f"融合后特征维度: {train_combined.shape[1]}")
    
    # 训练随机森林
    print("训练随机森林分类器...")
    forest_combined = RandomForestClassifier(n_estimators=150, max_depth=50, random_state=42)
    forest_combined.fit(train_combined, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    result_combined = forest_combined.predict(test_combined)
    
    # 生成提交文件
    output_combined = pd.DataFrame({"id": test["id"], "sentiment": result_combined})
    output_combined.to_csv("Combined_Features.csv", index=False, quoting=3)
    print("特征融合提交文件已生成: Combined_Features.csv")
    
    # 方法4：支持向量机
    print("\n方法4: 支持向量机（SVM）")
    # 使用TF-IDF特征
    print("训练支持向量机分类器...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(train_tfidf, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    result_svm = svm.predict(test_tfidf)
    
    # 生成提交文件
    output_svm = pd.DataFrame({"id": test["id"], "sentiment": result_svm})
    output_svm.to_csv("SVM_TFIDF.csv", index=False, quoting=3)
    print("SVM+TF-IDF提交文件已生成: SVM_TFIDF.csv")
    
    print("\n情感分析模型改进任务完成！")

if __name__ == "__main__":
    main()
