import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer

# 改进的文本清洗函数
def review_to_wordlist(review, remove_stopwords=True):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
    
    # 2. 保留情感相关的标点符号
    # 替换感叹号和问号为特殊标记
    review_text = review_text.replace('!', ' exclamationmark ')
    review_text = review_text.replace('?', ' questionmark ')
    
    # 3. Remove non-letters and numbers (保留数字可能有助于情感分析)
    review_text = re.sub("[^a-zA-Z0-9]"," ", review_text)
    
    # 4. Convert words to lower case and split them
    words = review_text.lower().split()
    
    # 5. Remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        # 保留一些可能有情感色彩的停用词
        important_stops = {'not', 'no', 'never', 'nor', 'nothing', 'none'}
        stops = stops - important_stops
        words = [w for w in words if not w in stops]
    
    # 6. 处理重复字符（例如 "sooooo good" -> "so good"）
    words = [re.sub(r'(.)\1{2,}', r'\1\1', word) for word in words]
    
    # 7. Return a list of words
    return words

# 加载Word2Vec模型
def load_word2vec_model():
    model_name = "300features_40minwords_10context"
    return gensim.models.Word2Vec.load(model_name)

# 改进的向量平均法（带TF-IDF加权）
def makeFeatureVec(words, model, num_features, tfidf_dict=None):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)
    
    for word in words:
        if word in index2word_set:
            nwords += 1
            # 如果有TF-IDF权重，使用权重加权
            if tfidf_dict and word in tfidf_dict:
                weight = tfidf_dict[word]
            else:
                weight = 1.0
            featureVec = np.add(featureVec, model.wv[word] * weight)
    
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features, tfidf_dict=None):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000 == 0:
            print(f"处理评论 {counter}/{len(reviews)}")
        reviewFeatureVecs[counter] = makeFeatureVec(review_to_wordlist(review), model, num_features, tfidf_dict)
        counter += 1
    return reviewFeatureVecs

# 生成TF-IDF词典
def get_tfidf_dict(reviews, max_features=10000):
    print("生成TF-IDF词典...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        max_features=max_features
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_dict = {}
    for i, word in enumerate(feature_names):
        tfidf_dict[word] = tfidf_vectorizer.idf_[i]
    return tfidf_dict

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 加载Word2Vec模型
    print("加载Word2Vec模型...")
    model = load_word2vec_model()
    num_features = 300
    
    # 生成TF-IDF词典用于加权
    tfidf_dict = get_tfidf_dict(train["review"])
    
    # 生成Word2Vec特征（带TF-IDF加权）
    print("\n生成Word2Vec特征...")
    train_w2v = getAvgFeatureVecs(train["review"], model, num_features, tfidf_dict)
    test_w2v = getAvgFeatureVecs(test["review"], model, num_features, tfidf_dict)
    
    # 标准化特征
    print("\n标准化特征...")
    scaler = StandardScaler()
    train_w2v_scaled = scaler.fit_transform(train_w2v)
    test_w2v_scaled = scaler.transform(test_w2v)
    
    # 调优逻辑回归（更细致的参数搜索）
    print("\n调优逻辑回归...")
    param_grid = {
        'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=3000),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(train_w2v_scaled, train["sentiment"])
    best_model = grid_search.best_estimator_
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"交叉验证准确率: {grid_search.best_score_}")
    
    # 预测测试集
    print("\n预测测试集...")
    predictions = best_model.predict(test_w2v_scaled)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("Word2Vec_Average_LR_Improved.csv", index=False, quoting=3)
    
    print("\n任务完成！")
    print(f"提交文件已生成: Word2Vec_Average_LR_Improved.csv")

if __name__ == "__main__":
    main()