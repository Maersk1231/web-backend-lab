import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import gensim
import os

# 文本清洗函数
def review_to_wordlist(review, remove_stopwords=False):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
    # 2. Remove URLs
    review_text = re.sub(r'http\S+|www\S+|https\S+', '', review_text, flags=re.MULTILINE)
    # 3. Handle negation
    review_text = re.sub(r"n't", " not", review_text)
    # 4. Remove non-letters
    review_text = re.sub(r'[^a-zA-Z\s]', ' ', review_text)
    # 5. Convert words to lower case and split them
    words = review_text.lower().split()
    # 6. Remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english")) - set(['not', 'no', 'never'])
        words = [w for w in words if not w in stops]
    # 7. Return a list of words
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
    
    # 加载Word2Vec特征
    word2vec_features_file = "word2vec_features_memory.npz"
    if os.path.exists(word2vec_features_file):
        print("\n加载已保存的Word2Vec特征...")
        with np.load(word2vec_features_file) as data:
            train_features = data['train_features']
            test_features = data['test_features']
        print(f"加载完成，训练特征形状: {train_features.shape}, 测试特征形状: {test_features.shape}")
    else:
        # 加载Word2Vec模型
        print("\n加载Word2Vec模型...")
        model_name = "300features_40minwords_10context"
        model = gensim.models.Word2Vec.load(model_name)
        num_features = model.vector_size
        
        # 生成Word2Vec特征
        print(f"\n生成Word2Vec特征 (维度: {num_features})...")
        train_features = getAvgFeatureVecs(train["review"], model, num_features)
        test_features = getAvgFeatureVecs(test["review"], model, num_features)
        
        # 保存特征到文件
        print("\n保存Word2Vec特征到文件...")
        np.savez(word2vec_features_file, train_features=train_features, test_features=test_features)
        print(f"Word2Vec特征已保存到: {word2vec_features_file}")
    
    # 标准化特征
    print("\n标准化特征...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # 网格搜索最佳参数
    print("\n进行网格搜索...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    grid_search = GridSearchCV(
        XGBClassifier(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    grid_search.fit(train_features_scaled, train["sentiment"])
    
    # 打印最佳参数
    print("\n最佳参数:")
    print(grid_search.best_params_)
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 使用最佳模型预测
    print("\n使用最佳模型预测测试集...")
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_features_scaled)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("Word2Vec_XGBoost.csv", index=False, quoting=3)
    print("Word2Vec+XGBoost提交文件已生成: Word2Vec_XGBoost.csv")
    print("任务完成！")

if __name__ == "__main__":
    main()
