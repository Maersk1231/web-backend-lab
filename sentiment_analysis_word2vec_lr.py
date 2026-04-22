import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
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
    
    # 加载Word2Vec模型
    print("\n加载Word2Vec模型...")
    model_name = "300features_40minwords_10context"
    model = gensim.models.Word2Vec.load(model_name)
    num_features = 300
    
    # 生成Word2Vec特征
    print("\n生成Word2Vec特征...")
    train_features = getAvgFeatureVecs(train["review"], model, num_features)
    test_features = getAvgFeatureVecs(test["review"], model, num_features)
    
    # 网格搜索最佳参数
    print("\n进行网格搜索...")
    param_grid = {
        'C': [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=2000),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(train_features, train["sentiment"])
    
    # 打印最佳参数
    print("\n最佳参数:")
    print(grid_search.best_params_)
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 使用最佳模型预测
    print("\n使用最佳模型预测测试集...")
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_features)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("Word2Vec_Average_LR_Best.csv", index=False, quoting=3)
    print("最佳参数模型提交文件已生成: Word2Vec_Average_LR_Best.csv")
    print("任务完成！")

if __name__ == "__main__":
    main()
