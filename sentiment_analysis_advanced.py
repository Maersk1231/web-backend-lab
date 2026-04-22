import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import gensim
from sklearn.metrics import accuracy_score

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
    
    # 生成TF-IDF特征
    print("\n生成TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word', 
        tokenizer=None, 
        preprocessor=None, 
        stop_words='english', 
        max_features=15000, 
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8
    )
    
    train_tfidf = tfidf_vectorizer.fit_transform(train["review"])
    test_tfidf = tfidf_vectorizer.transform(test["review"])
    
    # 生成Word2Vec特征
    print("\n生成Word2Vec特征...")
    model = load_word2vec_model()
    num_features = 300
    train_w2v = getAvgFeatureVecs(train["review"], model, num_features)
    test_w2v = getAvgFeatureVecs(test["review"], model, num_features)
    
    # 标准化Word2Vec特征
    scaler = StandardScaler()
    train_w2v_scaled = scaler.fit_transform(train_w2v)
    test_w2v_scaled = scaler.transform(test_w2v)
    
    # 融合特征
    print("\n融合特征...")
    train_tfidf_dense = train_tfidf.toarray()
    test_tfidf_dense = test_tfidf.toarray()
    
    train_combined = np.hstack([train_tfidf_dense, train_w2v_scaled])
    test_combined = np.hstack([test_tfidf_dense, test_w2v_scaled])
    
    print(f"融合后特征维度: {train_combined.shape[1]}")
    
    # 1. 调优逻辑回归
    print("\n1. 调优逻辑回归...")
    param_grid_lr = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'solver': ['liblinear', 'saga']
    }
    grid_search_lr = GridSearchCV(
        LogisticRegression(max_iter=2000),
        param_grid=param_grid_lr,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_lr.fit(train_tfidf, train["sentiment"])
    best_lr = grid_search_lr.best_estimator_
    print(f"最佳逻辑回归参数: {grid_search_lr.best_params_}")
    print(f"交叉验证准确率: {grid_search_lr.best_score_}")
    
    # 2. 调优梯度提升树
    print("\n2. 调优梯度提升树...")
    param_grid_gbm = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }
    grid_search_gbm = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid=param_grid_gbm,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_gbm.fit(train_w2v_scaled, train["sentiment"])
    best_gbm = grid_search_gbm.best_estimator_
    print(f"最佳梯度提升树参数: {grid_search_gbm.best_params_}")
    print(f"交叉验证准确率: {grid_search_gbm.best_score_}")
    
    # 3. 调优支持向量机
    print("\n3. 调优支持向量机...")
    param_grid_svm = {
        'C': [0.1, 0.5, 1.0],
        'kernel': ['linear']
    }
    # 使用LinearSVC更高效
    grid_search_svm = GridSearchCV(
        LinearSVC(max_iter=10000),
        param_grid=param_grid_svm,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_svm.fit(train_tfidf, train["sentiment"])
    best_svm = grid_search_svm.best_estimator_
    print(f"最佳SVM参数: {grid_search_svm.best_params_}")
    print(f"交叉验证准确率: {grid_search_svm.best_score_}")
    
    # 4. 调优随机森林（融合特征）
    print("\n4. 调优随机森林（融合特征）...")
    param_grid_rf = {
        'n_estimators': [150, 200],
        'max_depth': [50, 70],
        'min_samples_split': [2, 4]
    }
    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid_rf,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_rf.fit(train_combined, train["sentiment"])
    best_rf = grid_search_rf.best_estimator_
    print(f"最佳随机森林参数: {grid_search_rf.best_params_}")
    print(f"交叉验证准确率: {grid_search_rf.best_score_}")
    
    # 5. 模型集成 - 投票分类器
    print("\n5. 模型集成 - 投票分类器...")
    # 训练所有基础模型
    lr_model = best_lr
    svm_model = best_svm
    gbm_model = best_gbm
    rf_model = best_rf
    
    # 创建投票分类器
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('svm', svm_model),
            ('gbm', gbm_model),
            ('rf', rf_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # 训练投票分类器
    print("训练投票分类器...")
    voting_clf.fit(train_tfidf, train["sentiment"])
    
    # 6. 多层感知器（神经网络）
    print("\n6. 训练多层感知器...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        alpha=0.001,
        random_state=42
    )
    mlp.fit(train_combined, train["sentiment"])
    
    # 预测并生成提交文件
    print("\n生成预测结果...")
    
    # 逻辑回归
    print("逻辑回归预测...")
    result_lr = best_lr.predict(test_tfidf)
    output_lr = pd.DataFrame({"id": test["id"], "sentiment": result_lr})
    output_lr.to_csv("TFIDF_LogisticRegression_Tuned.csv", index=False, quoting=3)
    
    # 支持向量机
    print("SVM预测...")
    result_svm = best_svm.predict(test_tfidf)
    output_svm = pd.DataFrame({"id": test["id"], "sentiment": result_svm})
    output_svm.to_csv("SVM_TFIDF_Tuned.csv", index=False, quoting=3)
    
    # 梯度提升树
    print("梯度提升树预测...")
    result_gbm = best_gbm.predict(test_w2v_scaled)
    output_gbm = pd.DataFrame({"id": test["id"], "sentiment": result_gbm})
    output_gbm.to_csv("Word2Vec_GradientBoosting_Tuned.csv", index=False, quoting=3)
    
    # 随机森林（融合特征）
    print("随机森林预测...")
    result_rf = best_rf.predict(test_combined)
    output_rf = pd.DataFrame({"id": test["id"], "sentiment": result_rf})
    output_rf.to_csv("Combined_Features_Tuned.csv", index=False, quoting=3)
    
    # 投票分类器
    print("投票分类器预测...")
    result_voting = voting_clf.predict(test_tfidf)
    output_voting = pd.DataFrame({"id": test["id"], "sentiment": result_voting})
    output_voting.to_csv("Voting_Classifier.csv", index=False, quoting=3)
    
    # 多层感知器
    print("多层感知器预测...")
    result_mlp = mlp.predict(test_combined)
    output_mlp = pd.DataFrame({"id": test["id"], "sentiment": result_mlp})
    output_mlp.to_csv("MLP_Combined.csv", index=False, quoting=3)
    
    # 7. 最终集成：所有模型的加权投票
    print("\n7. 最终集成：加权投票...")
    # 获取所有模型的概率预测
    lr_proba = best_lr.predict_proba(test_tfidf)[:, 1]
    svm_proba = best_svm._predict_proba_lr(test_tfidf)[:, 1]
    gbm_proba = best_gbm.predict_proba(test_w2v_scaled)[:, 1]
    rf_proba = best_rf.predict_proba(test_combined)[:, 1]
    mlp_proba = mlp.predict_proba(test_combined)[:, 1]
    
    # 加权融合
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # 可以根据交叉验证分数调整权重
    final_proba = (
        weights[0] * lr_proba +
        weights[1] * svm_proba +
        weights[2] * gbm_proba +
        weights[3] * rf_proba +
        weights[4] * mlp_proba
    )
    final_pred = (final_proba > 0.5).astype(int)
    
    # 生成最终提交文件
    output_final = pd.DataFrame({"id": test["id"], "sentiment": final_pred})
    output_final.to_csv("Final_Ensemble.csv", index=False, quoting=3)
    
    print("\n所有模型预测完成！")
    print("生成的提交文件：")
    print("1. TFIDF_LogisticRegression_Tuned.csv")
    print("2. SVM_TFIDF_Tuned.csv")
    print("3. Word2Vec_GradientBoosting_Tuned.csv")
    print("4. Combined_Features_Tuned.csv")
    print("5. Voting_Classifier.csv")
    print("6. MLP_Combined.csv")
    print("7. Final_Ensemble.csv")

if __name__ == "__main__":
    main()