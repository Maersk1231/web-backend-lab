import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

# 文本清洗函数
def review_to_wordlist(review, remove_stopwords=True):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    # 4. Remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return words

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 生成TFIDF特征（优化版本）
    print("\n生成TFIDF特征...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word', 
        tokenizer=None, 
        preprocessor=None, 
        stop_words='english', 
        max_features=15000,  # 减少特征数量
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7
    )
    
    train_tfidf = tfidf_vectorizer.fit_transform(train["review"])
    test_tfidf = tfidf_vectorizer.transform(test["review"])
    
    print(f"TFIDF特征维度: {train_tfidf.shape[1]}")
    
    # 1. 训练逻辑回归
    print("\n训练逻辑回归模型...")
    lr_model = LogisticRegression(C=5.0, solver='liblinear', max_iter=2000)
    lr_model.fit(train_tfidf, train["sentiment"])
    
    # 2. 训练LinearSVC
    print("\n训练LinearSVC模型...")
    svc_model = LinearSVC(C=0.5, max_iter=10000, random_state=42)
    svc_model.fit(train_tfidf, train["sentiment"])
    
    # 3. 模型集成 - 投票分类器
    print("\n创建投票分类器...")
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('svc', svc_model)
        ],
        voting='hard',
        n_jobs=-1
    )
    
    # 交叉验证
    cv_score = cross_val_score(voting_clf, train_tfidf, train["sentiment"], cv=5, scoring='accuracy')
    print(f"投票分类器交叉验证准确率: {cv_score.mean()}")
    
    # 训练投票分类器
    voting_clf.fit(train_tfidf, train["sentiment"])
    
    # 4. 预测
    print("\n预测测试集...")
    # 逻辑回归预测
    lr_pred = lr_model.predict(test_tfidf)
    # LinearSVC预测
    svc_pred = svc_model.predict(test_tfidf)
    # 投票分类器预测
    voting_pred = voting_clf.predict(test_tfidf)
    
    # 5. 生成提交文件
    print("\n生成提交文件...")
    # 逻辑回归
    output_lr = pd.DataFrame({"id": test["id"], "sentiment": lr_pred})
    output_lr.to_csv("TFIDF_LogisticRegression_Optimized.csv", index=False, quoting=3)
    
    # LinearSVC
    output_svc = pd.DataFrame({"id": test["id"], "sentiment": svc_pred})
    output_svc.to_csv("TFIDF_LinearSVC_Optimized.csv", index=False, quoting=3)
    
    # 投票分类器
    output_voting = pd.DataFrame({"id": test["id"], "sentiment": voting_pred})
    output_voting.to_csv("TFIDF_VotingClassifier.csv", index=False, quoting=3)
    
    print("\n任务完成！")
    print("生成的提交文件：")
    print("1. TFIDF_LogisticRegression_Optimized.csv")
    print("2. TFIDF_LinearSVC_Optimized.csv")
    print("3. TFIDF_VotingClassifier.csv")

if __name__ == "__main__":
    main()