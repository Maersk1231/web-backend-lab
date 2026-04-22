import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

# 文本清洗函数
def review_to_text(review):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
    # 2. Remove URLs
    review_text = re.sub(r'http\S+|www\S+|https\S+', '', review_text, flags=re.MULTILINE)
    # 3. Handle negation
    review_text = re.sub(r"n't", " not", review_text)
    # 4. Remove non-letters but keep some punctuation
    review_text = re.sub(r'[^a-zA-Z\s!?-]', ' ', review_text)
    # 5. Convert to lower case
    review_text = review_text.lower()
    # 6. Return cleaned text
    return review_text

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 清洗文本
    print("\n清洗文本...")
    train["clean_review"] = train["review"].apply(review_to_text)
    test["clean_review"] = test["review"].apply(review_to_text)
    
    # 使用TF-IDF向量化器（使用短语模式）
    print("\n生成TF-IDF特征（使用短语模式）...")
    # 自定义停用词列表，保留否定词
    custom_stopwords = set(stopwords.words("english")) - set(['not', 'no', 'never', 'very', 'so', 'too'])
    
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=list(custom_stopwords),
        ngram_range=(1, 3),  # 使用1-3元短语
        max_features=20000,
        min_df=3,
        max_df=0.7
    )
    
    # 拟合训练数据并转换
    train_features = tfidf_vectorizer.fit_transform(train["clean_review"])
    test_features = tfidf_vectorizer.transform(test["clean_review"])
    
    print(f"特征维度: {train_features.shape[1]}")
    
    # 定义基础模型
    print("\n定义基础模型...")
    base_models = [
        ('lr', LogisticRegression(C=10.0, max_iter=2000, penalty='l2', solver='liblinear')),
        ('svc', LinearSVC(C=1.0, max_iter=2000)),
        ('nb', MultinomialNB(alpha=0.1)),
        ('lgbm', LGBMClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42))
    ]
    
    # 定义元模型
    meta_model = LogisticRegression(C=10.0, max_iter=2000, penalty='l2', solver='liblinear')
    
    # 创建Stacking分类器
    print("\n创建Stacking分类器...")
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    
    # 训练Stacking分类器
    print("训练Stacking分类器...")
    stacking_clf.fit(train_features, train["sentiment"])
    
    # 预测测试集
    print("预测测试集...")
    predictions = stacking_clf.predict(test_features)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("Stacking_Ensemble.csv", index=False, quoting=3)
    print("Stacking集成学习提交文件已生成: Stacking_Ensemble.csv")
    print("任务完成！")

if __name__ == "__main__":
    main()
