import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

# 文本清洗函数（保留否定词）
def review_to_text(review):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
    # 2. Remove URLs
    review_text = re.sub(r'http\S+|www\S+|https\S+', '', review_text, flags=re.MULTILINE)
    # 3. Handle negation (保留否定词的完整形式)
    review_text = re.sub(r"n't", " not", review_text)
    # 4. Remove non-letters but keep some punctuation
    review_text = re.sub(r'[^a-zA-Z\s!\?\.]', ' ', review_text)
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
        max_features=15000,
        min_df=5,
        max_df=0.8
    )
    
    # 拟合训练数据并转换
    train_features = tfidf_vectorizer.fit_transform(train["clean_review"])
    test_features = tfidf_vectorizer.transform(test["clean_review"])
    
    print(f"特征维度: {train_features.shape[1]}")
    
    # 方法1：逻辑回归
    print("\n方法1: 逻辑回归")
    param_grid_lr = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'solver': ['liblinear'],
        'max_iter': [2000]
    }
    
    grid_search_lr = GridSearchCV(
        LogisticRegression(),
        param_grid_lr,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    grid_search_lr.fit(train_features, train["sentiment"])
    
    print("最佳参数:")
    print(grid_search_lr.best_params_)
    print(f"最佳交叉验证分数: {grid_search_lr.best_score_:.4f}")
    
    # 预测测试集
    predictions_lr = grid_search_lr.best_estimator_.predict(test_features)
    
    # 生成提交文件
    output_lr = pd.DataFrame({"id": test["id"], "sentiment": predictions_lr})
    output_lr.to_csv("TFIDF_LogisticRegression_Phrase.csv", index=False, quoting=3)
    print("逻辑回归提交文件已生成: TFIDF_LogisticRegression_Phrase.csv")
    
    # 方法2：线性回归（需要将标签转换为连续值）
    print("\n方法2: 线性回归")
    # 训练线性回归模型
    linear_reg = LinearRegression()
    linear_reg.fit(train_features, train["sentiment"])
    
    # 预测并转换为0-1
    predictions_lin = linear_reg.predict(test_features)
    predictions_lin = [1 if pred >= 0.5 else 0 for pred in predictions_lin]
    
    # 生成提交文件
    output_lin = pd.DataFrame({"id": test["id"], "sentiment": predictions_lin})
    output_lin.to_csv("TFIDF_LinearRegression_Phrase.csv", index=False, quoting=3)
    print("线性回归提交文件已生成: TFIDF_LinearRegression_Phrase.csv")
    
    print("\n简单模型情感分析任务完成！")

if __name__ == "__main__":
    main()
