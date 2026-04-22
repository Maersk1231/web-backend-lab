import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 使用TF-IDF向量化器
    print("\n生成TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5
    )
    
    # 拟合训练数据并转换
    train_features = tfidf_vectorizer.fit_transform(train["review"])
    test_features = tfidf_vectorizer.transform(test["review"])
    
    # 训练逻辑回归模型
    print("训练逻辑回归分类器...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='liblinear'
    )
    model.fit(train_features, train["sentiment"])
    
    # 预测测试集
    print("预测测试集情感...")
    predictions = model.predict(test_features)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("TFIDF_LogisticRegression.csv", index=False, quoting=3)
    print("提交文件已生成: TFIDF_LogisticRegression.csv")
    print("任务完成！")

if __name__ == "__main__":
    main()
