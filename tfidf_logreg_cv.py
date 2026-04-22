import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 创建管道
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
    ])
    
    # 定义参数网格
    parameters = {
        'tfidf__max_features': [5000, 10000, 15000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [3, 5],
        'clf__C': [0.5, 1.0, 1.5]
    }
    
    # 网格搜索
    print("\n进行网格搜索...")
    grid_search = GridSearchCV(
        pipeline, 
        parameters, 
        cv=3, 
        n_jobs=-1, 
        verbose=1
    )
    
    # 拟合训练数据
    grid_search.fit(train["review"], train["sentiment"])
    
    # 打印最佳参数
    print("\n最佳参数:")
    print(grid_search.best_params_)
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 使用最佳模型预测
    print("\n使用最佳模型预测测试集...")
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test["review"])
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("TFIDF_LogisticRegression_Best.csv", index=False, quoting=3)
    print("最佳参数模型提交文件已生成: TFIDF_LogisticRegression_Best.csv")
    print("任务完成！")

if __name__ == "__main__":
    main()
