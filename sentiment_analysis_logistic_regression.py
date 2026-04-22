import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 改进的文本清洗函数（保留否定词）
def review_to_wordlist(review, remove_stopwords=True):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
    
    # 2. 处理情感相关的标点符号
    review_text = review_text.replace('!', ' exclamation ')
    review_text = review_text.replace('?', ' question ')
    review_text = review_text.replace('...', ' ellipsis ')
    
    # 3. 处理否定词（重要：保留否定词）
    review_text = review_text.replace("can't", "cannot")
    review_text = review_text.replace("won't", "will not")
    review_text = review_text.replace("n't", " not")
    
    # 4. 移除URL
    review_text = re.sub(r'http\S+|www\S+|https\S+', '', review_text, flags=re.MULTILINE)
    
    # 5. Remove non-letters and numbers
    review_text = re.sub("[^a-zA-Z0-9]"," ", review_text)
    
    # 6. Convert words to lower case and split them
    words = review_text.lower().split()
    
    # 7. 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 8. Remove stop words（保留否定词）
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        # 保留情感相关的停用词，特别是否定词
        important_stops = {'not', 'no', 'never', 'nor', 'nothing', 'none', 'hardly', 'scarcely', 'barely'}
        stops = stops - important_stops
        words = [w for w in words if not w in stops]
    
    # 9. 处理重复字符
    words = [re.sub(r'(.)\1{2,}', r'\1\1', word) for word in words]
    
    # 10. 过滤短词
    words = [word for word in words if len(word) > 2]
    
    # 11. 返回字符串（用于TF-IDF的短语模式）
    return ' '.join(words)

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 文本预处理
    print("预处理文本...")
    train['clean_review'] = train['review'].apply(review_to_wordlist)
    test['clean_review'] = test['review'].apply(review_to_wordlist)
    
    # 生成TF-IDF特征（使用短语模式）
    print("生成TF-IDF特征（使用短语模式）...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        max_features=25000,
        ngram_range=(1, 3),  # 使用1-3gram短语模式
        min_df=2,
        max_df=0.7,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    train_tfidf = tfidf_vectorizer.fit_transform(train['clean_review'])
    test_tfidf = tfidf_vectorizer.transform(test['clean_review'])
    
    print(f"TFIDF特征维度: {train_tfidf.shape[1]}")
    
    # 准备数据
    X = train_tfidf
    y = train['sentiment'].values
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. 训练逻辑回归模型
    print("\n训练逻辑回归模型...")
    param_grid_lr = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2']
    }
    
    grid_search_lr = GridSearchCV(
        LogisticRegression(max_iter=5000),
        param_grid=param_grid_lr,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search_lr.fit(X_train, y_train)
    best_lr = grid_search_lr.best_estimator_
    
    print(f"最佳逻辑回归参数: {grid_search_lr.best_params_}")
    print(f"交叉验证准确率: {grid_search_lr.best_score_:.4f}")
    
    # 验证逻辑回归模型
    y_pred_lr = best_lr.predict(X_val)
    accuracy_lr = accuracy_score(y_val, y_pred_lr)
    print(f"逻辑回归验证准确率: {accuracy_lr:.4f}")
    
    # 2. 训练线性回归模型（作为对比）
    print("\n训练线性回归模型...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # 验证线性回归模型
    y_pred_linear = linear_model.predict(X_val)
    y_pred_linear_class = (y_pred_linear > 0.5).astype(int)
    accuracy_linear = accuracy_score(y_val, y_pred_linear_class)
    print(f"线性回归验证准确率: {accuracy_linear:.4f}")
    
    # 3. 训练完整的逻辑回归模型
    print("\n训练完整的逻辑回归模型...")
    final_lr = LogisticRegression(
        C=grid_search_lr.best_params_['C'],
        solver=grid_search_lr.best_params_['solver'],
        penalty=grid_search_lr.best_params_['penalty'],
        max_iter=5000
    )
    final_lr.fit(X, y)
    
    # 4. 预测测试集
    print("\n预测测试集...")
    predictions = final_lr.predict(test_tfidf)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("LogisticRegression_TFIDF_Phrase.csv", index=False, quoting=3)
    
    print("\n任务完成！")
    print(f"提交文件已生成: LogisticRegression_TFIDF_Phrase.csv")
    
    # 分析特征重要性
    print("\n分析重要特征...")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coef = final_lr.coef_[0]
    top_positive = np.argsort(coef)[-20:]
    top_negative = np.argsort(coef)[:20]
    
    print("\nTop 20 积极特征:")
    for i in top_positive[::-1]:
        print(f"{feature_names[i]}: {coef[i]:.4f}")
    
    print("\nTop 20 消极特征:")
    for i in top_negative:
        print(f"{feature_names[i]}: {coef[i]:.4f}")

if __name__ == "__main__":
    main()