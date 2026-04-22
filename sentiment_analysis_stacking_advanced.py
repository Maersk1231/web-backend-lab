import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from bs4 import BeautifulSoup
import re

# 自定义停用词列表
def get_stopwords():
    return set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
        'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
        'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
        'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
        'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    ]) - set(['not', 'no', 'never', 'very', 'so', 'too'])

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
    # 获取自定义停用词列表
    custom_stopwords = get_stopwords()
    
    # 生成更丰富的特征
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=list(custom_stopwords),
        ngram_range=(1, 3),  # 使用1-3元短语
        max_features=25000,  # 增加特征数量
        min_df=2,  # 减少最小文档频率
        max_df=0.6,  # 减少最大文档频率
        strip_accents='unicode',
        analyzer='word'
    )
    
    # 拟合训练数据并转换
    train_features = tfidf_vectorizer.fit_transform(train["clean_review"])
    test_features = tfidf_vectorizer.transform(test["clean_review"])
    
    print(f"特征维度: {train_features.shape[1]}")
    
    # 定义基础模型（增加更多类型的模型）
    print("\n定义基础模型...")
    base_models = [
        ('lr', LogisticRegression(C=10.0, max_iter=2000, penalty='l2', solver='liblinear')),
        ('svc', LinearSVC(C=1.0, max_iter=2000)),
        ('nb', MultinomialNB(alpha=0.1)),
        ('lgbm', LGBMClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42))
    ]
    
    # 定义更强大的元模型
    meta_model = LogisticRegression(C=100.0, max_iter=3000, penalty='l2', solver='liblinear')
    
    # 创建Stacking分类器
    print("\n创建Stacking分类器...")
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,  # 增加交叉验证折数
        passthrough=True  # 将原始特征也传递给元模型
    )
    
    # 训练Stacking分类器
    print("训练Stacking分类器...")
    stacking_clf.fit(train_features, train["sentiment"])
    
    # 预测测试集
    print("预测测试集...")
    predictions = stacking_clf.predict(test_features)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("Stacking_Ensemble_Advanced.csv", index=False, quoting=3)
    print("高级Stacking集成学习提交文件已生成: Stacking_Ensemble_Advanced.csv")
    print("任务完成！")

if __name__ == "__main__":
    main()
