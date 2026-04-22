import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 改进的文本清洗函数
def review_to_wordlist(review, remove_stopwords=True):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
    
    # 2. 处理情感相关的标点符号
    review_text = review_text.replace('!', ' exclamation ')
    review_text = review_text.replace('?', ' question ')
    review_text = review_text.replace('...', ' ellipsis ')
    
    # 3. 处理否定词
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
    
    # 8. Remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        # 保留情感相关的停用词
        important_stops = {'not', 'no', 'never', 'nor', 'nothing', 'none', 'hardly', 'scarcely', 'barely'}
        stops = stops - important_stops
        words = [w for w in words if not w in stops]
    
    # 9. 处理重复字符
    words = [re.sub(r'(.)\1{2,}', r'\1\1', word) for word in words]
    
    # 10. 过滤短词
    words = [word for word in words if len(word) > 2]
    
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
    
    # 生成TF-IDF特征（优化版本）
    print("生成TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        max_features=30000,  # 增加特征数量
        ngram_range=(1, 3),
        min_df=1,  # 减少最小文档频率
        max_df=0.8,
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
    
    # 1. 训练LightGBM模型
    print("\n训练LightGBM模型...")
    lgbm_model = LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective='binary',
        metric='binary_logloss'
    )
    
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgbm.log_evaluation(50), lgbm.early_stopping(20)]
    )
    
    # 验证LightGBM模型
    y_pred_lgbm = lgbm_model.predict(X_val)
    accuracy_lgbm = accuracy_score(y_val, y_pred_lgbm)
    print(f"LightGBM验证准确率: {accuracy_lgbm:.4f}")
    
    # 2. 训练XGBoost模型
    print("\n训练XGBoost模型...")
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # 验证XGBoost模型
    y_pred_xgb = xgb_model.predict(X_val)
    accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
    print(f"XGBoost验证准确率: {accuracy_xgb:.4f}")
    
    # 3. 模型集成
    print("\n创建模型集成...")
    voting_clf = VotingClassifier(
        estimators=[
            ('lgbm', lgbm_model),
            ('xgb', xgb_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # 训练集成模型
    voting_clf.fit(X_train, y_train)
    
    # 验证集成模型
    y_pred_voting = voting_clf.predict(X_val)
    accuracy_voting = accuracy_score(y_val, y_pred_voting)
    print(f"集成模型验证准确率: {accuracy_voting:.4f}")
    
    # 4. 训练完整模型
    print("\n训练完整模型...")
    # 训练完整的LightGBM模型
    final_lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective='binary',
        metric='binary_logloss'
    )
    final_lgbm.fit(X, y)
    
    # 5. 预测测试集
    print("\n预测测试集...")
    # LightGBM预测
    predictions_lgbm = final_lgbm.predict(test_tfidf)
    
    # 生成提交文件
    output_lgbm = pd.DataFrame({"id": test["id"], "sentiment": predictions_lgbm})
    output_lgbm.to_csv("LightGBM_TFIDF_Sentiment.csv", index=False, quoting=3)
    
    print("\n任务完成！")
    print(f"提交文件已生成: LightGBM_TFIDF_Sentiment.csv")

if __name__ == "__main__":
    main()