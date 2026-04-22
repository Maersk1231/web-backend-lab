import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import gensim
import nltk
import os

# 下载必要的NLTK数据
nltk.download('wordnet')
nltk.download('punkt')

# 文本清洗函数（高级版）
def review_to_wordlist(review, remove_stopwords=False):
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="lxml").get_text()
    # 2. Remove URLs
    review_text = re.sub(r'http\S+|www\S+|https\S+', '', review_text, flags=re.MULTILINE)
    # 3. Handle negation (e.g., "don't" -> "do not")
    review_text = re.sub(r"n't", " not", review_text)
    # 4. Remove non-letters but keep sentiment-carrying punctuation
    review_text = re.sub(r'[^a-zA-Z\s!\?\.\']', ' ', review_text)
    # 5. Convert words to lower case and split them
    words = review_text.lower().split()
    # 6. Stem and lemmatize words
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]
    # 7. Optionally remove stop words
    if remove_stopwords:
        # Custom stopwords list with some sentiment-related words kept
        stops = set(stopwords.words("english")) - set(['not', 'no', 'never'])
        words = [w for w in words if not w in stops]
    # 8. Return a list of words
    return words

# 向量平均法（加权版）
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords += 1
            # 使用词频作为权重（简单版）
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

# 训练Word2Vec模型（高级参数）
def train_word2vec():
    # 读取数据
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    
    # 准备句子（更精细的句子分割）
    sentences = []
    tokenizer = nltk.data.load('tokenizers/punkt_tab/english.pickle')
    
    for review in train["review"]:
        raw_sentences = tokenizer.tokenize(review.strip())
        for sentence in raw_sentences:
            if len(sentence) > 0:
                sentences.append(review_to_wordlist(sentence, remove_stopwords=False))
    
    for review in unlabeled_train["review"]:
        raw_sentences = tokenizer.tokenize(review.strip())
        for sentence in raw_sentences:
            if len(sentence) > 0:
                sentences.append(review_to_wordlist(sentence, remove_stopwords=False))
    
    print(f"准备了 {len(sentences)} 个句子用于训练Word2Vec")
    
    # 训练Word2Vec模型（高级参数）
    model = gensim.models.Word2Vec(
        sentences,
        vector_size=500,      # 增加向量维度
        min_count=20,         # 调整最小词频
        window=20,             # 增加上下文窗口
        workers=4,
        sg=1,                 # 使用skip-gram架构
        negative=10,           # 增加负采样数量
        alpha=0.025,
        min_alpha=0.0001,
        epochs=15,            # 增加训练轮数
        sample=1e-4,          # 调整高频词下采样
        hs=0                  # 使用负采样而非层次softmax
    )
    
    # 保存模型
    model_name = "500features_20minwords_20context_skipgram"
    model.save(model_name)
    print(f"训练的Word2Vec模型已保存为: {model_name}")
    return model

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")
    
    # 检查是否已经有保存的Word2Vec特征
    word2vec_features_file = "word2vec_features_advanced.npz"
    if os.path.exists(word2vec_features_file):
        print("\n加载已保存的Word2Vec特征...")
        with np.load(word2vec_features_file) as data:
            train_features = data['train_features']
            test_features = data['test_features']
        print(f"加载完成，训练特征形状: {train_features.shape}, 测试特征形状: {test_features.shape}")
    else:
        # 训练Word2Vec模型
        print("\n训练Word2Vec模型...")
        model = train_word2vec()
        num_features = model.vector_size
        
        # 生成Word2Vec特征
        print(f"\n生成Word2Vec特征 (维度: {num_features})...")
        train_features = getAvgFeatureVecs(train["review"], model, num_features)
        test_features = getAvgFeatureVecs(test["review"], model, num_features)
        
        # 保存特征到文件
        print("\n保存Word2Vec特征到文件...")
        np.savez(word2vec_features_file, train_features=train_features, test_features=test_features)
        print(f"Word2Vec特征已保存到: {word2vec_features_file}")
    
    # 标准化特征
    print("\n标准化特征...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # 网格搜索最佳参数（更全面的网格）
    print("\n进行网格搜索...")
    param_grid = {
        'C': [1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced'],
        'max_iter': [3000]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    grid_search.fit(train_features_scaled, train["sentiment"])
    
    # 打印最佳参数
    print("\n最佳参数:")
    print(grid_search.best_params_)
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 使用最佳模型预测
    print("\n使用最佳模型预测测试集...")
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_features_scaled)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("Word2Vec_Average_LR_Advanced.csv", index=False, quoting=3)
    print("高级Word2Vec+逻辑回归提交文件已生成: Word2Vec_Average_LR_Advanced.csv")
    print("任务完成！")

if __name__ == "__main__":
    main()
