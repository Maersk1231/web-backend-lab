import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# 下载停用词
nltk.download('stopwords')

# 文本清洗函数
def review_to_words(raw_review):
    # 1. 去掉HTML标签
    review_text = re.sub('<br />', ' ', raw_review)
    # 2. 只保留英文字母
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. 转小写
    words = letters_only.lower().split()
    # 4. 去掉无意义的停用词
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    # 5. 把单词串起来
    return ' '.join(meaningful_words)

# 1. 读取训练数据
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
print(f'训练数据形状: {train.shape}')
print(f'训练数据前5行:\n{train.head()}')

# 2. 批量清理训练集评论
print('清理训练集评论...')
train['clean_review'] = train['review'].apply(review_to_words)
print('训练集清理完成')

# 3. 构建词袋模型
print('构建词袋模型...')
vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(train['clean_review'])
train_data_features = train_data_features.toarray()
print(f'词袋模型特征形状: {train_data_features.shape}')

# 4. 训练随机森林
print('训练随机森林分类器...')
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest = forest.fit(train_data_features, train['sentiment'])
print('模型训练完成')

# 5. 处理测试集
print('处理测试集...')
test = pd.read_csv('testData.tsv', header=0, delimiter='\t', quoting=3)
print(f'测试数据形状: {test.shape}')

# 清理测试集评论
test['clean_review'] = test['review'].apply(review_to_words)

# 转换测试集为词袋向量
test_data_features = vectorizer.transform(test['clean_review'])
test_data_features = test_data_features.toarray()

# 6. 预测测试集
print('预测测试集情感...')
predicted = forest.predict(test_data_features)

# 7. 生成提交文件
print('生成提交文件...')
output = pd.DataFrame({'id': test['id'], 'sentiment': predicted})
output.to_csv('Bag_of_Words_model.csv', index=False, quoting=3)
print('提交文件已生成: Bag_of_Words_model.csv')
print('情感分析任务完成！')