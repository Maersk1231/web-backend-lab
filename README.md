# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：马艺杰
- **学号**：112311170336
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：词袋遇上爆米花袋 (Bag of Words Meets Bags of Popcorn)
- **比赛链接**：https://www.kaggle.com/c/word2vec-nlp-tutorial
- **提交日期**：2026-04-22

- **GitHub 仓库地址**：https://github.com/Maersk1231/web-backend-lab
- **GitHub README 地址**：https://github.com/Maersk1231/web-backend-lab/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.89472
- **Private Score**（如有）：
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
1. 移除 HTML 标签：使用 BeautifulSoup 库移除文本中的 HTML 标签
2. 移除 URL：使用正则表达式移除文本中的 URL 链接
3. 处理否定词：将缩写形式的否定词（如 "n't"）转换为完整形式（如 " not"）
4. 保留情感相关的标点：移除非字母字符，但保留情感相关的标点符号（如 !, ?, -）
5. 转小写：将所有文本转换为小写
6. 移除停用词：使用自定义的停用词列表，保留否定词（如 not, no, never）

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
1. 自己训练 Word2Vec 模型：使用 labeledTrainData.tsv（25,000 条评论）和 unlabeledTrainData.tsv（50,000 条评论）共计 75,000 条评论进行训练
2. 词向量维度：300 维
3. 模型参数：
   - 最小词频：40
   - 窗口大小：10
   - 下采样率：0.001
   - 训练算法：skip-gram
   - 工作线程：4
4. 句子向量表示：使用向量平均法，将评论中所有词的词向量取平均值作为句子向量

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
1. 尝试了多种分类模型：
   - 逻辑回归 (Logistic Regression)
   - 随机森林 (Random Forest)
   - 线性 SVC
   - XGBoost
   - LightGBM
   - 集成学习（Stacking、Voting）
2. 最终采用的模型：Stacking 集成学习模型
   - 基础模型：逻辑回归、线性 SVC、LightGBM、XGBoost
   - 元模型：逻辑回归
   - 交叉验证折数：5

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程：**  
1. 读取数据：读取 labeledTrainData.tsv、unlabeledTrainData.tsv 和 testData.tsv
2. 文本预处理：对所有评论进行清洗，包括移除 HTML 标签、URL，处理否定词，保留情感标点，转小写，移除停用词
3. 训练 Word2Vec 模型：使用 75,000 条评论训练 300 维的 Word2Vec 模型
4. 生成句子向量：使用向量平均法将每条评论转换为 300 维的句子向量
5. 特征提取：使用 TF-IDF 提取文本特征，包括 1-3 元短语
6. 模型训练：
   - 训练多个基础模型（逻辑回归、线性 SVC、LightGBM、XGBoost）
   - 构建 Stacking 集成学习模型
7. 预测与提交：在测试集上预测情感标签，生成提交文件并提交到 Kaggle

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
数据5/
├─ labeledTrainData.tsv：带标签的训练数据
├─ unlabeledTrainData.tsv：无标签的训练数据
├─ testData.tsv：测试数据
├─ word2vec_training.py：Word2Vec 模型训练脚本
├─ word2vec_sentiment.py：Word2Vec 情感分析脚本
├─ sentiment_analysis_word2vec_lr.py：Word2Vec+逻辑回归模型
├─ sentiment_analysis_stacking.py：Stacking 集成学习模型
├─ sentiment_analysis_stacking_optimized.py：优化的 Stacking 模型
├─ Stacking_Ensemble.csv：Stacking 模型的提交文件
├─ Word2Vec_AverageVectors.csv：Word2Vec 平均向量模型的提交文件
├─ images/：存放 Kaggle 截图
└─ readme_机器学习实验2模板.md：实验报告
```

