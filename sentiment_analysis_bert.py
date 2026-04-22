import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

# 自定义数据集类
class SentimentDataset(Dataset):
    def __init__(self, reviews, labels=None):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        encoding = self.tokenizer(
            review,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# 主函数
def main():
    # 读取数据
    print("读取数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    
    # 准备数据集
    train_dataset = SentimentDataset(train['review'], train['sentiment'])
    test_dataset = SentimentDataset(test['review'])
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 加载预训练BERT模型
    print("加载BERT模型...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 训练模型
    print("训练BERT模型...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # 预测测试集
    print("预测测试集...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).numpy()
            predictions.extend(preds)
    
    # 生成提交文件
    output = pd.DataFrame({"id": test["id"], "sentiment": predictions})
    output.to_csv("BERT_Sentiment.csv", index=False, quoting=3)
    print("BERT模型提交文件已生成: BERT_Sentiment.csv")
    print("任务完成！")

if __name__ == "__main__":
    main()
