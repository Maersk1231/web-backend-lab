import gensim

# 加载训练好的模型
model_name = "300features_40minwords_10context"
model = gensim.models.Word2Vec.load(model_name)

print("Word2Vec 模型探索")
print("=" * 50)

# 测试 doesnt_match 函数
print("\n1. 测试 doesnt_match 函数:")
test_cases = [
    ["man", "woman", "child", "kitchen"],
    ["paris", "london", "berlin", "tokyo"],
    ["cat", "dog", "bird", "fish"]
]

for case in test_cases:
    try:
        result = model.doesnt_match(case)
        print(f"   集合 {case} 中最不匹配的词: {result}")
    except Exception as e:
        print(f"   集合 {case} 测试失败: {e}")

# 测试 most_similar 函数
print("\n2. 测试 most_similar 函数:")
test_words = ["queen", "king", "happy", "sad"]

for word in test_words:
    try:
        similar_words = model.wv.most_similar(word, topn=5)
        print(f"   与 '{word}' 最相似的词:")
        for similar_word, similarity in similar_words:
            print(f"     - {similar_word}: {similarity:.4f}")
    except Exception as e:
        print(f"   词 '{word}' 测试失败: {e}")

print("\n3. 测试语义关系:")
try:
    # 测试 king - man + woman = queen
    result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
    print("   king - man + woman = ")
    for word, similarity in result:
        print(f"     - {word}: {similarity:.4f}")
except Exception as e:
    print(f"   语义关系测试失败: {e}")

print("\nWord2Vec 模型探索完成！")
