import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from llm.api import get_embedding

# 读取jsonl文件中的response字段
def load_responses(jsonl_file):
    responses = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            responses.append(data['response'])
    return responses

# 获取OpenAI嵌入
def get_openai_embeddings(texts):
    embeddings = []
    for text in tqdm(texts, desc="获取嵌入"):
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings

# 计算余弦相似度并显示进度
def calculate_cosine_similarity(embeddings1, embeddings2):
    similarities = []
    for emb1, emb2 in tqdm(zip(embeddings1, embeddings2), total=len(embeddings1), desc="计算余弦相似度"):
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        similarities.append(similarity)
        print(f"相似度: {similarity:.4f}")
    return similarities

# 主函数，计算两个JSONL文件中response的相似度
def calculate_average_cosine_similarity(file1, file2):
    # 加载两个文件的responses
    responses1 = load_responses(file1)
    responses2 = load_responses(file2)

    if len(responses1) != len(responses2):
        raise ValueError("两个文件中的response数量不相同，无法比较！")

    # 获取每个response的嵌入
    embeddings1 = get_openai_embeddings(responses1)
    embeddings2 = get_openai_embeddings(responses2)

    # 计算每对response的余弦相似度并显示
    similarities = calculate_cosine_similarity(embeddings1, embeddings2)

    # 计算平均相似度
    avg_similarity = np.mean(similarities)
    return avg_similarity


if __name__ == '__main__':
    jsonl_file1 = '../dataset/val.jsonl'  # 替换为你的第一个jsonl文件路径
    jsonl_file2 = '../dataset/val_with_response_zhipu.jsonl'  # 替换为你的第二个jsonl文件路径
    average_similarity = calculate_average_cosine_similarity(jsonl_file1, jsonl_file2)

    print(f"两个JSONL文件中response的平均余弦相似度为: {average_similarity:.4f}")
