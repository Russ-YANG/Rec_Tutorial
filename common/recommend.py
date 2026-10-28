"""
推荐结果的公共处理。所有召回算法的 recommend 都返回 [(item_id, score), ...]，分数从高到低。
"""
import torch


def top_n(item_scores, n, exclude=()):
    """从 {item: score} 中去掉 exclude 里的物品，按分数取前 n 个"""
    candidates = [(item_id, score) for item_id, score in item_scores.items() if item_id not in exclude]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]


@torch.no_grad()
def recommend_by_embedding(user_vector, item_vectors, item_ids, n, exclude=()):
    """
    向量召回: 用户向量和所有物品向量做内积，取分数最高的 n 个。
    user_vector: (dim,)
    item_vectors: (num_items, dim)，第 i 行对应 item_ids[i]
    """
    scores = item_vectors @ user_vector
    item_scores = dict(zip(item_ids, scores.tolist()))
    return top_n(item_scores, n, exclude)
