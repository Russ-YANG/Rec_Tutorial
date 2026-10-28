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


@torch.no_grad()
def score_all_items(score_fn, user_idx, num_items, device, batch_size=1024):
    """
    排序模型不能拆成两个向量，只能把 (用户, 物品) 一对对送进模型打分。
    score_fn(user_idx, item_idx) -> (batch,) 或 (batch, k)
    返回 (num_items,) 或 (num_items, k)，第 i 行是用户对下标为 i 的物品的分数
    """
    scores = []
    for start in range(0, num_items, batch_size):
        item_idx = torch.arange(start, min(start + batch_size, num_items), device=device)
        users = torch.full_like(item_idx, user_idx)
        scores.append(score_fn(users, item_idx))
    return torch.cat(scores)
