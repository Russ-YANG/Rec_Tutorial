"""
UserCF：和用户 u 相似的用户喜欢物品 j，就把 j 推荐给 u。

1. user_similarity       两个用户的相似度
2. build_similar_users   每个用户取最相似的 top_k 个，得到用户-用户索引
3. score_items           用相似用户交互过的物品给候选物品打分
"""
import math
from collections import defaultdict

from common.data import build_recent_items, build_user_items, load_ratings
from common.recommend import top_n


def user_similarity(items_u, items_v):
    """sim(u, v) = |I_u ∩ I_v| / sqrt(|I_u| * |I_v|)"""
    common_items = set(items_u) & set(items_v)
    if not common_items:
        return 0.0
    return len(common_items) / math.sqrt(len(items_u) * len(items_v))


def build_similar_users(user_items, top_k):
    """
    user_items: {user: 交互过的物品}
    返回 {user: [(相似用户, 相似度), ...]}，按相似度从高到低，最多 top_k 个
    """
    user_ids = list(user_items.keys())
    user_sim = defaultdict(dict)

    for a in range(len(user_ids)):
        for b in range(a + 1, len(user_ids)):
            user_u, user_v = user_ids[a], user_ids[b]
            sim = user_similarity(user_items[user_u], user_items[user_v])
            if sim > 0:
                user_sim[user_u][user_v] = sim
                user_sim[user_v][user_u] = sim

    similar_users = {}
    for user_id, sims in user_sim.items():
        similar_users[user_id] = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return similar_users


def score_items(similar_users, user_items):
    """
    score(j) = Σ_{v ∈ 相似用户, j ∈ I_v} sim(u, v)
    similar_users: [(相似用户, 相似度), ...]
    """
    scores = defaultdict(float)
    for user_v, sim in similar_users:
        for item_id in user_items[user_v]:
            scores[item_id] += sim
    return scores


class UserCF:
    def __init__(self, top_k=10, lastn=15):
        """
        top_k: 每个用户保留最相似的 top_k 个用户
        lastn: 每个用户保留最近交互的 lastn 个物品，相似度也只用这些物品算
        """
        self.top_k = top_k
        self.lastn = lastn

    def fit(self, ratings):
        self.user_items = build_user_items(ratings)
        self.recent_items = build_recent_items(ratings, self.lastn)       # 用户-物品索引
        self.similar_users = build_similar_users(self.recent_items, self.top_k)  # 用户-用户索引
        return self

    def get_similar_users(self, user_id):
        return self.similar_users.get(user_id, [])

    def recommend(self, user_id, n_rec=10):
        scores = score_items(self.get_similar_users(user_id), self.recent_items)
        return top_n(scores, n_rec, exclude=self.user_items[user_id])


if __name__ == "__main__":
    model = UserCF(top_k=10, lastn=15).fit(load_ratings())

    user_id = 1
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in model.recommend(user_id, n_rec=10):
        print(f"物品ID: {item_id}, 分数: {score:.4f}")
