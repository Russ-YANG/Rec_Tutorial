"""
ItemCF：用户喜欢物品 i，就给他推荐和 i 相似的物品 j。

1. item_similarity       两个物品的相似度
2. build_similar_items   每个物品取最相似的 top_k 个，得到物品-物品索引
3. score_items           用用户最近交互的物品给候选物品打分
"""
import math
from collections import defaultdict

from common.data import build_item_users, build_recent_items, build_user_items, load_ratings
from common.recommend import top_n


def item_similarity(users_i, users_j):
    """
    sim(i, j) = |U_i ∩ U_j| / sqrt(|U_i| * |U_j|)
    users_i: 喜欢物品 i 的用户集合

    这里没有考虑喜欢程度。如果考虑评分，就是余弦相似度:
    Σ_u r_ui * r_uj / (sqrt(Σ_u r_ui^2) * sqrt(Σ_u r_uj^2))
    """
    common_users = set(users_i) & set(users_j)
    if not common_users:
        return 0.0
    return len(common_users) / math.sqrt(len(users_i) * len(users_j))


def build_similar_items(item_users, similarity, top_k):
    """
    item_users: {item: 喜欢它的用户}
    similarity: 计算两个物品相似度的函数，传入两个用户集合
    返回 {item: [(相似物品, 相似度), ...]}，按相似度从高到低，最多 top_k 个
    """
    item_ids = list(item_users.keys())
    item_sim = defaultdict(dict)

    # 相似度是对称的，只算 i < j 的一半
    for a in range(len(item_ids)):
        for b in range(a + 1, len(item_ids)):
            item_i, item_j = item_ids[a], item_ids[b]
            sim = similarity(item_users[item_i], item_users[item_j])
            if sim > 0:
                item_sim[item_i][item_j] = sim
                item_sim[item_j][item_i] = sim

    similar_items = {}
    for item_id, sims in item_sim.items():
        similar_items[item_id] = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return similar_items


def score_items(recent_items, user_ratings, similar_items):
    """
    score(j) = Σ_{i ∈ 最近交互的物品} rating(u, i) * sim(i, j)
    即 用户对历史物品的兴趣 * 候选物品和该物品的相似度
    """
    scores = defaultdict(float)
    for item_i in recent_items:
        for item_j, sim in similar_items.get(item_i, []):
            scores[item_j] += user_ratings[item_i] * sim
    return scores


class ItemCF:
    def __init__(self, top_k=10, lastn=20):
        """
        top_k: 每个物品保留最相似的 top_k 个物品
        lastn: 每个用户保留最近交互的 lastn 个物品
        """
        self.top_k = top_k
        self.lastn = lastn

    def item_similarity(self, users_i, users_j):
        # 子类（比如 Swing）只需要换掉这一步
        return item_similarity(users_i, users_j)

    def fit(self, ratings):
        self.user_items = build_user_items(ratings)       # 用户 -> {物品: 评分}
        self.item_users = build_item_users(ratings)       # 物品 -> {用户: 评分}
        self.recent_items = build_recent_items(ratings, self.lastn)  # 用户-物品索引
        self.similar_items = build_similar_items(self.item_users, self.item_similarity, self.top_k)  # 物品-物品索引
        return self

    def get_similar_items(self, item_id):
        return [item for item, _ in self.similar_items.get(item_id, [])]

    def recommend(self, user_id, n_rec=10):
        scores = score_items(self.recent_items[user_id], self.user_items[user_id], self.similar_items)
        # 跳过用户已经交互过的物品
        return top_n(scores, n_rec, exclude=self.user_items[user_id])


if __name__ == "__main__":
    model = ItemCF(top_k=20, lastn=50).fit(load_ratings())

    user_id = 1
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in model.recommend(user_id, n_rec=10):
        print(f"物品ID: {item_id}, 分数: {score:.4f}")
