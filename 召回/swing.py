"""
Swing：ItemCF 的变体，只换掉了物品相似度的算法。

ItemCF 里，两个物品被很多人同时喜欢就算相似。但如果这些人本身就很像
（比如同一个小圈子），这份相似度就有水分。Swing 会给重合度高的用户对降权。
"""
from common.data import load_ratings
from 召回.item_cf import ItemCF


def swing_similarity(users_i, users_j, user_items, alpha):
    """
    sim(i, j) = Σ_{u1 < u2 ∈ U_i ∩ U_j} 1 / (alpha + overlap(u1, u2))
    overlap(u1, u2) = |I_u1 ∩ I_u2|，两个用户共同喜欢的物品数
    user_items: {user: 喜欢的物品}
    """
    common_users = list(set(users_i) & set(users_j))
    similarity = 0.0
    for a in range(len(common_users)):
        for b in range(a + 1, len(common_users)):
            items_1 = user_items[common_users[a]]
            items_2 = user_items[common_users[b]]
            overlap = set(items_1) & set(items_2)
            similarity += 1.0 / (alpha + len(overlap))
    return similarity


class Swing(ItemCF):
    def __init__(self, top_k=10, lastn=30, alpha=10):
        super().__init__(top_k=top_k, lastn=lastn)
        self.alpha = alpha

    def item_similarity(self, users_i, users_j):
        return swing_similarity(users_i, users_j, self.user_items, self.alpha)


if __name__ == "__main__":
    # 全量数据要算很久，这里只取前 10000 条
    model = Swing(top_k=10, lastn=20).fit(load_ratings().head(10000))

    user_id = 1
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in model.recommend(user_id, n_rec=10):
        print(f"物品ID: {item_id}, 分数: {score:.4f}")
