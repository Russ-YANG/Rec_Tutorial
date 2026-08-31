import pandas as pd
from collections import defaultdict

# swing，itemcf的变体
class Swing:
    def __init__(self, 
                 dataset_path="ml-100k/ratings.csv", 
                 top_k=10,
                 lastn=30,
                 alpha=10):
        """
        取最相似的10个物品，保留用户最近交互的30个物品
        """
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.lastn = lastn
        self.item_sim = defaultdict(dict)
        self.alpha = alpha
        self.item_sim_sorted = None
        self.user_items = None
        self.user_item_index = None
        self.item_item_index = None
    
    def load_data(self):
        try:
            self.ratings = pd.read_csv(self.dataset_path)
            print(f"共{len(self.ratings)}条数据")
            return True
        except FileNotFoundError:
            # print(111)
            return False
    

    def get_user_items(self):
        # 返回用户-物品的交互字典 {user: {item: rating}}
        self.user_items = defaultdict(dict)
        for _, row in self.ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            rating = row.get("rating", 1.0)  # 如果没有rating就当1
            self.user_items[user_id][item_id] = rating
        return self.user_items


    # 获取共同用户，返回一个字典，key是物品，value是同时喜欢该物品的用户集合
    def get_co_users(self):
        self.item_users = defaultdict(list)       

        for _, row in self.ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            self.item_users[item_id].append(user_id)
        # 返回的格式是 {item_id: [user_id1, user_id2, ...]}
        return self.item_users
    
    # 计算用户的overlap
    def get_overlap(self, users1, users2):
        # Overlap(u1,u2) = u1喜欢的物品和u2喜欢物品的交集
        return set(users1) & set(users2)
    
    def compute_item_similarity(self):
        # 两个物品的 Swing 相似度是所有共同用户对贡献的和。
        # 每对用户的贡献 = 1 / (alpha + overlap(u1, u2))
        # 不需要考虑rating
            
        self.get_user_items() # 返回如{user_id: {item_id: rating}}
        self.get_co_users()   # 返回如{item_id: [user_id1, user_id2, ...]}

        item_ids = list(self.item_users.keys())
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                item_i = item_ids[i]
                item_j = item_ids[j]
                common_users = list(self.get_overlap(self.item_users[item_i], self.item_users[item_j]))

                similarity = 0.0
                for first in range(len(common_users)):
                    for second in range(first + 1, len(common_users)):
                        user_1 = common_users[first]
                        user_2 = common_users[second]
                        overlap = self.get_overlap(self.user_items[user_1], self.user_items[user_2])
                        similarity += 1.0 / (self.alpha + len(overlap))

                if similarity > 0.0:
                    self.item_sim[item_i][item_j] = similarity
                    self.item_sim[item_j][item_i] = similarity

        # 排序
        self.item_sim_sorted = {}
        for item_id, similarities in self.item_sim.items():
            sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            self.item_sim_sorted[item_id] = sorted_items[:self.top_k]
        return self.item_sim_sorted

    def recommend_for_user(self, user_id, n_rec=10):
        interacted_items = self.user_items[user_id]  # {item: rating}
        item_scores = defaultdict(float)

        recent_item_ids = self.user_item_index[user_id]
        for item_i in recent_item_ids:
            rating_i = interacted_items[item_i]
            # 用户对历史交互物品的兴趣 * 待计算物品和该物品的相似度
            if item_i not in self.item_sim:
                continue
            for item_j, sim_ij in self.item_sim_sorted[item_i]:
                if item_j in interacted_items:  # 跳过用户已交互的物品
                    continue
                item_scores[item_j] += rating_i * sim_ij
        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]
    
    def build_user_item_index(self):
        self.user_item_index = {}
        
        sorted_ratings = self.ratings.sort_values('timestamp')
        for user_id, group in sorted_ratings.groupby('userId'):
            self.user_item_index[user_id] = group['movieId'].tail(self.lastn).tolist()
    
    def build_item_item_index(self):
        """
        构建物品到物品的索引，返回每个物品最相似的K个物品
        """
        self.item_item_index = {}
        
        for item_id, similar_items in self.item_sim_sorted.items():
            self.item_item_index[item_id] = [item for item, _ in similar_items]
    
    def get_similar_items(self, item_id):
        return self.item_item_index[item_id]
    
    def get_user_interacted_items(self, user_id):
        return self.user_item_index[user_id]
    
    def train(self):
        self.compute_item_similarity()
        # 构建索引
        self.build_user_item_index()
        self.build_item_item_index()
        
        return self
    

if __name__ == "__main__":
    model = Swing(dataset_path="ml-100k/ratings.csv", top_k=10, lastn=20)
    if model.load_data():
        model.train()
        user_id = 1

        recommendations = model.recommend_for_user(user_id, n_rec=10)
        print(f"为用户 {user_id} 推荐的物品: {recommendations}")
