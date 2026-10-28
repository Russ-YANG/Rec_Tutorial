"""
各个算法共用的数据处理。

评分数据统一是 DataFrame，列为 userId, movieId, rating, timestamp。
"""
import random
from collections import defaultdict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DEFAULT_RATINGS_PATH = "ml-100k/ratings.csv"


def load_ratings(path=DEFAULT_RATINGS_PATH):
    ratings = pd.read_csv(path)
    print(f"共{len(ratings)}条数据")
    return ratings


def split_ratings(ratings, test_size=0.2, random_state=42):
    """随机划分训练集和测试集"""
    return train_test_split(ratings, test_size=test_size, random_state=random_state)


def build_id_maps(ratings):
    """原始 id -> 从 0 开始的连续下标，给 Embedding 用"""
    user_map = {user_id: idx for idx, user_id in enumerate(ratings['userId'].unique())}
    item_map = {item_id: idx for idx, item_id in enumerate(ratings['movieId'].unique())}
    return user_map, item_map


def build_user_items(ratings):
    """{user: {item: rating}}"""
    user_items = defaultdict(dict)
    for user_id, item_id, rating in zip(ratings['userId'], ratings['movieId'], ratings['rating']):
        user_items[user_id][item_id] = rating
    return user_items


def build_item_users(ratings):
    """{item: {user: rating}}，物品到用户的倒排表"""
    item_users = defaultdict(dict)
    for user_id, item_id, rating in zip(ratings['userId'], ratings['movieId'], ratings['rating']):
        item_users[item_id][user_id] = rating
    return item_users


def build_recent_items(ratings, lastn):
    """{user: [item, ...]}，按时间顺序保留每个用户最近交互的 lastn 个物品"""
    recent_items = {}
    sorted_ratings = ratings.sort_values('timestamp')
    for user_id, group in sorted_ratings.groupby('userId'):
        recent_items[user_id] = group['movieId'].tail(lastn).tolist()
    return recent_items


def sample_negative(all_items, interacted_items):
    """随机采一个用户没交互过的物品"""
    while True:
        item_id = random.choice(all_items)
        if item_id not in interacted_items:
            return item_id


class RatingDataset(Dataset):
    """显式评分样本: (用户, 物品, 评分)"""

    def __init__(self, ratings, user_map, item_map):
        self.users = [user_map[user_id] for user_id in ratings['userId']]
        self.items = [item_map[item_id] for item_id in ratings['movieId']]
        self.ratings = ratings['rating'].tolist()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user_idx': torch.tensor(self.users[idx], dtype=torch.long),
            'item_idx': torch.tensor(self.items[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }


class PointwiseDataset(Dataset):
    """
    隐式反馈样本: 每个正样本后面跟 negative_sampling 个负样本。
    下标 idx 对应第 idx // (1 + negative_sampling) 个正样本，
    idx % (1 + negative_sampling) == 0 时是正样本，否则是负样本。
    """

    def __init__(self, ratings, user_map, item_map, user_items, negative_sampling=3):
        self.user_ids = ratings['userId'].tolist()
        self.item_ids = ratings['movieId'].tolist()
        self.user_map = user_map
        self.item_map = item_map
        self.all_items = list(item_map.keys())
        # 负采样要排除用户在全部数据里交互过的物品，不只是训练集
        self.user_items = user_items
        self.negative_sampling = negative_sampling

    def __len__(self):
        return len(self.user_ids) * (1 + self.negative_sampling)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.negative_sampling)
        is_positive = idx % (1 + self.negative_sampling) == 0

        user_id = self.user_ids[pos_idx]
        if is_positive:
            item_id = self.item_ids[pos_idx]
        else:
            item_id = sample_negative(self.all_items, self.user_items[user_id])

        return {
            'user_idx': torch.tensor(self.user_map[user_id], dtype=torch.long),
            'item_idx': torch.tensor(self.item_map[item_id], dtype=torch.long),
            'label': torch.tensor(1.0 if is_positive else 0.0, dtype=torch.float)
        }
