import pandas as pd
import torch

from conftest import CPU
from 召回.dssm import DSSM, DSSMRecommender, MovieLensDataset


def test_数据集长度包含负样本(small_csv):
    dataset = MovieLensDataset(small_csv, train=True, negative_sampling=3)
    assert len(dataset) == len(dataset.data) * 4


def test_正负样本的位置和标签(small_csv):
    dataset = MovieLensDataset(small_csv, train=True, negative_sampling=3)
    _, item_map = dataset.get_user_item_maps()
    inverse_item_map = {index: item_id for item_id, index in item_map.items()}

    positive = dataset[0]
    assert positive["label"].item() == 1.0
    assert inverse_item_map[positive["item_idx"].item()] == dataset.data.iloc[0]["movieId"]

    for idx in range(1, 4):
        assert dataset[idx]["label"].item() == 0.0


def test_负样本是用户没交互过的物品(small_csv):
    dataset = MovieLensDataset(small_csv, train=True, negative_sampling=3)
    for _ in range(50):
        user_id = dataset.data.iloc[0]["userId"]
        negative = dataset.sample_negative(user_id, dataset.data.iloc[0]["movieId"])
        assert negative not in dataset.user_items[user_id]


def test_输出是余弦相似度():
    model = DSSM(num_users=5, num_items=6, embedding_dim=8, hidden_layers=[16, 8])
    model.eval()
    scores = model(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    assert scores.shape == (3,)
    assert torch.all(scores <= 1.0 + 1e-5) and torch.all(scores >= -1.0 - 1e-5)


def test_用户和物品向量做了L2归一化():
    model = DSSM(num_users=5, num_items=6, embedding_dim=8, hidden_layers=[16, 8])
    model.eval()
    user_vectors = model.get_user_embedding(torch.tensor([0, 1]))
    item_vectors = model.get_item_embedding(torch.tensor([2, 3]))
    assert user_vectors.shape == (2, 8)
    assert torch.allclose(user_vectors.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(item_vectors.norm(dim=1), torch.ones(2), atol=1e-5)


def test_双塔可以分开算再做内积():
    model = DSSM(num_users=5, num_items=6, embedding_dim=8, hidden_layers=[16, 8])
    model.eval()
    users, items = torch.tensor([0, 1]), torch.tensor([2, 3])
    separate = (model.get_user_embedding(users) * model.get_item_embedding(items)).sum(dim=1)
    assert torch.allclose(model(users, items), separate, atol=1e-6)


def test_推荐排除已交互物品(small_csv):
    recommender = DSSMRecommender(path=small_csv, embedding_dim=8, hidden_layers=[16, 8], device=CPU)
    recommender.load_data()
    recommendations = recommender.recommend_for_user(1, n_rec=5)
    rated = set(pd.read_csv(small_csv).query("userId == 1")["movieId"])
    assert len(recommendations) == 5
    assert not rated & {item_id for item_id, _ in recommendations}
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)
