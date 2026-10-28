import pandas as pd
import pytest
import torch

from conftest import CPU
from 召回.matrix_completion import MatrixCompletion, MatrixCompletionModel, MovieLensDataset


def test_数据集映射从0开始连续编号(small_csv):
    dataset = MovieLensDataset(small_csv, train=True)
    n_users, n_items = dataset.get_num_users_items()
    user_map, item_map = dataset.get_user_item_maps()
    assert sorted(user_map.values()) == list(range(n_users))
    assert sorted(item_map.values()) == list(range(n_items))


def test_训练集和测试集按比例划分(small_csv):
    total = len(pd.read_csv(small_csv))
    train = MovieLensDataset(small_csv, train=True, test_size=0.25)
    test = MovieLensDataset(small_csv, train=False, test_size=0.25)
    assert len(train) + len(test) == total
    assert len(test) == pytest.approx(total * 0.25, abs=1)


def test_样本格式(small_csv):
    dataset = MovieLensDataset(small_csv, train=True)
    sample = dataset[0]
    assert set(sample.keys()) == {"user_idx", "item_idx", "rating"}
    assert sample["user_idx"].dtype == torch.long
    assert sample["rating"].dtype == torch.float


def test_原始id和索引可以互相转换(small_csv):
    dataset = MovieLensDataset(small_csv, train=True)
    user_map, item_map = dataset.get_user_item_maps()
    user_id, item_id = next(iter(user_map)), next(iter(item_map))
    assert dataset.get_original_ids(user_map[user_id], item_map[item_id]) == (user_id, item_id)


def test_预测评分是两个向量的内积():
    model = MatrixCompletionModel(n_users=3, n_items=2, emb_size=2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 2.0], [0.0, 1.0], [3.0, 0.0]]))
        model.item_embedding.weight.copy_(torch.tensor([[2.0, 1.0], [1.0, -1.0]]))
    prediction = model(torch.tensor([0, 2]), torch.tensor([0, 1]))
    assert torch.allclose(prediction, torch.tensor([4.0, 3.0]))


def test_嵌入维度(small_csv):
    model = MatrixCompletionModel(n_users=5, n_items=7, emb_size=16)
    assert model.get_user_embedding(torch.tensor([0, 1])).shape == (2, 16)
    assert model.get_item_embedding(torch.tensor([3])).shape == (1, 16)


def test_训练能降低损失(small_csv):
    model = MatrixCompletion(path=small_csv, emb_size=8, batch_size=32, learning_rate=0.05, n_epochs=1)
    model.device = CPU
    model.load_data()
    model.model.to(CPU)
    first = model.train()["train_loss"]
    model.n_epochs = 30
    last = model.train()["train_loss"]
    assert last < first, "多训练几轮，训练集 MSE 应该下降"


def test_推荐排除已评分物品且有序(small_csv):
    model = MatrixCompletion(path=small_csv, emb_size=8)
    model.device = CPU
    model.load_data()
    model.model.to(CPU)
    recommendations = model.recommend_for_user(1, n_rec=5)
    rated = set(pd.read_csv(small_csv).query("userId == 1")["movieId"])
    assert len(recommendations) == 5
    assert not rated & {item_id for item_id, _ in recommendations}
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)


def test_predict和模型输出一致(small_csv):
    model = MatrixCompletion(path=small_csv, emb_size=8)
    model.device = CPU
    model.load_data()
    model.model.to(CPU)
    user_map, item_map = model.train_dataset.get_user_item_maps()
    expected = model.model(torch.tensor([user_map[1]]), torch.tensor([item_map[10]])).item()
    assert model.predict(1, 10) == pytest.approx(expected)
