import itertools

import pytest
import torch

from conftest import CPU
from 召回.dr import DeepRetrieval, DeepRetrievalRecommender, PathDataset, generate_item_paths, path_loss


def test_每个物品有固定数量的路径():
    item_to_paths, _ = generate_item_paths([1, 2, 3], num_paths=3, num_levels=4, nodes_per_level=10)
    for paths in item_to_paths.values():
        assert len(paths) == 3
        assert all(len(path) == 4 for path in paths)


def test_每一层的节点编号互不重叠():
    item_to_paths, _ = generate_item_paths(range(20), num_paths=3, num_levels=3, nodes_per_level=10)
    for paths in item_to_paths.values():
        for path in paths:
            for level, node in enumerate(path):
                assert level * 10 <= node < (level + 1) * 10


def test_路径到物品是反向索引():
    item_to_paths, path_to_items = generate_item_paths(range(20), num_paths=3, num_levels=3, nodes_per_level=2)
    for item_id, paths in item_to_paths.items():
        for path in paths:
            assert item_id in path_to_items[path]


def test_样本格式(tiny_ratings):
    user_map = {1: 0, 2: 1, 3: 2, 4: 3}
    item_to_paths, _ = generate_item_paths([10, 20, 30, 40], num_paths=2, num_levels=3, nodes_per_level=5)
    sample = PathDataset(tiny_ratings, user_map, item_to_paths)[0]
    assert sample["user_idx"].item() == 0
    assert sample["target_paths"].tolist() == [list(path) for path in item_to_paths[10]]


def make_model(num_levels=3):
    model = DeepRetrieval(num_users=4, num_levels=num_levels, nodes_per_level=4, embedding_dim=8, hidden_dim=16)
    model.eval()
    return model


def all_paths(num_levels=3, nodes_per_level=4):
    levels = [range(level * nodes_per_level, (level + 1) * nodes_per_level) for level in range(num_levels)]
    return torch.tensor(list(itertools.product(*levels)))


def test_第l层的输入包含前l层节点():
    model = make_model()
    assert [level[0].in_features for level in model.levels] == [8, 16, 24]


def test_一层内所有节点概率和为1():
    model = make_model()
    log_probs = model.level_log_probs(1, model.user_embedding(torch.tensor([0, 1])), torch.tensor([[2], [3]]))
    assert log_probs.shape == (2, 4)
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(2), atol=1e-6)


def test_所有路径概率和为1():
    model = make_model()
    paths = all_paths().unsqueeze(0)  # (1, 64, 3)
    log_probs = model.path_log_probs(torch.tensor([0]), paths)
    assert log_probs.shape == (1, 64)
    assert log_probs.exp().sum().item() == pytest.approx(1.0, abs=1e-5)


def test_路径概率是每层条件概率的乘积():
    model = make_model()
    user_emb = model.user_embedding(torch.tensor([0]))
    path = torch.tensor([[1, 6, 11]])
    expected = (
        model.level_log_probs(0, user_emb, path[:, :0])[0, 1]
        + model.level_log_probs(1, user_emb, path[:, :1])[0, 2]
        + model.level_log_probs(2, user_emb, path[:, :2])[0, 3]
    )
    assert torch.allclose(model.path_log_probs(torch.tensor([0]), path.unsqueeze(0))[0, 0], expected, atol=1e-6)


def test_物品概率是多条路径概率之和():
    model = make_model()
    paths = torch.tensor([[[1, 6, 11], [3, 4, 8]]])
    item_log_prob = model(torch.tensor([0]), paths)
    expected = torch.logsumexp(model.path_log_probs(torch.tensor([0]), paths), dim=1)
    assert torch.allclose(item_log_prob, expected)


def test_损失是负对数似然():
    model = make_model()
    batch = {"user_idx": torch.tensor([0, 1]), "target_paths": torch.tensor([[[1, 6, 11]], [[3, 4, 8]]])}
    assert path_loss(model, batch).item() == pytest.approx(-model(batch["user_idx"], batch["target_paths"]).mean().item())


def test_beam_search返回合法路径且有序():
    model = make_model()
    paths, log_probs = model.beam_search(torch.tensor([0, 1]), beam_size=3)
    assert paths.shape == (2, 3, 3)
    assert log_probs.shape == (2, 3)
    for scores in log_probs.tolist():
        assert scores == sorted(scores, reverse=True)
    for level in range(3):
        assert torch.all((paths[:, :, level] >= level * 4) & (paths[:, :, level] < (level + 1) * 4))


def test_beam_search的概率和路径对得上():
    model = make_model()
    paths, log_probs = model.beam_search(torch.tensor([0]), beam_size=3)
    assert torch.allclose(model.path_log_probs(torch.tensor([0]), paths), log_probs, atol=1e-5)


def test_beam足够大时找到概率最高的路径():
    model = make_model()
    paths, log_probs = model.beam_search(torch.tensor([0]), beam_size=64)
    exhaustive = model.path_log_probs(torch.tensor([0]), all_paths().unsqueeze(0))
    assert log_probs[0, 0].item() == pytest.approx(exhaustive.max().item(), abs=1e-5)


def test_推荐排除已交互物品(small_ratings):
    recommender = DeepRetrievalRecommender(embedding_dim=8, hidden_dim=16, n_epochs=1, num_paths_per_item=3,
                                           num_levels=2, nodes_per_level=3, device=CPU).fit(small_ratings)
    recommendations = recommender.recommend(1, n_rec=5, beam_size=5)
    rated = set(small_ratings.query("userId == 1")["movieId"])
    assert 0 < len(recommendations) <= 5
    assert not rated & {item_id for item_id, _ in recommendations}
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)
