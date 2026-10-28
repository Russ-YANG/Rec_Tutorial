import torch

from 召回.dr import DeepRetrieval, MovieLensDataset


def test_每个物品有固定数量的路径(small_csv):
    dataset = MovieLensDataset(small_csv, num_paths_per_item=3, path_length=3)
    for item_id in dataset.item_ids:
        paths = dataset.item_to_paths[item_id]
        assert len(paths) == 3
        assert all(len(path) == 3 for path in paths)


def test_每一层的节点编号互不重叠(small_csv):
    dataset = MovieLensDataset(small_csv, num_paths_per_item=3, path_length=3)
    width = dataset.num_nodes_per_level
    for paths in dataset.item_to_paths.values():
        for path in paths:
            for level, node in enumerate(path):
                assert level * width <= node < (level + 1) * width


def test_路径到物品是物品到路径的反向索引(small_csv):
    dataset = MovieLensDataset(small_csv, num_paths_per_item=3, path_length=3)
    for item_id, paths in dataset.item_to_paths.items():
        for path in paths:
            assert item_id in dataset.path_to_items[path]


def test_样本格式(small_csv):
    dataset = MovieLensDataset(small_csv, num_paths_per_item=3, path_length=3)
    sample = dataset[0]
    assert sample["target_paths"].shape == (3, 3)
    assert dataset.get_num_users_nodes()[1] == dataset.num_nodes_per_level * 3


def make_model():
    model = DeepRetrieval(num_users=4, num_nodes=30, embedding_dim=8, hidden_dims=[16, 8])
    model.eval()
    return model


def test_训练时输出路径和物品的对数概率():
    model = make_model()
    users = torch.tensor([0, 1])
    # 每层 10 个节点: 第1层 0-9，第2层 10-19，第3层 20-29
    paths = torch.tensor([
        [[1, 12, 25], [3, 14, 29]],
        [[0, 10, 20], [9, 19, 29]],
    ])
    output = model(users, target_paths=paths)
    assert output["path_log_probs"].shape == (2, 2)
    assert output["item_log_probs"].shape == (2,)
    assert torch.all(output["path_log_probs"] <= 0)


def test_物品概率是多条路径概率之和():
    model = make_model()
    paths = torch.tensor([[[1, 12, 25], [3, 14, 29]]])
    output = model(torch.tensor([0]), target_paths=paths)
    expected = torch.logsumexp(output["path_log_probs"], dim=1)
    assert torch.allclose(output["item_log_probs"], expected)


def test_一层的所有节点概率之和为1():
    model = make_model()
    # 固定第1、2层，把第3层遍历一遍，概率和应该等于前两层的概率乘积
    paths = torch.tensor([[[2, 13, 20 + node] for node in range(10)]])
    output = model(torch.tensor([0]), target_paths=paths)
    level1 = torch.softmax(output["level1_logits"][0], dim=0)[2]
    level2 = torch.softmax(output["level2_logits"][0], dim=0)[3]
    total = output["path_log_probs"].exp().sum()
    assert torch.allclose(total, level1 * level2, atol=1e-5)


def test_beam_search返回合法路径():
    model = make_model()
    with torch.no_grad():
        output = model(torch.tensor([0, 1]), beam_size=3)
    assert len(output["paths"]) == 2
    for paths, probs in zip(output["paths"], output["probs"]):
        assert len(paths) == 3
        assert probs == sorted(probs, reverse=True)
        assert all(0 < prob <= 1 for prob in probs)
        for path in paths:
            for level, node in enumerate(path):
                assert level * 10 <= node < (level + 1) * 10
