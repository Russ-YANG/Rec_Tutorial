import torch

from 序列.sim import SIM


def test_hard_search只找同类别的行为():
    model = SIM(num_items=20, embedding_dim=4, search_top_k=5)
    history_items = torch.tensor([[1, 2, 3, 4, 5, 6]])
    history_categories = torch.tensor([[7, 8, 7, 8, 7, 9]])
    searched, lengths = model.hard_search(torch.tensor([7]), history_items, history_categories, torch.tensor([6]))
    assert lengths.tolist() == [3]
    # 从最近的行为开始找
    assert searched[0, :3].tolist() == [5, 3, 1]
    assert searched[0, 3:].tolist() == [0, 0]


def test_hard_search最多返回top_k个():
    model = SIM(num_items=20, embedding_dim=4, search_top_k=2)
    history_items = torch.tensor([[1, 2, 3, 4]])
    history_categories = torch.tensor([[7, 7, 7, 7]])
    searched, lengths = model.hard_search(torch.tensor([7]), history_items, history_categories, torch.tensor([4]))
    assert searched.shape == (1, 2)
    assert lengths.tolist() == [2]
    assert searched[0].tolist() == [4, 3]


def test_hard_search忽略序列长度之外的padding():
    model = SIM(num_items=20, embedding_dim=4, search_top_k=5)
    history_items = torch.tensor([[1, 2, 9, 9]])
    history_categories = torch.tensor([[7, 8, 7, 7]])
    searched, lengths = model.hard_search(torch.tensor([7]), history_items, history_categories, torch.tensor([2]))
    assert lengths.tolist() == [1]
    assert searched[0, 0].item() == 1


def test_exact_search权重只分给有效行为():
    model = SIM(num_items=20, embedding_dim=4, search_top_k=4)
    searched = torch.tensor([[3, 5, 0, 0], [1, 2, 4, 6]])
    interest, weights = model.exact_search(torch.tensor([7, 8]), searched, torch.tensor([2, 4]))
    assert interest.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(weights[0, 2:], torch.zeros(2))


def test_没有同类行为时兴趣向量为0():
    model = SIM(num_items=20, embedding_dim=4, search_top_k=3)
    searched = torch.zeros(1, 3, dtype=torch.long)
    interest, weights = model.exact_search(torch.tensor([7]), searched, torch.tensor([0]))
    assert torch.allclose(interest, torch.zeros(1, 4))


def test_SIM输出():
    model = SIM(num_items=20, embedding_dim=4, search_top_k=3)
    prediction, weights, searched = model(
        torch.tensor([10, 11]),
        torch.tensor([1, 2]),
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
        torch.tensor([[1, 2, 1, 1], [2, 1, 2, 0]]),
        torch.tensor([4, 3]),
    )
    assert prediction.shape == (2,)
    assert torch.all((prediction > 0) & (prediction < 1))
    assert weights.shape == (2, 3) and searched.shape == (2, 3)
