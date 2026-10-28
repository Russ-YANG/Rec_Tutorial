import torch

from conftest import CPU
from GNN.lightgcn import LightGCN, LightGCNRecommender, bpr_loss, build_normalized_adjacency, propagate

# 2 个用户，3 个物品
# 用户0: 物品0、1   用户1: 物品1、2
USERS = torch.tensor([0, 0, 1, 1])
ITEMS = torch.tensor([0, 1, 1, 2])


def dense_adjacency():
    return build_normalized_adjacency(2, 3, USERS, ITEMS).to_dense()


def test_邻接矩阵形状和对称性():
    adjacency = dense_adjacency()
    assert adjacency.shape == (5, 5)
    assert torch.allclose(adjacency, adjacency.T)


def test_用户之间和物品之间没有边():
    adjacency = dense_adjacency()
    assert torch.all(adjacency[:2, :2] == 0)
    assert torch.all(adjacency[2:, 2:] == 0)


def test_对称归一化():
    # A_ui = 1 / sqrt(d_u * d_i)
    adjacency = dense_adjacency()
    # 用户0度数2，物品0度数1
    assert torch.isclose(adjacency[0, 2], torch.tensor(1 / 2 ** 0.5))
    # 用户0度数2，物品1度数2
    assert torch.isclose(adjacency[0, 3], torch.tensor(0.5))
    assert adjacency[0, 4] == 0


def test_0层时就是原始嵌入():
    model = LightGCN(num_users=2, num_items=3, embedding_dim=4, num_layers=0)
    adjacency = build_normalized_adjacency(2, 3, USERS, ITEMS)
    user_embeds, item_embeds = model(torch.tensor([0, 1]), torch.tensor([2]), adjacency)
    assert torch.allclose(user_embeds, model.user_embedding.weight)
    assert torch.allclose(item_embeds, model.item_embedding.weight[2:3])


def test_最终嵌入是各层嵌入的平均():
    model = LightGCN(num_users=2, num_items=3, embedding_dim=4, num_layers=2)
    adjacency = build_normalized_adjacency(2, 3, USERS, ITEMS)
    dense = adjacency.to_dense()
    e0 = torch.cat([model.user_embedding.weight, model.item_embedding.weight])
    e1 = dense @ e0
    e2 = dense @ e1
    expected = (e0 + e1 + e2) / 3
    user_embeds, item_embeds = model(torch.tensor([0, 1]), torch.tensor([0, 1, 2]), adjacency)
    assert torch.allclose(user_embeds, expected[:2], atol=1e-6)
    assert torch.allclose(item_embeds, expected[2:], atol=1e-6)


def test_预测分数是内积():
    model = LightGCN(num_users=2, num_items=3, embedding_dim=4, num_layers=1)
    adjacency = build_normalized_adjacency(2, 3, USERS, ITEMS)
    users, items = torch.tensor([0, 1]), torch.tensor([2, 0])
    user_embeds, item_embeds = model(users, items, adjacency)
    assert torch.allclose(model.predict(users, items, adjacency), (user_embeds * item_embeds).sum(dim=1))


def test_BPR损失公式():
    model = LightGCN(num_users=2, num_items=3, embedding_dim=4, num_layers=1)
    adjacency = build_normalized_adjacency(2, 3, USERS, ITEMS)
    users, positives, negatives = torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([2, 0])
    positive_scores = model.predict(users, positives, adjacency)
    negative_scores = model.predict(users, negatives, adjacency)
    expected = -torch.log(torch.sigmoid(positive_scores - negative_scores)).mean()
    loss = model.bpr_loss(users, positives, negatives, adjacency)
    assert loss.dim() == 0
    assert torch.allclose(loss, expected)


def test_训练BPR能拉开正负样本():
    model = LightGCN(num_users=2, num_items=3, embedding_dim=8, num_layers=1)
    adjacency = build_normalized_adjacency(2, 3, USERS, ITEMS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    users, positives, negatives = torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([2, 0])
    first = model.bpr_loss(users, positives, negatives, adjacency).item()
    for _ in range(50):
        loss = model.bpr_loss(users, positives, negatives, adjacency)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first


def test_传播k层():
    adjacency = build_normalized_adjacency(2, 3, USERS, ITEMS)
    embeddings = torch.randn(5, 4)
    layers = propagate(adjacency, embeddings, num_layers=2)
    assert len(layers) == 3
    assert torch.allclose(layers[0], embeddings)
    assert torch.allclose(layers[2], adjacency.to_dense() @ adjacency.to_dense() @ embeddings, atol=1e-6)


def test_BPR损失函数():
    positive, negative = torch.tensor([2.0, 0.0]), torch.tensor([1.0, 1.0])
    expected = -torch.log(torch.sigmoid(positive - negative)).mean()
    assert torch.allclose(bpr_loss(positive, negative), expected)
    # 正样本分数远高于负样本时损失接近 0
    assert bpr_loss(torch.tensor([20.0]), torch.tensor([0.0])).item() < 1e-6


def test_推荐排除已交互物品且有序(small_ratings):
    recommender = LightGCNRecommender(embedding_dim=8, num_layers=2, n_epochs=2, device=CPU).fit(small_ratings)
    recommendations = recommender.recommend(1, n_rec=5)
    rated = set(small_ratings.query("userId == 1")["movieId"])
    assert len(recommendations) == 5
    assert not rated & {item_id for item_id, _ in recommendations}
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)


def test_图只用训练集的边(small_ratings):
    recommender = LightGCNRecommender(embedding_dim=8, num_layers=1, n_epochs=1, test_size=0.25, device=CPU).fit(small_ratings)
    # 每条交互对应两条有向边
    assert recommender.adjacency._nnz() == 2 * (len(small_ratings) - round(len(small_ratings) * 0.25))
