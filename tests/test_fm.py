import numpy as np
import pytest

from 特征交叉.fm import FactorizationMachine


def brute_force_predict(model, x):
    # y = w0 + Σ w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j
    result = model.w0 + np.dot(model.w, x)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            result += np.dot(model.v[i], model.v[j]) * x[i] * x[j]
    return result


def test_二阶交叉项的化简公式和暴力计算一致():
    model = FactorizationMachine(n_features=5, n_factors=3)
    model.w0 = 0.3
    model.w = np.random.randn(5)
    x = np.array([1.0, 0.0, 2.0, -1.0, 0.5])
    assert model.predict_one(x) == pytest.approx(brute_force_predict(model, x))


def test_全零输入只剩偏置():
    model = FactorizationMachine(n_features=4)
    model.w0 = 1.5
    assert model.predict_one(np.zeros(4)) == pytest.approx(1.5)


def test_特征交叉权重是隐向量内积():
    model = FactorizationMachine(n_features=4, n_factors=2)
    assert model.get_feature_interactions(0, 2) == pytest.approx(np.dot(model.v[0], model.v[2]))


def test_predict批量预测():
    model = FactorizationMachine(n_features=3)
    X = np.random.randn(6, 3)
    predictions = model.predict(X)
    assert predictions.shape == (6,)
    assert predictions[2] == pytest.approx(model.predict_one(X[2]))


def test_训练能拟合带交叉项的数据():
    X = np.random.randint(0, 2, size=(100, 6)).astype(float)
    y = 0.5 + X[:, 0] - X[:, 1] + 2 * X[:, 2] * X[:, 3]
    model = FactorizationMachine(n_features=6, n_factors=4, learning_rate=0.02, epochs=30)
    losses = model.fit(X, y)
    assert len(losses) == 30
    assert losses[-1] < losses[0] * 0.2, "训练后 MSE 应该明显下降"
