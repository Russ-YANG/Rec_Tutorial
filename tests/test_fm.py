import numpy as np
import pytest

from 特征交叉.fm import FactorizationMachine, fm_gradients, fm_predict, interaction_term


def brute_force_interaction(x, v):
    # Σ_{i<j} <v_i, v_j> x_i x_j
    result = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            result += np.dot(v[i], v[j]) * x[i] * x[j]
    return result


def test_交叉项化简公式和暴力计算一致():
    x = np.array([1.0, 0.0, 2.0, -1.0, 0.5])
    v = np.random.randn(5, 3)
    assert interaction_term(x, v) == pytest.approx(brute_force_interaction(x, v))


def test_只有一个非零特征时没有交叉项():
    x = np.array([0.0, 3.0, 0.0])
    assert interaction_term(x, np.random.randn(3, 2)) == pytest.approx(0.0)


def test_预测是线性项加交叉项():
    x = np.array([1.0, 2.0])
    w = np.array([0.5, -1.0])
    v = np.array([[1.0, 0.0], [2.0, 1.0]])
    # 0.3 + (0.5 - 2) + <v0, v1> * 1 * 2 = 0.3 - 1.5 + 4
    assert fm_predict(x, 0.3, w, v) == pytest.approx(2.8)


def numeric_gradient(f, value, eps=1e-6):
    grad = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        original = value[index]
        value[index] = original + eps
        plus = f()
        value[index] = original - eps
        minus = f()
        value[index] = original
        grad[index] = (plus - minus) / (2 * eps)
    return grad


def test_梯度和数值梯度一致():
    x = np.array([1.0, 0.0, 2.0, -0.5])
    w = np.random.randn(4)
    v = np.random.randn(4, 3)
    _, grad_w, grad_v = fm_gradients(x, 1.0, v)
    assert np.allclose(grad_w, numeric_gradient(lambda: fm_predict(x, 0.0, w, v), w), atol=1e-5)
    assert np.allclose(grad_v, numeric_gradient(lambda: fm_predict(x, 0.0, w, v), v), atol=1e-5)


def test_梯度乘上误差():
    x = np.array([1.0, 2.0])
    v = np.random.randn(2, 2)
    grad_w0, grad_w, grad_v = fm_gradients(x, 3.0, v)
    _, unit_w, unit_v = fm_gradients(x, 1.0, v)
    assert grad_w0 == 3.0
    assert np.allclose(grad_w, 3 * unit_w)
    assert np.allclose(grad_v, 3 * unit_v)


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
