"""
FM (Factorization Machine)：线性模型 + 二阶特征交叉。

y = w0 + Σ_i w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j

交叉项的权重矩阵 W (d × d) 近似为 V V^T，V 是 d × k，k 越大越近似、计算量越大。

1. interaction_term    二阶交叉项，用化简公式 O(kd) 计算
2. fm_predict          线性项 + 交叉项
3. fm_gradients        平方损失对 w0, w, V 的梯度
4. FactorizationMachine  SGD 训练
"""
import numpy as np


def interaction_term(x, v):
    """
    Σ_{i<j} <v_i, v_j> x_i x_j = 1/2 * Σ_f [ (Σ_i v_if x_i)^2 - Σ_i v_if^2 x_i^2 ]
    x: (d,)  v: (d, k)
    """
    sum_vx = v.T @ x                 # (k,)
    sum_v2x2 = (v ** 2).T @ (x ** 2)  # (k,)
    return 0.5 * np.sum(sum_vx ** 2 - sum_v2x2)


def fm_predict(x, w0, w, v):
    linear_term = w0 + np.dot(w, x)
    return linear_term + interaction_term(x, v)


def fm_gradients(x, error, v):
    """
    平方损失 1/2 (y_pred - y)^2 的梯度，error = y_pred - y
    ∂y/∂w0 = 1
    ∂y/∂w_i = x_i
    ∂y/∂v_if = x_i * Σ_j v_jf x_j - v_if * x_i^2
    """
    grad_w0 = error
    grad_w = error * x
    sum_vx = v.T @ x  # (k,)
    grad_v = error * (np.outer(x, sum_vx) - v * (x ** 2)[:, None])
    return grad_w0, grad_w, grad_v


class FactorizationMachine:
    def __init__(self, n_features, n_factors=10, learning_rate=0.01, epochs=10):
        self.n_features = n_features
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w0 = 0.0                       # 全局偏置
        self.w = np.zeros(n_features)       # 一阶权重
        self.v = np.random.normal(scale=0.1, size=(n_features, n_factors))  # 隐向量

    def predict_one(self, x):
        return fm_predict(x, self.w0, self.w, self.v)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def fit(self, X, y):
        """逐个样本做 SGD，返回每个 epoch 的 MSE"""
        losses = []
        for epoch in range(self.epochs):
            for x, y_true in zip(X, y):
                error = self.predict_one(x) - y_true
                # 梯度必须基于同一个更新前的参数状态计算
                grad_w0, grad_w, grad_v = fm_gradients(x, error, self.v)
                self.w0 -= self.learning_rate * grad_w0
                self.w -= self.learning_rate * grad_w
                self.v -= self.learning_rate * grad_v
            losses.append(np.mean((y - self.predict(X)) ** 2))
        return losses

    def get_feature_interactions(self, i, j):
        """特征 i 和 j 的交叉权重 <v_i, v_j>"""
        return np.dot(self.v[i], self.v[j])
