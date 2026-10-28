import numpy as np
import pytest

from 多样性.dpp import DPP, MMR, evaluate_diversity, evaluate_relevance, normalize_rows


def test_行归一化():
    normalized = normalize_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(normalized, [[0.6, 0.8], [0.0, 1.0]])


def test_核矩阵公式():
    # L_ij = q_i * cos(v_i, v_j) * q_j
    embeddings = np.array([[1.0, 0.0], [1.0, 1.0]])
    scores = np.array([2.0, 3.0])
    kernel = DPP().compute_kernel_matrix(embeddings, scores)
    expected = np.array([[4.0, 6.0 / np.sqrt(2)], [6.0 / np.sqrt(2), 9.0]])
    assert np.allclose(kernel, expected)


def test_核矩阵半正定():
    embeddings = np.random.randn(6, 4)
    kernel = DPP().compute_kernel_matrix(embeddings, np.random.rand(6))
    assert np.allclose(kernel, kernel.T)
    assert np.all(np.linalg.eigvalsh(kernel) > -1e-8)


def test_正交向量张成的体积是1():
    assert DPP().compute_hyperparallelepiped_volume(np.eye(3)) == pytest.approx(1.0)


def test_平行向量张成的体积是0():
    vectors = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert DPP().compute_hyperparallelepiped_volume(vectors) == pytest.approx(0.0, abs=1e-6)


def test_二维体积是平行四边形面积():
    vectors = np.array([[2.0, 0.0], [1.0, 3.0]])
    assert DPP().compute_hyperparallelepiped_volume(vectors) == pytest.approx(6.0)


# 物品0和1完全一样且最相关，物品2和它们正交但相关性低一些
DUPLICATE_EMBEDDINGS = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
DUPLICATE_SCORES = np.array([0.9, 0.8, 0.5])


def test_贪心选择第一个是最相关的():
    selected = DPP().greedy_select(DUPLICATE_EMBEDDINGS, DUPLICATE_SCORES, k=1)
    assert selected == [0]


def test_贪心选择会避开重复物品():
    selected = DPP().greedy_select(DUPLICATE_EMBEDDINGS, DUPLICATE_SCORES, k=2)
    assert selected == [0, 2]


def test_贪心选择返回k个不重复下标():
    embeddings = np.random.randn(10, 4)
    selected = DPP().greedy_select(embeddings, np.random.rand(10), k=5)
    assert len(selected) == 5 and len(set(selected)) == 5


def test_滑动窗口只在窗口内选择():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    # 窗口为1时退化成按相关性排序
    assert DPP().sliding_window_select(embeddings, scores, k=3, window_size=1) == [0, 1, 2]
    assert DPP().sliding_window_select(embeddings, scores, k=2, window_size=3) == [0, 2]


def test_MMR_lambda为1时按相关性排序():
    scores = np.array([0.2, 0.9, 0.5])
    assert MMR().select(np.random.randn(3, 2), scores, k=3, lambda_param=1.0) == [1, 2, 0]


def test_MMR会避开重复物品():
    assert MMR().select(DUPLICATE_EMBEDDINGS, DUPLICATE_SCORES, k=2, lambda_param=0.5) == [0, 2]


def test_多样性和相关性评估():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert evaluate_diversity(embeddings, [0, 1]) == pytest.approx(1.0)
    assert evaluate_diversity(embeddings, [0, 2]) == pytest.approx(0.0)
    assert evaluate_relevance(np.array([0.2, 0.4, 0.9]), [0, 2]) == pytest.approx(0.55)
