"""
多样性：在相关性之外，让推荐结果彼此不那么像。

DPP (Determinantal Point Process)
1. compute_kernel_matrix  L_ij = q_i * sim(i, j) * q_j，q 是相关性，sim 是余弦相似度
2. 一组物品的行列式 det(L_S) = 质量^2 × 向量张成的体积^2，物品越相关、越不相似，行列式越大
3. greedy_select          每次选一个让 det(L_S) 最大的物品
4. sliding_window_select  只在相关性最高的几个物品里选，计算更快

MMR (Maximal Marginal Relevance)
score = λ * 相关性 - (1 - λ) * 和已选物品的最大相似度
"""
import numpy as np


def normalize_rows(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def cosine_similarity_matrix(embeddings):
    normalized = normalize_rows(embeddings)
    return normalized @ normalized.T


def compute_kernel_matrix(embeddings, relevance_scores):
    """L_ij = q_i * similarity_ij * q_j"""
    return np.outer(relevance_scores, relevance_scores) * cosine_similarity_matrix(embeddings)


def hyperparallelepiped_volume(embeddings):
    """向量张成的超平行体体积 = sqrt(det(V V^T))"""
    squared_volume = np.linalg.det(embeddings @ embeddings.T)
    return np.sqrt(max(squared_volume, 0.0))  # 数值误差可能出现很小的负数


def best_candidate(kernel_matrix, selected_indices, candidate_indices):
    """在候选物品中选一个，加入后让 det(L_S) 最大"""
    best_index = candidate_indices[0]
    best_determinant = -np.inf
    for candidate_index in candidate_indices:
        subset = selected_indices + [candidate_index]
        determinant = np.linalg.det(kernel_matrix[np.ix_(subset, subset)])
        if determinant > best_determinant:
            best_determinant = determinant
            best_index = candidate_index
    return best_index


class DPP:
    def compute_kernel_matrix(self, embeddings, relevance_scores):
        return compute_kernel_matrix(embeddings, relevance_scores)

    def compute_hyperparallelepiped_volume(self, embeddings):
        return hyperparallelepiped_volume(embeddings)

    def greedy_select(self, embeddings, relevance_scores, k):
        kernel_matrix = compute_kernel_matrix(embeddings, relevance_scores)
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))
        while len(selected_indices) < k and remaining_indices:
            best_index = best_candidate(kernel_matrix, selected_indices, remaining_indices)
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
        return selected_indices

    def sliding_window_select(self, embeddings, relevance_scores, k, window_size=3):
        """只在剩余物品里相关性最高的 window_size 个中选"""
        kernel_matrix = compute_kernel_matrix(embeddings, relevance_scores)
        remaining_indices = np.argsort(relevance_scores)[::-1].tolist()
        selected_indices = []
        while len(selected_indices) < k and remaining_indices:
            window = remaining_indices[:window_size]
            best_index = best_candidate(kernel_matrix, selected_indices, window)
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
        return selected_indices


def mmr_score(relevance, max_similarity, lambda_param):
    return lambda_param * relevance - (1.0 - lambda_param) * max_similarity


class MMR:
    def select(self, embeddings, relevance_scores, k, lambda_param=0.5):
        similarity = cosine_similarity_matrix(embeddings)
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))
        while len(selected_indices) < k and remaining_indices:
            best_index = remaining_indices[0]
            best_score = -np.inf
            for candidate_index in remaining_indices:
                # 相似度为负时按 0 算，不奖励"反方向"的物品
                max_similarity = max([0.0] + [similarity[candidate_index, i] for i in selected_indices])
                score = mmr_score(relevance_scores[candidate_index], max_similarity, lambda_param)
                if score > best_score:
                    best_score = score
                    best_index = candidate_index
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
        return selected_indices


def generate_test_data(n_items=50, embedding_dim=10, n_clusters=5):
    """物品分成几簇，离第 0 簇越近相关性越高"""
    np.random.seed(42)
    cluster_centers = np.random.randn(n_clusters, embedding_dim)
    embeddings = []
    relevance_scores = []
    for _ in range(n_items):
        cluster_id = np.random.randint(0, n_clusters)
        embedding = cluster_centers[cluster_id] + np.random.normal(0, 0.3, embedding_dim)
        embeddings.append(embedding)
        distance = np.linalg.norm(embedding - cluster_centers[0])
        relevance_scores.append(np.exp(-distance / 2.0))
    return np.array(embeddings), np.array(relevance_scores)


def evaluate_diversity(embeddings, selected_indices):
    """两两之间 1 - 余弦相似度 的平均值"""
    similarity = cosine_similarity_matrix(embeddings[selected_indices])
    upper = np.triu_indices(len(selected_indices), k=1)
    return np.mean(1.0 - similarity[upper])


def evaluate_relevance(relevance_scores, selected_indices):
    return np.mean(relevance_scores[selected_indices])


if __name__ == "__main__":
    embeddings, relevance_scores = generate_test_data()
    k = 10
    methods = {
        "按相关性排序": np.argsort(relevance_scores)[::-1][:k].tolist(),
        "DPP 贪心": DPP().greedy_select(embeddings, relevance_scores, k),
        "DPP 滑动窗口": DPP().sliding_window_select(embeddings, relevance_scores, k, window_size=5),
        "MMR": MMR().select(embeddings, relevance_scores, k, lambda_param=0.5),
    }
    for name, selected in methods.items():
        print(f"{name:<10} 相关性: {evaluate_relevance(relevance_scores, selected):.4f}  "
              f"多样性: {evaluate_diversity(embeddings, selected):.4f}")
