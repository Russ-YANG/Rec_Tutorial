import numpy as np


def normalize_rows(matrix):
    normalized = np.zeros_like(matrix, dtype=float)
    for i in range(len(matrix)):
        length = np.linalg.norm(matrix[i])
        normalized[i] = matrix[i] / length
    return normalized


class DPP:

    def compute_kernel_matrix(self, embeddings, relevance_scores):
        normalized_embeddings = normalize_rows(embeddings)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        # DPP 的 L 核：L_ij = q_i * similarity_ij * q_j。
        # q 表示物品质量，这里直接使用相关性分数。
        quality_matrix = np.outer(relevance_scores, relevance_scores)
        kernel_matrix = quality_matrix * similarity_matrix
        return kernel_matrix

    def compute_hyperparallelepiped_volume(self, embeddings):
        gram_matrix = np.dot(embeddings, embeddings.T)
        squared_volume = np.linalg.det(gram_matrix)
        if squared_volume < 0.0 and squared_volume > -1e-10:
            squared_volume = 0.0
        return np.sqrt(squared_volume)

    def greedy_select(self, embeddings, relevance_scores, k):
        kernel_matrix = self.compute_kernel_matrix(embeddings, relevance_scores)
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))

        while len(selected_indices) < k and len(remaining_indices) > 0:
            best_index = remaining_indices[0]
            best_determinant = -1.0

            for candidate_index in remaining_indices:
                subset = selected_indices + [candidate_index]
                submatrix = kernel_matrix[np.ix_(subset, subset)]
                determinant = np.linalg.det(submatrix)

                if determinant > best_determinant:
                    best_determinant = determinant
                    best_index = candidate_index

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        return selected_indices

    def sliding_window_select(self, embeddings, relevance_scores, k, window_size=3):
        kernel_matrix = self.compute_kernel_matrix(embeddings, relevance_scores)
        remaining_indices = np.argsort(relevance_scores)[::-1].tolist()
        selected_indices = []

        while len(selected_indices) < k and len(remaining_indices) > 0:
            window_indices = remaining_indices[:window_size]
            best_index = window_indices[0]
            best_determinant = -1.0

            for candidate_index in window_indices:
                subset = selected_indices + [candidate_index]
                submatrix = kernel_matrix[np.ix_(subset, subset)]
                determinant = np.linalg.det(submatrix)

                if determinant > best_determinant:
                    best_determinant = determinant
                    best_index = candidate_index

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        return selected_indices


class MMR:

    def select(self, embeddings, relevance_scores, k, lambda_param=0.5):
        normalized_embeddings = normalize_rows(embeddings)
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))

        while len(selected_indices) < k and len(remaining_indices) > 0:
            best_index = remaining_indices[0]
            best_score = -np.inf

            for candidate_index in remaining_indices:
                max_similarity = 0.0
                for selected_index in selected_indices:
                    similarity = np.dot(
                        normalized_embeddings[candidate_index],
                        normalized_embeddings[selected_index]
                    )
                    if similarity > max_similarity:
                        max_similarity = similarity

                relevance = relevance_scores[candidate_index]
                score = lambda_param * relevance - (1.0 - lambda_param) * max_similarity

                if score > best_score:
                    best_score = score
                    best_index = candidate_index

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        return selected_indices


def generate_test_data(n_items=50, embedding_dim=10, n_clusters=5):
    np.random.seed(42)
    cluster_centers = np.random.randn(n_clusters, embedding_dim)
    embeddings = []
    relevance_scores = []

    for _ in range(n_items):
        cluster_id = np.random.randint(0, n_clusters)
        noise = np.random.normal(0, 0.3, embedding_dim)
        embedding = cluster_centers[cluster_id] + noise
        embeddings.append(embedding)

        distance = np.linalg.norm(embedding - cluster_centers[0])
        relevance = np.exp(-distance / 2.0)
        relevance_scores.append(relevance)

    return np.array(embeddings), np.array(relevance_scores)


def evaluate_diversity(embeddings, selected_indices):
    selected_embeddings = embeddings[selected_indices]
    normalized_embeddings = normalize_rows(selected_embeddings)
    total_distance = 0.0
    pair_count = 0

    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            similarity = np.dot(normalized_embeddings[i], normalized_embeddings[j])
            total_distance += 1.0 - similarity
            pair_count += 1

    return total_distance / pair_count


def evaluate_relevance(relevance_scores, selected_indices):
    return np.mean(relevance_scores[selected_indices])
