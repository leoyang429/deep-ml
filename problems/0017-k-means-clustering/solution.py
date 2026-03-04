import numpy as np
import numpy.linalg as LA

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
    pts = np.array(points, dtype=np.float32) # n * d
    centroids = np.array(initial_centroids, dtype=np.float32) # k * d
    for _ in range(max_iterations):
        dist = LA.norm(pts[:, None] - centroids[None, ...], axis=-1) # n * k
        idx = np.argmin(dist, axis=1) # n,
        
        new_centroids = np.zeros_like(centroids, dtype=np.float32) # k * d
        cnt = np.zeros_like(centroids[:, 0], dtype=np.float32) # k,
        np.add.at(new_centroids, idx, pts)
        np.add.at(cnt, idx, 1)
        is_zero = cnt == 0
        new_centroids[~is_zero] = new_centroids[~is_zero] / cnt[~is_zero][:, None]
        new_centroids[is_zero] = centroids[is_zero]
        centroids = new_centroids
        
    return centroids