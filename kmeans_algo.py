# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:46:58 2024

Step !: Research
Step @: Work
Step #: bug fixes
Step $: $$$

Step 1: create a KMeans algo. from scratch.
Step 2: create image_segmentation, image_compression.py.
Step 3: create a direct executable for testing.
Step 4: tests and bug fixes
Step 5: get more template
Step 6: fix os stuff
Step 7: impleemnt 'argparse' learnt in IoP from Dr. Aarij.
Step 8: remove uselses libraries.
Step 9: add graphs.
Step 9: create introduction and pause.
Step 10: template.
Step 11: example files.
Step 12: code cleanup.
Step 13: pre-done files.
Step 14: presenation.
Step 15: documentation.

"""

import numpy as np
import matplotlib.pyplot as plt
import os

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        self.history.append((self.centroids.copy(), None))

        for i in range(self.max_iter):
            self.labels = self._assign_clusters(X)
            self.history.append((self.centroids.copy(), self.labels.copy()))  # for plotting
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])
        
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break
            
            self.centroids = new_centroids

        self.history.append((self.centroids.copy(), self.labels.copy()))  # Final state

    def _initialize_centroids(self, X):
        np.random.seed(42)
        centroids = [X[np.random.choice(range(X.shape[0]))]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            i = np.searchsorted(cumulative_probs, r)
            centroids.append(X[i])
        return np.array(centroids)

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)

    def plot_clusters(self, X, output_folder, filename):
        os.makedirs(output_folder, exist_ok=True)
        colors = plt.get_cmap('tab20b', self.n_clusters)
        
        for i, (centroids, labels) in enumerate(self.history):
            plt.figure()
            if labels is not None:
                for j in range(self.n_clusters):
                    plt.scatter(X[j:, 0], X[j:, 1], label=f'Cluster {j}', color=colors(j / self.n_clusters), s=25)
            else:
                plt.scatter(X[:, 0], X[:, 1], color='black', label='Data points', s=25)
            plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=100, label='Centroids')
            plt.title(f'{filename} - Iteration {i}')
            plt.legend()
            plt.savefig(os.path.join(output_folder, f'{filename}_iteration_{i}.png'))
            plt.close()
    
    '''
    EXTRA: INCASE ABOVE ISNT ENUF
    
    def _manhattan_distance(self, X):
        # Calculate Manhattan distance between X and centroids
        distances = np.sum(np.abs(X[:, np.newaxis] - self.centroids), axis=2)
        return np.argmin(distances, axis=1)

    def _cosine_similarity(self, X):
        # Calculate cosine similarity between X and centroids
        norms_X = np.linalg.norm(X, axis=1)
        norms_centroids = np.linalg.norm(self.centroids, axis=1)
        similarity = np.dot(X, self.centroids.T) / np.outer(norms_X, norms_centroids)
        return np.argmax(similarity, axis=1)
    '''