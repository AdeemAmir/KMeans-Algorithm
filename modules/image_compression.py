# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:45:07 2024
"""

import os
import numpy as np
from PIL import Image
from kmeans_algo import KMeans

class ImageCompressor:
    def __init__(self, n_colors=16):
        self.n_colors = n_colors
        self.kmeans = KMeans(n_clusters=n_colors)

    def compress_folder(self, input_folder, output_folder, plot=False):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                self.compress(image_path, output_path, plot, filename)

    def compress(self, image_path, output_path, plot=False, filename=''):
        image = Image.open(image_path)
        image = np.array(image) / 255.0

        w, h, d = image.shape
        image_array = np.reshape(image, (w * h, d))

        print(f"Image shape: {image.shape}")
        unique_colors = len(np.unique(image_array, axis=0))
        print(f"Original number of unique colors: {unique_colors}")

        self.kmeans.fit(image_array)
        if plot:
            self.kmeans.plot_clusters(image_array, os.path.join('execution', 'compressed'), filename)

        labels = self.kmeans.predict(image_array)

        compressed_image = self._recreate_image(self.kmeans.centroids, labels, w, h)
        compressed_image = (compressed_image * 255).astype(np.uint8)
        compressed_image = Image.fromarray(compressed_image)
        compressed_image.save(output_path)
        print(f"Compressed image saved to: {output_path}")

    def _recreate_image(self, centroids, labels, w, h):
        d = centroids.shape[1]
        image = centroids[labels].reshape(w, h, d)
        return image
