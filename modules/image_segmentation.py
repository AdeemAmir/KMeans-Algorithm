# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:35:14 2024
"""

import os
import numpy as np
from PIL import Image
from kmeans_algo import KMeans

class ImageSegmenter:
    def __init__(self, n_segments=3):
        self.n_segments = n_segments
        self.kmeans = KMeans(n_clusters=n_segments)

    def segment_folder(self, input_folder, output_folder, plot=False):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                self.segment(image_path, output_path, plot, filename)

    def segment(self, image_path, output_path, plot=False, filename=''):
        print(f"Segmenting image: {image_path}")
        image = Image.open(image_path)
        image = np.array(image) / 255.0
        w, h, d = image.shape
        image_array = np.reshape(image, (w * h, d))

        unique_colors = len(np.unique(image_array, axis=0))
        print(f"RAW unique colors: {unique_colors}")

        print("Fitting KMeans")
        self.kmeans.fit(image_array)
        print("KMeans fitted\nSegmenting...")

        if plot:
            self.kmeans.plot_clusters(image_array, os.path.join('execution', 'segmentation'), filename)

        labels = self.kmeans.predict(image_array)
        segmented_image = self._recreate_image(self.kmeans.centroids, labels, w, h)
        segmented_image = (segmented_image * 255).astype(np.uint8)
        segmented_image = Image.fromarray(segmented_image)
        segmented_image.save(output_path)
        print(f"Segmented image saved to: {output_path}")

    def _recreate_image(self, centroids, labels, w, h):
        d = centroids.shape[1]
        image = centroids[labels].reshape(w, h, d)
        return image
