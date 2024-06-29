# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:57:22 2024

@author: demon
"""

import os
from modules.image_compression import ImageCompressor
from modules.image_segmentation import ImageSegmenter

input_folder = 'input'
output_folder_compressed = 'output/compressed'
output_folder_segmented = 'output/segmented'

# Ensure output directories exist
os.makedirs(output_folder_compressed, exist_ok=True)
os.makedirs(output_folder_segmented, exist_ok=True)

# Compress images in the input folder
compressor = ImageCompressor(n_colors=16)
compressor.compress_folder(input_folder, output_folder_compressed)

# Segment images in the input folder
segmenter = ImageSegmenter(n_segments=3)
segmenter.segment_folder(input_folder, output_folder_segmented)

