import os
import argparse
from modules.image_compression import ImageCompressor
from modules.image_segmentation import ImageSegmenter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Compression and Segmentation')
    parser.add_argument('-a', action='store_true', help='Perform both segmentation and compression')
    parser.add_argument('-c', action='store_true', help='Perform compression only')
    parser.add_argument('-s', action='store_true', help='Perform segmentation only')
    parser.add_argument('--plot', action='store_true', help='Plot clustering process')
    return parser.parse_args()

def get_user_input(prompt, default):
    user_input = input(prompt).strip()
    return int(user_input) if user_input else default

print("\nhttps://github.com/AdeemAmir/KMeans-Algorithm\n")
print("     KMeans Algorithm")
print("Numerical Computing Project\n")
print("Program: BS(CS)")
print("By: Adeem Amir, 16050")
print("    Muhammad Naimatullah, 12809\n")

input_folder = 'input'
print('Ensure to place files in the "input" folder.')

output_folder_compressed = 'output/compressed'
os.makedirs(output_folder_compressed, exist_ok=True)
print('Output for compressed images will be in "output/compressed".')

output_folder_segmented = 'output/segmented'
os.makedirs(output_folder_segmented, exist_ok=True)
print('Output for segmented images will be in "output/segmented".\n')

args = parse_arguments()

change_kmeans_settings = input('Do you want to change the base settings of the KMeans algorithm? (y/n): ').strip().lower()
if change_kmeans_settings == 'y':
    max_iter = get_user_input('Enter the maximum number of iterations for KMeans (default: 300): ', 300)
    tol = float(get_user_input('Enter the tolerance for KMeans (default: 1e-4): ', 1e-4))
else:
    max_iter = 300
    tol = 1e-4
col = get_user_input('Enter the number of colors for image compression (default: 16): ', 16) 
seg = get_user_input('Enter the number of segments for image segmentation (default: 3): ', 3)

if args.a or args.c:
    print(f'Starting image compression with {col} colors')
    compressor = ImageCompressor(n_colors=col)
    compressor.kmeans.max_iter = max_iter
    compressor.kmeans.tol = tol
    compressor.compress_folder(input_folder, output_folder_compressed, plot=args.plot)
    print('Image compression completed.')

if args.a or args.s:    
    print(f'Starting image segmentation with {seg} segments')
    segmenter = ImageSegmenter(n_segments=seg)
    segmenter.kmeans.max_iter = max_iter
    segmenter.kmeans.tol = tol
    segmenter.segment_folder(input_folder, output_folder_segmented, plot=args.plot)
    print('Image segmentation completed.')
