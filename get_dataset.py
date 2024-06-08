import os
import random
import urllib.request
import tarfile
from tqdm import tqdm

def download_and_extract(url, extract_to='.'):
    tar_path, _ = urllib.request.urlretrieve(url)
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)

def get_images():
    # Download and extract the dataset
    dataset_url = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
    print('Downloading images...')
    download_and_extract(dataset_url, 'caltech101')
    print('finished downloading')

    # Sample 500 images
    image_dir = 'caltech101/101_ObjectCategories'
    images = []
    for root, _, files in tqdm(os.walk(image_dir)):
        for file in files:
            if file.endswith('.jpg'):
                images.append(os.path.join(root, file))
                
    return image_dir
