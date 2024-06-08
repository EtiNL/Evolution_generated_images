import os
import random
import urllib.request
import tarfile
from tqdm import tqdm
import pickle
import numpy as np
import shutil
from PIL import Image

def download_and_extract(url, extract_to='.'):
    tar_path, _ = urllib.request.urlretrieve(url)
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)

def get_images():
    dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    download_and_extract(dataset_url, 'cifar100')
    convert_cifar100_to_images('cifar100/cifar-100-python')
    return 'cifar100_images'

def convert_cifar100_to_images(cifar100_path):
    with open(os.path.join(cifar100_path, 'train'), 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
    with open(os.path.join(cifar100_path, 'test'), 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')

    images = np.concatenate([train_data[b'data'], test_data[b'data']])
    labels = np.concatenate([train_data[b'fine_labels'], test_data[b'fine_labels']])
    filenames = np.concatenate([train_data[b'filenames'], test_data[b'filenames']])

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    output_dir = 'cifar100_images'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for idx, img in enumerate(images):
        label = labels[idx]
        filename = filenames[idx].decode('utf-8')
        label_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        img_path = os.path.join(label_dir, filename)
        Image.fromarray(img).save(img_path)
