# get_dataset.py
import os
from PIL import Image
import shutil
import opendatasets as od

def get_images():
    # download_dataset.py

    dataset_url = 'https://www.kaggle.com/datasets/ikarus777/unsplash-lite'
    od.download(dataset_url)
    dataset_path = 'unsplash-lite'
    output_dir = 'high_res_images'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                if img.size[0] >= 300 and img.size[1] >= 300:
                    img_resized = img.resize((300, 300))
                    img_resized.save(os.path.join(output_dir, file))

    return output_dir
