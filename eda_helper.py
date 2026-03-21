import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import zipfile
from tqdm import tqdm


# ZIP EXTRACTION
def extract_zip_if_exists():
    """Check for archive.zip in dataset folder and extract if found"""
    zip_path = "dataset/archive.zip"
    
    if os.path.exists(zip_path):
        print(f"Found {zip_path}, extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("dataset/")
            print(f"Successfully extracted {zip_path}")
            return True
        except Exception as e:
            print(f"Error extracting zip file: {e}")
            return False
    else:
        print(f"Error: {zip_path} not found! Please ensure the zip file exists in the dataset folder.")
        return False


# UTILITIES
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_text(output_dir, filename, content):
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(content)


# CLASS DISTRIBUTION
def get_class_distribution(data_dir):
    class_counts = {}

    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            class_counts[cls] = len(os.listdir(cls_path))

    return class_counts


def plot_distribution(counts, title, output_dir, filename):
    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# SAMPLE IMAGES
def save_samples(data_dir, output_dir, filename, samples_per_class=3):
    classes = os.listdir(data_dir)

    plt.figure(figsize=(12, 8))
    plot_index = 1

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = os.listdir(cls_path)

        sample_images = random.sample(images, min(samples_per_class, len(images)))

        for img_name in sample_images:
            img_path = os.path.join(cls_path, img_name)
            img = Image.open(img_path)

            plt.subplot(len(classes), samples_per_class, plot_index)
            plt.imshow(img)
            plt.title(cls)
            plt.axis('off')

            plot_index += 1

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# IMAGE SIZE ANALYSIS
def analyze_image_sizes(data_dir):
    widths = []
    heights = []

    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        files = os.listdir(cls_path)

        for file in tqdm(files, desc=f"Sizes ({cls})"):
            try:
                img = Image.open(os.path.join(cls_path, file))
                w, h = img.size
                widths.append(w)
                heights.append(h)
            except:
                continue

    return widths, heights


# CHANNEL CHECK
def check_image_channels(data_dir):
    shapes = defaultdict(int)

    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        images = os.listdir(cls_path)

        for file in tqdm(images, desc=f"Channels ({cls})"):
            try:
                img = Image.open(os.path.join(cls_path, file))
                arr = np.array(img)
                shapes[str(arr.shape)] += 1
            except:
                continue

    return shapes


# CORRUPTED IMAGES
def find_corrupted_images(data_dir):
    corrupted = []

    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        files = os.listdir(cls_path)

        for file in tqdm(files, desc=f"Corrupted ({cls})"):
            path = os.path.join(cls_path, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                corrupted.append(path)

    return corrupted


# PIXEL DISTRIBUTION
def pixel_distribution(data_dir, output_dir, filename):
    pixels = []

    classes = os.listdir(data_dir)

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        files = os.listdir(cls_path)

        for file in tqdm(files, desc=f"Pixels ({cls})"):
            try:
                img = Image.open(os.path.join(cls_path, files)).convert('L')
                pixels.extend(np.array(img).flatten())
            except:
                continue

    plt.figure()
    plt.hist(pixels, bins=50)
    plt.title("Pixel Intensity Distribution")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()