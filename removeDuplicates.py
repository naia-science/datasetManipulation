import os
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

def compute_dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_LINEAR)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.uint8)

def hamming_distance(hash1, hash2):
    return np.mean(hash1 != hash2)

class Duplicate:
    def __init__(self, hash_size=8):
        self.image_hashes = {}
        self.hash_size=hash_size

    def hash_image_folder(self, image_folder):
        n_image = 0
    
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            
            try:
                # Load the image and convert to grayscale
                image = Image.open(file_path).convert('L')
                image = np.array(image)
                img_hash = compute_dhash(image, self.hash_size)
                self.image_hashes[file_path] = img_hash
                n_image += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        print(f'images hashed: {n_image}')

    def find_near_duplicates(self, threshold=0.1):
        duplicates = defaultdict(list)

        # Compare hashes for near-duplicates
        file_list = list(self.image_hashes.keys())
        for i, file1 in enumerate(file_list):
            for j in range(i + 1, len(file_list)):
                file2 = file_list[j]
                dist = hamming_distance(self.image_hashes[file1], self.image_hashes[file2])
                if dist <= threshold:
                    duplicates[file1].append(file2)
    
        return duplicates

