"""
OCT Dataset Exploration Script
-------------------------------
Quick analysis of the Kermany OCT dataset structure and class distribution.
This helps verify the dataset is properly organized before training.
"""

import os
import matplotlib.pyplot as plt

# Dataset paths - using the Kermany 2018 OCT dataset structure
base_dir = r"data/raw/OCT2017"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Four classes from the Kermany dataset
categories = ["CNV", "DME", "DRUSEN", "NORMAL"]

print("--- Dataset Overview ---")
print(f"Base directory: {base_dir}")

def count_images(directory, set_name):
    """Count images in each category for a given dataset split."""
    print(f"\n[{set_name} Set]")
    total = 0
    counts = []

    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return [], 0

    for category in categories:
        path = os.path.join(directory, category)
        try:
            num_files = len(os.listdir(path))
            print(f"  - {category}: {num_files:,} images")
            counts.append(num_files)
            total += num_files
        except FileNotFoundError:
            print(f"  - {category}: FOLDER MISSING")
            counts.append(0)

    print(f"  TOTAL: {total:,} images")
    return counts, total

# Count images in both training and test sets
train_counts, train_total = count_images(train_dir, "Training")
test_counts, test_total = count_images(test_dir, "Test")

# Visualize the class distribution (helpful for understanding imbalance)
try:
    plt.figure(figsize=(10, 5))
    plt.bar(categories, train_counts, color=['#ef4444', '#3b82f6', '#f97316', '#22c55e'])
    plt.title(f"Training Set Distribution (Total: {train_total:,} images)")
    plt.xlabel("Disease Category")
    plt.ylabel("Number of Images")
    plt.savefig("data_distribution.png", dpi=150, bbox_inches='tight')
    print("\nVisualization saved: data_distribution.png")
except Exception as e:
    print(f"\nCouldn't generate visualization: {e}")
