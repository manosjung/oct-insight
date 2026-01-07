"""
ResNet50 Baseline Model Training
----------------------------------
Train a ResNet50 classifier on the Kermany OCT dataset using FastAI.
Uses transfer learning from ImageNet for faster convergence.

Expected training time: ~10-30 minutes depending on hardware.
"""

from fastai.vision.all import *
import os

if __name__ == "__main__":
    # Setup paths
    path = Path("data/raw/OCT2017")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    print("--- Starting Model Training ---")

    # 1. Data Loading and Preprocessing
    print("1. Preparing data pipeline...")

    # Define how to load and process the OCT images
    octs = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        # Split based on folder structure (train/ vs test/)
        splitter=GrandparentSplitter(train_name='train', valid_name='test'),
        # Label comes from parent folder name (CNV, DME, DRUSEN, NORMAL)
        get_y=parent_label,
        # Resize all images to 224x224 (standard ResNet input size)
        item_tfms=Resize(224),
        # Apply data augmentation to reduce overfitting
        batch_tfms=aug_transforms(size=224, min_scale=0.75)
    )

    # Create dataloaders - handles batching and loading images during training
    # Batch size of 64 works well for most GPUs, reduce to 32 or 16 if you get OOM errors
    dls = octs.dataloaders(path, batch_size=64, num_workers=0)

    print(f"   Training images: {len(dls.train_ds):,}")
    print(f"   Validation images: {len(dls.valid_ds):,}")
    print(f"   Classes: {dls.vocab}")

    # 2. Model Setup
    print("2. Initializing ResNet50 architecture...")
    # Using pretrained ResNet50 from ImageNet as starting point
    learn = vision_learner(dls, resnet50, metrics=accuracy, path=Path("."))

    # 3. Training
    # fine_tune() uses discriminative learning rates - trains the head first,
    # then gradually unfreezes and trains the entire network
    print("3. Training model (this may take a while)...")

    # Start with 1 epoch for quick testing, increase to 5-10 for better results
    learn.fine_tune(1)

    # 4. Save Model
    print("4. Exporting trained model...")
    learn.export('models/baseline_model.pkl')
    print(f"   Model saved to: {model_dir}/baseline_model.pkl")

    print("--- Training Complete ---")
    print("\nNext steps:")
    print("  - Run app.py to test the model")
    print("  - Check validation metrics above to assess performance")
    print("  - Consider training for more epochs if accuracy is low")
