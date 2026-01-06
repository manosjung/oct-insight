"""
Debug script to inspect the FastAI model structure
Run this to see the exact architecture of your baseline_model.pkl
"""

from fastai.vision.all import *
import pathlib

# Fix for Windows Path issues
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load model
print("Loading model...")
learn = load_learner('models/baseline_model.pkl')

print("\n" + "="*80)
print("MODEL STRUCTURE INSPECTION")
print("="*80)

# Get the PyTorch model
pytorch_model = learn.model

print("\n1. Type of model:")
print(f"   {type(pytorch_model)}")

print("\n2. Model components (top level):")
for i, module in enumerate(pytorch_model.children()):
    print(f"   [{i}] {type(module).__name__}")

print("\n3. Checking for ResNet attributes:")
attributes_to_check = ['layer1', 'layer2', 'layer3', 'layer4', 'conv1', 'bn1', 'relu', 'maxpool']
for attr in attributes_to_check:
    has_attr = hasattr(pytorch_model, attr)
    print(f"   {attr}: {has_attr}")

print("\n4. If wrapped in Sequential, checking first element:")
if hasattr(pytorch_model, '__getitem__'):
    try:
        first_elem = pytorch_model[0]
        print(f"   Type of pytorch_model[0]: {type(first_elem)}")
        print(f"   Checking pytorch_model[0] for ResNet attributes:")
        for attr in attributes_to_check:
            has_attr = hasattr(first_elem, attr)
            print(f"     {attr}: {has_attr}")
    except Exception as e:
        print(f"   Could not access pytorch_model[0]: {e}")

print("\n5. All named modules (first 20):")
for i, (name, module) in enumerate(pytorch_model.named_modules()):
    if i >= 20:
        print("   ... (truncated, see full output for more)")
        break
    print(f"   {name}: {type(module).__name__}")

print("\n6. Looking for 'layer4' in module names:")
layer4_modules = [(name, type(module).__name__) for name, module in pytorch_model.named_modules() if 'layer4' in name]
if layer4_modules:
    print("   Found the following modules with 'layer4':")
    for name, mtype in layer4_modules:
        print(f"     {name}: {mtype}")
else:
    print("   No modules found with 'layer4' in name")

print("\n7. All Conv2d layers (for Grad-CAM fallback):")
conv_layers = []
for name, module in pytorch_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        conv_layers.append((name, module))

print(f"   Found {len(conv_layers)} Conv2d layers")
if conv_layers:
    print(f"   Last Conv2d layer: {conv_layers[-1][0]}")
    print(f"   Last 5 Conv2d layers:")
    for name, _ in conv_layers[-5:]:
        print(f"     {name}")

print("\n8. Vocabulary (classes):")
print(f"   Classes: {learn.dls.vocab}")

print("\n" + "="*80)
print("END OF MODEL STRUCTURE INSPECTION")
print("="*80)
print("\nRecommended target layer for Grad-CAM:")

if layer4_modules:
    # Find the last layer4 module that's a leaf (no children)
    for name, mtype in reversed(layer4_modules):
        print(f"  Use: pytorch_model.{name.replace('.', '][')} (type: {mtype})")
        break
elif conv_layers:
    print(f"  Use last Conv2d layer: {conv_layers[-1][0]}")
else:
    print("  Could not determine appropriate layer - model structure is unusual")
