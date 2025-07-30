Brain Tumor MRI Classification Model Stress Testing
This README provides instructions to stress-test a pre-trained brain tumor MRI classification model using adversarial attacks (FGSM, PGD, AutoAttack) on the BrainTumor-7K dataset. The model is designed for 4-class classification (glioma, meningioma, notumor, pituitary) and optimized for robustness. This proof-of-concept allows users to evaluate the model’s performance under adversarial conditions without requiring training.
Prerequisites

Environment: Kaggle Notebook with 2x T4 GPUs (recommended for efficiency).
Dependencies:
Python 3.8+
PyTorch (torch, torchvision)
NumPy
scikit-learn
tqdm
AutoAttack (pip install autoattack)


Hardware: At least 1 GPU (2x NVIDIA T4 preferred, 32GB VRAM total).
Dataset: BrainTumor-7K dataset, available at /kaggle/input/brain-tumor-mri-dataset/ or via Kaggle.

Weights: .pth file provided in RobustnessBenchmarks

Setup Instructions

Create a Kaggle Notebook:

Go to Kaggle Notebooks and create a new notebook.
In “Settings” > “Accelerator,” select GPU T4 x2 to enable 2 GPUs.
Save and start the session.


Add the Dataset:

Click “+ Add Data” in the notebook’s right sidebar.
Search for “brain-tumor-mri-dataset” by Masoud Nickparvar and add it.
Verify the dataset is at /kaggle/input/brain-tumor-mri-dataset/ with Training and Testing folders containing subfolders: glioma, meningioma, notumor, pituitary.


Install AutoAttack:
!pip install autoattack


Download the Model:

Obtain the pre-trained model file (evolutionary_model.pth) from the provided source (e.g., conference repository or shared link).
Upload the model to your Kaggle notebook’s /kaggle/working/ directory.


Verify GPU Availability:
import torch
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")

Expected output:
Number of GPUs: 2
GPU 0: NVIDIA T4
GPU 1: NVIDIA T4



Stress Testing the Model
Dataset Preparation
The BrainTumor-7K dataset contains MRI images for 4-class classification. The following script loads the dataset, applies necessary transforms, and prepares a test set for evaluation.
Model Architecture
The model is a modified ResNet18, adapted for single-channel MRI images, with a final layer outputting 4 classes. It’s pre-trained and loaded from evolutionary_model.pth.
Stress Testing Script
Save the following code as stress_test.py in your Kaggle notebook or /kaggle/working/ directory. It evaluates the model’s clean accuracy and robustness against FGSM, PGD, and AutoAttack.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from autoattack import AutoAttack as AutoAttackLib
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPU(s)")

# Define transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset
dataset_root = '/kaggle/input/brain-tumor-mri-dataset/'
testing_path = os.path.join(dataset_root, 'Testing')
try:
    test_dataset = ImageFolder(root=testing_path, transform=test_transform)
    print(f"Class mapping: {test_dataset.class_to_idx}")
    class_names = [name for name, _ in sorted([(v, k) for k, v in test_dataset.class_to_idx.items()])]
    print(f"Classes: {class_names}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Ensure dataset is at '/kaggle/input/brain-tumor-mri-dataset/'")
    exit(1)

# Data loader
batch_size = 32 * num_gpus
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define model
class TumorResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorResNet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet(x)

# Load model
model = TumorResNet().to(device)
if num_gpus > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.load_state_dict(torch.load('/kaggle/working/evolutionary_model.pth', map_location=device))
model.eval()

# Loss function
loss_fn = nn.CrossEntropyLoss()

# FGSM attack
def fgsm_attack(model, loss_fn, images, labels, epsilon):
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    perturbed_images = images + epsilon * grad.sign()
    perturbed_images = torch.clamp(perturbed_images, -1, 1)
    return perturbed_images

# PGD attack
def pgd_attack(model, loss_fn, images, labels, epsilon, alpha, num_steps):
    images = images.to(device)
    labels = labels.to(device)
    perturbed_images = images.clone().detach() + torch.zeros_like(images).uniform_(-epsilon/2, epsilon/2).to(device)
    perturbed_images = torch.clamp(perturbed_images, -1, 1)
    for _ in range(num_steps):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = perturbed_images.grad.data
        curr_alpha = alpha * (1.0 if _ > num_steps // 2 else 0.5)
        perturbed_images = perturbed_images + curr_alpha * grad.sign()
        perturbation = torch.clamp(perturbed_images - images, -epsilon, epsilon)
        perturbed_images = images + perturbation
        perturbed_images = torch.clamp(perturbed_images, -1, 1)
        perturbed_images = perturbed_images.detach()
    return perturbed_images

# AutoAttack evaluation
def evaluate_autoattack(model, test_loader, epsilon, norm='Linf', batch_size=32, log_path='autoattack_log.txt'):
    model_eval = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_eval.eval()
    adversary = AutoAttackLib(model_eval, norm=norm, eps=epsilon, version='custom', 
                             attacks_to_run=['apgd-ce', 'apgd-dlr', 'fab', 'square'], log_path=log_path)
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc=f'AutoAttack Eval (ε={epsilon})'):
        images = images.to(device)
        labels = labels.to(device)
        images = (images * 0.5 + 0.5).clamp(0, 1)  # Denormalize to [0, 1]
        adv_images = adversary.run_standard_evaluation(images, labels, bs=batch_size)
        adv_images = (adv_images - 0.5) / 0.5  # Renormalize to [-1, 1]
        adv_images = adv_images.to(device)
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# General evaluation
def evaluate(model, loader, return_details=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            if return_details:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    if return_details:
        cm = confusion_matrix(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        return accuracy, cm, precision, recall, f1, macro_precision, macro_recall, macro_f1
    return accuracy

# Adversarial evaluation
def evaluate_adversarial(model, test_loader, attack_type='pgd', epsilon=0.01, alpha=0.002, num_steps=10):
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc=f'Adv. Eval ({attack_type})'):
        images = images.to(device)
        labels = labels.to(device)
        if attack_type == 'fgsm':
            adv_images = fgsm_attack(model, loss_fn, images, labels, epsilon)
        elif attack_type == 'pgd':
            adv_images = pgd_attack(model, loss_fn, images, labels, epsilon, alpha, num_steps)
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Run evaluations
print("\n=== Model Stress Test ===")
# Clean accuracy
clean_acc, cm, precision, recall, f1, macro_prec, macro_rec, macro_f1 = evaluate(model, test_loader, return_details=True)
print(f"Clean Accuracy: {clean_acc:.4f}")
print(f"Macro Precision: {macro_prec:.4f}, Recall: {macro_rec:.4f}, F1: {macro_f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nPer-class metrics:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")

# Adversarial attacks
attack_results = {}
for attack_type in ['fgsm', 'pgd']:
    attack_results[attack_type] = {}
    for eps in [0.01, 0.02, 0.03, 0.05]:
        adv_acc = evaluate_adversarial(
            model, test_loader, attack_type=attack_type, epsilon=eps,
            alpha=eps/5 if attack_type == 'pgd' else None, num_steps=10 if attack_type == 'pgd' else None
        )
        attack_results[attack_type][eps] = adv_acc

# AutoAttack
autoattack_results = {}
for eps in [0.01, 0.02, 0.03, 0.05]:
    acc = evaluate_autoattack(model, test_loader, epsilon=eps, norm='Linf', batch_size=batch_size, 
                              log_path=f'autoattack_log_eps_{eps}.txt')
    autoattack_results[eps] = acc
    print(f"AutoAttack Accuracy (ε={eps}): {acc:.4f}")

# Print results
print("\n=== Robustness Results ===")
for attack_type, results in attack_results.items():
    print(f"\n{attack_type.upper()} Attack:")
    for eps, acc in results.items():
        print(f"  Epsilon = {eps:.3f}: Accuracy = {acc:.4f}")

print("\nAutoAttack Robustness:")
for eps, acc in autoattack_results.items():
    print(f"  Epsilon = {eps:.3f}: Accuracy = {acc:.4f}")

print("\nResults saved. Check '/kaggle/working/' for logs.")

Running the Stress Test

Save the Script:

Copy the code into a Kaggle notebook cell or save as stress_test.py in /kaggle/working/.


Execute:
!python stress_test.py


Monitor GPU Usage:
!nvidia-smi

Ensure both GPUs are utilized (Python processes on GPU 0 and 1).

Outputs:

Logs: AutoAttack logs (autoattack_log_eps_*.txt) in /kaggle/working/.
Results: Printed clean accuracy, per-class metrics, confusion matrix, and adversarial accuracies for FGSM, PGD, and AutoAttack.



Expected Results
Based on prior evaluations, expect:

Clean Accuracy: ~98.58%
FGSM (ε=0.01–0.05): 58.40–96.72%
PGD (ε=0.01–0.05): 37.61–96.44%
AutoAttack (ε=0.01–0.05): 97.86–98.86%

These reflect the model’s robustness to adversarial perturbations, with AutoAttack showing particularly strong performance.
Troubleshooting

Dataset Error: Ensure /kaggle/input/brain-tumor-mri-dataset/Testing exists. Re-add the dataset if missing.
Model Loading Error: Verify evolutionary_model.pth is in /kaggle/working/.
Memory Error: Reduce batch_size to 32 or 16:batch_size = 32  # or 16


AutoAttack Error: Check logs (autoattack_log_*.txt) or update AutoAttack:!pip install --upgrade autoattack


GPU Issues: Confirm “GPU T4 x2” is selected in Kaggle settings. Restart the notebook if only one GPU is detected.

Notes

The model is evaluated on the test set only; training is not required.
Outputs are saved in /kaggle/working/. Download logs or results from the notebook’s “Output” section.
For further details, contact the model provider or refer to the associated conference paper (IEEE AI-SI 2025).
