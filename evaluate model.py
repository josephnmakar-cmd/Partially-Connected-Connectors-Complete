import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIGURATION ---
TEST_DIR = 'D:\AI and Automation- University West\Master Thesis\Project code\Thesis Project\Test_FullFrame'
MODEL_PATH = 'full_image_ai.pth'
CLASS_NAMES = ['fail_disconnected', 'fail_partial', 'pass_seated'] 
# ---------------------

def evaluate_system():
    # 1. CHECK FOR FOLDERS
    if not os.path.exists(TEST_DIR):
        print(f"Error: Could not find the '{TEST_DIR}' folder.")
        return

    # 2. PREPARE THE DATA PIPELINE
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading test images...")
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=preprocess)
    print(f"\n>>> CRITICAL CHECK - PyTorch found these folders: {test_dataset.classes}\n")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. LOAD THE BRAIN
    print("Loading AI Model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval() # CRITICAL: No learning, just testing!

    # 4. RUN THE INFERENCE
    true_labels = []
    predictions = []

    print("Running evaluation (this will only take a few seconds)...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Save the results to our lists
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    # 5. CALCULATE METRICS
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (Performance Metrics)")
    print("="*50)
    # This automatically calculates Precision, Recall, and F1-Score!
    report = classification_report(true_labels, predictions, target_names=CLASS_NAMES)
    print(report)

    # 6. GENERATE THE CONFUSION MATRIX
    cm = confusion_matrix(true_labels, predictions)

    # Make it beautiful using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, 
                annot_kws={"size": 14}) # Make the numbers big and readable
    
    plt.title('AI Inspection - Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Status (Ground Truth)', fontsize=14)
    plt.xlabel('AI Predicted Status', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the graph as a high-quality image for your thesis!
    plt.savefig('confusion_matrix.png', dpi=300)
    print(">>> Confusion matrix graph saved as 'confusion_matrix.png'")
    
    # Show the graph on screen
    plt.show()

if __name__ == "__main__":
    evaluate_system()