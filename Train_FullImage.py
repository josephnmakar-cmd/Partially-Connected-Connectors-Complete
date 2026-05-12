import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import copy

# --- CONFIGURATION ---
DATA_DIR = "D:\\AI and Automation- University West\\Master Thesis\\Project code\\Thesis Project\\Full_ConnectorFrame_Dataset" 
BATCH_SIZE = 16 
# ---------------------

# 1. DEFINE THE IMAGE TRANSFORMATIONS
# We have to standardize the images before the AI sees them
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),       # Force image to 224x224 square
        transforms.RandomHorizontalFlip(),   # Data Augmentation (flips image randomly)
        transforms.ToTensor(),               # Convert from image to PyTorch Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ResNet colors
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),       
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# 2. LOAD THE DATASETS FROM FOLDERS
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'])
}

# 3. CREATE THE DATALOADERS
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False)
}

# Get the names of our classes directly from your folder names!
class_names = image_datasets['train'].classes
print(f"Classes found: {class_names}")

# Check if we have a Graphics Card (GPU) available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
# 4. DOWNLOAD THE BRAIN (Transfer Learning)
print("Downloading pre-trained ResNet-18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Find out how many connections are going into the final layer
num_ftrs = model.fc.in_features

# Chop off the old layer and replace it with a new one that has exactly 3 outputs
model.fc = nn.Linear(num_ftrs, len(class_names))

# Send the model to the Graphics Card (or CPU)
model = model.to(device)

# 5. DEFINE HOW THE BRAIN LEARNS
# CrossEntropyLoss is the industry standard for classifying categories
criterion = nn.CrossEntropyLoss()

# Adam is a highly efficient Optimizer. 'lr' stands for Learning Rate (how big of a step it takes when learning)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 6. THE TRAINING ENGINE
num_epochs = 25          # We can safely raise this now!
best_acc = 0.0  
patience = 4             # How many epochs to wait without improvement
epochs_no_improve = 0    # Counter

print("\n--- Starting Training ---")
start_time = time.time()

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  
        else:
            model.eval()   

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device) 
            labels = labels.to(device) 

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) 
                loss = criterion(outputs, labels) 

                if phase == 'train':
                    loss.backward()   
                    optimizer.step()  

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 7. EARLY STOPPING & CHECKPOINTING LOGIC
        if phase == 'val':
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'full_image_ai.pth')
                print(">>> New best model saved! Resetting patience counter.")
                epochs_no_improve = 0  # Reset the counter because we got a new high score
            else:
                epochs_no_improve += 1
                print(f">>> No improvement for {epochs_no_improve} epoch(s).")

    print() # Print a blank line for readability

    # 8. THE KILL SWITCH
    # If we have gone 'patience' epochs without beating our high score, stop the loop!
    if epochs_no_improve >= patience:
        print(f"🛑 EARLY STOPPING TRIGGERED! The AI stopped improving after Epoch {epoch+1 - patience}.")
        break

time_elapsed = time.time() - start_time
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best Validation Accuracy: {best_acc:.4f}')