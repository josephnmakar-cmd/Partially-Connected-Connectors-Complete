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

# 1. DEFINE THE IMAGE TRANSFORMATIONS (With Heavy Augmentation)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),       
        # --- NEW AUGMENTATIONS ---
        transforms.RandomHorizontalFlip(p=0.5),             # 50% chance to mirror image
        transforms.RandomRotation(degrees=10),              # Simulates the camera getting bumped
        transforms.ColorJitter(brightness=0.3, contrast=0.3), # Simulates clouds passing or factory lights changing
        # -------------------------
        transforms.ToTensor(),               
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
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
num_epochs = 10
best_acc = 0.0  # We will use this to track our highest score

print("\n--- Starting Training ---")
start_time = time.time()

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # Every epoch has a 'train' phase and a 'val' (test) phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Tell the brain: "You are allowed to change your neurons"
        else:
            model.eval()   # Tell the brain: "Pens down, this is a test. No changing neurons."

        running_loss = 0.0
        running_corrects = 0

        # Feed the batches of images to the AI
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device) # Send images to the RTX 3050
            labels = labels.to(device) # Send correct answers to the RTX 3050

            # Step 1: Clear the old math from the previous batch
            optimizer.zero_grad()

            # Step 2: The Forward Pass (The AI makes a guess)
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) # Which of the 3 classes got the highest score?
                loss = criterion(outputs, labels) # How wrong was the guess?

                # Step 3: The Backward Pass (The AI learns from its mistakes - ONLY IN TRAIN MODE)
                if phase == 'train':
                    loss.backward()   # Calculate the exact neuron changes needed
                    optimizer.step()  # Apply the changes to the brain

            # Calculate statistics for the screen
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate the final grade for this Epoch
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 7. SAVE THE BEST BRAIN
        # If this test score is the best one we've seen so far, save the model to the hard drive!
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_connector_ai.pth')
            print(">>> New best model saved to hard drive!")

    print() # Print a blank line for readability

time_elapsed = time.time() - start_time
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best Validation Accuracy: {best_acc:4f}')