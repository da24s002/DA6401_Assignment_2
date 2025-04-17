import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import os
from tqdm import tqdm


data_dir = "nature_12K/inaturalist_12K/train"
test_data_source = "nature_12K/inaturalist_12K/val"
phase_epochs = 3

lr_phase_1 = 0.001
lr_phase_2 = 0.0001
lr_phase_3 = 0.00001
lr_phase_4 = 0.000001


#################################### Function for finding testing accuracy using best saved model till now ################################
def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Evaluate the model on the test dataset
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    test_loss = running_loss / total_samples
    test_acc = running_corrects.double() / total_samples
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    return test_loss, test_acc

################################################################################################################################



############################################## Define transformations for the training data and create dataset and dataloader  ################################

"""
Here the first parameter, transformas.Resize(256), resizes the input image dimension to 256x256x3.

transforms.CenterCrop(224), takes a 224x224 patch from it, making the final dimension 224x224x3.

transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) is used, since, 
these specific mean ([0.485, 0.456, 0.406]) and standard deviation ([0.229, 0.224, 0.225]) values are 
the channel-wise (RGB) statistics calculated from the ImageNet dataset, so we are converting our image to match them for better results.

"""

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


########################################### Function to train for one phase ##################################################################
def train_one_phase(model, criterion, optimizer, train_loader, val_loader, num_epochs=3, device='cuda'):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')


        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_gradual_unfreezing.pth')
            
        print()
    

####################################################################################################################################
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # load dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
        
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    ############################################################################################################

    ############################################################################################################

    # Get the number of classes
    num_classes = 10
    print(f"Number of classes: {num_classes}")

    # Load pre-trained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Initially freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # Make sure the fc layer parameters require gradients
    for param in model.fc.parameters():
        param.requires_grad = True

    # Move model to device
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    ############ Phase 1: Train only the fully connected layer ###########
    print("Phase 1: Training only the fully connected layer")
    # Make sure all other layers are frozen
    for name, param in model.named_parameters():
        if "fc" not in name:  # if the parameter is not in the fc layer
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Create an optimizer that only updates the fc parameters
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr_phase_1)
    train_one_phase(model, criterion, optimizer, train_loader, val_loader, num_epochs=phase_epochs, device=device)

    ########### Phase 2: Unfreeze layer4 ###############
    print("Phase 2: Unfreezing layer4")
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Create an optimizer with different learning rates
    
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr_phase_2)
    train_one_phase(model, criterion, optimizer, train_loader, val_loader, num_epochs=phase_epochs, device=device)

    ################## Phase 3: Unfreeze layer3 ###############
    print("Phase 3: Unfreezing layer3")
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Create an optimizer with different learning rates
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr_phase_3)
    train_one_phase(model, criterion, optimizer, train_loader, val_loader, num_epochs=phase_epochs, device=device)

    ############### Phase 4: Unfreeze all remaining layers ###############
    print("Phase 4: Unfreezing all remaining layers")
    for param in model.parameters():
        param.requires_grad = True

    # Create an optimizer with different learning rates
    optimizer = optim.Adam(model.parameters(), lr=lr_phase_4)
    train_one_phase(model, criterion, optimizer, train_loader, val_loader, num_epochs=phase_epochs, device=device)

    print("Gradual unfreezing training complete")
    ####################################################################################################################################

    ########################################### Testing with the best model ########################################################################


    # Load the best model for testing
    # Load the test dataset
    test_dir = test_data_source
    test_dataset = datasets.ImageFolder(root=test_dir, transform=train_transforms)
    print(f"Test dataset loaded successfully with {len(test_dataset)} samples")

    # Create data loader for testing
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model.load_state_dict(torch.load("best_model_gradual_unfreezing.pth"))

    # Evaluate the model on the test dataset
    print("Evaluating on test dataset...")
    test_loss, test_acc = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Final Test Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()