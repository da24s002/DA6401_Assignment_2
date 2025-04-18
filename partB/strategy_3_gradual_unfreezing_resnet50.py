import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets
from evaluation import evaluate_model
from transformation import train_transforms
from phase_training import train_one_phase


data_dir = "nature_12K/inaturalist_12K/train"
test_data_source = "nature_12K/inaturalist_12K/val"
phase_epochs = 3

lr_phase_1 = 0.001
lr_phase_2 = 0.0001
lr_phase_3 = 0.00001
lr_phase_4 = 0.000001




##################################### unfreeze and train given the layer names #####################################################
def check_unfreeze(name, list_of_names_to_unfreeze):
    for unfreeze_name in list_of_names_to_unfreeze:
        if unfreeze_name in name:
            return True
    return False


def unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze, lr, phase_no):
    ############ Phase 1: Train only the fully connected layer ###########
    print("Phase "+str(phase_no)+": Training")
    # Make sure all other layers are frozen
    for name, param in model.named_parameters():
        if check_unfreeze(name, layer_names_to_unfreeze):
            # print(name, False)
            param.requires_grad = True
        else:
            # print(name, True)
            param.requires_grad = False
        

    # Create an optimizer that only updates the fc parameters
    print()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
    train_one_phase(model, criterion, optimizer, train_loader, val_loader, num_epochs=phase_epochs, device=device)
    return model


#####################################################################################################################################
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

    ####### sequentially unfreeze each part of the network and train for 'num_epochs=3' ###################
    model = unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze=['fc'], lr=lr_phase_1, phase_no=1)
    model = unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze=['layer4', 'fc'], lr=lr_phase_2, phase_no=2)
    model = unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze=['layer3','layer4', 'fc'], lr=lr_phase_3, phase_no=3)
    model = unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze=['conv1', 'bn1','layer1','layer2','layer3','layer4', 'fc'], lr=lr_phase_4, phase_no=4)


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