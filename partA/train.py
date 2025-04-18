import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import traceback
import argparse
import wandb
from torchinfo import summary

from model import FlexibleCNN
from transformation import TransformDataset, get_transforms

os.environ['WANDB_TIMEOUT'] = '60'
wandb.login()

activation_dict = {
    "ReLU": nn.ReLU,
    "GeLU": nn.GELU,
    "SiLU": nn.SiLU,
    "Mish": nn.Mish
}


data_source = "nature_12K/inaturalist_12K/train"
test_data_source = "nature_12K/inaturalist_12K/val"
train_data_split_frac = 0.8
best_model_name = "best_model.pth"
num_blocks = 5
filter_size = 3




############################### train procedure ################################

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Train the model for the specified number of epochs
    """
    # Move model to device
    model = model.to(device)
    
    # Track best validation accuracy
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Use tqdm for progress bar
        for inputs, labels in tqdm(train_loader, desc="Training"):
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
            total_samples += inputs.size(0)
        
        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        val_loss = running_loss / total_samples
        val_acc = running_corrects.double() / total_samples
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_name)
        
        print()

        wandb.log(
            {
                "epoch": epoch,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "validation_accuracy": val_acc,
                "validation_loss": val_loss,
            }
        )
    
    print(f'Best val Acc: {best_val_acc:.4f}')
    return model
############################################################################

######################### evaluation on test data procedure ############################

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

##########################################################################


def main(args):

    ################################### Set the hyper-params from command line arguments ###################################
    
    num_filters = args.num_filters
    if (args.filter_organization == "same"):
        num_filters = [num_filters] * num_blocks
    elif (args.filter_organization == "double"):
        num_filters_arr = []
        for i in range(num_blocks):
            num_filters_arr.append(num_filters * (2**i))
        num_filters = num_filters_arr
    elif (args.filter_organization == "half"):
        num_filters_arr = []
        for i in range(num_blocks):
            num_filters_arr.append(int(num_filters / (2**i)))
        num_filters = num_filters_arr

    activation = activation_dict[args.activation]
    num_epochs = args.epochs

    lr = args.learning_rate
    fully_connected_neurons = args.fully_connected_neurons
    batch_size = args.batch_size

    batch_normalization = args.batch_normalization
    drop_out = args.drop_out
    data_augmentation = args.data_augmentation

    ##################################################################################


    wandb.run.name = f"fc_{fully_connected_neurons}_act_{args.activation}_id_{wandb.run.id}_nf_{args.num_filters}_fo_{args.filter_organization}_lr_{args.learning_rate}"


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations based on whether data augmentation is enabled
    train_transform = get_transforms(data_augmentation==data_augmentation)
    val_transform = get_transforms(data_augmentation=="No")
    
    # Load the dataset
    data_dir = data_source
    try:
        
        # Load the entire dataset
        # full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        full_dataset = datasets.ImageFolder(root=data_dir)
        
        # Split into train and validation
        train_size = int(train_data_split_frac * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        # train_dataset.dataset.transform = train_transform
        # val_dataset.dataset.transform = val_transform
        
        # Apply different transforms
        train_dataset = TransformDataset(train_dataset_raw, train_transform)
        val_dataset = TransformDataset(val_dataset_raw, val_transform)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Initialize the model
        # num_classes = len(train_dataset.classes)
        print("num filters", num_filters)
        num_classes = len(full_dataset.classes)
        model = FlexibleCNN(
            input_channels=3, ## rgb
            num_classes=num_classes,
            num_filters=num_filters,
            filter_size=filter_size,
            activation_fn=activation,
            dense_neurons=fully_connected_neurons,
            drop_out=drop_out,
            batch_normalization=batch_normalization
        )


        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train the model
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device
        )
        
        print("Training complete")


        # Load the best model for testing
        # Load the test dataset
        test_dir = test_data_source
        # Define transformations for the dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        print(f"Test dataset loaded successfully with {len(test_dataset)} samples")
        
        # Create data loader for testing
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model.load_state_dict(torch.load(best_model_name))
        
        # Evaluate the model on the test dataset
        print("Evaluating on test dataset...")
        test_loss, test_acc = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        print(f"Final Test Accuracy: {test_acc:.4f}")
        
        
    except Exception as e:
        print(f"Error: {e}")
        stack_trace_string = traceback.format_exc()
        print(stack_trace_string)



###########################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    default_dict = {
        "wandb_project": "DA6401_Assignment_2",
        "wandb_entity": "da24s002-indian-institute-of-technology-madras",
        "activation":"GeLU",
        "batch_size":64,
        "epochs":15,
        "learning_rate":0.00015670728191997078,
        "num_filters": 64,
        "fully_connected_neurons": 256,
        "filter_organization": "double",
        "data_augmentation": "No",
        "batch_normalization": "Yes",
        "drop_out": 0
    }


    
    parser.add_argument("-wp", "--wandb_project", type=str, default=default_dict["wandb_project"], help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument("-we", "--wandb_entity", type=str, default=default_dict["wandb_entity"], help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-e", "--epochs", type=int, default=default_dict["epochs"], help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=default_dict["batch_size"], help="Batch size used to train neural network.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=default_dict["learning_rate"], help="Learning rate used to optimize model parameters.")
    parser.add_argument("-a", "--activation", type=str, default=default_dict["activation"], choices=["ReLU", "GeLU", "Mish", "SiLU"], help="Which activation functions to use.")
    parser.add_argument("-nf", "--num_filters", type=int, default=default_dict["num_filters"], help="number of filters in first layer")
    parser.add_argument("-fcn", "--fully_connected_neurons", type=int, default=default_dict["fully_connected_neurons"], help="number of neurons in the fully connected hidden layer.")
    parser.add_argument("-fo", "--filter_organization", type=str, default=default_dict["filter_organization"], choices=["same", "double", "half"], help="How the number of filters count changes with changing layers.")
    parser.add_argument("-do", "--data_augmentation", type=str, default=default_dict["data_augmentation"], choices=["Yes", "No"], help="Include data augmentation in training.")
    parser.add_argument("-bn", "--batch_normalization", type=str, default=default_dict["batch_normalization"], choices=["Yes", "No"], help="Include batch normalization in nn layers.")
    parser.add_argument("-dr", "--drop_out", type=float, default=default_dict["drop_out"], help="Add dropout regularization to the layers.")

    args = parser.parse_args()



    wandb.init(project=args.wandb_project)
    main(args)
###########################################################################################