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
from matplotlib import pyplot as plt

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
best_model_name = "best_hyper_param_tuned_model.pth"
num_blocks = 5
filter_size = 3

def pick_one_image_per_class(dataset):
    class_to_idx = {}  # Dictionary to track which classes we've seen
    samples = []  # List to store one sample per class
    
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if label not in class_to_idx:
            class_to_idx[label] = idx
            samples.append((image, label))
        
        # Stop once we have one sample from each class
        if len(class_to_idx) == len(dataset.classes):
            break
            
    return samples



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

    ##################################################################################


    wandb.run.name = f"inference_run_{wandb.run.id}"


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    data_dir = data_source
    try:
        
        # Load the entire dataset
        # full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        full_dataset = datasets.ImageFolder(root=data_dir)
        
        # Split into train and validation
        
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



        model.load_state_dict(torch.load(best_model_name))
        model.to(device)
        model.eval()
        # Load the test dataset
        
        # Define transformations for the dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        test_dir = test_data_source
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        # Load the best model for testing
        samples = pick_one_image_per_class(test_dataset)
        print(f"Selected {len(samples)} samples, one per class")
        
        # Process each image and store predictions
        results = []
        table_data = []
        columns = ["Image", "Predicted Class", "Actual Class"]
        classes = test_dataset.classes
        for index, (image, label) in enumerate(samples):
            # Prepare the image for the model
            image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()
                print(predicted_class, label)
                
            # Store results
            results.append({
                'true_label': label,
                'true_class': classes[label],
                'predicted_label': predicted_class,
                'predicted_class': classes[predicted_class],
                'correct': (label == predicted_class)
            })

            table_data.append([wandb.Image(image), classes[predicted_class], classes[predicted_class]])
        table = wandb.Table(data=table_data, columns=columns)
        wandb.log({"prediction_table": table})
        
        
        
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