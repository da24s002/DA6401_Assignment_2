import torch
from tqdm import tqdm


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