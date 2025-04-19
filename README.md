# DA6401_Assignment_2
CNN assignment for DA6401 course

# Part A:

Implementation of a CNN model having 5 blocks

In order to run the model you can run the following command :

runs with the best arguments found while experimenting with the INaturalist dataset.<br>
## python train.py 

arguments supported :

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | DA6401_Assignment_1 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | puspak  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 64 | Batch size used to train neural network. | 
| `-lr`, `--learning_rate` | 0.00021970549167311983 | Learning rate used to optimize model parameters | 
| `-a`, `--activation` | relu | choices:  ["sigmoid", "tanh", "relu"] |
| `-nf`, `--num_filters` | 64 | Number of filters in the first layer |
| `-fcn`, `--fully_connected_neurons` | 256 | choices: [128, 256] |
| `-fo`, `--filter_organization` | double | choices: [same, half, double] |
| `-do`, `--data_augmentation` | No | choices: [Yes, No] |
| `-bn`, `--batch_normalization` | Yes | choices: [Yes, No] |
| `-dr`, `--drop_out` | 0 | choices: [0, 0.2, 0.3] |

The final test accuracy is reported in console after all the epochs are run.

Please go through the wandb init workflow in your current directory before running the code, as the code is configured to automatically log the run into wandb.

Run the command:
## wandb init

<br>
then provide the options asked, (note that the default project name is provided in the key value pair : "wandb_project": "DA6401_Assignment_2", you can ## change it while providing command line argument or directly in the dictionary, if you use a different wandb project while initialization)

==========================================================================================

Running a wandb sweep,

eg:

## wandb sweep config.yaml
<br>
the link of the sweep created along with the id will be provided after you run the above command, just add the argument --count after the command provided. following is an example of such a command
<br>

## wandb agent da24s002-indian-institute-of-technology-madras/DA6401_Assignment_2/o5t3ukso --count 35
<br>

run the above to start a wandb sweep.
by default the sweep uses the list of hyperparams written in config.yaml


========================================================================================
```yaml
program: "train.py"
name: "DA6401_Assignment2_sweep"
method: "bayes"
metric:
  goal: maximize
  name: validation_accuracy
parameters:
  epochs:
    values: [10, 15]
  num_filters:
    values: [32, 64]
  batch_size:
    values: [16, 32, 64]
  learning_rate: 
    max: 0.001
    min: 0.0001
    distribution: log_uniform_values
  activation:
    values: ["ReLU", "GeLU", "SiLU", "Mish"]
  filter_organization:
    values: ["same", "double", "half"]
  data_augmentation:
    values: ["Yes", "No"]
  batch_normalization:
    values: ["Yes", "No"]
  drop_out:
    values: [0, 0.2, 0.3]
  fully_connected_neurons:
    values: [128, 256]
```
## Setting up the data  <br>
Please make sure you download the INaturalist dataset from the link: https://storage.googleapis.com/wandb_datasets/nature_12K.zip <br>
Extract the zip, and paste the folder: nature_12K where your train.py is:<br>
Your folder structure should look like this: <br>

_________________<br>
|&nbsp;&nbsp;train.py<br>
|&nbsp;&nbsp;config.yaml<br>
|&nbsp;&nbsp;model.py<br>
|&nbsp;&nbsp;transformation.py<br>
|&nbsp;&nbsp;best_hyper_param_tuned_model.pth<br>
|&nbsp;&nbsp;nature_12K<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;inaturalist_12K<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;train<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;val<br>
_________________ <br>
The train.py file is the file containing the main method, and is the entry point to the training where we can pass our hyper-params<br>
via command line arguments. <br>
config.yaml is the hyper-parameter sweeping file, containing all options explored for hyper-param tuning. <br>
model.py contains the cnn model definition which is being read by train.py to train the network. <br>
best_hyper_param_tuned_model.pth is the best model I have found using hyper-param tuning, this can be loaded with model.py<br>
and inferences can be made on any data without training.<br>
transformation.py contains the transformation applied to the input image, if we want different transformations to be applied, we can<br>
add it to this file.<br>


========================================================================================<br>
The table made in Question4 (Provide a 10×3 grid containing sample images from the test data and predictions made by your best model),<br>
is made using the inference_code.py python file.<br>

The model used (best model using hyper-params) is named as best_hyper_param_tuned_model.pth<br>

<br>
Link to Github Repository:<br>
https://github.com/da24s002/DA6401_Assignment_2/tree/main/partA<br>

==========================================================================================================================================<br>

# Part B<br>
Taking a pre-trained model (ResNet50 in our case), and fine tuning it for the INaturalist dataset.<br>

In order to run the model you can run the following command :<br>

python strategy_3_gradual_unfreezing_resnet50.py <br>

## Setting up the data  <br>
Please make sure you download the INaturalist dataset from the link: https://storage.googleapis.com/wandb_datasets/nature_12K.zip <br>
Extract the zip, and paste the folder: nature_12K where your strategy_3_gradual_unfreezing_resnet50.py is:<br>
Your folder structure should look like this: <br>

_________________<br>
|&nbsp;&nbsp;strategy_3_gradual_unfreezing_resnet50.py<br>
|&nbsp;&nbsp;evaluation.py<br>
|&nbsp;&nbsp;transformation.py<br>
|&nbsp;&nbsp;phase_training.py<br>
|&nbsp;&nbsp;best_model_gradual_unfreezing.pth<br>
|&nbsp;&nbsp;nature_12K<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;inaturalist_12K<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;train<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;val<br>
_________________<br>

The strategy_3_gradual_unfreezing_resnet50.py file is the file containing the main method, and is the entry point to the training where we have used the ResNet50 fine-tuning logic<br>
best_model_gradual_unfreezing.pth is the best model I have found using strategy 3<br>
transformation.py contains the transformation applied to the input image, if we want different transformations to be applied, we can<br>
add it to this file.<br>
evaluation.py contains the method for calculating test accuracy after fine-tuning<br>
phase_training.py file contains the method which can be used to train different phases in strategy 3, we can pass the layers we want to unfreeze<br>
and this function will only train those layers.<br>

This implements the third strategy discussed in the report, which is gradual unfreezing<br>
We can freeze or unfreeze layers or add phases of training by modifying the following snippet as we want <br>

```python
model = unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze=['fc'], lr=lr_phase_1, phase_no=1)
model = unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze=['layer4', 'fc'], lr=lr_phase_2, phase_no=2)
model = unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze=['layer3','layer4', 'fc'], lr=lr_phase_3, phase_no=3)
model = unfreeze_and_train(model, criterion, train_loader, val_loader, device, layer_names_to_unfreeze=['conv1', 'bn1','layer1','layer2','layer3','layer4', 'fc'], lr=lr_phase_4, phase_no=4)
```

<br>
Here the unfreeze_and_train function can take in the names of layers to unfreeze and the learning rates to apply to them. <br>
The best model is saved with the name best_model_gradual_unfreezing.pth<br>

The test accuracy with the best model is logged at the end of the code. <br>
Link to Github Repository:<br>
https://github.com/da24s002/DA6401_Assignment_2/tree/main/partB<br>


========================================================================================<br>
