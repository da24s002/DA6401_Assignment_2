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
|&nbsp;&nbsp;nature_12K<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;inaturalist_12K<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;train<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;val<br>
_________________


========================================================================================<br>
The table made in Question4 (Provide a 10×3 grid containing sample images from the test data and predictions made by your best model),<br>
is made using the inference_code.py python file.<br>

<br>
Link to Github Repository:<br>
https://github.com/da24s002/DA6401_Assignment_2/tree/main/partA<br>

==========================================================================================================================================<br>

# Part B<br>
Taking a pre-trained model (ResNet50 in our case), and fine tuning it for the INaturalist dataset.<br>

In order to run the model you can run the following command :<br>

python strategy_3_gradual_unfreezing_resnet50.py <br>

This implements the third strategy discussed in the report, which is gradual unfreezing<br>

The test accuracy with the best model is logged at the end of the code. <br>


========================================================================================<br>
