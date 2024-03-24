# Gated Transformer Networks for Time Series Classification: A Study on Resource Optimization
## Introduction
<p>In the context of time series classification tasks, Transformer-based architecture has shown its superb performance compared with other deep learning methods. However, huge resource consumption has always been an non-negligible issue when deploying this kind of models to the applications in embedded-AI-friendly scenarios. In this work, we studied a Transformer-based model called Gated Transformer Networks (GTN), which is a powerful model raised recently specializing in time series classification tasks. Like many other Transformer-based models, GTN achieves outstanding efficiency through its Gating mechanism and Two-tower Strategy. However, it consumes large memory space by model parameters and incurs high computational complexity, which grows quadratically with the sequence length in attention modules. This lays an obstacle when researchers want to deploy this state-of-the-art architecture on resource-constrained devices. To address the high memory and computational demands of GTN, we also applied resource optimization on this model. Below are the details of how to deploy a virtual environment and how to train the GTN model.

## Environments
Python 3.10.12
<P>Visual Studio Code

## Requirements
matplotlib==3.8.2<br>
numpy==1.26.3<br>
pandas==2.1.4<br>
torch==2.1.2<br>
wandb==0.16.12<br>
scikit-learn==1.4.1.post1 <br>
scipy==1.11.4 <br>
seaborn==0.13.2 <br>
transformers @ git+http://github.com/ibeltagy/transformers.git@52d6236dc15ad5142b4146ff74d2ec973fa3da22<br>
longformer @ git+https://github.com/allenai/longformer.git@caefee668e39cacdece7dd603a0bebf24df6d8ca<br>

## Instruction
<p>To conduct the experiment, we first need to deploy the virtual environment.</p>
To deploy the virtual evrionment,

  ```
    python3 -m pip install --user virtualenv
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
  ```

<p>After the venv is set up, we can run the programm with the script test_default.sh and train_default.sh, also you can change the hyper parameters in the script if you want.</p>
To conduct the training,

  ```
  source train_default.sh
  ```
To conduct only testing,
  ```
  source train_default.sh
  ```
<p>After the training or testing is done, you can find related plottings in the folder Confusion_Matrix and Heat_Map. The plottings will be divided by the name of the datasets into different folders. The learning curves and other metrices can be seen on Weights & Bias.</p>

## File Structure
### main.py<br>
- the main function of the programm where lays the sequence of the whole project<br>
### model1
- this folder contains the GTN model with vanilla attention.
### model2
- this folder contains the GTN model with ProbSparse attention.
### model3
- this folder contains the GTN model with Longformer attention.
### modules
- datast_process_longformer.py: used to preprocess data for Longformer attention.
- dataset_process.py: used to preprocess data for ProbSparse attention and vanilla attention.
- evaluation.py: used for evaluation phase during trainning.
- early_stopping.py: used to conduct early stopping in validation step.
- plot.py: used to plot the confusion matrices and heatmaps.
- train.pu: used for training phase.
### shellScript
- test_default.sh: used for only conducting testing.
- train_default.sh: used for conduct both training and testing.

## 超参描述
超参|描述|
----|---|
d_model|The model deals with time series data rather than natural language, so the encoding of words used in NLP is omitted. Instead, only a linear layer is employed to map the input into a dense vector of dimension d_model. Additionally, d_model ensures that the dimensions remain consistent at each module junction.|
d_hidden|The dimension of the hidden layers in Position-wise FeedForword.| 
d_input|The length of the timesteps, which is actually the longest timesteps of the datasets. It is directly defined by the data preprocessing.|
d_channel|The number of the channels of the time series data.|
d_output|The number of the classifications.|
q,v|The dimensionality of the linear layer mapping in Multi-Head Attention.|
h|The number of heads in Multi-Head Attention.|
N|The number of the encoders.|
dropout|The drop out rate.|
EPOCH| The number of epochs in the training.|
BATCH_SIZE|Mini-batch size.|
LR|Learning rate, which is defined as 1e-4.|
optimizer_name|The selection of Optimizers, **Adagrad**和Adam are recommended.|
attn_type| The selection of the attention modules.
num_exps| The number of experiments.
## contact
- email: xinyu.xie@stud.uni-due.de
