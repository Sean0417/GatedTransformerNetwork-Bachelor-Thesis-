# LSTM for predicting future Ozone
## Introduction
<p>This is a project using LSTM to predict future Ozone quantity. In this project we use every 6 days' Ozone data to predict the seventh day's Ozone quantity.

## Environments
Python 3.10.9
<P>Visual Studio Code

## Requirements
matplotlib==3.8.0<br>
numpy==1.26.1<br>
pandas==2.1.2<br>
torch==2.1.0<br>
wandb==0.15.12<br>

## Instruction
<p>To conduct the experiment, we first need to deploy the virtual environment.</p>
To deploy the virtual evrionment,

  ```
    python3 -m pip install --user virtualenv
    python3 -m venv venv
    source venv_test/bin/activate
    pip install -r requirements.txt
  ```

<p>After the venv is set up, we can run the programm with the script start_training.sh, also you can change the hyper parameters in the script if you want.</p>
To run the programm,

  ```
  source start_py.sh
  ```
<p>After the training is done, you can find related plottings in the folder pic, which includes loss.png and results.png. loss.png shows the train loss and validation loss of the best model. results.png shows the trend differences between labels and predictions.</p>

## File Structure
### main.py<br>
- the main function of the programm where lays the sequence of the whole project<br>
### dataloader.py <br>
- sort_data_by_slidingwindow(): sort the data in sliding window, the return value type are two arrays.
- train_validate_test_data_split(): divide the data into training set, validation set and test set.
- dataPrepare(): get the required dataloader.
### model.py
-  LSTM_Regression: the class of the training model, which consists of a lstm layer and a linear layer.
### train.py
- training_cycle(model,epoch_sum,train_loader,val_loader,learningRate,patience,size_average=True): function used to conduct the training cycle.
- change_best_model_name():change the best model's file name in order to call it in evaluation module.
### early_stopping.py
- used to conduct early stopping in validation step.
### evaluation.py
- evalution(): perform the test process after training.

### start_py.sh
- shell script used to run python project. you can also change the parameters here.
### plot.py
- plot_prediction_curve(): plot the prediction results with the data on test set.
- plot_Train_and_validation_loss():plot the learning curve(training loss and validation loss).
## contact
- email: xinyu.xie@stud.uni-due.de
