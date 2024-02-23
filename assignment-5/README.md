## Assignment 5

<p>This folder contains the solutions to assignment-5 of ERAv2 course.</p>

<p>We have separated the original code into [model.py](./model.py) (with the convNet model alone) and rest of the train/test loop and plotting functions into the [utils.py](./utils.py) file.</p>

<p>The [S5.ipynb](./S5.ipynb) file contains an object of the model file and then starts training the model (training loop) and runs the test loop.</p>

<p>The model params and size for our simple convnet are here (also present in the notebook) :</p>




     |   Layer (type)    |        Output Shape    |     Param # |
     | ----------------- | ---------------------- | ----------- |
     |      Conv2d-1     |     [-1, 32, 26, 26]   |         288 |
     |      Conv2d-2     |     [-1, 64, 24, 24]   |      18,432 |
     |      Conv2d-3     |    [-1, 128, 10, 10]   |      73,728 |
     |      Conv2d-4     |      [-1, 256, 8, 8]   |     294,912 |
     |      Linear-5     |             [-1, 50]   |     204,800 |
     |      Linear-6     |             [-1, 10]   |         500 |

----------------------------------------------------------------
Total params: 592,660  
Trainable params: 592,660   
Non-trainable params: 0  

----------------------------------------------------------------
 Input size (MB): 0.00  
 Forward/backward pass size (MB): 0.67  
 Params size (MB): 2.26  
 Estimated Total Size (MB): 2.93  

----------------------------------------------------------------


<p>The test accuracy has readched 99+ % for the last few epochs as seen in the notebook.</p>