# Regularization methods in probabilistic neural networks and general regression neural network

This repository was created as part of the author's bachelor's thesis. Here you can find the implemented GRNN and PNN methods as well as modified versions with implemented L1 and L2 regularization. Also in the ```.ipynb``` files you can see code and experimental results, which are also part of the thesis.

## Repository structure
 - ```\GRNN``` contains both ```GRNN.GRNN``` and modified ```GRNN.TrainableGRNN``` classes.
 - ```\PNN``` contains both ```PNN.PNN``` and modified ```PNN.TrainablePNN``` classes.
 - ```\base``` contains base code parts, used in both ```GRNN``` and ```PNN``` including and also abstract classes for consitency of code:
 - - ```\Distance``` - distance metrics comptation
   - ```\Layers``` - classes for the layers of networks
   - ```\Kernels``` - kernles classes.
