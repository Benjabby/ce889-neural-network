# ce889-neural-network
Code for a neural network assignment to implement a single hidden-layer neural network in C++ using the C++ standard library. The neural network is then applied to an Aria robot using the Aria API.

If you have access to the ARIA API and are running on a mobile robot, use the RunRobot build configuration. 

Otherwise use the RunNetwork (default) configuration to train a demo neural network.

## `MatrixFast`
Contains custom implemented, fast matrix functionality.

## `NeuralNetworkManager`
Provides functionality to train and test multiple neural networks with different hyper-parameters.

## `NeuralNetwork`
Represents a single hidden layer neural network for the provided training and validation data sets. 

Feedforward process is achieved with the following code;

    MatrixFast::multiply(&X, &hiddenWeights, &VH);
    VH += hiddenBias;
    VH.applyFunction(hiddenActivationFunction->activation,LAMBDA, &H);
    
    MatrixFast::multiply(&H, &outputWeights, &V);
    V += outputBias;
    V.applyFunction(outputActivationFunction->activation, LAMBDA, &Y);`

This represents the following equations:

<img src="https://latex.codecogs.com/svg.latex?\\\mathbf{V}^h=\mathbf{XW}^h+\mathbf{B}^h\\\mathbf{H}=\varphi^h(\mathbf{V}^h)\\\mathbf{V}=\mathbf{HW}^h+\mathbf{B}\\\mathbf{Y}=\varphi(\mathbf{V}) "/>

where **X** is a 1 by 2 column vector of inputs. **W**^h^ is a 2 by n matrix of hidden weights. **B**^h^ is a 1 by n column vector of hidden biases. <img src="https://latex.codecogs.com/svg.latex?\varphi^h" /> is the activation function for the hidden layer. **W** is an n by 2 matrix of output weights. **B** is a 1 by 2 column vector of output biases. <img src="https://latex.codecogs.com/svg.latex?\varphi" /> is the activation function for the output.

Thus the full feedforward equation is:

<img src="https://latex.codecogs.com/svg.latex?\mathbf{Y}=\varphi\left(\left(\varphi^h\left(\mathbf{XW}^h+\mathbf{B}^h\right)\right)\mathbf{W}+\mathbf{B}\right) " />


Backpropagation is calculated with the following code:

    MatrixFast::subtract(&T, &Y, &Error);
    
    V.applyFunction(outputActivationFunction->derivative, LAMBDA, &localGradient);
    localGradient.hadamard(&Error);
    
    MatrixFast G2(1, numHiddenNeurons);
    VH.applyFunction(hiddenActivationFunction->derivative, LAMBDA,&G2);
    
    MatrixFast::multiply(&localGradient, &outputWeights, &localHiddenGradient, false, true);
    localHiddenGradient.hadamard(&G2);
    
    MatrixFast dOW(numHiddenNeurons, numOutput);
    MatrixFast::multiply(&H, &localGradient, &dOW, true, false);
    dOW.scale(ETA);
    
    MatrixFast dHW(numInput, numHiddenNeurons);
    MatrixFast::multiply(&X, &localHiddenGradient, &dHW, true, false);
    dHW.scale(ETA);
    
    MatrixFast dOB(1, numOutput);
    localGradient.scale(ETA, &dOB);
    
    MatrixFast dHB(1, numHiddenNeurons);
    localHiddenGradient.scale(ETA, &dHB);
    
    
    dOW += alphaOutputWeight;
    outputWeights += dOW;
    
    dOB += alphaOutputBias;
    outputBias += dOB;

    dHW += alphaHiddenWeight;
    hiddenWeights += dHW;

    dHB += alphaHiddenBias;
    hiddenBias += dHB;

    dOW.scale(ALPHA, &alphaOutputWeight);
    dOB.scale(ALPHA, &alphaOutputBias);
    dHW.scale(ALPHA, &alphaHiddenWeight);
    dHB.scale(ALPHA, &alphaHiddenBias);



which represents the following equations:

<img src="https://latex.codecogs.com/svg.latex?\\ \mathbf{E}=\mathbf{T-Y}\\\boldsymbol{\delta}=\mathbf{E}\circ \varphi^\prime({\mathbf{V}})\\\Delta\mathbf{W}=\eta\mathbf{H^T}\boldsymbol{\delta}+\alpha\Delta\mathbf{W_{t-1}}\\\Delta\mathbf{B}=\eta\boldsymbol{\delta}+\alpha\Delta\mathbf{B_{t-1}}\\\boldsymbol{\delta}^h=\boldsymbol{\delta}\mathbf{W^T}\circ \varphi^{h^\prime}(\mathbf{V}^h)\\\Delta\mathbf{W}^h=\eta\mathbf{X^T}\boldsymbol{\delta}^h+\alpha\Delta\mathbf{W}^h_\mathbf{t-1}\\\Delta\mathbf{B}^h=\eta\boldsymbol{\delta}^h+\alpha\Delta\mathbf{B}^h_\mathbf{t-1}" />
<br/><br/>
<img src="https://latex.codecogs.com/svg.latex?\\\Delta\mathbf{W_{t-1}}=\textbf{W}+\Delta\mathbf{W}\\ \Delta\mathbf{W}^h_\mathbf{t-1}=\textbf{W}^h+\Delta\mathbf{W^h}\\ \Delta\mathbf{B_{t-1}}=\textbf{B}+\Delta\mathbf{B}\\\Delta\mathbf{B}^h_\mathbf{t-1}=\textbf{B}^h+\Delta\mathbf{B}^h " />

Where <img src="https://latex.codecogs.com/svg.latex?\circ" /> denotes the Hadamard product, and <img src="https://latex.codecogs.com/svg.latex?\mathbf{M^T} " /> denotes the transposition of matrix <img src="https://latex.codecogs.com/svg.latex?\mathbf{M}" />. <img src="https://latex.codecogs.com/svg.latex?\mathbf{T}" /> is a 1 by 2 column vector of target values. <img src="https://latex.codecogs.com/svg.latex?\varphi^\prime" /> is the derivative of the output activation function. <img src="https://latex.codecogs.com/svg.latex?\varphi^{h^\prime}" /> is the derivative of the hidden activation function. <img src="https://latex.codecogs.com/svg.latex?\eta" /> and <img src="https://latex.codecogs.com/svg.latex?\alpha" /> are scalars representing the learning rate and momentum respectively.
To speed up hyper-parameter searching, parallel processing was introduced using OpenMP.


## `TrainedNetwork`
This class represents a trained a `NeuralNetwork` object

## All Other Classes
All other classes remain unusued. They are from the initial inefficient scalar-based implementation
