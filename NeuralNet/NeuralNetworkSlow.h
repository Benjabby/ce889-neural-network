#pragma once

#include <vector>
#include <cassert>

// TEMP
#include <sstream>
#include "Utilities.h"


//constexpr int NUM_INPUT = 2;
//constexpr int NUM_OUTPUT = 2;

//const ActivationFunction LOGISTIC = ActivationFunction(ActivationFunction::logisticActivation, ActivationFunction::logisticDerivative);
//const ActivationFunction SOFTPLUS = ActivationFunction(ActivationFunction::softPlusActivation, ActivationFunction::logisticActivation);
//const ActivationFunction LINEAR = ActivationFunction(ActivationFunction::linearActivation, ActivationFunction::linearDerivative);


class NeuralNetworkSlow
{
public:

	// Constructors

	// In the future this could be expanded to include other parameters, but this is sufficient for our needs.
	NeuralNetworkSlow(std::vector<LabeledTuple> &trainSet, std::vector<LabeledTuple> &validateSet, int numHiddenNeurons, double eta, double alpha);
	// Test Constructor
	//NeuralNetworkSlow(int numHiddenNeurons);


	// Functions



	void printWeights();

	// Other stopping criteria to come
	void startTraining(int totalEpochs);


	// Destructors

	~NeuralNetworkSlow();

private:
	// Variables

#pragma region Per-Network Variables

	const double ETA;
	const double ALPHA;

	// Number of input neurons = Dimension of input data (in this case will always be 2)
	int numInput = NUM_INPUT;
	// Number of hidden neurons
	int numHiddenNeurons;
	// Number of output neurons = Dimension of output data (in this case will always be 2)
	int numOutput = NUM_OUTPUT;

	// If backpropagation has finished
	bool trained = false;

	// Data used to train this network
	std::vector<LabeledTuple> trainSet;
	// Data used to validate this network
	std::vector<LabeledTuple> validateSet;

	ActivationFunction hiddenActivationFunction;
	ActivationFunction outputActivationFunction;

	// Current epoch
	int epoch = 0;

	std::vector<TuplePair> trainingError;
	std::vector<TuplePair> validationError;

#pragma endregion

#pragma region Per-Epoch Variables
	// Current index of training data
	int t = 0;
#pragma endregion 

#pragma region Per-Sample (t) Variables

	// Matrix of weights connecting input to hidden layer
	// Dimensions: (numInput+1, numHiddenNeuron)
	MatrixSlow hiddenWeights;
	// Matrix of weights connecting hidden layer to outputs
	// Dimensions: (numHiddenNeuron+1, numOutput)
	MatrixSlow outputWeights;

	MatrixSlow X;

	// Row vector containing pre-activation function values of the hidden neurons
	// Dimensions: (1, numHiddenNeuron)
	MatrixSlow VH;
	// Row vector containing pre-activation function values of the output neurons
	// Dimensions: (1, numOutput)
	MatrixSlow V;

	// Row vector containing the values of the hidden neurons
	// Dimensions: (1, numHiddenNeuron)
	MatrixSlow H;
	// Row vector containing values of the output neurons
	// Dimensions: (1, numOutput)
	MatrixSlow Y;

	// Row vector containing the error of outputs
	// Dimensions: (1, numOutput)
	MatrixSlow Error;

	// Row vector containing pre-activation function values of hidden neurons
	// Dimensions: (1, numOutput)
	MatrixSlow localGradient;
	// Row vector containing pre-activation function values of hidden neurons
	// Dimensions: (1, numHiddenNeuron)
	MatrixSlow localHiddenGradient;

	MatrixSlow deltaHiddenWeight;
	MatrixSlow deltaOutputWeight;

	// Functions
	MatrixSlow predict(const TuplePair& input);

	void runEpoch();
	void feedforward();
	void backpropagate();

	static double error(double input);

	// Weight initializations
	static double initRandom();
	static double initRoundedRandom();
	static double initDebug();

#pragma region OLD
	//std::vector<HiddenNeuron*> hiddenNeurons;
	//std::vector<OutputNeuron*> outputNeurons;
#pragma endregion

};

