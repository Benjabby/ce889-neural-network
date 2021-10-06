#include "NeuralNetworkSlow.h"
#include <limits>
#include <algorithm>
#include <iostream>
#include <string>
#include <iomanip>
#include <ctime>

NeuralNetworkSlow::NeuralNetworkSlow(std::vector<LabeledTuple>& trainSet, std::vector<LabeledTuple>& validateSet,int numHiddenNeurons, double eta, double alpha)
	: ETA(eta), ALPHA(alpha), numHiddenNeurons(numHiddenNeurons), trainSet(trainSet), validateSet(validateSet),
hiddenActivationFunction(LOGISTIC), outputActivationFunction(LOGISTIC)
, hiddenWeights(numInput + 1, numHiddenNeurons), outputWeights(numHiddenNeurons + 1, numOutput), X(1,numHiddenNeurons+1)
, VH(1, numHiddenNeurons), V(1, numOutput), H(1, numHiddenNeurons), Y(1, numOutput), Error(1,numOutput)
, localGradient(1,numOutput),  localHiddenGradient(1,numHiddenNeurons), deltaHiddenWeight(numInput + 1, numHiddenNeurons)
, deltaOutputWeight(numHiddenNeurons + 1, numOutput)
{
	std::cout << "Neural Network Created\n" << "Size of Training Data: " << trainSet.size() << "\n" << "Size of Validation Data: " << validateSet.size() << "\n";
	std::cout << "Randomzing Weights\n";
	// Randomize weights
	hiddenWeights = hiddenWeights.applyFunction(initRandom);
	outputWeights = outputWeights.applyFunction(initRandom);
}


MatrixSlow NeuralNetworkSlow::predict(const TuplePair& input)
{
	X = MatrixSlow(input);
	Y = ((X*hiddenWeights).applyFunction(LOGISTIC.activation).insertBiasColumn()*outputWeights).applyFunction(LOGISTIC.activation);
	return Y;
}


void NeuralNetworkSlow::startTraining(int totalEpochs)
{
	if (trained)
	{
		std::cerr << "Network already trained\n";
		return;
	}
	std::cout << "Training Started, total epochs to run: " << totalEpochs << "\n";
	while (epoch < totalEpochs)
	{
		runEpoch();
		std::cout << "Epoch " << (epoch + 1) << " Complete\n";
		std::cout << "Training Error: " << trainingError[epoch].rmse(trainSet.size()) << '\n';
		std::cout << "Validation Error: " << validationError[epoch].rmse(validateSet.size()) << '\n';
		epoch++;
	}
	trained = true;
}

void NeuralNetworkSlow::runEpoch()
{
	t = 0;
	trainingError.emplace_back(TuplePair(0, 0));
	while (t < trainSet.size())
	{
		try
		{
			feedforward();
			backpropagate();
		}
		catch (std::runtime_error& e) { std::cout << e.what(); }

		t++;
		trainingError[epoch] += Error.applyFunction(error).toTuplePair();
	}

	validationError.emplace_back(TuplePair(0, 0));
	for (const LabeledTuple& sample : validateSet)
	{
		MatrixSlow out = predict(sample.x);

		Error = MatrixSlow(sample.y,false) - out;
		
		validationError[epoch] += Error.applyFunction(error).toTuplePair();
	}

}

void NeuralNetworkSlow::feedforward()
{
	 X = MatrixSlow(trainSet[t].x);
	 VH = (X*hiddenWeights);
	 H = VH.applyFunction(LOGISTIC.activation);
	 V = (H.insertBiasColumn()*outputWeights);
	 Y = V.applyFunction(LOGISTIC.activation);
}

void NeuralNetworkSlow::backpropagate()
{
	 const MatrixSlow y(trainSet[t].y,false);
	
	 Error = (y - Y);
	
	 localGradient = elementWise(Error, V.applyFunction(LOGISTIC.derivative));
	 localHiddenGradient = elementWise(localGradient*(transpose(outputWeights).removeBiasColumn()), VH.applyFunction(LOGISTIC.derivative));
	
	 // H Doesn't have bias column, need to add before transposing
	 const MatrixSlow dOW = ((transpose(H.insertBiasColumn())*localGradient)*ETA) + (deltaOutputWeight*ALPHA);
	 // X already has bias column so no need to add before transposing.
	 const MatrixSlow dHW = ((transpose(X)*localHiddenGradient)*ETA) + (deltaHiddenWeight*ALPHA);
	
	 outputWeights = outputWeights + dOW;
	 hiddenWeights = hiddenWeights + dHW;
	
	 deltaOutputWeight = dOW;
	 deltaHiddenWeight = dHW;
}

void NeuralNetworkSlow::printWeights()
{
	//std::cout << hiddenWeights;
	//std::cout << outputWeights;
}

double NeuralNetworkSlow::initRandom()
{
	return (rand() / double(RAND_MAX)) * 2 - 1;
}

double NeuralNetworkSlow::initRoundedRandom()
{
	return round(initRandom() * 100) / 100;
}

double NeuralNetworkSlow::initDebug()
{
	return 0.5;
}

double NeuralNetworkSlow::error(double input)
{
	return (input*input)*0.5;
}

NeuralNetworkSlow::~NeuralNetworkSlow() = default;
