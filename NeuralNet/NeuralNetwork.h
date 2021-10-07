#pragma once

#include <vector>
#include <cassert>
#include <random>

// TEMP
#include <sstream>
#include "Utilities.h"

constexpr bool QUIET = false;

class TrainedNetwork;

// I know I've got too many variable members and need to refactor this.
class NeuralNetwork
{
public:
	friend class TrainedNetwork;
	// Constructors

	//NeuralNetwork(std::vector<LabeledTuple> &trainSet, std::vector<LabeledTuple> &validateSet, int numHiddenNeurons, double eta, double alpha);
	//NeuralNetwork(std::vector<LabeledTuple> &trainSet, std::vector<LabeledTuple> &validateSet, int numHiddenNeurons, double eta, double alpha, double lambda);
	NeuralNetwork(std::vector<LabeledTuple> &trainSet, std::vector<LabeledTuple> &validateSet, int numHiddenNeurons, double eta, double alpha, double lambda=0.8, const ActivationFunction* hiddenActivation=LOGISTIC, const ActivationFunction* outputActivation=LOGISTIC);
	// Test Constructor
	//NeuralNetwork(int numHiddenNeurons);


	// Functions


	// Other stopping criteria to come
	void startTraining(int totalEpochs);

	const MatrixContig& validate(const MatrixContig& input);
	std::string identifierString() const;

	static double error(double input);

	friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& nn);
	
	// Destructors
	
	~NeuralNetwork();

private:
	// Variables

	#pragma region Per-Network Variables

	std::default_random_engine shuffler;
	
	const double ETA;
	const double ALPHA;
	const double LAMBDA;

	// When the error of validation goes up, how many epochs to continue running before ending.
	// If the error falls below the current minimum validation error, we continue anyway
	const int PATIENCE = 20;
	int stopCheck = 0;
	double bestError = DBL_MAX;
	int bestErrorEpoch = 0;

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

	const ActivationFunction* hiddenActivationFunction;
	const ActivationFunction* outputActivationFunction;

	MatrixContig bestHiddenWeights;
	MatrixContig bestHiddenBias;
	MatrixContig bestOutputWeights;
	MatrixContig bestOutputBias;

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


	// Row vector containing current input
	// Dimensions: (1, numInput)
	MatrixContig X;

	// Row vector containing current target
	// Dimensions: (1, numOutput)
	MatrixContig T;
	
	// Matrix of weights connecting input to hidden layer
	// Dimensions: (numInput, numHiddenNeuron)
	MatrixContig hiddenWeights;
	// Row vector of hidden layer biases
	// Dimensions: (1, numHiddenNeuron)
	MatrixContig hiddenBias;
	// Matrix of weights connecting hidden layer to outputs
	// Dimensions: (numHiddenNeuron, numOutput) 
	MatrixContig outputWeights;
	// Row vector of output layer biases
	// Dimensions: (1, numOutput)
	MatrixContig outputBias;

	// Row vector containing pre-activation function values of the hidden neurons
	// Dimensions: (1, numHiddenNeuron)
	MatrixContig VH;
	// Row vector containing pre-activation function values of the output neurons
	// Dimensions: (1, numOutput)
	MatrixContig V;

	// Row vector containing the values of the hidden neurons
	// Dimensions: (1, numHiddenNeuron)
	MatrixContig H;
	// Row vector containing values of the output neurons
	// Dimensions: (1, numOutput)
	MatrixContig Y;

	// Row vector containing the error of outputs
	// Dimensions: (1, numOutput)
	MatrixContig Error;

	// Row vector containing pre-activation function values of hidden neurons
	// Dimensions: (1, numOutput)
	MatrixContig localGradient;
	// Row vector containing pre-activation function values of hidden neurons
	// Dimensions: (1, numHiddenNeuron)
	MatrixContig localHiddenGradient;

	// Dimensions: (numInput, numHiddenNeuron)
	MatrixContig alphaHiddenWeight;
	// Dimensions: (1, numHiddenNeuron)
	MatrixContig alphaHiddenBias;

	// Dimensions: (numHiddenNeuron, numOutput) 
	MatrixContig alphaOutputWeight;
	// Dimensions: (1, numOutput) 
	MatrixContig alphaOutputBias;
	

	// Functions
	
	void runEpoch();
	void feedforward();
	void backpropagate();
	bool stopEarly();

	// Weight initializations
	static double initRandom();
	static double initRandomPositive();
	static double initRoundedRandom();
	static double initDebug();

#pragma region OLD
	//std::vector<HiddenNeuron*> hiddenNeurons;
	//std::vector<OutputNeuron*> outputNeurons;
#pragma endregion

};

class TrainedNetwork
{
public:
	TrainedNetwork(NeuralNetwork& nn, double* inputScale, double* outputScale);
	TrainedNetwork(const char* weightFilename);
	void setTestScore(double rmse);
	double* getInputScale() const;
	void changeScale(int i, int i1);
	TrainedNetwork(const TrainedNetwork& tn);
	TrainedNetwork& operator=(const TrainedNetwork& tn) = default;
	~TrainedNetwork();
	TuplePair predict(TuplePair& input, bool denormalizeOutput = true) const;
	void exportWeights(const char* filename) const;
	double getTestScore() const;
	friend std::ostream& operator<<(std::ostream& os, const TrainedNetwork& tn);
	friend bool operator<(const TrainedNetwork& s1, const TrainedNetwork& s2);
private:
	const ActivationFunction* hiddenActivationFunction;
	const ActivationFunction* outputActivationFunction;
	int numInput{};
	int numHiddenNeurons{};
	int numOutput{};
	MatrixContig* HW;
	MatrixContig* HB;
	MatrixContig* OW;
	MatrixContig* OB;
	double LAMBDA{};
	double* inputScale;
	double* outputScale;
	std::string identifier;
	double testScore;
};