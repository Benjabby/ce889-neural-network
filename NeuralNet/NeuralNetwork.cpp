#include "NeuralNetwork.h"
#include <limits>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <chrono>

// I know I've got too many variable members and need to refactor this
NeuralNetwork::NeuralNetwork(std::vector<LabeledTuple>& trainSet, std::vector<LabeledTuple>& validateSet, int numHiddenNeurons, double eta, double alpha, double lambda, const ActivationFunction* hiddenActivation, const ActivationFunction* outputActivation) :
	shuffler(std::chrono::system_clock::now().time_since_epoch().count()), ETA(eta), ALPHA(alpha), LAMBDA(lambda),
	numHiddenNeurons(numHiddenNeurons), trainSet(trainSet), validateSet(validateSet),
	hiddenActivationFunction(hiddenActivation), outputActivationFunction(outputActivation),
	bestHiddenWeights(numInput, numHiddenNeurons), bestHiddenBias(1, numHiddenNeurons),
	bestOutputWeights(numHiddenNeurons, numOutput), bestOutputBias(1, numOutput),
	X(1, numInput), T(1, numOutput),
	hiddenWeights(numInput, numHiddenNeurons, initRandom), hiddenBias(1, numHiddenNeurons, initRandom),
	outputWeights(numHiddenNeurons, numOutput, initRandom), outputBias(1, numOutput, initRandom), VH(1, numHiddenNeurons), V(1, numOutput), H(1, numHiddenNeurons), Y(1, numOutput),
	Error(1, numOutput), localGradient(1, numOutput), localHiddenGradient(1, numHiddenNeurons),
	alphaHiddenWeight(numInput, numHiddenNeurons), alphaHiddenBias(1, numHiddenNeurons),
	alphaOutputWeight(numHiddenNeurons, numOutput), alphaOutputBias(1, numOutput)
{
	if (!QUIET) std::cout << "Neural Network " << *this << " Created\nETA:" << ETA << "\nALPHA" << ALPHA << "\n";
}

/* OLD
 *NeuralNetwork::NeuralNetwork(std::vector<LabeledTuple>& trainSet, std::vector<LabeledTuple>& validateSet,int numHiddenNeurons)
	: numHiddenNeurons(numHiddenNeurons), trainSet(trainSet), validateSet(validateSet),
	  hiddenActivationFunction(LOGISTIC), outputActivationFunction(LOGISTIC)
	, hiddenWeights(numInput + 1, numHiddenNeurons), outputWeights(numHiddenNeurons + 1, numOutput), X(1,numHiddenNeurons+1)
	, VH(1, numHiddenNeurons), V(1, numOutput), H(1, numHiddenNeurons), Y(1, numOutput), Error(1,numOutput)
	, localGradient(1,numOutput),  localHiddenGradient(1,numHiddenNeurons), alphaHiddenWeight(numInput + 1, numHiddenNeurons)
	, alphaOutputWeight(numHiddenNeurons + 1, numOutput)
{
	std::cout << "Neural Network Created\n" << "Size of Training Data: " << trainSet.size() << "\n" << "Size of Validation Data: " << validateSet.size() << "\n";
	std::cout << "Randomzing Weights\n";
	// Randomize weights
	hiddenWeights = hiddenWeights.applyFunction(initRandom);
	outputWeights = outputWeights.applyFunction(initRandom);
}

NeuralNetwork::NeuralNetwork(int numHiddenNeurons)
	: numHiddenNeurons(numHiddenNeurons), hiddenActivationFunction(LOGISTIC), outputActivationFunction(LOGISTIC)
	, hiddenWeights(numInput + 1, numHiddenNeurons), outputWeights(numHiddenNeurons + 1, numOutput), X(1, numHiddenNeurons + 1)
	, VH(1, numHiddenNeurons), V(1, numOutput), H(1, numHiddenNeurons), Y(1, numOutput), Error(1, numOutput)
	, localGradient(1, numOutput), localHiddenGradient(1, numHiddenNeurons), alphaHiddenWeight(numInput + 1, numHiddenNeurons)
	, alphaOutputWeight(numHiddenNeurons + 1, numOutput)
{
	// Randomize weights
	hiddenWeights = hiddenWeights.applyFunction(initRandom);
	outputWeights = outputWeights.applyFunction(initRandom);
}*/

const MatrixContig& NeuralNetwork::validate(const MatrixContig& input)
{
	if(!trained)
	{
		std::cerr << "Network not trained\n";
		return {0,0};
	}
	X.set(input);
	feedforward();
	return Y;
}


void NeuralNetwork::startTraining(int totalEpochs)
{
	if (trained)
	{
		std::cerr << "Network already trained\n";
		return;
	}
	if (!QUIET)
	{
		std::cout << "\t\t***** TRAINING STARTED *****\n";
		std::cout << "Total epochs to run: " << totalEpochs << "\n";
	}
	while (epoch < totalEpochs)
	{
		runEpoch();

		if (!QUIET)
		{
			std::cout << "Epoch " << (epoch + 1) << " Complete\n";
			std::cout << "Training Error: " << trainingError[epoch].rmse(trainSet.size()) << '\n';
			std::cout << "Validation Error: " << validationError[epoch].rmse(validateSet.size()) << '\n';
		}

		if (stopEarly())
		{
			if (!QUIET) std::cout << "Stopping early as Validation Error has not decreased for " << PATIENCE << " epochs\n";
			else std::cout << "Stopping at epoch " << (epoch + 1) << " with best result at epoch " << bestErrorEpoch << " as Validation Error has not decreased for " << PATIENCE << " epochs\n";
			break;
		}

		if (stopCheck > 0 && !QUIET) std::cout << "Validation Error increasing. Will stop early in " << (PATIENCE - stopCheck + 1) << " epochs unless error falls below current best (" << bestError << ")\n";

		epoch++;
		
	}
	trained = true;

	if (!QUIET)
	{
		std::cout << "\t\t***** TRAINING COMPLETE *****\n";
		std::cout << "Best Validation Error: " << bestError << " at Epoch " << bestErrorEpoch << '\n';
		std::cout << "\t\t*****************************\n";
	}
	else
	{
		//To appease the parallel gods, must use a single << call to cout.
		std::stringstream ss;
		ss << *this << " Trained. Best Validation Error: " << bestError << '\n';
		std::cout << ss.str();
	}
	
	hiddenBias.set(bestHiddenBias);
	hiddenWeights.set(bestHiddenWeights);
	outputBias.set(bestOutputBias);
	outputWeights.set(bestOutputWeights);
}

void NeuralNetwork::runEpoch()
{
	// TRAINING
	t = 0;
	trainingError.emplace_back(TuplePair(0,0));
	while (t < trainSet.size())
	{
		try
		{
			X.set(trainSet[t].x);
			T.set(trainSet[t].y);
			
			feedforward();
			backpropagate();
		}
		catch (std::runtime_error& e) { std::cout << e.what(); }
		
		t++;
		Error.applyFunction(error);
		trainingError[epoch] += Error.getTuple();
	}
	
	std::shuffle(trainSet.begin(), trainSet.end(), shuffler);
	
	// VALIDATION
	validationError.emplace_back(TuplePair(0, 0));
	for (const LabeledTuple& sample : validateSet)
	{
		try
		{
			X.set(sample.x);
			T.set(sample.y);
			feedforward();

			MatrixContig::subtract(&T, &Y, &Error);
			Error.applyFunction(error);

			validationError[epoch] += Error.getTuple();
		}
		catch (std::runtime_error& e) { std::cout << e.what(); }
	}
	
	std::shuffle(validateSet.begin(), validateSet.end(), shuffler);
	
	if (validationError[epoch].rmse(validateSet.size()) < bestError)
	{
		bestError = validationError[epoch].rmse(validateSet.size());
		bestErrorEpoch = epoch+1;

		bestHiddenBias.set(hiddenBias);
		bestHiddenWeights.set(hiddenWeights);
		bestOutputBias.set(outputBias);
		bestOutputWeights.set(outputWeights);
	}
}

bool NeuralNetwork::stopEarly()
{
	if (validationError[epoch].rmse(validateSet.size()) > bestError) return (stopCheck++ >= PATIENCE);
	
	stopCheck = 0;

	return false;
}

void NeuralNetwork::feedforward()
{
	MatrixContig::multiply(&X, &hiddenWeights, &VH); // VH = X*hiddenWeights
	VH += hiddenBias;
	VH.applyFunction(hiddenActivationFunction->activation,LAMBDA, &H);
	
	MatrixContig::multiply(&H, &outputWeights, &V); // V = H*outputWeights
	V += outputBias;
	V.applyFunction(outputActivationFunction->activation, LAMBDA, &Y);
}

void NeuralNetwork::backpropagate()
{
	
	MatrixContig::subtract(&T, &Y, &Error); // Error = T-Y

	// localGradient = (logistic derivative at V) .* Error
	V.applyFunction(outputActivationFunction->derivative, LAMBDA, &localGradient);
	localGradient.hadamard(&Error);

	MatrixContig G2(1, numHiddenNeurons);
	VH.applyFunction(hiddenActivationFunction->derivative, LAMBDA,&G2);

	MatrixContig::multiply(&localGradient, &outputWeights, &localHiddenGradient, false, true);
	localHiddenGradient.hadamard(&G2);

	MatrixContig dOW(numHiddenNeurons, numOutput);
	MatrixContig::multiply(&H, &localGradient, &dOW, true, false);
	dOW.scale(ETA);

	MatrixContig dHW(numInput, numHiddenNeurons);
	MatrixContig::multiply(&X, &localHiddenGradient, &dHW, true, false);
	dHW.scale(ETA);

	MatrixContig dOB(1, numOutput);
	localGradient.scale(ETA, &dOB);

	MatrixContig dHB(1, numHiddenNeurons);
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
}


std::string NeuralNetwork::identifierString() const
{
	std::stringstream ss;
	ss << *this;
	return ss.str();
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& nn)
{
	if(nn.trained) return os << "[ NEURONS=" << nn.numHiddenNeurons << " | ETA=" << nn.ETA << " | ALPHA=" << nn.ALPHA << " | LAMBDA=" << nn.LAMBDA << " ]";

	return os << "[ NEURONS=" << nn.numHiddenNeurons << " | ETA=" << nn.ETA << " | ALPHA=" << nn.ALPHA << " | LAMBDA=" << nn.LAMBDA << " | NOT TRAINED ]";
}

double NeuralNetwork::initRandom()
{
	return (rand() / double(RAND_MAX))*2 - 1;
}

double NeuralNetwork::initRandomPositive()
{
	return (rand() / double(RAND_MAX));
}

double NeuralNetwork::initRoundedRandom()
{
	return round(initRandom() * 100) / 100;
}

double NeuralNetwork::initDebug()
{
	return 0.5;
}

double NeuralNetwork::error(double input)
{
	return (input*input)*0.5;
}

NeuralNetwork::~NeuralNetwork() = default;

#pragma region Trained Network

TrainedNetwork::TrainedNetwork(NeuralNetwork& nn, double* inputScale, double* outputScale) : hiddenActivationFunction(nn.hiddenActivationFunction), outputActivationFunction(nn.outputActivationFunction), numInput(nn.numInput), numHiddenNeurons(nn.numHiddenNeurons), numOutput(nn.numOutput), HW(new MatrixContig(nn.bestHiddenWeights)), HB(new MatrixContig(nn.bestHiddenBias)), OW(new MatrixContig(nn.bestOutputWeights)), OB(new MatrixContig(nn.bestOutputBias)), LAMBDA(nn.LAMBDA), inputScale(inputScale), outputScale(outputScale), identifier(nn.identifierString()), testScore(-1)
{
	if (!nn.trained) throw std::invalid_argument("NeuralNetwork must be trained to create a trained network");
}

// This is terrible. I'm genuinely ashamed of this code.
// I should do proper tokenizing, make a better output format, or at least make a function template for the loops.
// but I just need to get it done for now.
TrainedNetwork::TrainedNetwork(const char* weightFilename) : hiddenActivationFunction(LOGISTIC), outputActivationFunction(LOGISTIC)
{
	std::ifstream weightFile(weightFilename);
	std::vector<std::string> lines;
	if (weightFile)
	{
		std::string line;
		while (std::getline(weightFile, line)) lines.emplace_back(line);
	}
	weightFile.close();

	if (lines.size() != 7) throw std::runtime_error("Weight file format is incorrect");


	testScore = -1;
	std::istringstream iss(lines[0]);
	char c;
	iss >> numInput >> c >> numHiddenNeurons >> c >> numOutput >> c >> LAMBDA;

	inputScale = new double[4];
	iss = std::istringstream(lines[1]);
	iss >> inputScale[0] >> c >> inputScale[1] >> c >> inputScale[2] >> c >> inputScale[3]>>c;

	outputScale = new double[4];
	iss = std::istringstream(lines[2]);
	iss >> outputScale[0] >> c >> outputScale[1] >> c >> outputScale[2] >> c >> outputScale[3]>>c;
	
	std::vector<std::vector<double>> tempHW(numInput, std::vector<double>(numHiddenNeurons));
	iss = std::istringstream(lines[3]);
	iss.ignore(4);
	for (auto i = 0; i < numInput; i++)
	{
		for (auto j = 0; j < numHiddenNeurons; j++)
		{
			iss >> tempHW[i][j] >> c;
		}
		iss.ignore(1);
	}

	HW = new MatrixContig(tempHW);

	std::vector<std::vector<double>> tempHB(1, std::vector<double>(numHiddenNeurons));
	iss = std::istringstream(lines[4]);
	iss.ignore(4);
	for (auto j = 0; j < numHiddenNeurons; j++)
	{
		iss >> tempHB[0][j] >> c;
	}

	HB = new MatrixContig(tempHB);

	std::vector<std::vector<double>> tempOW(numHiddenNeurons, std::vector<double>(numOutput));
	iss = std::istringstream(lines[5]);
	iss.ignore(4);
	for (auto i = 0; i < numHiddenNeurons; i++)
	{
		for (auto j = 0; j < numOutput; j++)
		{
			iss >> tempOW[i][j] >> c;
		}
		iss.ignore(1);
	}

	OW = new MatrixContig(tempOW);

	std::vector<std::vector<double>> tempOB(1, std::vector<double>(numOutput));
	iss = std::istringstream(lines[6]);
	iss.ignore(4);
	for (auto j = 0; j < numOutput; j++)
	{
		iss >> tempOB[0][j] >> c;
	}

	OB = new MatrixContig(tempOB);

}
double TrainedNetwork::getTestScore() const
{
	return testScore;
}
TuplePair TrainedNetwork::predict(TuplePair& input, bool denormalizeOutput) const
{
	if (!input.isNormalized()) input.normalize(inputScale);
	MatrixContig X(input);
	MatrixContig H(1, numHiddenNeurons);
	MatrixContig out(1, numOutput);
	
	MatrixContig::multiply(&X, HW, &H);
	H += *HB;
	H.applyFunction(hiddenActivationFunction->activation,LAMBDA);
	MatrixContig::multiply(&H, OW, &out);
	out += *OB;
	out.applyFunction(outputActivationFunction->activation,LAMBDA);

	const TuplePair outTuple = out.getTuple();

	return denormalizeOutput ? outTuple.denormalize(outputScale) : outTuple;	
}

void TrainedNetwork::exportWeights(const char * filename) const
{
	std::ofstream weightFile(filename);
	std::cout << *OB << *OW << *HB << *HW << ' ' << numOutput << ' ' << numHiddenNeurons << ' ' << numInput;
	if (weightFile.good())
	{
		weightFile << numInput << ',' << numHiddenNeurons << ',' << numOutput << ',' << LAMBDA << '\n';
		for (auto i = 0; i < 4; i++) weightFile << inputScale[i] << ',';
		weightFile << '\n';
		for (auto o = 0; o < 4; o++) weightFile << outputScale[o] << ',';
		weightFile << '\n';
		weightFile << "HW";
		weightFile << *HW << '\n';
		weightFile << "HB";
		weightFile << *HB << '\n';
		weightFile << "OW";
		weightFile << *OW << '\n';
		weightFile << "OB";
		weightFile << *OB << '\n';
	}
}

void TrainedNetwork::changeScale(int i, int i1)
{
	outputScale[0] = i;
	outputScale[1] = i1;

	outputScale[2] = i;
	outputScale[3] = i1;
}

void TrainedNetwork::setTestScore(const double rmse)
{
	testScore = rmse;
}

double* TrainedNetwork::getInputScale() const
{
	return inputScale;
}

bool operator<(const TrainedNetwork& s1, const TrainedNetwork& s2)
{
	return s1.testScore < s2.testScore;
}

std::ostream& operator<<(std::ostream& os, const TrainedNetwork& tn)
{
	if (tn.testScore < 0) return os << tn.identifier;
	return  os << tn.identifier << " TEST SCORE: " << tn.testScore;
}

TrainedNetwork::TrainedNetwork(const TrainedNetwork& tn) : hiddenActivationFunction(tn.hiddenActivationFunction), outputActivationFunction(tn.outputActivationFunction), numInput(tn.numInput), numHiddenNeurons(tn.numHiddenNeurons), numOutput(tn.numOutput), HW(new MatrixContig(*tn.HW)), HB(new MatrixContig(*tn.HB)), OW(new MatrixContig(*tn.OW)), OB(new MatrixContig(*tn.OB)), LAMBDA(tn.LAMBDA), inputScale(tn.inputScale), outputScale(tn.outputScale), identifier(tn.identifier), testScore(tn.testScore)
{
	
}


TrainedNetwork::~TrainedNetwork()
{
	
}

#pragma endregion 