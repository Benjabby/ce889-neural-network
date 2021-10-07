#pragma once
#include <vector>
#include "NeuralNetwork.h"

constexpr bool USE12 = true;
constexpr bool MONORM = false;


class NeuralNetworkManager
{
public:
	NeuralNetworkManager(std::vector<LabeledTuple> &labeledData);

	~NeuralNetworkManager();
private:
	// Variables

	std::vector<NeuralNetwork> networks;
	std::vector<TrainedNetwork> trainedNetworks;
	
	std::vector<LabeledTuple> allData;
	
	// Data used to train networks
	std::vector<LabeledTuple> trainSet;
	// Data used to validate networks
	std::vector<LabeledTuple> validateSet;
	// Data used to test the final networks. Withheld from network until they're trained
	std::vector<LabeledTuple> testSet;

	/*	Min/Max scale parameters for input labeled data. Retained to rescale new input data
	Size 4
	0: Min of X1 set
	1: Max of X1 set
	2: Min of X2 set
	3: Max of X2 set
	*/
	double inputScale[4]{ DBL_MAX ,-DBL_MAX ,DBL_MAX ,-DBL_MAX };
	/*	Min/Max scale parameters for output labeled data. Retained to rescale new output data
	Size 4
	0: Min of Y1 set
	1: Max of Y1 set
	2: Min of Y2 set
	3: Max of Y2 set
	*/
	double outputScale[4]{ DBL_MAX ,-DBL_MAX ,DBL_MAX ,-DBL_MAX };
	
	// Functions
	
	// Initializes inputScale and outputScale using 'allData' then normalizes
	void normalizeData();
	// Splits allData randomly into trainSet, validateSet, and testSet with 70%, 15%, 15% ratios respectively
	void splitDataSets();
	void testNetwork(TrainedNetwork& trained, bool test=false);
	void exhaustiveTests(int epochs);
	void exhaustiveTimeEstimation(int epochs);
	// void parameterTests();
	// void speedTest();
};
