#include "NeuralNetworkManager.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <locale>
#include <chrono>

#ifdef RUNROBOT
#include "Aria.h"
#endif


int main(int argc, char* argv[])
{
#ifdef RUNROBOT
		bool running = true;
		try
		{
			TrainedNetwork t("v5.w");
			double* inputScale = t.getInputScale();
			//**ROBOT SETUP & CONNECTION**
			
			// create instances
			Aria::init();
			ArRobot robot;
			// parse command line arguments
			ArArgumentParser argParser(&argc, argv);
			argParser.loadDefaultArguments();
			ArRobotConnector robotConnector(&argParser, &robot);
			if (robotConnector.connectRobot())
				std::cout << "Robot Connected!" << std::endl;
			robot.runAsync(false);
			robot.lock();
			robot.enableMotors();
			robot.unlock();

			if (MONORM) t.changeScale(0, 500);
			
			while(running)
			{
				double left = max(inputScale[0], min(min(robot.getSonarReading(0)->getRange(), USE12 ? robot.getSonarReading(1)->getRange() : robot.getSonarReading(0)->getRange()), inputScale[1]));
				double right = max(inputScale[2], min(USE12 ? robot.getSonarReading(1)->getRange() : robot.getSonarReading(1)->getRange(), inputScale[3]));
				TuplePair input(left, right);
				TuplePair output = t.predict(input);
				robot.setVel2(output[0], output[1]);
				ArUtil::sleep(20);
			}
		}
		catch (std::exception const& ex)
		{
			std::cout << "Weight file format is incorrect";
			throw std::runtime_error(std::string("Weight file format is incorrect"));
		}
#else
		std::vector<LabeledTuple> GatheredData = std::vector<LabeledTuple>();
		std::ifstream inData("data.csv");
		if (inData)
		{
			std::string line;
			while (std::getline(inData, line))
			{
				std::stringstream sep(line);
				std::string field;
				std::vector<double> values;
				while (std::getline(sep, field, ','))
				{
					field.erase(std::remove_if(field.begin(), field.end(), ::isspace), field.end());
					values.emplace_back(std::stod(field));
				}
				GatheredData.emplace_back(values[0], values[1], values[2], values[3]);
			}
		}
		inData.close();

		NeuralNetworkManager NNM(GatheredData);

#endif
	return 0;
}

NeuralNetworkManager::NeuralNetworkManager(std::vector<LabeledTuple>& labeledData) : allData(labeledData)
{
	normalizeData();
	splitDataSets();

	//speedTest();

	//try
	//{
	//	TrainedNetwork T1("v1.w");
	//	testNetwork(T1);

	//	TrainedNetwork T2("v2.w");
	//	testNetwork(T2);

	//	TrainedNetwork T3("v3.w");
	//	testNetwork(T3);

	//	std::cout << "\n\n\n";
	//	
	//	testNetwork(T1,true);
	//	testNetwork(T2, true);
	//	testNetwork(T3,true);
	//}
	//catch (std::exception const& ex)
	// {
	// 	throw std::runtime_error(std::string("Weight file format is incorrect"));
	// }
	

	
	 NeuralNetwork nn(trainSet, validateSet, 10, 0.8, 0.8, 0.8);
	 nn.startTraining(500);
	 TrainedNetwork tn(nn, inputScale, outputScale);
	 testNetwork(tn);
	 std::cout << tn << '\n';
	 tn.exportWeights("demo.w");

	
	
	 //std::cout << "Running Neural Networks\nSize of Training Data: " << trainSet.size() << "\n" << "Size of Validation Data: " << validateSet.size() << "\n" << "Size of Test Data: " << testSet.size() << "\n";
	
	 //exhaustiveTests(100);
	
	 //if (!trainedNetworks.empty())
	 //{
	 //	std::cout << "***** TESTING (ORDERED) *****\n";
	
	 int i;
	 //	#pragma omp parallel private(i)
	 //	#pragma omp for
	 //	for(i = 0; i < trainedNetworks.size(); ++i)
	 //	{
	 //		testNetwork(trainedNetworks.at(i));
	 //	}
	 //	
	 //	//for (TrainedNetwork & network : trainedNetworks) testNetwork(network);
	
	 //	std::sort(trainedNetworks.begin(), trainedNetworks.end());
	
	 //	for (TrainedNetwork & network : trainedNetworks) std::cout << network << '\n';
	
	 //	std::cout << "Best network is " << trainedNetworks.front() << '\n';
	
	 //	const TrainedNetwork best = trainedNetworks.front();
	
	 //	best.exportWeights("demo.w");
	 //}
}

void NeuralNetworkManager::testNetwork(TrainedNetwork& trained, bool test)
{
	try
	{
		double e(0);
		for (auto & t : testSet)
		{
			TuplePair out = trained.predict(t.x, false);			
			const double e1 = (t.y[0] - out[0]*(test?1.5:1));
			const double e2 = (t.y[1] - out[1]);
			//std::cout << "OUT:" << out.denormalize(outputScale) << '\n';
			//std::cout << "TARGET:" << t.y.denormalize(outputScale) << '\n'<<'\n';
			
			e += (e1*e1 + e2*e2)*0.5;
		}
		const auto rmse(sqrt(e / testSet.size()));
		if (!QUIET) std::cout << "Network [" << trained << "] Final Test Error: " << rmse << '\n';

		trained.setTestScore(rmse);
	}
	catch (std::runtime_error& e) { std::cout << e.what(); }
}

void NeuralNetworkManager::exhaustiveTimeEstimation(int epochs)
{
	NeuralNetwork bestCase(trainSet, validateSet, 2, 0.8, 0.4, 0.8);
	NeuralNetwork worstCase(trainSet, validateSet, 10, 0.8, 0.4, 0.8);

	const auto bcstart = std::chrono::high_resolution_clock::now();
	bestCase.startTraining(epochs);
	TrainedNetwork bestCaseTrained(bestCase, inputScale, outputScale);
	testNetwork(bestCaseTrained);
	const auto bcstop = std::chrono::high_resolution_clock::now();
	const auto bcduration = std::chrono::duration_cast<std::chrono::seconds>(bcstop - bcstart);

	const auto wcstart = std::chrono::high_resolution_clock::now();
	worstCase.startTraining(epochs);
	TrainedNetwork worstCaseTrained(worstCase, inputScale, outputScale);
	testNetwork(worstCaseTrained);
	const auto wcstop = std::chrono::high_resolution_clock::now();
	const auto wcduration = std::chrono::duration_cast<std::chrono::seconds>(wcstop - wcstart);

	const auto acduration = (bcduration + wcduration) / 2;

	int count = 0;
	for (auto l = 0; l < 4; l++)
	{
		double L = 0.2*l + 0.2;
		for (auto e = 0; e < 4; e++)
		{
			double E = 0.2*e + 0.2;

			for (auto a = 0; a < 4; a++)
			{
				double A = 0.2*a + 0.2;
				for (auto n = 2; n <= 10; n+=2)
				{
					count++;
				}
			}
		}
	}
	std::cout << "Total Count: " << count << '\n';
	std::cout << "Best Case: " << std::chrono::duration_cast<std::chrono::minutes>(bcduration*count).count() << '\n';
	std::cout << "Worst Case: " << std::chrono::duration_cast<std::chrono::minutes>(wcduration*count).count() << '\n';
	std::cout << "Avg Case: " << std::chrono::duration_cast<std::chrono::minutes>(acduration*count).count() << '\n';
}

void NeuralNetworkManager::exhaustiveTests(int epochs)
{
	//for (auto l = 0; l < 4; l++)
	//{
	double L = 0.8;//0.2*l + 0.2;
		for (auto e = 0; e <= 5; e++)
		{
			double E = 0.04*e + 0.6;
	
			for (auto a = 0; a <= 5; a++)
			{
				double A = 0.04*a + 0.6;
				for (auto n = 2; n <= 10; n += 2)
				{
					
					//NeuralNetwork nn(trainSet, validateSet, n, E, A, L);
					//nn.startTraining(epochs);
					//trainedNetworks.emplace_back(TrainedNetwork(nn, inputScale, outputScale));
					networks.emplace_back(NeuralNetwork(trainSet, validateSet, n, E, A, L));
				}
			}
		}
	//}
	
	std::cout << "***** TRAINING " << networks.size() << " NETWORKS *****\n";
	std::cout << "***** MAX " << epochs << " EPOCHS *****\n";
	
	//For the networks, train. Do in parallel.
	// IT ACTUALLY WORKS. THANK YOU OPENMP.
	#pragma omp parallel
	#pragma omp for
	for(auto i = 0; i<networks.size(); ++i) // need to use this kind of loop to do parallelism
	{
		networks.at(i).startTraining(epochs);
	}
	
	// Create trained networks
	for(NeuralNetwork nn : networks)  trainedNetworks.emplace_back(TrainedNetwork(nn, inputScale, outputScale));
}

// template<class T, class C>
// void parallel_for_each(std::vector<T>& ts, C callable)
// {
// 	size_t index = ;
// 	std::vector<std::future> results;
//
// 	for (auto const& value : ts)
// 	{
// 		results.emplace_back(async(callable, value));
// 	}
// 	for (auto& result : results)
// 	{
// 		result.wait();
// 	}
// }

// void NeuralNetworkManager::parameterTests()
// {
	////50 each
	//// 0.2, 0.2
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.2, 0.2));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.2, 0.2));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.2, 0.2));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.2, 0.4));
	//// 0.4, 0.2
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.4, 0.2));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.4, 0.2));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.4, 0.2));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.4, 0.2));
	//// 0.6, 0.2
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.6, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.6, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.6, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.6, 0.4));
	//
	////// 0.2, 0.4
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.2, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.2, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.2, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.2, 0.4));
	//// 0.4, 0.4
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.4, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.4, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.4, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.4, 0.4));
	//// 0.6, 0.4
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.6, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.6, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.6, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.6, 0.4));
	//
	////// 0.2, 0.6
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.2, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.2, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.2, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.2, 0.6));
	//// 0.4, 0.6
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.4, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.4, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.4, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.4, 0.6));
	//// 0.6, 0.6
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.6, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 3, 0.6, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.6, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.6, 0.6));

	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.8, 0.8));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.8, 0.8));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 6, 0.8, 0.8));
	
	// 500 each
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.6, 0.2));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 5, 0.8, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 6, 0.8, 0.4));

	// 500 each, then 50 each
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 7, 0.8, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 9, 0.8, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 10, 0.8, 0.4));

	// 50 each
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 1, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.6, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.4, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.2, 0.4));

	// 50 each
	/*networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 1, 0.5));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.4));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.6, 0.3));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.4, 0.2));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.2, 0.1));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 1));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.8));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.6));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.4));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.2));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.0));*/

	// 50 each
	/*networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.8));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.6));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.4));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 2, 0.8, 0.8));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 4, 0.8, 0.8));
	networks.emplace_back(NeuralNetwork(trainSet, validateSet, 6, 0.8, 0.8));*/

	// 5000 each
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.8));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.6));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 8, 0.8, 0.4));
	//networks.emplace_back(NeuralNetwork(trainSet, validateSet, 6, 0.8, 0.8));
// }

// void NeuralNetworkManager::speedTest()
// {
// 	srand(1);
// 	NeuralNetwork nn(trainSet, validateSet, 2, 0.2, 0.1,0.8);
//
// 	auto start = std::chrono::high_resolution_clock::now();
// 	nn.startTraining(10);
// 	auto stop = std::chrono::high_resolution_clock::now();
// 	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
// 	std::cout << '\n' << "Fast Matrix took: " << duration.count() << '\n' << '\n';
//
// 	srand(1);
// 	NeuralNetworkSlow nns(trainSet, validateSet, 2, 0.2, 0.1);
//
// 	start = std::chrono::high_resolution_clock::now();
// 	nns.startTraining(10);
// 	stop = std::chrono::high_resolution_clock::now();
// 	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
// 	std::cout << '\n' << "Slow Matrix took: " << duration.count() << '\n' << '\n';
//
// 	// Result
// 	// Slow matrix:	30624666 ms
// 	// New matrix:	 8638061 ms
// 	// Using new matrix is 3.54 times faster.
// }

void NeuralNetworkManager::normalizeData()
{
	
	for (auto const& value : allData)
	{
		inputScale[0] = std::min(inputScale[0], value.x[0]);
		inputScale[1] = std::max(inputScale[1], value.x[0]);
		inputScale[2] = std::min(inputScale[2], value.x[1]);
		inputScale[3] = std::max(inputScale[3], value.x[1]);

		outputScale[0] = std::min(outputScale[0], value.y[0]);
		outputScale[1] = std::max(outputScale[1], value.y[0]);
		outputScale[2] = std::min(outputScale[2], value.y[1]);
		outputScale[3] = std::max(outputScale[3], value.y[1]);
	}

	std::cout << "INPUT SCALE: " << inputScale[0] << ", " << inputScale[1] << ", " << inputScale[2] << ", " << inputScale[3] << "\n";
	std::cout << "OUTPUT SCALE: " << outputScale[0] << ", " << outputScale[1] << ", " << outputScale[2] << ", " << outputScale[3] << "\n";
	
	for (auto &value : allData)
	{
		value.x.normalize(inputScale);
		value.y.normalize(outputScale);
	}

}

void NeuralNetworkManager::splitDataSets()
{
	std::shuffle(allData.begin(), allData.end(), std::mt19937(std::random_device()()));
	// Training Set 70%
	trainSet = std::vector<LabeledTuple>(allData.begin(), allData.begin() + allData.size()*0.7);
	// Validation Set 15%
	validateSet = std::vector<LabeledTuple>(allData.begin() + allData.size()*0.7,allData.end() - allData.size()*0.15);
	// Test Set 15%
	testSet = std::vector<LabeledTuple>( allData.end() - allData.size()*0.15, allData.end());
}

NeuralNetworkManager::~NeuralNetworkManager() = default;
