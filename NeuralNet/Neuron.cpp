#ifndef NEURON
#define NEURON
#include "Neuron.h"
#endif

Neuron::Neuron(int numPrevious) : first(true), numPrevious(numPrevious), weights(std::vector<double>())
{
	initWeights();
}

Neuron::Neuron(const std::vector<Neuron*>& previous) : first(false), previousLayer(previous), numPrevious(previous.size()), weights(std::vector<double>())
{
	initWeights();
}

double Neuron::feedforward(const double* input)
{
	double sum = 0;
	// Bias is always index 0
	sum += weights.at(0);
	for(auto i=0; i<numPrevious; i++)
	{
		const double in = first ? input[i] : previousLayer.at(i)->feedforward(input);
		sum += weights.at(i + 1)*in;
	}
	return activationFunctions[DEF_ACTIVATION].activation(sum);
	//return activationFunctions[TEST_ACTIVATION].activation(sum);
}

std::string Neuron::debugWeights()
{
	std::string out("Bias=");
	out.append(std::to_string(weights.at(0)));
	for (auto i = 0; i < numPrevious; i++)
	{
		out.append(" W"+std::to_string(i)+"="+std::to_string(weights.at(i+1)));
	}
	return out;
}

void Neuron::initWeights()
{
	for(auto i=0; i<numPrevious+1; ++i)
	{
		weights.emplace_back(double(initRoundedRandom()));
		//weights.emplace_back(double(initDebug()));
	}
}

double Neuron::initRandom()
{
	return (rand() / double(RAND_MAX));
}

double Neuron::initRoundedRandom()
{
	return round(initRandom() * 100) / 100;
}

double Neuron::initDebug()
{
	return 0.5;
}

Neuron::~Neuron() = default;



