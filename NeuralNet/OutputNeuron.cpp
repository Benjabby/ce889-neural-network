#include "OutputNeuron.h"


OutputNeuron::OutputNeuron(const std::vector<HiddenNeuron*>& previous) : Neuron(std::vector<Neuron*>(previous.begin(),previous.end()))
{
}

double OutputNeuron::generateOutput(const double * input)
{
	return feedforward(input);
}

