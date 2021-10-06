#pragma once
#include "Neuron.h"
#include "HiddenNeuron.h"

class OutputNeuron : public Neuron
{
public:
	OutputNeuron(const std::vector<HiddenNeuron*>& previous);

	double generateOutput(const double* input);
};

