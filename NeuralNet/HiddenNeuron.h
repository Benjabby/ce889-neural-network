#pragma once

#include "Neuron.h"

class HiddenNeuron : public Neuron
{
public:

	// First hidden layer constructor with specified input count. We're only ever using two inputs for this network though.
	HiddenNeuron(int numPrevious);

	// Second or greater hidden layer constructor. Not Needed for this network. We're only ever using one hidden layer. Included for possible future expansion
	//HiddenNeuron(const std::vector<Neuron*>& previous);
};

