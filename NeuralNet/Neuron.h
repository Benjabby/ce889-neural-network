#pragma once
#include <cmath>
#include <vector>
#include <map>
#include <string>


constexpr double LAMBDA = 1;

constexpr int SOFTPLUS = 0;
constexpr int LOGISTIC = 1;
constexpr int LINEAR = 2;

constexpr int DEF_ACTIVATION = LOGISTIC;
constexpr int TEST_ACTIVATION = LINEAR;

class ActivationFunction
{
	typedef double(*AFunction)(double);
	
public:
	
	AFunction activation;
	AFunction derivative;
	//AFunction inverse;

	ActivationFunction(const AFunction a, const AFunction d) : activation(a), derivative(d){}
	
	static double softPlusActivation(double input) { return log(1 + exp(input*LAMBDA)); };
	static double logisticActivation(double input) { return 1.0 / (1 + exp(-LAMBDA*input)); };
	static double linearActivation(double input) { return LAMBDA*input; };
	static double logisticDerivative(double input) { return logisticActivation(input)*(1 - logisticActivation(input)); }
	static double linearDerivative(double input) { return LAMBDA; };
	
};

static const std::vector<ActivationFunction> activationFunctions{ ActivationFunction(ActivationFunction::softPlusActivation,ActivationFunction::logisticActivation), ActivationFunction(ActivationFunction::logisticActivation,ActivationFunction::logisticDerivative), ActivationFunction(ActivationFunction::linearActivation,ActivationFunction::linearDerivative) };


class Neuron
{
public:
	
	
	~Neuron();

	// Public Functions

	std::string debugWeights();
	
protected:

	// First layer neuron (hidden) constructor
	Neuron(int numPrevious);

	// Second or greater layer constructor. Only needed for output layer in this network. We're only ever using one hidden layer. Included for possible future expansion
	Neuron(const std::vector<Neuron*>& previous);

	// Feedforward starts from the output node and works backwards recursively.
	double feedforward(const double* input);

	static double initRandom();
	static double initRoundedRandom();
	static double initDebug();

private:
	// Private Variables

	const bool first;
	std::vector<Neuron*> previousLayer;
	int numPrevious;
	std::vector<double> weights;
	// Private Functions

	void initWeights();
	
};

