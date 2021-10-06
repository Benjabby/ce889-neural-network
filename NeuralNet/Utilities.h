#pragma once
#include <vector>
#include <iostream>

constexpr int NUM_INPUT = 2;
constexpr int NUM_OUTPUT = 2;


const double TLNT = 2.0*std::log(2.0);

// Contains an activation function and its derivative.
// The pairings of these are currently stored in 
class ActivationFunction
{
	typedef double(*AFunction)(double,double);

public:

	const AFunction activation;
	const AFunction derivative;
	//const AFunction inverse;

	ActivationFunction(const AFunction a, const AFunction d);

	static double softPlusActivation(const double input, double lambda);
	static double logisticActivation(const double input, double lambda);
	static double linearActivation(const double input, double lambda);
	static double logisticDerivative(const double input, double lambda);
	static double softPlusDerivative(const double input, double lambda);
	static double linearDerivative(double input, double lambda);

};

extern const ActivationFunction* LOGISTIC;
extern const ActivationFunction* SOFTPLUS;
extern const ActivationFunction* LINEAR;


// Both these tuple classes honestly should just be replaced with matrices, but have more important stuff to finish
class TuplePair
{
public:
	TuplePair(double a, double b, bool scaled = false);

	//TuplePair(double *a, double *b) : a(*(a)), b(*(b)) {}

	const double& operator[](int index) const;

	bool isNormalized() const;

	void normalize(const double* scale);

	TuplePair denormalize(const double* scale) const;

	TuplePair& operator+=(const TuplePair& o);

	TuplePair& operator/=(const double& d);

	friend std::ostream& operator<<(std::ostream& os, const TuplePair tp);
	double rmse(const double& n) const;

private:
	bool scaled;
	double a;
	double b;
};


class LabeledTuple
{
public:
	LabeledTuple(const double x1, const double x2, const double y1, const double y2);

	TuplePair x;
	TuplePair y;
	
	friend std::vector<TuplePair> splitX(std::vector<LabeledTuple> const& data);
	friend std::vector<TuplePair> splitY(std::vector<LabeledTuple> const& data);
	friend std::ostream& operator<<(std::ostream& os, const LabeledTuple lt);

	~LabeledTuple();
};

/*
 * OLDER SLOW VERSION
 * A wrapper for std::vector<std::vector<double>> to perform Matrix operations.
 */
class MatrixSlow
{
private:
	int rows;
	int cols;
	std::vector<std::vector<double>> backing;
public:
	MatrixSlow(int rows, int cols);

	// has dimensions (1,3), Includes a 1 at index [0][0] for bias
	MatrixSlow(TuplePair tuple, bool appendBias = true);

	friend MatrixSlow operator+(const MatrixSlow &m1, const MatrixSlow &m2);

	friend MatrixSlow operator-(const MatrixSlow &m1, const MatrixSlow &m2);

	// Matrix multiplication
	friend MatrixSlow operator*(const MatrixSlow &m1, const MatrixSlow &m2);

	// Scalar multiplication
	friend MatrixSlow operator*(const MatrixSlow &m1, const double &scalar);

	// Hadamard product
	friend MatrixSlow elementWise(MatrixSlow const& m1, MatrixSlow const& m2);

	friend MatrixSlow transpose(MatrixSlow const& m);
	
	// Subscript operator
	std::vector<double>& operator[](int index);
	
	// Const version of subscript operator
	const std::vector<double>& operator[] (int index) const;

	MatrixSlow applyFunction(double(*f)(double)) const;
	MatrixSlow applyFunction(double(*f)()) const;

	MatrixSlow insertBiasColumn() const;

	MatrixSlow removeBiasColumn() const;

	TuplePair toTuplePair() const;

};


/*
 * This is probably full of awful C++ practices, and it's also very inconsistent with its methods, but it works quickly so I'm pretty pleased with it. 
 */
class MatrixFast
{
	const int rows;
	const int cols;
	// Want vectors to be constant, because I just do. That also makes their inner type (effectively) const as well, so the pointer will be const which is good, but because it's still a pointer the actual value is not constant. Yay. 
	const std::vector<std::vector<double*>> backing;
public:
	
	MatrixFast(int rows, int cols);

	MatrixFast(int rows, int cols, double(*f)());
	// Copy constructor, creates new vector backing and copies the values from c
	MatrixFast(const MatrixFast& c);
	MatrixFast(TuplePair& t);
	MatrixFast(std::vector<std::vector<double>>& vecVec);

	void set(TuplePair tuple);
	void set(const MatrixFast& other);

	TuplePair getTuple() const;

	//void set(int r, int c, double v);

	// M.value(r,c) is just short/longhand for *(M[r][c]). Don't know if I ever use it.
	double value(int r, int c) const;
	
	// Applies a function of two inputs, using each element in matrix as first output. Outputs matrix to output.
	void applyFunction(double(*f)(double, double), double l, MatrixFast* output) const;
	// Applies a function of two inputs, to each element in place.
	void applyFunction(double(*f)(double, double), double l);

	// Applies a function to each element of the matrix. Outputs matrix to output.
	void applyFunction(double(*f)(double), MatrixFast* output) const;
	// Applies a function to each element in place.
	void applyFunction(double(*f)(double));

	void hadamard(const MatrixFast* h);

	void scale(double s, MatrixFast* output) const;
	void scale(double s);

	static void add(const MatrixFast* lhs, const MatrixFast* rhs, MatrixFast* output);
	static void subtract(const MatrixFast* lhs, const MatrixFast* rhs, MatrixFast* output);
	
	static void multiply(const MatrixFast* lhs, const MatrixFast* rhs, MatrixFast* output, bool transposeLHS = false, bool transposeRHS = false);
	static void hadamard(const MatrixFast* lhs, const  MatrixFast* rhs, MatrixFast* output);

	// Subscript operators. Potentially not needed
	const std::vector<double*>& operator[](int index);
	const std::vector<double*>& operator[] (int index) const;

	MatrixFast& operator+=(const MatrixFast& a);
	MatrixFast& operator-=(const MatrixFast& a);

	void clear();

	friend std::ostream& operator<<(std::ostream& os, const MatrixFast m);

	~MatrixFast();

private:
	std::vector<std::vector<double*>> allocate();
	std::vector<std::vector<double*>> allocate(double(*f)());
	std::vector<std::vector<double*>> allocate(const MatrixFast& c);
	std::vector<std::vector<double*>> allocate(const TuplePair & t);
	std::vector<std::vector<double*>> allocate(std::vector<std::vector<double>>& vecVec);
};

