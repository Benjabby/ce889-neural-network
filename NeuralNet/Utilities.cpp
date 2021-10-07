#include "Utilities.h"
#include <iomanip>
#include <functional>
#include <Windows.h>
#include <ppl.h>

#pragma region Tuple Pair

TuplePair::TuplePair(const double a, const double b, bool scaled) : scaled(scaled), a(a), b(b) {}

const double& TuplePair::operator[](int index) const
{
	if (index == 0) return a;
	if (index == 1) return b;
	throw std::out_of_range("TuplePair only has subscripts 0 or 1");
}

bool TuplePair::isNormalized() const
{
	return scaled;
}

void TuplePair::normalize(const double* scale)
{
	if (scaled) return;

	scaled = true;

	a = (a - scale[0]) / (scale[1] - scale[0]);
	b = (b - scale[2]) / (scale[3] - scale[2]);
}

TuplePair TuplePair::denormalize(const double* scale) const
{
	return (!scaled ? *this : TuplePair(a*(scale[1] - scale[0]) + scale[0], b*(scale[3] - scale[2]) + scale[2]));
}

TuplePair& TuplePair::operator+=(const TuplePair& o)
{
	a += o.a;
	b += o.b;
	return *this;
}

TuplePair& TuplePair::operator/=(const double& d)
{
	a /= d;
	b /= d;
	return *this;
}

double TuplePair::rmse(const double& n) const
{
	return sqrt((a + b)/n);
}

#pragma endregion

#pragma region Labeled Tuple
LabeledTuple::LabeledTuple(const double x1, const double x2, const double y1, const double y2) : x(TuplePair(x1, x2)), y(TuplePair(y1, y2))
{
	
}

std::ostream & operator<<(std::ostream & os, const TuplePair tp)
{
	return os << std::setprecision(10) << "[ " << tp.a << ", " << tp.b << " ]";
}

std::vector<TuplePair> splitX(std::vector<LabeledTuple> const& data)
{
	std::vector<TuplePair> xValues;
	xValues.reserve(data.size());
	for (LabeledTuple e : data) xValues.emplace_back(e.x);
	return xValues;
}

std::vector<TuplePair> splitY(std::vector<LabeledTuple> const& data)
{
	std::vector<TuplePair> yValues;
	yValues.reserve(data.size());
	for (LabeledTuple e : data) yValues.emplace_back(e.y);
	return yValues;
}

std::ostream& operator<<(std::ostream& os, const LabeledTuple lt)
{
	os << std::setprecision(10) << "{ X:" << lt.x << ", Y:" << lt.y << " }";
	return os;
}

LabeledTuple::~LabeledTuple() = default;
#pragma endregion

#pragma region OLD Matrix
MatrixSlow::MatrixSlow(int rows, int cols) : rows(rows), cols(cols), backing(rows, std::vector<double>(cols)) { }
// has dimensions (1,3), Includes a 1 at index [0][0] for bias
MatrixSlow::MatrixSlow(TuplePair tuple, bool appendBias) : rows(1), cols(appendBias ? 3 : 2), backing(rows, std::vector<double>(cols))
{
	if (appendBias)
	{
		backing[0][0] = 1;
		backing[0][1] = tuple[0];
		backing[0][2] = tuple[1];
	}
	else
	{
		backing[0][0] = tuple[0];
		backing[0][1] = tuple[1];
	}
}

MatrixSlow operator+(const MatrixSlow &m1, const MatrixSlow &m2)
{
	if (m1.cols != m2.cols || m1.rows != m2.rows) throw std::runtime_error("Incompatible dimensions for Matrix addition");
	MatrixSlow result(m1.rows, m1.cols);
	for (auto i = 0; i < result.rows; i++)
	{
		for (auto j = 0; j < result.cols; j++)
		{
			result[i][j] = m1[i][j] + m2[i][j];
		}
	}

	return result;
}

MatrixSlow operator-(const MatrixSlow &m1, const MatrixSlow &m2)
{
	if (m1.cols != m2.cols || m1.rows != m2.rows) throw std::runtime_error("Incompatible dimensions for Matrix subtraction");
	MatrixSlow result(m1.rows, m1.cols);
	for (auto i = 0; i < result.rows; i++)
	{
		for (auto j = 0; j < result.cols; j++)
		{
			result[i][j] = m1[i][j] - m2[i][j];
		}
	}

	return result;
}

// Matrix multiplication
MatrixSlow operator*(const MatrixSlow &m1, const MatrixSlow &m2)
{
	if (m1.cols != m2.rows) throw std::runtime_error("Incompatible dimensions for Matrix multiplication");

	MatrixSlow result(m1.rows, m2.cols);

	for (auto i = 0; i < m1.rows; i++)
	{
		for (auto j = 0; j < m2.cols; j++)
		{
			for (auto k = 0; k < m1.cols; k++)
			{
				result[i][j] += m1[i][k] * m2[k][j];
			}

		}
	}

	return result;
}

// Scalar multiplication
MatrixSlow operator*(const MatrixSlow &m1, const double &scalar)
{
	MatrixSlow result(m1.rows, m1.cols);
	for (auto i = 0; i < result.rows; i++)
	{
		for (auto j = 0; j < result.cols; j++)
		{
			result[i][j] = m1[i][j] * scalar;
		}
	}
	return result;
}

// Hadamard product
MatrixSlow elementWise(MatrixSlow const& m1, MatrixSlow const& m2)
{
	if (m1.cols != m2.cols || m1.rows != m2.rows) throw std::runtime_error("Incompatible dimensions for element-wise Matrix multiplication");
	MatrixSlow result(m1.rows, m1.cols);
	for (auto i = 0; i < result.rows; i++)
	{
		for (auto j = 0; j < result.cols; j++)
		{
			result[i][j] = m1[i][j] * m2[i][j];
		}
	}

	return result;
}

MatrixSlow transpose(MatrixSlow const& m)
{
	MatrixSlow result(m.cols, m.rows);

	for (int i = 0; i < m.cols; i++)
	{
		for (int j = 0; j < m.rows; j++)
		{
			result[i][j] = m[j][i];
		}
	}
	return result;
}

// Subscript operator
std::vector<double>& MatrixSlow::operator[](int index)
{
	return backing[index];
}
// Const version of subscript operator
const std::vector<double>& MatrixSlow::operator[] (int index) const
{
	return backing[index];
}

MatrixSlow MatrixSlow::applyFunction(double(*f)(double)) const
{
	MatrixSlow result(rows, cols);

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			result[i][j] = f(backing[i][j]);
		}
	}

	return result;
}

MatrixSlow MatrixSlow::applyFunction(double(*f)()) const
{
	MatrixSlow result(rows, cols);

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			result[i][j] = f();
		}
	}

	return result;
}

MatrixSlow MatrixSlow::insertBiasColumn() const
{
	MatrixSlow result(rows, cols + 1);
	for (auto i = 0; i < rows; i++)
	{
		result[i][0] = 1;
		for (auto j = 1; j <= cols; j++)
		{
			result[i][j] = backing[i][j - 1];
		}
	}
	return result;
}

MatrixSlow MatrixSlow::removeBiasColumn() const
{
	MatrixSlow result(rows, cols - 1);
	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 1; j < cols; j++)
		{
			result[i][j - 1] = backing[i][j];
		}
	}
	return result;
}

TuplePair MatrixSlow::toTuplePair() const
{
	if (rows != 1 || cols != 2) throw std::runtime_error("Matrix must have dimensions (1,2) to convert to TuplePair");
	return{ backing[0][0], backing[0][1], true };
}

#pragma endregion

#pragma region New, faster Matrix

std::vector<std::vector<double*>> MatrixFast::allocate()
{
	std::vector<std::vector<double*>> N(rows, std::vector<double*>(cols));

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			N[i][j] = new double;
			*N[i][j] = 0;
		}
	}
	return N;
}

std::vector<std::vector<double*>> MatrixFast::allocate(double(*f)())
{
	std::vector<std::vector<double*>> N(rows, std::vector<double*>(cols));

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			N[i][j] = new double;
			*N[i][j] = f();
		}
	}
	return N;
}

std::vector<std::vector<double*>> MatrixFast::allocate(const MatrixFast & c)
{
	std::vector<std::vector<double*>> N(rows, std::vector<double*>(cols));

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			N[i][j] = new double;
			*N[i][j] = *c.backing[i][j];
		}
	}
	return N;
}

std::vector<std::vector<double*>> MatrixFast::allocate(const TuplePair & t)
{
	std::vector<std::vector<double*>> N(1, std::vector<double*>(2));

	N[0][0] = new double(t[0]);
	N[0][1] = new double(t[1]);
	
	return N;
}

std::vector<std::vector<double*>> MatrixFast::allocate(std::vector<std::vector<double>>& vecVec)
{
	std::vector<std::vector<double*>> N(rows, std::vector<double*>(cols));

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			N[i][j] = new double;
			*N[i][j] = vecVec[i][j];
		}
	}
	return N;
}

MatrixFast::MatrixFast(const int rows, const int cols) : rows(rows), cols(cols), backing(allocate())
{
}

MatrixFast::MatrixFast(TuplePair& t) : rows(1), cols(2), backing(allocate(t))
{
}

MatrixFast::MatrixFast(int rows, int cols, double(*f)()) : rows(rows), cols(cols), backing(allocate(f))
{
}

MatrixFast::MatrixFast(const MatrixFast& c) : rows(c.rows), cols(c.cols), backing(allocate(c))
{
}

MatrixFast::MatrixFast(std::vector<std::vector<double>>& vecVec) : rows(vecVec.size()), cols(vecVec[0].size()), backing(allocate(vecVec))
{
}

void MatrixFast::set(TuplePair tuple)
{
	if (rows != 1) throw std::runtime_error("Cannot set matrix from tuple if it is not a row vector");
	if (cols != 2) throw std::runtime_error("Cannot set matrix from tuple if it does not have 2 columns");

	*backing[0][0] = tuple[0];
	*backing[0][1] = tuple[1];
}

TuplePair MatrixFast::getTuple() const
{
	return{ *backing[0][0] ,*backing[0][1] , true};
}

double MatrixFast::value(int r, int c) const { return *backing[r][c]; }

void MatrixFast::applyFunction(double(*f)(double, double), double l, MatrixFast* output) const
{
	if (rows != output->rows) throw std::runtime_error("Output matrix must have the same rows to apply function");
	if (cols != output->cols) throw std::runtime_error("Output matrix must have the same columns to apply function");

	#pragma omp parallel for
	for (int i = 0; i < output->rows; i++)
	{
		for (auto j = 0; j < output->cols; j++)
		{
			*output->backing[i][j] = f(*backing[i][j],l);
		}
	}
}

void MatrixFast::applyFunction(double(*f)(double, double), double l)
{
	// Would this cause any problems?
	applyFunction(f,l, this);
}

void MatrixFast::applyFunction(double(*f)(double), MatrixFast* output) const
{
	if (rows != output->rows) throw std::runtime_error("Output matrix must have the same rows to apply function");
	if (cols != output->cols) throw std::runtime_error("Output matrix must have the same columns to apply function");

	#pragma omp parallel for
	for (int i = 0; i < output->rows; i++)
	{
		for (auto j = 0; j < output->cols; j++)
		{
			*output->backing[i][j] = f(*backing[i][j]);
		}
	}
}

void MatrixFast::applyFunction(double(*f)(double))
{
	// Would this cause any problems?
	applyFunction(f, this);
}

void MatrixFast::scale(double s, MatrixFast* output) const
{
	if (rows != output->rows) throw std::runtime_error("Output matrix must have the same rows to apply function");
	if (cols != output->cols) throw std::runtime_error("Output matrix must have the same columns to apply function");

	#pragma omp parallel for
	for (int i = 0; i < output->rows; i++)
	{
		for (auto j = 0; j < output->cols; j++)
		{
			*output->backing[i][j] = (*backing[i][j])*s;
		}
	}
}

void MatrixFast::scale(double s)
{
	// Again, would this cause any problems?
	scale(s, this);
}


void MatrixFast::add(const MatrixFast* lhs, const MatrixFast* rhs, MatrixFast* output)
{
	if(lhs->rows != rhs->rows) throw std::runtime_error("LHS and RHS matrices must have the same rows to add");
	if(lhs->cols != rhs->cols) throw std::runtime_error("LHS and RHS matrices must have the same columns to add");
	if(lhs->rows != output->rows) throw std::runtime_error("Output matrix must have the same rows as operands to add");
	if(lhs->cols != output->cols) throw std::runtime_error("Output matrix must have the same columns as operands to add");

	#pragma omp parallel for
	for (int i = 0; i < output->rows; i++)
	{
		for (auto j = 0; j < output->cols; j++)
		{
			*output->backing[i][j] = *lhs->backing[i][j] + *rhs->backing[i][j];
			// Not sure which one is C++ier, that ^ or that v which relies on subscript operator
			//*(*output)[i][j] = *(*lhs)[i][j] + *(*rhs)[i][j];

		}
	}

}

MatrixFast& MatrixFast::operator+=(const MatrixFast& a)
{
	add(this, &a, this);
	return *this;
}

MatrixFast& MatrixFast::operator-=(const MatrixFast& a)
{
	subtract(this, &a, this);
	return *this;
}

void MatrixFast::subtract(const MatrixFast* lhs, const MatrixFast* rhs, MatrixFast* output)
{
	if (lhs->rows != rhs->rows) throw std::runtime_error("LHS and RHS matrices must have the same rows to subtract");
	if (lhs->cols != rhs->cols) throw std::runtime_error("LHS and RHS matrices must have the same columns to subtract");
	if (lhs->rows != output->rows) throw std::runtime_error("Output matrix must have the same rows as operands to subtract");
	if (lhs->cols != output->cols) throw std::runtime_error("Output matrix must have the same columns as operands to subtract");

	#pragma omp parallel for
	for (int i = 0; i < output->rows; i++)
	{
		for (auto j = 0; j < output->cols; j++)
		{
			*output->backing[i][j] = *lhs->backing[i][j] - *rhs->backing[i][j];
		}
	}
}

void MatrixFast::hadamard(const MatrixFast* h)
{
	hadamard(this, h, this);
}

void MatrixFast::hadamard(const MatrixFast * lhs, const MatrixFast * rhs, MatrixFast * output)
{
	if (lhs->rows != rhs->rows) throw std::runtime_error("LHS and RHS matrices must have the same rows to use Hadamard product");
	if (lhs->cols != rhs->cols) throw std::runtime_error("LHS and RHS matrices must have the same columns to use Hadamard product");
	if (lhs->rows != output->rows) throw std::runtime_error("Output matrix must have the same rows as operands to use Hadamard product");
	if (lhs->cols != output->cols) throw std::runtime_error("Output matrix must have the same columns as operands to use Hadamard product");

	#pragma omp parallel for
	for (int i = 0; i < output->rows; i++)
	{
		for (auto j = 0; j < output->cols; j++)
		{
			*output->backing[i][j] = (*lhs->backing[i][j]) * (*rhs->backing[i][j]);
		}
	}
}

void MatrixFast::multiply(const MatrixFast* lhs, const MatrixFast* rhs, MatrixFast* output, const bool transposeLHS, const bool transposeRHS)
{
	const int lhsRows = transposeLHS ? lhs->cols : lhs->rows;
	const int lhsCols = transposeLHS ? lhs->rows : lhs->cols;

	const int rhsRows = transposeRHS ? rhs->cols : rhs->rows;
	const int rhsCols = transposeRHS ? rhs->rows : rhs->cols;
	
	if (lhsCols != rhsRows) throw std::runtime_error("LHS columns must equal RHS rows for matrix multiplication");

	output->clear();

	int i, j, k;
	#pragma omp parallel for private(i, j, k)
	for (i = 0; i < lhsRows; i++)
	{
		for (j = 0; j < rhsCols; j++)
		{
			for (k = 0; k < lhsCols; k++)
			{
				*output->backing[i][j] += (*lhs->backing[transposeLHS ? k : i][transposeLHS ? i : k]) * (*rhs->backing[transposeRHS ? j : k][transposeRHS ? k : j]);
			}
		}
	}

	// Slower...
	// Concurrency::parallel_for(int(0), lhsRows, [&](int i)
	// {
	// 	for (auto j = 0; j < rhsCols; j++)
	// 	{
	// 		for (auto k = 0; k < lhsCols; k++)
	// 		{
	// 			*output->backing[i][j] += (*lhs->backing[transposeLHS ? k : i][transposeLHS ? i : k]) * (*rhs->backing[transposeRHS ? j : k][transposeRHS ? k : j]);
	// 		}
	// 	}
	// });

}

void MatrixFast::clear()
{
	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			*backing[i][j] = 0;
		}
	}
}

const std::vector<double*>& MatrixFast::operator[](int index)
{
	return backing[index];
}

const std::vector<double*>& MatrixFast::operator[](int index) const
{
	return backing[index];
}

void MatrixFast::set(const MatrixFast & other)
{
	if (rows != other.rows) throw std::runtime_error("Matrices must have the same rows to set");
	if (cols != other.cols) throw std::runtime_error("Matrices must have the same columns to set");

	#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			*backing[i][j] = *other.backing[i][j];
		}
	}
}

std::ostream& operator<<(std::ostream& os, const MatrixFast m)
{
	os << "{";
	for (auto i = 0; i < m.rows; i++)
	{
		os << "[";
		for (auto j = 0; j < m.cols; j++)
		{
			os << *(m.backing[i][j]);
			if (j < m.cols - 1) os << ',';
		}
		os << "]";
	}
	return os << "}";
}

MatrixFast::~MatrixFast()
{
	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			delete backing[i][j];
		}
	}
}

#pragma endregion

#pragma region Contigous Array Matrix

void MatrixContig::allocate()
{
	backing = new double[rows*cols];

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			backing[i*cols + j] = 0;
		}
	}
}

void MatrixContig::allocate(double(*f)())
{
	backing = new double[rows*cols];

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			backing[i*cols + j] = f();
		}
	}
}

void MatrixContig::allocate(const MatrixContig & c)
{
	backing = new double[rows*cols];

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			backing[i*cols + j] = c.backing[i*cols + j];
		}
	}
}

void MatrixContig::allocate(const TuplePair & t)
{
	backing = new double[2];

	backing[0] = t[0];
	backing[1] = t[1];

}

void MatrixContig::allocate(std::vector<std::vector<double>>& vecVec)
{
	backing = new double[rows*cols];

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			backing[i*cols + j] = vecVec[i][j];
		}
	}
}


MatrixContig::MatrixContig(const int rows, const int cols) : rows(rows), cols(cols)
{
	allocate();
}

MatrixContig::MatrixContig(TuplePair& t) : rows(1), cols(2)
{
	allocate(t);
}

MatrixContig::MatrixContig(int rows, int cols, double(*f)()) : rows(rows), cols(cols)
{
	allocate(f);
}

MatrixContig::MatrixContig(const MatrixContig& c) : rows(c.rows), cols(c.cols)
{
	allocate(c);
}

MatrixContig::MatrixContig(std::vector<std::vector<double>>& vecVec) : rows(vecVec.size()), cols(vecVec[0].size())
{
	allocate(vecVec);
}

void MatrixContig::set(TuplePair tuple)
{
	if (rows != 1) throw std::runtime_error("Cannot set matrix from tuple if it is not a row vector");
	if (cols != 2) throw std::runtime_error("Cannot set matrix from tuple if it does not have 2 columns");

	backing[0] = tuple[0];
	backing[1] = tuple[1];
}

TuplePair MatrixContig::getTuple() const
{
	return{ backing[0], backing[1] , true };
}

double MatrixContig::value(int r, int c) const { return backing[r*cols + c]; }

void MatrixContig::applyFunction(double(*f)(double, double), double l, MatrixContig* output) const
{
	if (rows != output->rows) throw std::runtime_error("Output matrix must have the same rows to apply function");
	if (cols != output->cols) throw std::runtime_error("Output matrix must have the same columns to apply function");

#pragma omp parallel for
	for (int i = 0; i < output->rows; i++)
	{
		for (auto j = 0; j < output->cols; j++)
		{
			output->backing[i*cols + j] = f(backing[i*cols + j], l);
		}
	}
}

void MatrixContig::applyFunction(double(*f)(double, double), double l)
{
	// Would this cause any problems?
	applyFunction(f, l, this);
}

void MatrixContig::applyFunction(double(*f)(double), MatrixContig* output) const
{
	if (rows != output->rows) throw std::runtime_error("Output matrix must have the same rows to apply function");
	if (cols != output->cols) throw std::runtime_error("Output matrix must have the same columns to apply function");

#pragma omp parallel for
	for (int i = 0; i < output->rows; i++)
	{
		for (auto j = 0; j < output->cols; j++)
		{
			output->backing[i*cols + j] = f(backing[i*cols + j]);
		}
	}
}

void MatrixContig::applyFunction(double(*f)(double))
{
	// Would this cause any problems?
	applyFunction(f, this);
}

void MatrixContig::scale(double s, MatrixContig* output) const
{
	if (rows != output->rows) throw std::runtime_error("Output matrix must have the same rows to apply function");
	if (cols != output->cols) throw std::runtime_error("Output matrix must have the same columns to apply function");

	const int rows = output->rows;
	const int cols = output->cols;
#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			output->backing[i*cols + j] *= s;
		}
	}
}

void MatrixContig::scale(double s)
{
	// Again, would this cause any problems?
	scale(s, this);
}


void MatrixContig::add(const MatrixContig* lhs, const MatrixContig* rhs, MatrixContig* output)
{
	if (lhs->rows != rhs->rows) throw std::runtime_error("LHS and RHS matrices must have the same rows to add");
	if (lhs->cols != rhs->cols) throw std::runtime_error("LHS and RHS matrices must have the same columns to add");
	if (lhs->rows != output->rows) throw std::runtime_error("Output matrix must have the same rows as operands to add");
	if (lhs->cols != output->cols) throw std::runtime_error("Output matrix must have the same columns as operands to add");

	const int rows = output->rows;
	const int cols = output->cols;
#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			output->backing[i*cols + j] = lhs->backing[i*cols + j] + rhs->backing[i*cols + j];
		}
	}
}

MatrixContig& MatrixContig::operator+=(const MatrixContig& a)
{
	add(this, &a, this);
	return *this;
}

MatrixContig& MatrixContig::operator-=(const MatrixContig& a)
{
	subtract(this, &a, this);
	return *this;
}

void MatrixContig::subtract(const MatrixContig* lhs, const MatrixContig* rhs, MatrixContig* output)
{
	if (lhs->rows != rhs->rows) throw std::runtime_error("LHS and RHS matrices must have the same rows to subtract");
	if (lhs->cols != rhs->cols) throw std::runtime_error("LHS and RHS matrices must have the same columns to subtract");
	if (lhs->rows != output->rows) throw std::runtime_error("Output matrix must have the same rows as operands to subtract");
	if (lhs->cols != output->cols) throw std::runtime_error("Output matrix must have the same columns as operands to subtract");

	const int rows = output->rows;
	const int cols = output->cols;
#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			output->backing[i*cols + j] = lhs->backing[i*cols + j] - rhs->backing[i*cols + j];
		}
	}
}

void MatrixContig::hadamard(const MatrixContig* h)
{
	hadamard(this, h, this);
}

void MatrixContig::hadamard(const MatrixContig * lhs, const MatrixContig * rhs, MatrixContig * output)
{
	if (lhs->rows != rhs->rows) throw std::runtime_error("LHS and RHS matrices must have the same rows to use Hadamard product");
	if (lhs->cols != rhs->cols) throw std::runtime_error("LHS and RHS matrices must have the same columns to use Hadamard product");
	if (lhs->rows != output->rows) throw std::runtime_error("Output matrix must have the same rows as operands to use Hadamard product");
	if (lhs->cols != output->cols) throw std::runtime_error("Output matrix must have the same columns as operands to use Hadamard product");

	const int rows = output->rows;
	const int cols = output->cols;
#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			output->backing[i*cols + j] = (lhs->backing[i*cols + j]) * (rhs->backing[i*cols + j]);
		}
	}
}

void MatrixContig::multiply(const MatrixContig* lhs, const MatrixContig* rhs, MatrixContig* output, const bool transposeLHS, const bool transposeRHS)
{
	const int lhsRows = transposeLHS ? lhs->cols : lhs->rows;
	const int lhsCols = transposeLHS ? lhs->rows : lhs->cols;

	const int rhsRows = transposeRHS ? rhs->cols : rhs->rows;
	const int rhsCols = transposeRHS ? rhs->rows : rhs->cols;

	if (lhsCols != rhsRows) throw std::runtime_error("LHS columns must equal RHS rows for matrix multiplication");

	output->clear();

	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < lhsRows; i++)
	{
		for (j = 0; j < rhsCols; j++)
		{
			double sum = 0;
			for (k = 0; k < lhsCols; k++)
			{
				sum += lhs->backing[(transposeLHS ? k : i)*lhs->cols + (transposeLHS ? i : k)] * rhs->backing[(transposeRHS ? j : k)*rhs->cols + (transposeRHS ? k : j)];
			}
			output->backing[i*rhsCols + j] = sum;
		}
	}

	// Slower...
	// Concurrency::parallel_for(int(0), lhsRows, [&](int i)
	// {
	// 	for (auto j = 0; j < rhsCols; j++)
	// 	{
	// 		for (auto k = 0; k < lhsCols; k++)
	// 		{
	// 			*output->backing[i][j] += (*lhs->backing[transposeLHS ? k : i][transposeLHS ? i : k]) * (*rhs->backing[transposeRHS ? j : k][transposeRHS ? k : j]);
	// 		}
	// 	}
	// });

}

void MatrixContig::clear()
{
	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			backing[i*cols + j] = 0;
		}
	}
}


void MatrixContig::set(const MatrixContig & other)
{
	if (rows != other.rows) throw std::runtime_error("Matrices must have the same rows to set");
	if (cols != other.cols) throw std::runtime_error("Matrices must have the same columns to set");

#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			backing[i*cols + j] = other.backing[i*cols + j];
		}
	}
}

std::ostream& operator<<(std::ostream& os, const MatrixContig m)
{
	os << "{";
	for (auto i = 0; i < m.rows; i++)
	{
		os << "[";
		for (auto j = 0; j < m.cols; j++)
		{
			os << m.backing[i*m.cols + j];
			if (j < m.cols - 1) os << ',';
		}
		os << "]";
	}
	return os << "}";
}

MatrixContig::~MatrixContig()
{
	delete[] backing;
}

#pragma endregion

#pragma region Activation Function

const ActivationFunction* LOGISTIC = new ActivationFunction(ActivationFunction::logisticActivation, ActivationFunction::logisticDerivative);
const ActivationFunction* SOFTPLUS = new ActivationFunction(ActivationFunction::softPlusActivation, ActivationFunction::softPlusDerivative);
const ActivationFunction* LINEAR = new ActivationFunction(ActivationFunction::linearActivation, ActivationFunction::linearDerivative);

ActivationFunction::ActivationFunction(const AFunction a, const AFunction d) : activation(a), derivative(d) {}

double ActivationFunction::softPlusActivation(const double input, const double lambda) { return (log(1 + exp(input*lambda*TLNT)) / TLNT); };
double ActivationFunction::logisticActivation(const double input, const double lambda) { return 1.0 / (1 + exp(-lambda*input)); }
double ActivationFunction::linearActivation(const double input, const double lambda) { return lambda*input; };
double ActivationFunction::logisticDerivative(const double input, const double lambda)
{
	const double l = logisticActivation(input, lambda);
	return l*(1 - l);
}
double ActivationFunction::softPlusDerivative(const double input, const double lambda) { return logisticActivation(input, lambda*TLNT)*lambda; };
double ActivationFunction::linearDerivative(double input, const double lambda) { return lambda; };
#pragma endregion