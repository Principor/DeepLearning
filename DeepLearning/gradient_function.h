#pragma once
#include <vector>
#include <tuple>

class Tensor;

using gradientTuple = std::tuple<Tensor*, Tensor>;
using gradientList = std::vector<gradientTuple>;

class GradientFunction {
public:
	virtual gradientList calculateGradient(Tensor& previousGradient) const = 0;
};

class GetFunction : public GradientFunction {
private:
	Tensor* original;
	int index;
	int size;
public:
	GetFunction(Tensor* original, int index, int size);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class SetSingleFunction : public GradientFunction {
private:
	Tensor* original;
	int index;
	int size;
public:
	SetSingleFunction(Tensor* original, int index, int size);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class SetTensorFunction : public GradientFunction
{
private:
	Tensor* copyTo;
	Tensor* copyFrom;
	int index;
	int size;
	std::vector<int> broadcastShape;
	std::vector<int> broadcastIndices;
public:
	SetTensorFunction(Tensor* copyTo, Tensor* copyFrom, int index, int size, const std::vector<int>& broadcastShape,
		const std::vector<int>& broadcastIndices);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class AddSingleFunction : public GradientFunction {
private:
	Tensor* original;
public:
	AddSingleFunction(Tensor* original);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class AddTensorFunction : public GradientFunction
{
private:
	Tensor* original1, * original2;
	std::vector<int> broadcastIndices1, broadcastIndices2;
public:
	AddTensorFunction(Tensor* original1, Tensor* original2,
		const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class SubtractSingleFunction : public GradientFunction {
private:
	Tensor* original;
public:
	SubtractSingleFunction(Tensor* original);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class SubtractTensorFunction : public GradientFunction
{
private:
	Tensor* original1, * original2;
	std::vector<int> broadcastIndices1, broadcastIndices2;
public:
	SubtractTensorFunction(Tensor* original1, Tensor* original2,
		const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class MultiplySingleFunction : public GradientFunction {
private:
	Tensor* original;
	float value;
public:
	MultiplySingleFunction(Tensor* original, float value);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class MultiplyTensorFunction : public GradientFunction
{
private:
	Tensor* original1, * original2;
	std::vector<int> broadcastIndices1, broadcastIndices2;
public:
	MultiplyTensorFunction(Tensor* original1, Tensor* original2,
		const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class DivideSingleFunction : public GradientFunction {
private:
	Tensor* original;
	float value;
public:
	DivideSingleFunction(Tensor* original, float value);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class DivideTensorFunction : public GradientFunction
{
private:
	Tensor* original1, * original2;
	std::vector<int> broadcastIndices1, broadcastIndices2;
public:
	DivideTensorFunction(Tensor* original1, Tensor* original2,
		const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class TransposeFunction : public GradientFunction
{
private:
	Tensor* original;
	std::vector<int> transposeIndices;
public:
	TransposeFunction(Tensor* original, const std::vector<int>& transposeIndices);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class MatrixMultiplicationFunction : public GradientFunction
{
private:
	Tensor* original1, * original2;
	std::vector<int> broadcastIndices1, broadcastIndices2;
	int matrixWidth, matrixInner, matrixHeight;
public:
	MatrixMultiplicationFunction(Tensor* original1, Tensor* original2,
		const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2,
		int matrixWidth, int matrixInner, int matrixHeight);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class MaxSingleFunction : public GradientFunction
{
private:
	Tensor* original;
	float value;
public:
	MaxSingleFunction(Tensor* original, float value);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class MaxTensorFunction : public GradientFunction
{
private:
	Tensor* original1, * original2;
	std::vector<int> broadcastIndices1, broadcastIndices2;
public:
	MaxTensorFunction(Tensor* original1, Tensor* original2,
		const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class MinSingleFunction : public GradientFunction
{
private:
	Tensor* original;
	float value;
public:
	MinSingleFunction(Tensor* original, float value);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};

class MinTensorFunction : public GradientFunction
{
private:
	Tensor* original1, * original2;
	std::vector<int> broadcastIndices1, broadcastIndices2;
public:
	MinTensorFunction(Tensor* original1, Tensor* original2,
		const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2);
	gradientList calculateGradient(Tensor& previousGradient) const override;
};
