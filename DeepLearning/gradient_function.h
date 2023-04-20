#pragma once
#include <vector>
#include <tuple>

class Tensor;

using gradientTuple = std::tuple<Tensor*, Tensor>;
using gradientList = std::vector<gradientTuple>;

class GradientFunction {
public:
	virtual gradientList calculateGradient(const Tensor& previousGradient) const = 0;
};

class GetFunction : public GradientFunction {
private:
	Tensor* original;
	int index;
	int size;
public:
	GetFunction(Tensor* original, int index, int size);
	gradientList calculateGradient(const Tensor& previousGradient) const override;
};

class SetSingleFunction : public GradientFunction {
private:
	Tensor* original;
	int index;
	int size;
public:
	SetSingleFunction(Tensor* original, int index, int size);
	gradientList calculateGradient(const Tensor& previousGradient) const override;
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
	gradientList calculateGradient(const Tensor& previousGradient) const override;
};

class AddSingleFunction : public GradientFunction {
private:
	Tensor* original;
public:
	AddSingleFunction(Tensor* original);
	gradientList calculateGradient(const Tensor& previousGradient) const override;
};