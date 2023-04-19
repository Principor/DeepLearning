#pragma once
#include <vector>
#include <tuple>

class Tensor;

using gradientList = std::vector<std::tuple<Tensor*, Tensor>>;

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
public:
	SetTensorFunction(Tensor* copyTo, Tensor* copyFrom, int index, const std::vector<int>& broadcastShape,
		const std::vector<int>& broadcastIndices);
	gradientList calculateGradient(const Tensor& previousGradient) const override;
};