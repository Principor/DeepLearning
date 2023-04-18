#pragma once

class Tensor;

class GradientFunction {
public:
	virtual Tensor calculateGradient(const Tensor& previousGradient) const = 0;
};

class GetFunction : public GradientFunction {
private:
	const Tensor* original;
	int index;
	int size;
public:
	GetFunction(const Tensor* original, int index, int size);
	Tensor calculateGradient(const Tensor& previousGradient) const override;
};

class SetSingleFunction : public GradientFunction {
private:
	const Tensor* original;
	int index;
	int size;
public:
	SetSingleFunction(const Tensor* original, int index, int size);
	Tensor calculateGradient(const Tensor& previousGradient) const override;
};