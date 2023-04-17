#pragma once

class Tensor;

class GradientFunction {
};

class GetFunction : public GradientFunction {
private:
	const Tensor* original;
	int index;
	int size;
public:
	GetFunction(const Tensor* original, int index, int size);
};