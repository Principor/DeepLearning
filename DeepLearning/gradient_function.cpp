#include "gradient_function.h"
#include "deep_learning.h"

GetFunction::GetFunction(const Tensor* original, int index, int size) : original(original), index(index), size(size)
{

}

Tensor GetFunction::calculateGradient(const Tensor& previousGradient) const {
	int gradientSize = original->getSize();
	const std::vector<int>& shape = original->getShape();
	float* values = new float[gradientSize];
	return Tensor::fromValues(values, shape);
}