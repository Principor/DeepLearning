#include "gradient_function.h"
#include "deep_learning.h"

GetFunction::GetFunction(const Tensor* original, int index, int size) : original(original), index(index), size(size)
{

}

Tensor GetFunction::calculateGradient(const Tensor& previousGradient) const {
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradient = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		if (i - index >= 0 && i - index < size) {
			gradient[i] = previousGradient.at(i - index);
		}
		else {
			gradient[i] = 0.0f;
		}
	}
	return Tensor::fromValues(gradient, gradientShape);
}