#include "gradient_function.h"
#include "deep_learning.h"

GetFunction::GetFunction(const Tensor* original, int index, int size) : original(original), index(index), size(size)
{

}

Tensor GetFunction::calculateGradient(const Tensor& previousGradient) const {
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		if (i - index >= 0 && i - index < size) {
			gradientValues[i] = previousGradient.at(i - index);
		}
		else{
			gradientValues[i] = 0.0f;
		}
	}
	return Tensor::fromValues(gradientValues, gradientShape);
}

SetSingleFunction::SetSingleFunction(const Tensor* original, int index, int size) : original(original), index(index), size(size)
{

}

Tensor SetSingleFunction::calculateGradient(const Tensor& previousGradient) const {
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		if (i - index >= 0 && i - index < size) {
			gradientValues[i] = 0.0f;
		}
		else {
			gradientValues[i] = previousGradient.at(i);
		}
	}
	return Tensor::fromValues(gradientValues, gradientShape);
}

SetTensorFunction::SetTensorFunction(Tensor* copyTo, Tensor* copyFrom, int index, const std::vector<int>& broadcastShape, 
	const std::vector<int>& broadcastIndices) 
{

}

Tensor SetTensorFunction::calculateGradient(const Tensor& previousGradient) const
{
	return Tensor::zeroes({});
}
