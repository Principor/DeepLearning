#include "gradient_function.h"
#include "deep_learning.h"

GetFunction::GetFunction(Tensor* original, int index, int size) : original(original), index(index), size(size)
{

}

gradientList GetFunction::calculateGradient(const Tensor& previousGradient) const {
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
	return gradientList{ std::tuple<Tensor*, Tensor>(original, Tensor::fromValues(gradientValues, gradientShape))};
}

SetSingleFunction::SetSingleFunction(Tensor* original, int index, int size) : original(original), index(index), size(size)
{

}
gradientList SetSingleFunction::calculateGradient(const Tensor& previousGradient) const {
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
	return gradientList{ std::tuple<Tensor*, Tensor>(original, Tensor::fromValues(gradientValues, gradientShape)) };
}

SetTensorFunction::SetTensorFunction(Tensor* copyTo, Tensor* copyFrom, int index, const std::vector<int>& broadcastShape, 
	const std::vector<int>& broadcastIndices) 
{

}

gradientList SetTensorFunction::calculateGradient(const Tensor& previousGradient) const
{
	return gradientList();
}
