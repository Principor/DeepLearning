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
		else {
			gradientValues[i] = 0.0f;
		}
	}
	return gradientList{ gradientTuple(original, Tensor::fromValues(gradientValues, gradientShape)) };
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
	return gradientList{ gradientTuple(original, Tensor::fromValues(gradientValues, gradientShape)) };
}

SetTensorFunction::SetTensorFunction(Tensor* copyTo, Tensor* copyFrom, int index, int size, const std::vector<int>& broadcastShape,
	const std::vector<int>& broadcastIndices) : copyTo(copyTo), copyFrom(copyFrom), index(index), size(size),
	broadcastShape(broadcastShape), broadcastIndices(broadcastIndices)
{

}

gradientList SetTensorFunction::calculateGradient(const Tensor& previousGradient) const
{
	gradientList list{};

	//Copy-To gradient
	{
		int gradientSize = copyTo->getSize();
		const std::vector<int>& gradientShape = copyTo->getShape();
		float* gradientValues = new float[gradientSize];
		for (int i = 0; i < gradientSize; i++) {
			if (i - index >= 0 && i - index < size) {
				gradientValues[i] = 0.0f;
			}
			else {
				gradientValues[i] = previousGradient.at(i);
			}
		}
		list.push_back(gradientTuple{ copyTo, Tensor::fromValues(gradientValues, gradientShape) });
	}

	//Copy-From gradient
	{
		int gradientSize = copyFrom->getSize();
		const std::vector<int>& gradientShape = copyFrom->getShape();
		float* gradientValues = new float[gradientSize];
		for (int i = 0; i < gradientSize; i++) {
			gradientValues[i] = 0.0f;
		}
		for (int i = 0; i < size; i++) {
			int j = broadcastIndices[i];
			int k = index + i;
			gradientValues[broadcastIndices[i]] += previousGradient.at(index + i);
		}
		list.push_back(gradientTuple{ copyTo, Tensor::fromValues(gradientValues, gradientShape) });
	}
	return list;
}
