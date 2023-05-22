#include "gradient_function.h"
#include "deep_learning.h"

GetFunction::GetFunction(Tensor* original, int index, int size) : original(original), index(index), size(size)
{

}

gradientList GetFunction::calculateGradient(Tensor& previousGradient) const {
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

std::vector<Tensor*> GetFunction::getDependents() const {
	return { original };
}


SetSingleFunction::SetSingleFunction(Tensor* original, int index, int size) : original(original), index(index), size(size)
{

}

gradientList SetSingleFunction::calculateGradient(Tensor& previousGradient) const {
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

std::vector<Tensor*> SetSingleFunction::getDependents() const {
	return { original };
}


SetTensorFunction::SetTensorFunction(Tensor* copyTo, Tensor* copyFrom, int index, int size, const std::vector<int>& broadcastShape,
	const std::vector<int>& broadcastedIndices) : copyTo(copyTo), copyFrom(copyFrom), index(index), size(size),
	broadcastShape(broadcastShape), broadcastedIndices(broadcastedIndices)
{

}

gradientList SetTensorFunction::calculateGradient(Tensor& previousGradient) const
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
			int j = broadcastedIndices[i];
			int k = index + i;
			gradientValues[broadcastedIndices[i]] += previousGradient.at(index + i);
		}
		list.push_back(gradientTuple{ copyTo, Tensor::fromValues(gradientValues, gradientShape) });
	}
	return list;
}

std::vector<Tensor*> SetTensorFunction::getDependents() const {
	return { copyTo, copyFrom};
}

AddSingleFunction::AddSingleFunction(Tensor* original) : original(original)
{

}

gradientList AddSingleFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		gradientValues[i] = previousGradient.at(i);
	}
	return gradientList{ gradientTuple{original, Tensor::fromValues(gradientValues, gradientShape)} };
}

std::vector<Tensor*> AddSingleFunction::getDependents() const {
	return { original };
}

AddTensorFunction::AddTensorFunction(Tensor* original1, Tensor* original2, const std::vector<int>& broadcastedIndices1,
	const std::vector<int>& broadcastedIndices2) : original1(original1), original2(original2),
	broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2)
{

}

gradientList AddTensorFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize1 = original1->getSize();
	const std::vector<int>& gradientShape1 = original1->getShape();
	float* gradientValues1 = new float[gradientSize1];
	for (int i = 0; i < gradientSize1; i++) gradientValues1[i] = 0;

	int gradientSize2 = original2->getSize();
	const std::vector<int>& gradientShape2 = original2->getShape();
	float* gradientValues2 = new float[gradientSize2];
	for (int i = 0; i < gradientSize2; i++) gradientValues2[i] = 0;

	for (int i = 0; i < previousGradient.getSize(); i++) {
		gradientValues1[broadcastedIndices1[i]] += previousGradient.at(i);
		gradientValues2[broadcastedIndices2[i]] += previousGradient.at(i);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
}

std::vector<Tensor*> AddTensorFunction::getDependents() const {
	return { original1, original2 };
}



SubtractSingleFunction::SubtractSingleFunction(Tensor* original) : original(original)
{

}

gradientList SubtractSingleFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		gradientValues[i] = previousGradient.at(i);
	}
	return gradientList{ gradientTuple{original, Tensor::fromValues(gradientValues, gradientShape)} };
}


std::vector<Tensor*> SubtractSingleFunction::getDependents() const {
	return { original };
}

SubtractTensorFunction::SubtractTensorFunction(Tensor* original1, Tensor* original2, const std::vector<int>& broadcastedIndices1,
	const std::vector<int>& broadcastedIndices2) : original1(original1), original2(original2),
	broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2)
{

}

gradientList SubtractTensorFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize1 = original1->getSize();
	const std::vector<int>& gradientShape1 = original1->getShape();
	float* gradientValues1 = new float[gradientSize1];
	for (int i = 0; i < gradientSize1; i++) gradientValues1[i] = 0;

	int gradientSize2 = original2->getSize();
	const std::vector<int>& gradientShape2 = original2->getShape();
	float* gradientValues2 = new float[gradientSize2];
	for (int i = 0; i < gradientSize2; i++) gradientValues2[i] = 0;

	for (int i = 0; i < previousGradient.getSize(); i++) {
		gradientValues1[broadcastedIndices1[i]] += previousGradient.at(i);
		gradientValues2[broadcastedIndices2[i]] -= previousGradient.at(i);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
}

std::vector<Tensor*> SubtractTensorFunction::getDependents() const {
	return { original1, original2 };
}



MultiplySingleFunction::MultiplySingleFunction(Tensor* original, float value) : original(original), value(value)
{

}

gradientList MultiplySingleFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		gradientValues[i] = previousGradient.at(i) * value;
	}
	return gradientList{ gradientTuple{original, Tensor::fromValues(gradientValues, gradientShape)} };
}

std::vector<Tensor*> MultiplySingleFunction::getDependents() const {
	return { original };
}


MultiplyTensorFunction::MultiplyTensorFunction(Tensor* original1, Tensor* original2, const std::vector<int>& broadcastedIndices1,
	const std::vector<int>& broadcastedIndices2) : original1(original1), original2(original2),
	broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2)
{

}

gradientList MultiplyTensorFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize1 = original1->getSize();
	const std::vector<int>& gradientShape1 = original1->getShape();
	float* gradientValues1 = new float[gradientSize1];
	for (int i = 0; i < gradientSize1; i++) gradientValues1[i] = 0;

	int gradientSize2 = original2->getSize();
	const std::vector<int>& gradientShape2 = original2->getShape();
	float* gradientValues2 = new float[gradientSize2];
	for (int i = 0; i < gradientSize2; i++) gradientValues2[i] = 0;

	for (int i = 0; i < previousGradient.getSize(); i++) {
		int index1 = broadcastedIndices1[i], index2 = broadcastedIndices2[i];
		gradientValues1[index1] += previousGradient.at(i) * original2->at(index2);
		gradientValues2[index2] += previousGradient.at(i) * original1->at(index1);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
}

std::vector<Tensor*> MultiplyTensorFunction::getDependents() const {
	return { original1, original2 };
}



DivideSingleFunction::DivideSingleFunction(Tensor* original, float value) : original(original), value(value)
{

}

gradientList DivideSingleFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		gradientValues[i] = previousGradient.at(i) / value;
	}
	return gradientList{ gradientTuple{original, Tensor::fromValues(gradientValues, gradientShape)} };
}

std::vector<Tensor*> DivideSingleFunction::getDependents() const {
	return { original };
}


DivideTensorFunction::DivideTensorFunction(Tensor* original1, Tensor* original2, const std::vector<int>& broadcastedIndices1,
	const std::vector<int>& broadcastedIndices2) : original1(original1), original2(original2),
	broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2)
{

}

gradientList DivideTensorFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize1 = original1->getSize();
	const std::vector<int>& gradientShape1 = original1->getShape();
	float* gradientValues1 = new float[gradientSize1];
	for (int i = 0; i < gradientSize1; i++) gradientValues1[i] = 0;

	int gradientSize2 = original2->getSize();
	const std::vector<int>& gradientShape2 = original2->getShape();
	float* gradientValues2 = new float[gradientSize2];
	for (int i = 0; i < gradientSize2; i++) gradientValues2[i] = 0;

	for (int i = 0; i < previousGradient.getSize(); i++) {
		int index1 = broadcastedIndices1[i], index2 = broadcastedIndices2[i];
		gradientValues1[index1] += previousGradient.at(i) / original2->at(index2);
		gradientValues2[index2] -= previousGradient.at(i) * original1->at(index1) / (original2->at(index2) * original2->at(index2));
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
}

std::vector<Tensor*> DivideTensorFunction::getDependents() const {
	return { original1, original2 };
}


TransposeFunction::TransposeFunction(Tensor* original, const std::vector<int>& transposeIndices) : original(original), transposeIndices(transposeIndices)
{

}

gradientList TransposeFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		gradientValues[i] = previousGradient.at(transposeIndices[i]);
	}
	return gradientList{ gradientTuple(original, Tensor::fromValues(gradientValues, gradientShape)) };
}

std::vector<Tensor*> TransposeFunction::getDependents() const {
	return { original };
}


MatrixMultiplicationFunction::MatrixMultiplicationFunction(Tensor* original1, Tensor* original2,
	const std::vector<int>& broadcastedIndices1, const std::vector<int>& broadcastedIndices2,
	int matrixWidth, int matrixInner, int matrixHeight) : original1(original1), original2(original2),
	broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2), matrixWidth(matrixWidth),
	matrixInner(matrixInner), matrixHeight(matrixHeight)
{

}

gradientList MatrixMultiplicationFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize1 = original1->getSize();
	const std::vector<int>& gradientShape1 = original1->getShape();
	float* gradientValues1 = new float[gradientSize1];
	for (int i = 0; i < gradientSize1; i++) gradientValues1[i] = 0;
	Tensor transpose1 = original2->detached().transpose();
	Tensor unbroadcastedGradient1 = Tensor::matrixMultiply(previousGradient, transpose1);
	int matrixSize1 = matrixWidth * matrixInner;
	for (int i = 0; i < broadcastedIndices1.size(); i++) {
		for (int j = 0; j < matrixSize1; j++) {
			int broadcastedIndex = broadcastedIndices1[i] * matrixSize1 + j;
			int unbroadcastedIndex = i * matrixSize1 + j;
			gradientValues1[broadcastedIndex] += unbroadcastedGradient1.at(unbroadcastedIndex);
		}
	}

	int gradientSize2 = original2->getSize();
	const std::vector<int>& gradientShape2 = original2->getShape();
	float* gradientValues2 = new float[gradientSize2];
	for (int i = 0; i < gradientSize2; i++) gradientValues2[i] = 0;
	Tensor transpose2 = original1->detached().transpose();
	Tensor unbroadcastedGradient2 = Tensor::matrixMultiply(transpose2, previousGradient);
	int matrixSize2 = matrixInner * matrixHeight;
	for (int i = 0; i < broadcastedIndices2.size(); i++) {
		for (int j = 0; j < matrixSize2; j++) {
			int broadcastedIndex = broadcastedIndices2[i] * matrixSize2 + j;
			int unbroadcastedIndex = i * matrixSize2 + j;
			gradientValues2[broadcastedIndex] += unbroadcastedGradient2.at(unbroadcastedIndex);
		}
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
}

std::vector<Tensor*> MatrixMultiplicationFunction::getDependents() const {
	return { original1, original2 };
}

MaxSingleFunction::MaxSingleFunction(Tensor* original, float value) : original(original), value(value)
{

}

gradientList MaxSingleFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		gradientValues[i] = original->at(i) >= value ? previousGradient.at(i) : 0;
	}
	return gradientList{ gradientTuple{original, Tensor::fromValues(gradientValues, gradientShape)} };
}

std::vector<Tensor*> MaxSingleFunction::getDependents() const {
	return { original };
}

MaxTensorFunction::MaxTensorFunction(Tensor* original1, Tensor* original2,
	const std::vector<int>& broadcastedIndices1, const std::vector<int>& broadcastedIndices2) : original1(original1), original2(original2),
	broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2)
{
}

gradientList MaxTensorFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize1 = original1->getSize();
	const std::vector<int>& gradientShape1 = original1->getShape();
	float* gradientValues1 = new float[gradientSize1];
	for (int i = 0; i < gradientSize1; i++) gradientValues1[i] = 0;

	int gradientSize2 = original2->getSize();
	const std::vector<int>& gradientShape2 = original2->getShape();
	float* gradientValues2 = new float[gradientSize2];
	for (int i = 0; i < gradientSize2; i++) gradientValues2[i] = 0;

	for (int i = 0; i < previousGradient.getSize(); i++) {
		int index1 = broadcastedIndices1[i], index2 = broadcastedIndices2[i];
		if (original1->at(index1) >= original2->at(index2)) gradientValues1[index1] += previousGradient.at(i);
		if (original2->at(index2) >= original1->at(index1)) gradientValues2[index2] += previousGradient.at(i);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
}

std::vector<Tensor*> MaxTensorFunction::getDependents() const {
	return { original1, original2 };
}


MinSingleFunction::MinSingleFunction(Tensor* original, float value) : original(original), value(value)
{
}

gradientList MinSingleFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize = original->getSize();
	const std::vector<int>& gradientShape = original->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) {
		gradientValues[i] = original->at(i) <= value ? previousGradient.at(i) : 0;
	}
	return gradientList{ gradientTuple{original, Tensor::fromValues(gradientValues, gradientShape)} };
}

std::vector<Tensor*> MinSingleFunction::getDependents() const {
	return { original };
}

MinTensorFunction::MinTensorFunction(Tensor* original1, Tensor* original2,
	const std::vector<int>& broadcastedIndices1, const std::vector<int>& broadcastedIndices2) : original1(original1), original2(original2),
	broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2)
{
}

gradientList MinTensorFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize1 = original1->getSize();
	const std::vector<int>& gradientShape1 = original1->getShape();
	float* gradientValues1 = new float[gradientSize1];
	for (int i = 0; i < gradientSize1; i++) gradientValues1[i] = 0;

	int gradientSize2 = original2->getSize();
	const std::vector<int>& gradientShape2 = original2->getShape();
	float* gradientValues2 = new float[gradientSize2];
	for (int i = 0; i < gradientSize2; i++) gradientValues2[i] = 0;

	for (int i = 0; i < previousGradient.getSize(); i++) {
		int index1 = broadcastedIndices1[i], index2 = broadcastedIndices2[i];
		if (original1->at(index1) <= original2->at(index2)) gradientValues1[index1] += previousGradient.at(i);
		if (original2->at(index2) <= original1->at(index1)) gradientValues2[index2] += previousGradient.at(i);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
}

std::vector<Tensor*> MinTensorFunction::getDependents() const {
	return { original1, original2 };
}


MeanSquaredErrorLossFunction::MeanSquaredErrorLossFunction(Tensor* original1, Tensor* original2, int broadcastedSize,
	const std::vector<int>& broadcastedIndices1, const std::vector<int>& broadcastedIndices2) : original1(original1), original2(original2),
	broadcastedSize(broadcastedSize), broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2)
{
}

gradientList MeanSquaredErrorLossFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize1 = original1->getSize();
	const std::vector<int>& gradientShape1 = original1->getShape();
	float* gradientValues1 = new float[gradientSize1];
	for (int i = 0; i < gradientSize1; i++) gradientValues1[i] = 0;

	int gradientSize2 = original2->getSize();
	const std::vector<int>& gradientShape2 = original2->getShape();
	float* gradientValues2 = new float[gradientSize2];
	for (int i = 0; i < gradientSize2; i++) gradientValues2[i] = 0;

	float coefficient = 2.0f / broadcastedSize;
	for (int i = 0; i < broadcastedSize; i++) {
		int index1 = broadcastedIndices1[i], index2 = broadcastedIndices2[i];
		gradientValues1[index1] += coefficient * previousGradient.item() * (original1->at(index1) - original2->at(index2));
		gradientValues2[index2] += coefficient * previousGradient.item() * (original2->at(index2) - original1->at(index1));
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
}

std::vector<Tensor*> MeanSquaredErrorLossFunction::getDependents() const {
	return { original1, original2 };
}

CategoricalCrossEntropyLossFunction::CategoricalCrossEntropyLossFunction(Tensor* original1, const Tensor* original2, float* softmaxValues,
	int finalDimSize, int broadcastedSize, const std::vector<int>& broadcastedIndices1, const std::vector<int>& broadcastedIndices2) :
	original1(original1), original2(original2), softmaxValues(softmaxValues), finalDimSize(finalDimSize),
	broadcastedSize(broadcastedSize), broadcastedIndices1(broadcastedIndices1), broadcastedIndices2(broadcastedIndices2)
{
}

gradientList CategoricalCrossEntropyLossFunction::calculateGradient(Tensor& previousGradient) const
{
	int gradientSize = original1->getSize();
	const std::vector<int>& gradientShape = original1->getShape();
	float* gradientValues = new float[gradientSize];
	for (int i = 0; i < gradientSize; i++) gradientValues[i] = 0;

	for (int i = 0; i < broadcastedSize; i++) {
		int index1 = broadcastedIndices1[i], index2 = broadcastedIndices2[i];
		float pred = softmaxValues[index1], truth = original2->at(index2);
		gradientValues[index1] += previousGradient.item() * (pred - truth);
	}
	for (int i = 0; i < gradientSize; i++) gradientValues[i] /= (gradientSize / finalDimSize);

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues, gradientShape))
	};
}

std::vector<Tensor*> CategoricalCrossEntropyLossFunction::getDependents() const {
	return { original1 };
}