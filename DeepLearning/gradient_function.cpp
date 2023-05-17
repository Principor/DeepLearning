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


SetTensorFunction::SetTensorFunction(Tensor* copyTo, Tensor* copyFrom, int index, int size, const std::vector<int>& broadcastShape,
	const std::vector<int>& broadcastIndices) : copyTo(copyTo), copyFrom(copyFrom), index(index), size(size),
	broadcastShape(broadcastShape), broadcastIndices(broadcastIndices)
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
			int j = broadcastIndices[i];
			int k = index + i;
			gradientValues[broadcastIndices[i]] += previousGradient.at(index + i);
		}
		list.push_back(gradientTuple{ copyTo, Tensor::fromValues(gradientValues, gradientShape) });
	}
	return list;
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


AddTensorFunction::AddTensorFunction(Tensor* original1, Tensor* original2, const std::vector<int>& broadcastIndices1,
	const std::vector<int>& broadcastIndices2) : original1(original1), original2(original2),
	broadcastIndices1(broadcastIndices1), broadcastIndices2(broadcastIndices2)
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
		gradientValues1[broadcastIndices1[i]] += previousGradient.at(i);
		gradientValues2[broadcastIndices2[i]] += previousGradient.at(i);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
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


SubtractTensorFunction::SubtractTensorFunction(Tensor* original1, Tensor* original2, const std::vector<int>& broadcastIndices1,
	const std::vector<int>& broadcastIndices2) : original1(original1), original2(original2),
	broadcastIndices1(broadcastIndices1), broadcastIndices2(broadcastIndices2)
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
		gradientValues1[broadcastIndices1[i]] += previousGradient.at(i);
		gradientValues2[broadcastIndices2[i]] -= previousGradient.at(i);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
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


MultiplyTensorFunction::MultiplyTensorFunction(Tensor* original1, Tensor* original2, const std::vector<int>& broadcastIndices1,
	const std::vector<int>& broadcastIndices2) : original1(original1), original2(original2),
	broadcastIndices1(broadcastIndices1), broadcastIndices2(broadcastIndices2)
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
		int index1 = broadcastIndices1[i], index2 = broadcastIndices2[i];
		gradientValues1[index1] += previousGradient.at(i) * original2->at(index2);
		gradientValues2[index2] += previousGradient.at(i) * original1->at(index1);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
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


DivideTensorFunction::DivideTensorFunction(Tensor* original1, Tensor* original2, const std::vector<int>& broadcastIndices1,
	const std::vector<int>& broadcastIndices2) : original1(original1), original2(original2),
	broadcastIndices1(broadcastIndices1), broadcastIndices2(broadcastIndices2)
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
		int index1 = broadcastIndices1[i], index2 = broadcastIndices2[i];
		gradientValues1[index1] += previousGradient.at(i) / original2->at(index2);
		gradientValues2[index2] -= previousGradient.at(i) * original1->at(index1) / (original2->at(index2) * original2->at(index2));
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
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

MatrixMultiplicationFunction::MatrixMultiplicationFunction(Tensor* original1, Tensor* original2,
	const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2,
	int matrixWidth, int matrixInner, int matrixHeight) : original1(original1), original2(original2),
	broadcastIndices1(broadcastIndices1), broadcastIndices2(broadcastIndices2), matrixWidth(matrixWidth),
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
	for (int i = 0; i < broadcastIndices1.size(); i++) {
		for (int j = 0; j < matrixSize1; j++) {
			int broadcastedIndex = broadcastIndices1[i] * matrixSize1 + j;
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
	for (int i = 0; i < broadcastIndices2.size(); i++) {
		for (int j = 0; j < matrixSize2; j++) {
			int broadcastedIndex = broadcastIndices2[i] * matrixSize2 + j;
			int unbroadcastedIndex = i * matrixSize2 + j;
			gradientValues2[broadcastedIndex] += unbroadcastedGradient2.at(unbroadcastedIndex);
		}
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
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

MaxTensorFunction::MaxTensorFunction(Tensor* original1, Tensor* original2,
	const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2) : original1(original1), original2(original2),
	broadcastIndices1(broadcastIndices1), broadcastIndices2(broadcastIndices2)
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
		int index1 = broadcastIndices1[i], index2 = broadcastIndices2[i];
		if (original1->at(index1) >= original2->at(index2)) gradientValues1[index1] += previousGradient.at(i);
		if (original2->at(index2) >= original1->at(index1)) gradientValues2[index2] += previousGradient.at(i);
	}

	return gradientList{
		gradientTuple(original1, Tensor::fromValues(gradientValues1, gradientShape1)),
		gradientTuple(original2, Tensor::fromValues(gradientValues2, gradientShape2))
	};
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

MinTensorFunction::MinTensorFunction(Tensor* original1, Tensor* original2,
	const std::vector<int>& broadcastIndices1, const std::vector<int>& broadcastIndices2) : original1(original1), original2(original2),
	broadcastIndices1(broadcastIndices1), broadcastIndices2(broadcastIndices2)
{
}

gradientList MinTensorFunction::calculateGradient(Tensor& previousGradient) const
{
	return gradientList{};
}