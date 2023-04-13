#include "deep_learning.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape, float* values) : shape(shape)
{
	if (!validateShape(shape, size))
	{
		throw std::invalid_argument("Length of all dimensions must be greater than or equal to 1.");
	}
	if (values == NULL) this->values = new float[size];
	else this->values = values;
}

const std::vector<int>& Tensor::getShape() {
	return shape;
}

int Tensor::getSize() {
	return size;
}

float Tensor::item() {
	if (size > 1) throw std::length_error("Item can only be used on tensors of size 1.");
	return values[0];
}

Tensor& Tensor::reshape(const std::vector<int>& shape) {
	int size;
	if (!validateShape(shape, size))
	{
		throw std::invalid_argument("Length of all dimensions must be greater than or equal to 1.");
	}
	else if (this->size != size) {
		throw::std::length_error("New size does not match the current size.");
	}
	this->shape = shape;
	return *this;
}

bool Tensor::validateShape(const std::vector<int>& shape, int& size) {
	size = 1;
	bool valid = true;
	for (int dim : shape) {
		if (dim < 1)
			valid = false;
		size *= dim;
	}
	return valid;
}

Tensor Tensor::get(const std::vector<int>& indices) {
	int index = getIndex(indices);
	std::vector<int> newShape(shape.begin() + indices.size(), shape.end());
	return Tensor(newShape, values + index);
}

Tensor& Tensor::set(float value) {
	set({}, value);
}

Tensor& Tensor::set(const std::vector<int>& indices, float value) {
	int index = getIndex(indices);

	std::vector<int> assignmentShape(shape.begin() + indices.size(), shape.end());
	int assignmentSize = 1;
	for (int dim : assignmentShape) assignmentSize *= dim;

	for (int i = index; i < index + assignmentSize; i++) {
		values[i] = value;
	}
}

int Tensor::getIndex(const std::vector<int>& indices) {
	//Ensure indices are valid
	if (indices.size() > shape.size()) throw std::length_error("Number of indices cannot be greater than number of dimensions.");
	for (int i = 0; i < indices.size(); i++) {
		if (indices[i] >= shape[i] || indices[i] < 0) throw std::out_of_range("Index is not in the range of the tensor.");
	}

	//Calculate starting index
	std::vector<int> fullIndices(indices);
	fullIndices.resize(shape.size());
	int index = 0;
	int stepSize = 1;
	for (int i = shape.size() - 1; i >= 0; i--) {
		index += fullIndices[i] * stepSize;
		stepSize *= shape[i];
	}

	return index;
}