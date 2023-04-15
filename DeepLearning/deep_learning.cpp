#include "deep_learning.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape, int size, float* values) : shape(shape), size(size), values(values)
{
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
	int size = calculateSize(shape);
	if (this->size != size) {
		throw::std::length_error("New size does not match the current size.");
	}
	this->shape = shape;
	return *this;
}

int Tensor::calculateSize(const std::vector<int>& shape) {
	int size = 1;
	for (int dim : shape) {
		if (dim < 1) throw std::invalid_argument("Length of all dimensions must be greater than or equal to 1.");
		size *= dim;
	}
	return size;
}

Tensor Tensor::get(const std::vector<int>& indices) {
	int index = getIndex(indices);
	std::vector<int> newShape(shape.begin() + indices.size(), shape.end());
	int newSize = calculateSize(newShape);
	float* newValues = new float[newSize];
	for (int i = 0; i < newSize; i++) {
		newValues[i] = values[i + index];
	}
	return Tensor(newShape, newSize, newValues);
}

Tensor& Tensor::set(float value) {
	return set({}, value);
}

Tensor& Tensor::set(const std::vector<int>& indices, float value) {
	int index = getIndex(indices);

	std::vector<int> assignmentShape(shape.begin() + indices.size(), shape.end());
	int assignmentSize = 1;
	for (int dim : assignmentShape) assignmentSize *= dim;

	for (int i = index; i < index + assignmentSize; i++) {
		values[i] = value;
	}
	return *this;
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

Tensor Tensor::zeroes(const std::vector<int>& shape) {
	return full(shape, 0.0f);
}

Tensor Tensor::ones(const std::vector<int>& shape) {
	return full(shape, 1.0f);
}

Tensor Tensor::full(const std::vector<int>& shape, float value)
{
	int size = calculateSize(shape);
	float* values = new float[size];
	for (int i = 0; i < size; i++) {
		values[i] = value;
	}
	return Tensor(shape, size, values);
}
