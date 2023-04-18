#include <stdexcept>

#include "deep_learning.h"

Tensor::Tensor(const std::vector<int>& shape, int size, float* values) : shape(shape), size(size), values(values),
gradient(false), function(NULL)
{
}

Tensor::~Tensor()
{
	if (function != NULL) delete function;
}

const std::vector<int>& Tensor::getShape() const {
	return shape;
}

int Tensor::getSize() const {
	return size;
}

bool Tensor::getGradient() const {
	return gradient;
}

void Tensor::setGradient(bool gradient) {
	this->gradient = gradient;
}

float Tensor::item() const {
	if (size > 1) throw std::length_error("Item can only be used on tensors of size 1.");
	return values[0];
}

float Tensor::at(int index) const {
	return values[index];
}

const GradientFunction* Tensor::getFunction() const
{
	return function;
}

Tensor& Tensor::reshape(const std::vector<int>& shape) {
	int size = calculateSize(shape);
	if (this->size != size) {
		throw::std::length_error("New size does not match the current size.");
	}
	this->shape = shape;
	return *this;
}

Tensor Tensor::get(const std::vector<int>& indices) const {
	int index = getIndex(indices);
	std::vector<int> newShape(shape.begin() + indices.size(), shape.end());
	int newSize = calculateSize(newShape);
	float* newValues = new float[newSize];
	for (int i = 0; i < newSize; i++) {
		newValues[i] = values[i + index];
	}
	Tensor newTensor(newShape, newSize, newValues);
	if (gradient) {
		newTensor.gradient = true;
		newTensor.function = new GetFunction(this, index, newSize);
	}
	return newTensor;
}

Tensor Tensor::set(float value, const std::vector<int>& indices) {
	int index = getIndex(indices);

	std::vector<int> assignmentShape = getSubShape(shape, indices.size(), 0);
	int assignmentSize = 1;
	for (int dim : assignmentShape) assignmentSize *= dim;

	float* newValues = new float[size];
	for (int i = 0; i < size; i++) {
		if (i - index >= 0 && i - index < assignmentSize) newValues[i] = value;
		else newValues[i] = values[i];
	}

	Tensor newTensor(shape, size, newValues);
	if (gradient) {
		newTensor.gradient = true;
		newTensor.function = new SetSingleFunction(this, index, assignmentSize);
	}
	return newTensor;
}

Tensor Tensor::set(Tensor values, const std::vector<int>& indices)
{
	int index = getIndex(indices);

	return Tensor(shape, size, new float[size]);
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

Tensor Tensor::fromValues(float* values, const std::vector<int>& shape)
{
	return Tensor(shape, calculateSize(shape), values);
}

int Tensor::calculateSize(const std::vector<int>& shape) {
	int size = 1;
	for (int dim : shape) {
		if (dim < 1) throw std::invalid_argument("Length of all dimensions must be greater than or equal to 1.");
		size *= dim;
	}
	return size;
}

std::vector<int> Tensor::getSubShape(const std::vector<int>& shape, int frontRemoval, int endRemoval)
{
	return std::vector<int>(shape.begin() + frontRemoval, shape.end() - endRemoval);
}

int Tensor::getIndex(const std::vector<int>& indices) const {
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