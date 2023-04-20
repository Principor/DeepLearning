#include <stdexcept>

#include "deep_learning.h"
#include <functional>

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

Tensor Tensor::get(const std::vector<int>& indices) {
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
	int assignmentSize = calculateSize(assignmentShape);

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

Tensor Tensor::set(Tensor& values, const std::vector<int>& indices)
{
	int index = getIndex(indices);

	std::vector<int> assignmentShape = getSubShape(shape, indices.size(), 0);
	int assignmentSize = calculateSize(assignmentShape);

	std::vector<int> broadcastedShape = broadcastShapes(values.shape, assignmentShape, true);
	auto broadcastedIndices = broadcastIndices(values.shape, broadcastedShape);

	float* newValues = new float[size];
	for (int i = 0; i < size; i++) {
		int assignmentIndex = i - index;
		if (assignmentIndex >= 0 && assignmentIndex < assignmentSize) {
			newValues[i] = values.at(broadcastedIndices[assignmentIndex]);
		}
		else {
			newValues[i] = this->values[i];
		}
	}

	Tensor newTensor(shape, size, newValues);
	if (gradient) {
		newTensor.gradient = true;
		newTensor.function = new SetTensorFunction(this, &values, index, assignmentSize, broadcastedShape, broadcastedIndices);
	}
	return newTensor;
}

Tensor Tensor::add(float value) {
	float* values = new float[size];
	for (int i = 0; i < size; i++) {
		values[i] = this->values[i] + value;
	}
	Tensor newTensor(shape, size, values);
	if (gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new AddSingleFunction(this);
	}
	return newTensor;
}

Tensor Tensor::add(Tensor& values) {
	auto broadcastedShape = broadcastShapes(shape, values.shape);
	int broadcastedSize = calculateSize(broadcastedShape);
	auto broadcastedIndices1 = broadcastIndices(shape, broadcastedShape);
	auto broadcastedIndices2 = broadcastIndices(values.shape, broadcastedShape);
	
	float* newValues = new float[broadcastedSize];
	
	for (int i = 0; i < broadcastedSize; i++) {
		newValues[i] = this->values[broadcastedIndices1[i]] + values.values[broadcastedIndices2[i]];
	}

	Tensor newTensor(broadcastedShape, broadcastedSize, newValues);
	if (gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new AddTensorFunction(this, &values, broadcastedIndices1, broadcastedIndices2);
	}
	return newTensor;
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


std::vector<int> Tensor::broadcastShapes(std::vector<int> shape1, std::vector<int> shape2, bool oneWay) {
	while (shape1.size() > shape2.size()) shape2.insert(shape2.begin(), 1);
	while (shape1.size() < shape2.size()) shape1.insert(shape1.begin(), 1);
	int dims = shape1.size();
	for (int i = 0; i < dims; i++) {
		if (shape1[i] == shape2[i]) continue;
		if (shape1[i] > shape2[i] && oneWay) throw std::invalid_argument("Shape 1 cannot have dimensions larger than shape 2.");
		if (shape1[i] == 1) shape1[i] = shape2[i];
		else if (shape2[i] != 1) throw std::invalid_argument("Smaller dimension must have a size of 1.");
	}
	return shape1;
}

std::vector<int> Tensor::broadcastIndices(const std::vector<int>& originalShape, const std::vector<int>& broadcastedShape)
{
	std::vector<int> trueStrides(originalShape.size()), broadcastStrides(originalShape.size());
	{
		int stride = 1;
		for (int i = originalShape.size() - 1; i >= 0; i--) {
			if (originalShape[i] == 1) broadcastStrides[i] = 0;
			else broadcastStrides[i] = stride;
			trueStrides[i] = stride;
			stride *= originalShape[i];
		}
	}

	std::vector<int> broadcastedIndices(calculateSize(broadcastedShape));

	std::function<void(int, int, int)> recursiveIterate = [&](int depth, int trueIndex, int broadcastedIndex) {
		if (depth == trueStrides.size()) {
			broadcastedIndices[trueIndex] = broadcastedIndex;
			return;
		}
		for (int i = 0; i < broadcastedShape[depth]; i++) {
			recursiveIterate(depth + 1, trueIndex + i * trueStrides[depth], broadcastedIndex + i * broadcastStrides[depth]);
		}
	};

	if (originalShape.size() == 0) broadcastedIndices[0] = 0;
	else recursiveIterate(0, 0, 0);

	return broadcastedIndices;
}