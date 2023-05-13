#include <stdexcept>
#include <functional>
#include <random>

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

bool Tensor::requiresGradient() const {
	return gradient;
}

Tensor& Tensor::requireGradient() {
	this->gradient = true;
	return *this;
}

float Tensor::item() const {
	if (size > 1) throw std::length_error("Item can only be used on tensors of size 1.");
	return values[0];
}

float Tensor::at(int index) const {
	if (index < 0 || index >= size) throw std::out_of_range("Index must be within the range of the values.");
	return values[index];
}

float Tensor::at(const std::vector<int>& indices) const {
	if (indices.size() < shape.size()) throw std::length_error("Number of indices cannot be less than number of dimensions.");
	return values[getIndex(indices)];
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

Tensor Tensor::detached() const
{
	float* newValues = new float[size];
	for (int i = 0; i < size; i++)
	{
		newValues[i] = values[i];
	}
	return Tensor(shape, size, newValues);
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
	if (gradient || values.gradient) {
		newTensor.gradient = true;
		newTensor.function = new SetTensorFunction(this, &values, index, assignmentSize, broadcastedShape, broadcastedIndices);
	}
	return newTensor;
}

Tensor Tensor::transpose() {
	int numDims = shape.size();

	if (numDims < 2) {
		throw std::length_error("Must have at least 2 dimensions to transpose");
	}

	int newSize = size;
	auto newShape = getSubShape(shape, 0, 2);
	newShape.push_back(shape[numDims - 1]);
	newShape.push_back(shape[numDims - 2]);
	int transposeSize = shape[numDims - 1] * shape[numDims - 2];

	float* newValues = new float[newSize];
	std::vector<int> transposeIndices(newSize);

	for (int i = 0; i < size; i += transposeSize)
	{
		for (int x = 0; x < shape[numDims - 1]; x++)
		{
			for (int y = 0; y < shape[numDims - 2]; y++)
			{
				int index1 = i + x + y * shape[numDims - 1];
				int index2 = i + x * shape[numDims - 2] + y;
#pragma warning(disable : 6386)
				newValues[index2] = values[index1];
				transposeIndices[index1] = index2;
			}
		}
	}

	Tensor newTensor(newShape, newSize, newValues);
	if (gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new TransposeFunction(this, transposeIndices);
	}
	return newTensor;
}

Tensor Tensor::add(Tensor& input, float value) {
	float* other = new float[input.size];
	for (int i = 0; i < input.size; i++) {
		other[i] = input.values[i] + value;
	}
	Tensor newTensor(input.shape, input.size, other);
	if (input.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new AddSingleFunction(&input);
	}
	return newTensor;
}

Tensor Tensor::add(Tensor& input, Tensor& other) {
	auto broadcastedShape = broadcastShapes(input.shape, other.shape);
	int broadcastedSize = calculateSize(broadcastedShape);
	auto broadcastedIndices1 = broadcastIndices(input.shape, broadcastedShape);
	auto broadcastedIndices2 = broadcastIndices(other.shape, broadcastedShape);

	float* newValues = new float[broadcastedSize];

	for (int i = 0; i < broadcastedSize; i++) {
		newValues[i] = input.values[broadcastedIndices1[i]] + other.values[broadcastedIndices2[i]];
	}

	Tensor newTensor(broadcastedShape, broadcastedSize, newValues);
	if (input.gradient || other.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new AddTensorFunction(&input, &other, broadcastedIndices1, broadcastedIndices2);
	}
	return newTensor;
}

Tensor Tensor::subtract(Tensor& input, float value) {
	float* other = new float[input.size];
	for (int i = 0; i < input.size; i++) {
		other[i] = input.values[i] - value;
	}
	Tensor newTensor(input.shape, input.size, other);
	if (input.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new SubtractSingleFunction(&input);
	}
	return newTensor;
}

Tensor Tensor::subtract(Tensor& input, Tensor& other) {
	auto broadcastedShape = broadcastShapes(input.shape, other.shape);
	int broadcastedSize = calculateSize(broadcastedShape);
	auto broadcastedIndices1 = broadcastIndices(input.shape, broadcastedShape);
	auto broadcastedIndices2 = broadcastIndices(other.shape, broadcastedShape);

	float* newValues = new float[broadcastedSize];

	for (int i = 0; i < broadcastedSize; i++) {
		newValues[i] = input.values[broadcastedIndices1[i]] - other.values[broadcastedIndices2[i]];
	}

	Tensor newTensor(broadcastedShape, broadcastedSize, newValues);
	if (input.gradient || other.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new SubtractTensorFunction(&input, &other, broadcastedIndices1, broadcastedIndices2);
	}
	return newTensor;
}

Tensor Tensor::multiply(Tensor& input, float value) {
	float* other = new float[input.size];
	for (int i = 0; i < input.size; i++) {
		other[i] = input.values[i] * value;
	}
	Tensor newTensor(input.shape, input.size, other);
	if (input.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new MultiplySingleFunction(&input, value);
	}
	return newTensor;
}

Tensor Tensor::multiply(Tensor& input, Tensor& other) {
	auto broadcastedShape = broadcastShapes(input.shape, other.shape);
	int broadcastedSize = calculateSize(broadcastedShape);
	auto broadcastedIndices1 = broadcastIndices(input.shape, broadcastedShape);
	auto broadcastedIndices2 = broadcastIndices(other.shape, broadcastedShape);

	float* newValues = new float[broadcastedSize];

	for (int i = 0; i < broadcastedSize; i++) {
		newValues[i] = input.values[broadcastedIndices1[i]] * other.values[broadcastedIndices2[i]];
	}

	Tensor newTensor(broadcastedShape, broadcastedSize, newValues);
	if (input.gradient || other.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new MultiplyTensorFunction(&input, &other, broadcastedIndices1, broadcastedIndices2);
	}
	return newTensor;
}

Tensor Tensor::divide(Tensor& input, float value) {
	float* other = new float[input.size];
	for (int i = 0; i < input.size; i++) {
		other[i] = input.values[i] / value;
	}
	Tensor newTensor(input.shape, input.size, other);
	if (input.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new DivideSingleFunction(&input, value);
	}
	return newTensor;
}

Tensor Tensor::divide(Tensor& input, Tensor& other) {
	auto broadcastedShape = broadcastShapes(input.shape, other.shape);
	int broadcastedSize = calculateSize(broadcastedShape);
	auto broadcastedIndices1 = broadcastIndices(input.shape, broadcastedShape);
	auto broadcastedIndices2 = broadcastIndices(other.shape, broadcastedShape);

	float* newValues = new float[broadcastedSize];

	for (int i = 0; i < broadcastedSize; i++) {
		newValues[i] = input.values[broadcastedIndices1[i]] / other.values[broadcastedIndices2[i]];
	}

	Tensor newTensor(broadcastedShape, broadcastedSize, newValues);
	if (input.gradient || other.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new DivideTensorFunction(&input, &other, broadcastedIndices1, broadcastedIndices2);
	}
	return newTensor;
}

Tensor Tensor::matrixMultiply(Tensor& input, Tensor& other)
{
	if (input.shape.size() < 2 || other.shape.size() < 2)
		throw std::length_error("Tensors must have at least 2 dims for matrix multiplication.");

	std::vector<int> matrixShape1 = getSubShape(input.shape, input.shape.size() - 2, 0);
	std::vector<int> matrixShape2 = getSubShape(other.shape, other.shape.size() - 2, 0);

	if (matrixShape1[1] != matrixShape2[0])
		throw std::invalid_argument("Inner dimensions of matrixes must match.");

	std::vector<int> beforeShape1 = getSubShape(input.shape, 0, 2);
	std::vector<int> beforeShape2 = getSubShape(other.shape, 0, 2);

	std::vector<int> broadcastedShape = broadcastShapes(beforeShape1, beforeShape2);
	std::vector<int> broadcastedIndices1 = broadcastIndices(beforeShape1, broadcastedShape);
	std::vector<int> broadcastedIndices2 = broadcastIndices(beforeShape2, broadcastedShape);

	int matrixWidth = matrixShape1[0];
	int matrixInner = matrixShape1[1];
	int matrixHeight = matrixShape2[1];

	std::vector<int> newShape(broadcastedShape);
	newShape.push_back(matrixWidth);
	newShape.push_back(matrixHeight);
	int newSize = calculateSize(newShape);
	float* newValues = new float[newSize];

	int broadcastedSize = calculateSize(broadcastedShape);
	for (int i = 0; i < broadcastedSize; i++) {
		int startIndex1 = broadcastedIndices1[i] * matrixWidth * matrixInner;
		int startIndex2 = broadcastedIndices2[i] * matrixInner * matrixHeight;
		for (int x = 0; x < matrixWidth; x++) {
			for (int y = 0; y < matrixHeight; y++) {
				int sum = 0;
				for (int j = 0; j < matrixInner; j++) {
					sum += input.values[startIndex1 + x * matrixInner + j] * other.values[startIndex2 + j * matrixHeight + y];
				}
				newValues[i * matrixWidth * matrixHeight + x * matrixHeight + y] = sum;
			}
		}
	}

	Tensor newTensor(newShape, newSize, newValues);
	if (input.gradient || other.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new MatrixMultiplicationFunction(
			&input, &other, broadcastedIndices1, broadcastedIndices2, matrixWidth, matrixInner, matrixHeight
		);
	}
	return newTensor;
}

Tensor Tensor::max(Tensor& input, float value) {
	float* other = new float[input.size];
	for (int i = 0; i < input.size; i++) {
		other[i] = std::max(input.values[i], value);
	}
	Tensor newTensor(input.shape, input.size, other);
	if (input.gradient)
	{
		newTensor.gradient = true;
		newTensor.function = new MaxSingleFunction(&input, value);
	}
	return newTensor;
}

Tensor Tensor::max(Tensor& input, Tensor& other)
{
	auto broadcastedShape = broadcastShapes(input.shape, other.shape);
	int broadcastedSize = calculateSize(broadcastedShape);
	auto broadcastedIndices1 = broadcastIndices(input.shape, broadcastedShape);
	auto broadcastedIndices2 = broadcastIndices(other.shape, broadcastedShape);


	float* newValues = new float[broadcastedSize];

	Tensor newTensor(broadcastedShape, broadcastedSize, newValues);
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

Tensor Tensor::range(const std::vector<int>& shape, float start, float step)
{
	int size = calculateSize(shape);
	float* values = new float[size];
	for (int i = 0; i < size; i++) {
		values[i] = start + i * step;
	}
	return Tensor(shape, size, values);
}

Tensor Tensor::fromValues(float* values, const std::vector<int>& shape)
{
	return Tensor(shape, calculateSize(shape), values);
}

Tensor Tensor::uniform(const std::vector<int>& shape, float min, float max)
{
	int size = calculateSize(shape);
	float* values = new float[size];
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(min, max);
	for (int i = 0; i < size; i++)
	{
		values[i] = dist(gen);
	}
	return Tensor(shape, size, values);
}

Tensor Tensor::normal(const std::vector<int>& shape, float mean, float std)
{
	int size = calculateSize(shape);
	float* values = new float[size];
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> dist(mean, std);
	for (int i = 0; i < size; i++)
	{
		values[i] = dist(gen);
	}
	return Tensor(shape, size, values);
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

std::vector<int> Tensor::broadcastIndices(std::vector<int> originalShape, const std::vector<int>& broadcastedShape)
{
	while (broadcastedShape.size() > originalShape.size()) originalShape.insert(originalShape.begin(), 1);
	std::vector<int> originalShapeStrides(originalShape.size()), broadcastedShapeStrides(originalShape.size());
	{
		int originalShapeStride = 1, broadcastedShapeStride = 1;
		for (int i = originalShape.size() - 1; i >= 0; i--) {
			if (originalShape[i] == 1) broadcastedShapeStrides[i] = 0;
			else broadcastedShapeStrides[i] = originalShapeStride;
			originalShapeStrides[i] = broadcastedShapeStride;
			originalShapeStride *= originalShape[i];
			broadcastedShapeStride *= broadcastedShape[i];
		}
	}

	std::vector<int> broadcastedIndices(calculateSize(broadcastedShape));

	std::function<void(int, int, int)> recursiveIterate = [&](int depth, int trueIndex, int broadcastedIndex) {
		if (depth == originalShapeStrides.size()) {
			broadcastedIndices[trueIndex] = broadcastedIndex;
			return;
		}
		for (int i = 0; i < broadcastedShape[depth]; i++) {
			recursiveIterate(depth + 1,
				trueIndex + i * originalShapeStrides[depth],
				broadcastedIndex + i * broadcastedShapeStrides[depth]);
		}
	};

	if (originalShape.size() == 0) broadcastedIndices[0] = 0;
	else recursiveIterate(0, 0, 0);

	return broadcastedIndices;
}