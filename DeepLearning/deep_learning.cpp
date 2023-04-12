#include "deep_learning.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape) : shape(shape)
{
	if (!validateShape(shape, size))
	{
		throw std::invalid_argument("Length of all dimensions must be greater than or equal to 1.");
	}
}

const std::vector<int>& Tensor::getShape() {
	return shape;
}

int Tensor::getSize() {
	return size;
}

float Tensor::item() {
	if (size > 1) throw std::length_error("Item can only be used on tensors of size 1.");
	return 0;
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
	if (indices.size() > shape.size()) throw std::length_error("Number of indices cannot be greater than number of dimensions.");
	for (int i = 0; i < indices.size(); i++) {
		if (indices[i] >= shape[i] || indices[i] < 0) throw std::out_of_range("Index is not in the range of the tensor.");
	}
	std::vector<int> newShape(shape.begin() + indices.size(), shape.end());
	return Tensor(newShape);
}