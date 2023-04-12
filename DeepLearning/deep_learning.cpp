#include "deep_learning.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape) : shape(shape)
{
	size = 1;
	for (int dim : shape) {
		if (dim < 1)
			throw std::invalid_argument("Length of all dimensions must be greater than or equal to 1.");
		size *= dim;
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