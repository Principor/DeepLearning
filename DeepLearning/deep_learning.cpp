#include "deep_learning.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape) : shape(shape)
{
	for (int dim : shape) {
		if (dim < 1)
			throw std::invalid_argument("Length of all dimensions must be greater than or equal to 1.");
	}
}

const std::vector<int>& Tensor::getShape() {
	return shape;
}