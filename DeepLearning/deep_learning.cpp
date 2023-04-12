#include "deep_learning.h"

Tensor::Tensor(const std::vector<int>& shape) : shape(shape)
{
}

const std::vector<int>& Tensor::getShape() {
	return shape;
}