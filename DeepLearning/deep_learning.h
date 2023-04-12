#pragma once
#include <vector>

class Tensor {
private:
	std::vector<int> shape;
public:
	Tensor(const std::vector<int>& shape);
	const std::vector<int>& getShape();
};