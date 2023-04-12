#pragma once
#include <vector>

class Tensor {
private:
	std::vector<int> shape;
	int size;
public:
	Tensor(const std::vector<int>& shape);
	const std::vector<int>& getShape();
	int getSize();
};