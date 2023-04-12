#pragma once
#include <vector>

class Tensor {
private:
	std::vector<int> shape;
	int size;

	bool validateShape(const std::vector<int>& shape, int& size);
public:
	Tensor(const std::vector<int>& shape);
	const std::vector<int>& getShape();
	int getSize();
	float item();
	Tensor& reshape(const std::vector<int>& shape);
	Tensor get(const std::vector<int>& indices);
};