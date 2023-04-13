#pragma once
#include <vector>

class Tensor {
private:
	std::vector<int> shape;
	int size;
	float* values;

	bool validateShape(const std::vector<int>& shape, int& size);
	int getIndex(const std::vector<int>& indices);
public:
	Tensor(const std::vector<int>& shape, float* values = NULL);
	const std::vector<int>& getShape();
	int getSize();
	float item();
	Tensor& reshape(const std::vector<int>& shape);
	Tensor get(const std::vector<int>& indices);
	Tensor& set(const std::vector<int>& indices, float value);
	Tensor& set(float value);
};