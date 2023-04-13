#pragma once
#include <vector>

class Tensor {
private:
	std::vector<int> shape;
	int size;
	float* values;

	Tensor(const std::vector<int>& shape, int size, float* values);

	int getIndex(const std::vector<int>& indices);

	static int calculateSize(const std::vector<int>& shape);
public:
	const std::vector<int>& getShape();
	int getSize();
	float item();

	Tensor& reshape(const std::vector<int>& shape);
	Tensor get(const std::vector<int>& indices);
	Tensor& set(const std::vector<int>& indices, float value);
	Tensor& set(float value);

	static Tensor zeroes(const std::vector<int>& shape);
	static Tensor full(const std::vector<int>& shape, float value);
};