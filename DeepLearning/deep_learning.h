#pragma once
#include <vector>

class Tensor {
private:
	std::vector<int> shape;
	int size;
	float* values;
	bool gradient;

	Tensor(const std::vector<int>& shape, int size, float* values);

	int getIndex(const std::vector<int>& indices);

	static int calculateSize(const std::vector<int>& shape);
public:
	const std::vector<int>& getShape();
	int getSize();
	bool getGradient();
	void setGradient(bool gradient);
	float item();
	float at(int index);

	Tensor& reshape(const std::vector<int>& shape);
	Tensor get(const std::vector<int>& indices);
	Tensor& set(const std::vector<int>& indices, float value);
	Tensor& set(float value);

	static Tensor zeroes(const std::vector<int>& shape);
	static Tensor ones(const std::vector<int>& shape);
	static Tensor full(const std::vector<int>& shape, float value);

	static Tensor fromValues(float* values, const std::vector<int>& shape);
};