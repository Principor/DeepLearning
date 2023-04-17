#pragma 
#include <vector>

#include "gradient_function.h"

class Tensor {
private:
	std::vector<int> shape;
	int size;
	float* values;
	bool gradient;
	GradientFunction* function;

	Tensor(const std::vector<int>& shape, int size, float* values);

	int getIndex(const std::vector<int>& indices) const;

	static int calculateSize(const std::vector<int>& shape);
public:
	~Tensor();

	const std::vector<int>& getShape() const;
	int getSize() const;
	bool getGradient() const;
	void setGradient(bool gradient);
	float item() const;
	float at(int index) const;
	const GradientFunction* getFunction() const;

	Tensor& reshape(const std::vector<int>& shape);
	Tensor get(const std::vector<int>& indices) const;
	Tensor& set(const std::vector<int>& indices, float value);
	Tensor& set(float value);

	static Tensor zeroes(const std::vector<int>& shape);
	static Tensor ones(const std::vector<int>& shape);
	static Tensor full(const std::vector<int>& shape, float value);

	static Tensor fromValues(float* values, const std::vector<int>& shape);
};