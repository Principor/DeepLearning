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
	static std::vector<int> getSubShape(const std::vector<int>& shape, int frontRemoval, int endRemoval);
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
	Tensor set(float value, const std::vector<int>& indices = {});
	Tensor set(Tensor values, const std::vector<int>& indices = {});

	static Tensor zeroes(const std::vector<int>& shape);
	static Tensor ones(const std::vector<int>& shape);
	static Tensor full(const std::vector<int>& shape, float value);

	static Tensor fromValues(float* values, const std::vector<int>& shape);
};