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
	static std::vector<int> broadcastShapes(std::vector<int> shape0, std::vector<int> shape1, bool oneWay = false);
	static std::vector<int> broadcastIndices(std::vector<int> originalShape, const std::vector<int>& broadcastedShape);
public:
	~Tensor();

	const std::vector<int>& getShape() const;
	int getSize() const;
	bool requiresGradient() const;
	Tensor& requireGradient();
	float item() const;
	float at(int index) const;
	float at(const std::vector<int>& indices) const;
	const GradientFunction* getFunction() const;

	Tensor& reshape(const std::vector<int>& shape);

	Tensor get(const std::vector<int>& indices);
	Tensor set(float value, const std::vector<int>& indices = {});
	Tensor set(Tensor& values, const std::vector<int>& indices = {});
	Tensor add(float value);
	Tensor add(Tensor& values);
	Tensor subtract(float value);
	Tensor subtract(Tensor& values);
	Tensor multiply(float value);
	Tensor multiply(Tensor& values);
	Tensor divide(float value);
	Tensor divide(Tensor& values);
	Tensor transpose();
	Tensor matrixMultiply(Tensor& other);

	static Tensor zeroes(const std::vector<int>& shape);
	static Tensor ones(const std::vector<int>& shape);
	static Tensor full(const std::vector<int>& shape, float value);
	static Tensor range(const std::vector<int>& shape, float start=0, float step=1);

	static Tensor fromValues(float* values, const std::vector<int>& shape);
};