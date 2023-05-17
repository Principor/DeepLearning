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

	Tensor detached() const;

	Tensor get(const std::vector<int>& indices);
	Tensor set(float value, const std::vector<int>& indices = {});
	Tensor set(Tensor& values, const std::vector<int>& indices = {});
	Tensor transpose();

	static Tensor add(Tensor& input, float other);
	static Tensor add(Tensor& input, Tensor& other);
	static Tensor subtract(Tensor& input, float value);
	static Tensor subtract(Tensor& input, Tensor& other);
	static Tensor multiply(Tensor& input, float value);
	static Tensor multiply(Tensor& input, Tensor& other);
	static Tensor divide(Tensor& input, float value);
	static Tensor divide(Tensor& input, Tensor& other);
	static Tensor matrixMultiply(Tensor& input, Tensor& other);
	static Tensor max(Tensor& input, float other);
	static Tensor max(Tensor& input, Tensor& other);
	static Tensor min(Tensor& input, float other);

	static Tensor zeroes(const std::vector<int>& shape);
	static Tensor ones(const std::vector<int>& shape);
	static Tensor full(const std::vector<int>& shape, float value);
	static Tensor range(const std::vector<int>& shape, float start=0, float step=1);
	static Tensor uniform(const std::vector<int>& shape, float min, float max);
	static Tensor normal(const std::vector<int>& shape, float mean, float std);

	static Tensor fromValues(float* values, const std::vector<int>& shape);
};