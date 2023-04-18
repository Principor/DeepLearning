#include "pch.h"
#include "deep_learning.h"
#include "util.h"

namespace GradientFunctionTest {
	TEST_CLASS(GetFunctionTest) 
	{
	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 });
			tensor1a.setGradient(true);
			Tensor tensor1b = tensor1a.get({ 0 });
			Tensor gradient1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[3], { 1,3 })
			);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 });
			tensor2a.setGradient(true);
			Tensor tensor2b = tensor2a.get({});
			Tensor gradient2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[10], { 10 })
			);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}
	};
}