#include "pch.h"
#include "deep_learning.h"
#include "util.h"

namespace GradientFunctionTest
{
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

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 4, 2 });
			tensor1a.setGradient(true);
			Tensor tensor1b = tensor1a.get({ 1 });
			Tensor gradient1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[2] {1.0f, 2.0f}, { 1,3 })
			);
			CompareFloats(gradient1.at(0), 0.0f);
			CompareFloats(gradient1.at(1), 0.0f);
			CompareFloats(gradient1.at(2), 1.0f);
			CompareFloats(gradient1.at(3), 2.0f);
			CompareFloats(gradient1.at(4), 0.0f);
			CompareFloats(gradient1.at(5), 0.0f);
			CompareFloats(gradient1.at(6), 0.0f);
			CompareFloats(gradient1.at(7), 0.0f);

			Tensor tensor2a = Tensor::ones({ 3 });
			tensor2a.setGradient(true);
			Tensor tensor2b = tensor1a.get({});
			Tensor gradient2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[3] {3.0f, 2.0f, 1.0f}, { 3 })
			);
			CompareFloats(gradient2.at(0), 3.0f);
			CompareFloats(gradient2.at(1), 2.0f);
			CompareFloats(gradient2.at(2), 1.0f);
		}
	};

	TEST_CLASS(SetSingleFunctionTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 });
			tensor1a.setGradient(true);
			Tensor tensor1b = tensor1a.set(0.0f);
			Tensor gradient1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[3], { 2,1,3 })
			);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 });
			tensor2a.setGradient(true);
			Tensor tensor2b = tensor2a.set(1.0f);
			Tensor gradient2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[10], { 10 })
			);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}
	};
}