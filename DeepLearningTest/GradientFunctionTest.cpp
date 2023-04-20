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
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[3], { 1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 });
			tensor2a.setGradient(true);
			Tensor tensor2b = tensor2a.get({});
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[10], { 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 4, 2 });
			tensor1a.setGradient(true);
			Tensor tensor1b = tensor1a.get({ 1 });
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[2] {1.0f, 2.0f}, { 1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
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
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[3] {3.0f, 2.0f, 1.0f}, { 3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
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
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[6], { 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 });
			tensor2a.setGradient(true);
			Tensor tensor2b = tensor2a.set(1.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[10], { 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 4, 2 });
			tensor1a.setGradient(true);
			Tensor tensor1b = tensor1a.set(0.0f, { 1 });
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[8] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, { 4, 2 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 1.0f);
			CompareFloats(gradient1.at(1), 2.0f);
			CompareFloats(gradient1.at(2), 0.0f);
			CompareFloats(gradient1.at(3), 0.0f);
			CompareFloats(gradient1.at(4), 5.0f);
			CompareFloats(gradient1.at(5), 6.0f);
			CompareFloats(gradient1.at(6), 7.0f);
			CompareFloats(gradient1.at(7), 8.0f);

			Tensor tensor2a = Tensor::ones({ 3 });
			tensor2a.setGradient(true);
			Tensor tensor2b = tensor2a.set(1.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[3] {1.0f, 2.0f, 3.0f}, { 3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 0.0f);
			CompareFloats(gradient2.at(1), 0.0f);
			CompareFloats(gradient2.at(2), 0.0f);
		}
	};

	TEST_CLASS(SetTensorFunctionTest)
	{
	public:
		TEST_METHOD(CopyToShape)
		{
			Tensor tensor1a = Tensor::zeroes({ 4,2 });
			tensor1a.setGradient(true);
			Tensor tensor1b = Tensor::zeroes({ 1 });
			Tensor tensor1c = tensor1a.set(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,2 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], 4);
			Assert::AreEqual(gradient1.getShape()[1], 2);

			Tensor tensor2a = Tensor::zeroes({ 10, 1, 3 });
			tensor2a.setGradient(true);
			Tensor tensor2b = Tensor::zeroes({ 1, 1, 3 });
			Tensor tensor2c = tensor2a.set(tensor2b, { 3, 0 });
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 10,1,3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], 10);
			Assert::AreEqual(gradient2.getShape()[1], 1);
			Assert::AreEqual(gradient2.getShape()[2], 3);

		}

		TEST_METHOD(CopyToValues)
		{
			Tensor tensor1a = Tensor::zeroes({ 4,2 });
			tensor1a.setGradient(true);
			Tensor tensor1b = Tensor::zeroes({ 1 });
			Tensor tensor1c = tensor1a.set(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::ones({ 4,2 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 0.0f);
			CompareFloats(gradient1.at(1), 0.0f);
			CompareFloats(gradient1.at(2), 0.0f);
			CompareFloats(gradient1.at(3), 0.0f);
			CompareFloats(gradient1.at(4), 0.0f);
			CompareFloats(gradient1.at(5), 0.0f);
			CompareFloats(gradient1.at(6), 0.0f);
			CompareFloats(gradient1.at(7), 0.0f);

			Tensor tensor2a = Tensor::zeroes({ 2, 1, 3 });
			tensor2a.setGradient(true);
			Tensor tensor2b = Tensor::zeroes({ 1, 1, 3 });
			Tensor tensor2c = tensor2a.set(tensor2b, { 1, 0 });
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::ones({ 10,1,3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 1.0f);
			CompareFloats(gradient2.at(1), 1.0f);
			CompareFloats(gradient2.at(2), 1.0f);
			CompareFloats(gradient2.at(3), 0.0f);
			CompareFloats(gradient2.at(4), 0.0f);
			CompareFloats(gradient2.at(5), 0.0f);
		}

		TEST_METHOD(CopyFromShape)
		{
			Tensor tensor1a = Tensor::zeroes({ 4,2,1 });
			tensor1a.setGradient(true);
			Tensor tensor1b = Tensor::zeroes({ 1,2,1 });
			Tensor tensor1c = tensor1a.set(tensor1b, { 0 });
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::ones({ 4,2,1 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(gradient1.getShape()[0], 1);
			Assert::AreEqual(gradient1.getShape()[1], 2);
			Assert::AreEqual(gradient1.getShape()[2], 1);

			Tensor tensor2a = Tensor::zeroes({ 6,3 });
			tensor2a.setGradient(true);
			Tensor tensor2b = Tensor::zeroes({});
			Tensor tensor2c = tensor2a.set(tensor2b, {});
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::ones({ 6,3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			Assert::AreEqual((int)gradient2.getShape().size(), 0);
		}

		TEST_METHOD(CopyFromValues)
		{
			Tensor tensor1a = Tensor::zeroes({ 4,2,1 });
			tensor1a.setGradient(true);
			Tensor tensor1b = Tensor::zeroes({ 1,2,1 });
			Tensor tensor1c = tensor1a.set(tensor1b, { 0 });
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::fromValues(new float[8]{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }, { 4, 2, 1 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			CompareFloats(gradient1.at(0), 1.0f);
			CompareFloats(gradient1.at(1), 2.0f);

			Tensor tensor2a = Tensor::zeroes({ 6,3 });
			tensor2a.setGradient(true);
			Tensor tensor2b = Tensor::zeroes({});
			Tensor tensor2c = tensor2a.set(tensor2b, {});
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::ones({ 6,3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			CompareFloats(gradient2.at(0), 18.0f);
		}
	};

	TEST_CLASS(AddSingleFunctionTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 });
			tensor1a.setGradient(true);
			Tensor tensor1b = tensor1a.add(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[6], { 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 });
			tensor2a.setGradient(true);
			Tensor tensor2b = tensor2a.add(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[10], { 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 });
			tensor1a.setGradient(true);
			Tensor tensor1b = tensor1a.add(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[6]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2,1,3})
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 1.0f);
			CompareFloats(gradient1.at(1), 2.0f);
			CompareFloats(gradient1.at(2), 3.0f);
			CompareFloats(gradient1.at(3), 4.0f);
			CompareFloats(gradient1.at(4), 5.0f);
			CompareFloats(gradient1.at(5), 6.0f);

			Tensor tensor2a = Tensor::zeroes({ 5 });
			tensor2a.setGradient(true);
			Tensor tensor2b = tensor2a.add(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::fromValues(new float[5]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5})
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 1.0f);
			CompareFloats(gradient2.at(1), 2.0f);
			CompareFloats(gradient2.at(2), 3.0f);
			CompareFloats(gradient2.at(3), 4.0f);
			CompareFloats(gradient2.at(4), 5.0f);
		}
	};
}