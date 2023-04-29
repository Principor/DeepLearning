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
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.get({ 0 });
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = tensor2a.get({});
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 4, 2 }).requireGradient();
			Tensor tensor1b = tensor1a.get({ 1 });
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::range({ 1,2 }, 1)
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

			Tensor tensor2a = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2b = tensor2a.get({});
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::range({ 3 }, 3, -1)
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
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.set(0.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = tensor2a.set(1.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 4, 2 }).requireGradient();
			Tensor tensor1b = tensor1a.set(0.0f, { 1 });
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::range({ 4, 2 }, 1)
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

			Tensor tensor2a = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2b = tensor2a.set(1.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::range({ 3 }, 1)
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
			Tensor tensor1a = Tensor::zeroes({ 4,2 }).requireGradient();
			Tensor tensor1b = Tensor::zeroes({ 1 });
			Tensor tensor1c = tensor1a.set(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,2 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], 4);
			Assert::AreEqual(gradient1.getShape()[1], 2);

			Tensor tensor2a = Tensor::zeroes({ 10, 1, 3 }).requireGradient();
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
			Tensor tensor1a = Tensor::zeroes({ 4,2 }).requireGradient();
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

			Tensor tensor2a = Tensor::zeroes({ 2, 1, 3 }).requireGradient();
			Tensor tensor2b = Tensor::zeroes({ 1, 1, 3 });
			Tensor tensor2c = tensor2a.set(tensor2b, { 1, 0 });
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::range({ 2,1,3 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 1.0f);
			CompareFloats(gradient2.at(1), 2.0f);
			CompareFloats(gradient2.at(2), 3.0f);
			CompareFloats(gradient2.at(3), 0.0f);
			CompareFloats(gradient2.at(4), 0.0f);
			CompareFloats(gradient2.at(5), 0.0f);
		}

		TEST_METHOD(CopyFromShape)
		{
			Tensor tensor1a = Tensor::zeroes({ 4,2,1 }).requireGradient();
			Tensor tensor1b = Tensor::zeroes({ 1,2,1 });
			Tensor tensor1c = tensor1a.set(tensor1b, { 0 });
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::ones({ 4,2,1 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(gradient1.getShape()[0], 1);
			Assert::AreEqual(gradient1.getShape()[1], 2);
			Assert::AreEqual(gradient1.getShape()[2], 1);

			Tensor tensor2a = Tensor::zeroes({ 6,3 }).requireGradient();
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
			Tensor tensor1a = Tensor::zeroes({ 4,2,1 }).requireGradient();
			Tensor tensor1b = Tensor::zeroes({ 1,2,1 });
			Tensor tensor1c = tensor1a.set(tensor1b, { 0 });
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 2, 1 }, 1)
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			CompareFloats(gradient1.at(0), 1.0f);
			CompareFloats(gradient1.at(1), 2.0f);

			Tensor tensor2a = Tensor::zeroes({ 6,3 }).requireGradient();
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
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.add(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = tensor2a.add(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.add(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::range({ 2,1,3 }, 1)
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 1.0f);
			CompareFloats(gradient1.at(1), 2.0f);
			CompareFloats(gradient1.at(2), 3.0f);
			CompareFloats(gradient1.at(3), 4.0f);
			CompareFloats(gradient1.at(4), 5.0f);
			CompareFloats(gradient1.at(5), 6.0f);

			Tensor tensor2a = Tensor::zeroes({ 5 }).requireGradient();
			Tensor tensor2b = tensor2a.add(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::range({ 5 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 1.0f);
			CompareFloats(gradient2.at(1), 2.0f);
			CompareFloats(gradient2.at(2), 3.0f);
			CompareFloats(gradient2.at(3), 4.0f);
			CompareFloats(gradient2.at(4), 5.0f);
		}
	};

	TEST_CLASS(AddTensorFunctionTest)
	{
	public:
		TEST_METHOD(Shape1)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.add(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.add(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(1, gradient2.getShape()[0]);
		}

		TEST_METHOD(Shape2)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.add(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.add(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			Assert::AreEqual(2, gradient2.getShape()[0]);
			Assert::AreEqual(3, gradient2.getShape()[1]);
			Assert::AreEqual(1, gradient2.getShape()[2]);
		}

		TEST_METHOD(Values1)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.add(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 18.0f);
			CompareFloats(gradient1.at(1), 22.0f);
			CompareFloats(gradient1.at(2), 26.0f);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.add(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::ones({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 6.0f);
		}

		TEST_METHOD(Values2)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.add(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			CompareFloats(gradient1.at(0), 3.0f);
			CompareFloats(gradient1.at(1), 12.0f);
			CompareFloats(gradient1.at(2), 21.0f);
			CompareFloats(gradient1.at(3), 30.0f);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.add(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::range({ 2, 3, 1 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			CompareFloats(gradient2.at(0), 1.0f);
			CompareFloats(gradient2.at(1), 2.0f);
			CompareFloats(gradient2.at(2), 3.0f);
			CompareFloats(gradient2.at(3), 4.0f);
			CompareFloats(gradient2.at(4), 5.0f);
			CompareFloats(gradient2.at(5), 6.0f);
		}
	};

	TEST_CLASS(SubtractSingleFunctionTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.subtract(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = tensor2a.subtract(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.subtract(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::range({ 2,1,3 }, 1)
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 1.0f);
			CompareFloats(gradient1.at(1), 2.0f);
			CompareFloats(gradient1.at(2), 3.0f);
			CompareFloats(gradient1.at(3), 4.0f);
			CompareFloats(gradient1.at(4), 5.0f);
			CompareFloats(gradient1.at(5), 6.0f);

			Tensor tensor2a = Tensor::zeroes({ 5 }).requireGradient();
			Tensor tensor2b = tensor2a.subtract(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::range({ 5 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 1.0f);
			CompareFloats(gradient2.at(1), 2.0f);
			CompareFloats(gradient2.at(2), 3.0f);
			CompareFloats(gradient2.at(3), 4.0f);
			CompareFloats(gradient2.at(4), 5.0f);
		}
	};

	TEST_CLASS(SubtractTensorFunctionTest)
	{
	public:
		TEST_METHOD(Shape1)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.subtract(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.subtract(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(1, gradient2.getShape()[0]);
		}

		TEST_METHOD(Shape2)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.subtract(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.subtract(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			Assert::AreEqual(2, gradient2.getShape()[0]);
			Assert::AreEqual(3, gradient2.getShape()[1]);
			Assert::AreEqual(1, gradient2.getShape()[2]);
		}

		TEST_METHOD(Values1)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.subtract(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 18.0f);
			CompareFloats(gradient1.at(1), 22.0f);
			CompareFloats(gradient1.at(2), 26.0f);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.subtract(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::ones({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 6.0f);
		}

		TEST_METHOD(Values2)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.subtract(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			CompareFloats(gradient1.at(0), -3.0f);
			CompareFloats(gradient1.at(1), -12.0f);
			CompareFloats(gradient1.at(2), -21.0f);
			CompareFloats(gradient1.at(3), -30.0f);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.subtract(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::range({ 2, 3, 1 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			CompareFloats(gradient2.at(0), -1.0f);
			CompareFloats(gradient2.at(1), -2.0f);
			CompareFloats(gradient2.at(2), -3.0f);
			CompareFloats(gradient2.at(3), -4.0f);
			CompareFloats(gradient2.at(4), -5.0f);
			CompareFloats(gradient2.at(5), -6.0f);
		}
	};

	TEST_CLASS(MultiplySingleFunctionTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.multiply(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = tensor2a.multiply(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.multiply(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::range({ 2,1,3 }, 1)
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 2.0f);
			CompareFloats(gradient1.at(1), 4.0f);
			CompareFloats(gradient1.at(2), 6.0f);
			CompareFloats(gradient1.at(3), 8.0f);
			CompareFloats(gradient1.at(4), 10.0f);
			CompareFloats(gradient1.at(5), 12.0f);

			Tensor tensor2a = Tensor::zeroes({ 5 }).requireGradient();
			Tensor tensor2b = tensor2a.multiply(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::range({ 5 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 2.0f);
			CompareFloats(gradient2.at(1), 4.0f);
			CompareFloats(gradient2.at(2), 6.0f);
			CompareFloats(gradient2.at(3), 8.0f);
			CompareFloats(gradient2.at(4), 10.0f);
		}
	};

	TEST_CLASS(MultiplyTensorFunctionTest)
	{
	public:
		TEST_METHOD(Shape1)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.multiply(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.multiply(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(1, gradient2.getShape()[0]);
		}

		TEST_METHOD(Shape2)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.multiply(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.multiply(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			Assert::AreEqual(2, gradient2.getShape()[0]);
			Assert::AreEqual(3, gradient2.getShape()[1]);
			Assert::AreEqual(1, gradient2.getShape()[2]);
		}

		TEST_METHOD(Values1)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::range({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.multiply(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 42.0f);
			CompareFloats(gradient1.at(1), 48.0f);
			CompareFloats(gradient1.at(2), 54);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::ones({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.multiply(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::range({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 15.0);
		}

		TEST_METHOD(Values2)
		{
			Tensor tensor1a = Tensor::range({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.multiply(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			CompareFloats(gradient1.at(0), 5);
			CompareFloats(gradient1.at(1), 14.0f);
			CompareFloats(gradient1.at(2), 23.0f);
			CompareFloats(gradient1.at(3), 32.0f);

			Tensor tensor2a = Tensor::full({ 1, }, 2);
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.multiply(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::range({ 2, 3, 1 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			CompareFloats(gradient2.at(0), 2.0f);
			CompareFloats(gradient2.at(1), 4.0f);
			CompareFloats(gradient2.at(2), 6.0f);
			CompareFloats(gradient2.at(3), 8.0f);
			CompareFloats(gradient2.at(4), 10.0f);
			CompareFloats(gradient2.at(5), 12.0f);
		}
	};

	TEST_CLASS(DivideSingleFunctionTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.divide(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = tensor2a.divide(2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = tensor1a.divide(2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::range({ 2,1,3 }, 1)
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 0.5f);
			CompareFloats(gradient1.at(1), 1.0f);
			CompareFloats(gradient1.at(2), 1.5f);
			CompareFloats(gradient1.at(3), 2.0f);
			CompareFloats(gradient1.at(4), 2.5f);
			CompareFloats(gradient1.at(5), 3.0f);

			Tensor tensor2a = Tensor::zeroes({ 5 }).requireGradient();
			Tensor tensor2b = tensor2a.divide(0.5f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::range({ 5 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 2.0f);
			CompareFloats(gradient2.at(1), 4.0f);
			CompareFloats(gradient2.at(2), 6.0f);
			CompareFloats(gradient2.at(3), 8.0f);
			CompareFloats(gradient2.at(4), 10.0f);
		}
	};

	TEST_CLASS(DivideTensorFunctionTest)
	{
	public:
		TEST_METHOD(Shape1)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.divide(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.divide(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(1, gradient2.getShape()[0]);
		}

		TEST_METHOD(Shape2)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = tensor1a.divide(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = tensor2a.divide(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			Assert::AreEqual(2, gradient2.getShape()[0]);
			Assert::AreEqual(3, gradient2.getShape()[1]);
			Assert::AreEqual(1, gradient2.getShape()[2]);
		}

		TEST_METHOD(Values1)
		{
			Tensor tensor1a = Tensor::range({ 2, 1, 3 }, 0, 2);
			Tensor tensor1b = Tensor::full({ 1 }, 0.5f).requireGradient();
			Tensor tensor1c = tensor1a.divide(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 2, 1, 3 }, 1)
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at({ 0, 0, 0 }), 2.0f);
			CompareFloats(gradient1.at({ 0, 0, 1 }), 4.0f);
			CompareFloats(gradient1.at({ 0, 0, 2 }), 6.0f);
			CompareFloats(gradient1.at({ 1, 0, 0 }), 8.0f);
			CompareFloats(gradient1.at({ 1, 0, 1 }), 10.0f);
			CompareFloats(gradient1.at({ 1, 0, 2 }), 12.0f);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::full({ 2, 3, 1 }, 2.0f).requireGradient();
			Tensor tensor2c = tensor2a.divide(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::range({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 7.5f);
		}

		TEST_METHOD(Values2)
		{
			Tensor tensor1a = Tensor::range({ 1, 3 }, 1);
			Tensor tensor1b = Tensor::full({ 4, 1 }, 0.2f).requireGradient();
			Tensor tensor1c = tensor1a.divide(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			CompareFloats(gradient1.at(0), -200.0f);
			CompareFloats(gradient1.at(1), -650.0f);
			CompareFloats(gradient1.at(2), -1100.0f);
			CompareFloats(gradient1.at(3), -1550.0f);

			Tensor tensor2a = Tensor::full({ 1, }, 2);
			Tensor tensor2b = Tensor::full({ 2, 3, 1 }, -1.0f).requireGradient();
			Tensor tensor2c = tensor2a.divide(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::range({ 2, 3, 1 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			CompareFloats(gradient2.at(0), -2.0f);
			CompareFloats(gradient2.at(1), -4.0f);
			CompareFloats(gradient2.at(2), -6.0f);
			CompareFloats(gradient2.at(3), -8.0f);
			CompareFloats(gradient2.at(4), -10.0f);
			CompareFloats(gradient2.at(5), -12.0f);
		}
	};

	TEST_CLASS(MatrixMultiplicationFunctionTest)
	{
	public:
		TEST_METHOD(Shape1)
		{

			Tensor tensor1a = Tensor::zeroes({ 10, 3, 1 });
			Tensor tensor1b = Tensor::zeroes({ 1, 1, 2 }).requireGradient();
			Tensor tensor1c = tensor1a.matrixMultiply(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(10, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);
			Assert::AreEqual(1, gradient1.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 7, 5, 4, 2 });
			Tensor tensor2b = Tensor::zeroes({ 5, 2, 3 }).requireGradient();
			Tensor tensor2c = tensor2a.matrixMultiply(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(7, gradient2.getShape()[0]);
			Assert::AreEqual(5, gradient2.getShape()[1]);
			Assert::AreEqual(4, gradient2.getShape()[2]);
			Assert::AreEqual(2, gradient2.getShape()[3]);
		}

		TEST_METHOD(Shape2)
		{

			Tensor tensor1a = Tensor::zeroes({ 10, 3, 1 });
			Tensor tensor1b = Tensor::zeroes({ 1, 1, 2 }).requireGradient();
			Tensor tensor1c = tensor1a.matrixMultiply(tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);
			Assert::AreEqual(2, gradient1.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 7, 5, 4, 2 });
			Tensor tensor2b = Tensor::zeroes({ 5, 2, 3 }).requireGradient();
			Tensor tensor2c = tensor2a.matrixMultiply(tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			Assert::AreEqual(5, gradient2.getShape()[0]);
			Assert::AreEqual(2, gradient2.getShape()[1]);
			Assert::AreEqual(3, gradient2.getShape()[2]);
		}
	};

	TEST_CLASS(TransposeFunctionTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 4,3 }).requireGradient();
			Tensor tensor1b = tensor1a.transpose();
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 3,4 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], 4);
			Assert::AreEqual(gradient1.getShape()[1], 3);

			Tensor tensor2a = Tensor::zeroes({ 10, 1, 2, 5 }).requireGradient();
			Tensor tensor2b = tensor2a.transpose();
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10, 1, 5, 2 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], 10);
			Assert::AreEqual(gradient2.getShape()[1], 1);
			Assert::AreEqual(gradient2.getShape()[2], 2);
			Assert::AreEqual(gradient2.getShape()[3], 5);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::range({ 2,3 }).requireGradient();
			Tensor tensor1b = tensor1a.transpose();
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::range({ 3,2 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at({ 0,0 }), 0);
			CompareFloats(gradient1.at({ 1,0 }), 1);
			CompareFloats(gradient1.at({ 0,1 }), 2);
			CompareFloats(gradient1.at({ 1,1 }), 3);
			CompareFloats(gradient1.at({ 0,2 }), 4);
			CompareFloats(gradient1.at({ 1,2 }), 5);

			Tensor tensor2a = Tensor::range({ 10, 2, 1, 5 }).requireGradient();
			Tensor tensor2b = tensor2a.transpose();
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::range({ 10,2,5,1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			int correct = 0;
			for (int i = 0; i < 10; i++) {
				for (int j = 0; j < 2; j++) {
					for (int l = 0; l < 5; l++) {
						for (int k = 0; k < 1; k++) {
							CompareFloats(gradient2.at({ i,j,k,l }), correct++);
						}
					}
				}
			}
		}
	};
}