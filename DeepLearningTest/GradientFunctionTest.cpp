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
}