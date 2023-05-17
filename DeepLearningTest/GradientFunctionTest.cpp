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
			Tensor tensor1b = Tensor::add(tensor1a, 2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = Tensor::add(tensor2a, 2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = Tensor::add(tensor1a, 2.0f);
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
			Tensor tensor2b = Tensor::add(tensor2a, 2.0f);
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
			Tensor tensor1c = Tensor::add(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::add(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::add(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::add(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::add(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 18.0f);
			CompareFloats(gradient1.at(1), 22.0f);
			CompareFloats(gradient1.at(2), 26.0f);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::add(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::add(tensor1a, tensor1b);
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
			Tensor tensor2c = Tensor::add(tensor2a, tensor2b);
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
			Tensor tensor1b = Tensor::subtract(tensor1a, 2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = Tensor::subtract(tensor2a, 2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = Tensor::subtract(tensor1a, 2.0f);
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
			Tensor tensor2b = Tensor::subtract(tensor2a, 2.0f);
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
			Tensor tensor1c = Tensor::subtract(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::subtract(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::subtract(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::subtract(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::subtract(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 18.0f);
			CompareFloats(gradient1.at(1), 22.0f);
			CompareFloats(gradient1.at(2), 26.0f);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::subtract(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::subtract(tensor1a, tensor1b);
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
			Tensor tensor2c = Tensor::subtract(tensor2a, tensor2b);
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
			Tensor tensor1b = Tensor::multiply(tensor1a, 2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = Tensor::multiply(tensor2a, 2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = Tensor::multiply(tensor1a, 2.0f);
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
			Tensor tensor2b = Tensor::multiply(tensor2a, 2.0f);
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
			Tensor tensor1c = Tensor::multiply(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::multiply(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::multiply(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::multiply(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::multiply(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 4, 3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 42.0f);
			CompareFloats(gradient1.at(1), 48.0f);
			CompareFloats(gradient1.at(2), 54);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::ones({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::multiply(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::multiply(tensor1a, tensor1b);
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
			Tensor tensor2c = Tensor::multiply(tensor2a, tensor2b);
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
			Tensor tensor1b = Tensor::divide(tensor1a, 2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = Tensor::divide(tensor2a, 2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = Tensor::divide(tensor1a, 2.0f);
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
			Tensor tensor2b = Tensor::divide(tensor2a, 0.5f);
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
			Tensor tensor1c = Tensor::divide(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::divide(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::divide(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::divide(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::divide(tensor1a, tensor1b);
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
			Tensor tensor2c = Tensor::divide(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::divide(tensor1a, tensor1b);
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
			Tensor tensor2c = Tensor::divide(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::matrixMultiply(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 10,3,2 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(10, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);
			Assert::AreEqual(1, gradient1.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 7, 5, 4, 2 });
			Tensor tensor2b = Tensor::zeroes({ 5, 2, 3 }).requireGradient();
			Tensor tensor2c = Tensor::matrixMultiply(tensor2a, tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 7,5,4,3 })
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
			Tensor tensor1c = Tensor::matrixMultiply(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 10,3,2 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);
			Assert::AreEqual(2, gradient1.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 7, 5, 4, 2 });
			Tensor tensor2b = Tensor::zeroes({ 5, 2, 3 }).requireGradient();
			Tensor tensor2c = Tensor::matrixMultiply(tensor2a, tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 7,5,4,3 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			Assert::AreEqual(5, gradient2.getShape()[0]);
			Assert::AreEqual(2, gradient2.getShape()[1]);
			Assert::AreEqual(3, gradient2.getShape()[2]);
		}

		TEST_METHOD(Values1)
		{
			Tensor tensor1a = Tensor::range({ 2, 3 }, 1);
			Tensor tensor1b = Tensor::range({ 3, 4 }, 1).requireGradient();
			Tensor tensor1c = Tensor::matrixMultiply(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 2,4 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at({ 0, 0 }), 20.0f);
			CompareFloats(gradient1.at({ 0, 1 }), 44.0f);
			CompareFloats(gradient1.at({ 0, 2 }), 68.0f);
			CompareFloats(gradient1.at({ 1, 0 }), 60.0f);
			CompareFloats(gradient1.at({ 1, 1 }), 148.0f);
			CompareFloats(gradient1.at({ 1, 2 }), 236.0f);

			Tensor tensor2a = Tensor::range({ 2, 1, 1, 3 }, 1);
			{
				CompareFloats(tensor2a.at({ 0,0,0,0 }), 1);
				CompareFloats(tensor2a.at({ 0,0,0,1 }), 2);
				CompareFloats(tensor2a.at({ 0,0,0,2 }), 3);
				CompareFloats(tensor2a.at({ 1,0,0,0 }), 4);
				CompareFloats(tensor2a.at({ 1,0,0,1 }), 5);
				CompareFloats(tensor2a.at({ 1,0,0,2 }), 6);
			}
			Tensor tensor2b = Tensor::range({ 1, 3 ,3, 2 }, 1).requireGradient();
			{
				CompareFloats(tensor2b.at({ 0,0,0,0 }), 1);
				CompareFloats(tensor2b.at({ 0,0,0,1 }), 2);
				CompareFloats(tensor2b.at({ 0,0,1,0 }), 3);
				CompareFloats(tensor2b.at({ 0,0,1,1 }), 4);
				CompareFloats(tensor2b.at({ 0,0,2,0 }), 5);
				CompareFloats(tensor2b.at({ 0,0,2,1 }), 6);
				CompareFloats(tensor2b.at({ 0,1,0,0 }), 7);
				CompareFloats(tensor2b.at({ 0,1,0,1 }), 8);
				CompareFloats(tensor2b.at({ 0,1,1,0 }), 9);
				CompareFloats(tensor2b.at({ 0,1,1,1 }), 10);
				CompareFloats(tensor2b.at({ 0,1,2,0 }), 11);
				CompareFloats(tensor2b.at({ 0,1,2,1 }), 12);
				CompareFloats(tensor2b.at({ 0,2,0,0 }), 13);
				CompareFloats(tensor2b.at({ 0,2,0,1 }), 14);
				CompareFloats(tensor2b.at({ 0,2,1,0 }), 15);
				CompareFloats(tensor2b.at({ 0,2,1,1 }), 16);
				CompareFloats(tensor2b.at({ 0,2,2,0 }), 17);
				CompareFloats(tensor2b.at({ 0,2,2,1 }), 18);
			}
			Tensor tensor2c = Tensor::matrixMultiply(tensor2a, tensor2b);
			{
				CompareFloats(tensor2c.at({ 0,0,0,0 }), 22);
				CompareFloats(tensor2c.at({ 0,0,0,1 }), 28);
				CompareFloats(tensor2c.at({ 0,1,0,0 }), 58);
				CompareFloats(tensor2c.at({ 0,1,0,1 }), 64);
				CompareFloats(tensor2c.at({ 0,2,0,0 }), 94);
				CompareFloats(tensor2c.at({ 0,2,0,1 }), 100);
				CompareFloats(tensor2c.at({ 1,0,0,0 }), 49);
				CompareFloats(tensor2c.at({ 1,0,0,1 }), 64);
				CompareFloats(tensor2c.at({ 1,1,0,0 }), 139);
				CompareFloats(tensor2c.at({ 1,1,0,1 }), 154);
				CompareFloats(tensor2c.at({ 1,2,0,0 }), 229);
				CompareFloats(tensor2c.at({ 1,2,0,1 }), 244);
			}			
			Tensor tensor2d = Tensor::range({ 2,3,1,2 });
			{
				CompareFloats(tensor2d.at({ 0,0,0,0 }), 0);
				CompareFloats(tensor2d.at({ 0,0,0,1 }), 1);
				CompareFloats(tensor2d.at({ 0,1,0,0 }), 2);
				CompareFloats(tensor2d.at({ 0,1,0,1 }), 3);
				CompareFloats(tensor2d.at({ 0,2,0,0 }), 4);
				CompareFloats(tensor2d.at({ 0,2,0,1 }), 5);
				CompareFloats(tensor2d.at({ 1,0,0,0 }), 6);
				CompareFloats(tensor2d.at({ 1,0,0,1 }), 7);
				CompareFloats(tensor2d.at({ 1,1,0,0 }), 8);
				CompareFloats(tensor2d.at({ 1,1,0,1 }), 9);
				CompareFloats(tensor2d.at({ 1,2,0,0 }), 10);
				CompareFloats(tensor2d.at({ 1,2,0,1 }), 11);
			}
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				tensor2d
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at({ 0,0,0,0 }), 162.0f);
			CompareFloats(gradient2.at({ 0,0,0,1 }), 192.0f);
			CompareFloats(gradient2.at({ 0,0,0,2 }), 222.0f);
			CompareFloats(gradient2.at({ 1,0,0,0 }), 432.0f);
			CompareFloats(gradient2.at({ 1,0,0,1 }), 534.0f);
			CompareFloats(gradient2.at({ 1,0,0,2 }), 636.0f);
		}

		TEST_METHOD(Values2)
		{
			Tensor tensor1a = Tensor::range({ 2, 3 }, 1);
			Tensor tensor1b = Tensor::range({ 3, 4 }, 1).requireGradient();
			Tensor tensor1c = Tensor::matrixMultiply(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::range({ 2,4 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			CompareFloats(gradient1.at({ 0, 0 }), 16.0f);
			CompareFloats(gradient1.at({ 0, 1 }), 21.0f);
			CompareFloats(gradient1.at({ 0, 2 }), 26.0f);
			CompareFloats(gradient1.at({ 0, 3 }), 31.0f);

			CompareFloats(gradient1.at({ 1, 0 }), 20.0f);
			CompareFloats(gradient1.at({ 1, 1 }), 27.0f);
			CompareFloats(gradient1.at({ 1, 2 }), 34.0f);
			CompareFloats(gradient1.at({ 1, 3 }), 41.0f);

			CompareFloats(gradient1.at({ 2, 0 }), 24.0f);
			CompareFloats(gradient1.at({ 2, 1 }), 33.0f);
			CompareFloats(gradient1.at({ 2, 2 }), 42.0f);
			CompareFloats(gradient1.at({ 2, 3 }), 51.0f);

			Tensor tensor2a = Tensor::range({ 2, 1, 1, 3 }, 1);
			{
				CompareFloats(tensor2a.at({ 0,0,0,0 }), 1);
				CompareFloats(tensor2a.at({ 0,0,0,1 }), 2);
				CompareFloats(tensor2a.at({ 0,0,0,2 }), 3);
				CompareFloats(tensor2a.at({ 1,0,0,0 }), 4);
				CompareFloats(tensor2a.at({ 1,0,0,1 }), 5);
				CompareFloats(tensor2a.at({ 1,0,0,2 }), 6);
			}
			Tensor tensor2b = Tensor::range({ 1, 3 ,3, 2 }, 1).requireGradient();
			{
				CompareFloats(tensor2b.at({ 0,0,0,0 }), 1);
				CompareFloats(tensor2b.at({ 0,0,0,1 }), 2);
				CompareFloats(tensor2b.at({ 0,0,1,0 }), 3);
				CompareFloats(tensor2b.at({ 0,0,1,1 }), 4);
				CompareFloats(tensor2b.at({ 0,0,2,0 }), 5);
				CompareFloats(tensor2b.at({ 0,0,2,1 }), 6);
				CompareFloats(tensor2b.at({ 0,1,0,0 }), 7);
				CompareFloats(tensor2b.at({ 0,1,0,1 }), 8);
				CompareFloats(tensor2b.at({ 0,1,1,0 }), 9);
				CompareFloats(tensor2b.at({ 0,1,1,1 }), 10);
				CompareFloats(tensor2b.at({ 0,1,2,0 }), 11);
				CompareFloats(tensor2b.at({ 0,1,2,1 }), 12);
				CompareFloats(tensor2b.at({ 0,2,0,0 }), 13);
				CompareFloats(tensor2b.at({ 0,2,0,1 }), 14);
				CompareFloats(tensor2b.at({ 0,2,1,0 }), 15);
				CompareFloats(tensor2b.at({ 0,2,1,1 }), 16);
				CompareFloats(tensor2b.at({ 0,2,2,0 }), 17);
				CompareFloats(tensor2b.at({ 0,2,2,1 }), 18);
			}
			Tensor tensor2c = Tensor::matrixMultiply(tensor2a, tensor2b);
			{
				CompareFloats(tensor2c.at({ 0,0,0,0 }), 22);
				CompareFloats(tensor2c.at({ 0,0,0,1 }), 28);
				CompareFloats(tensor2c.at({ 0,1,0,0 }), 58);
				CompareFloats(tensor2c.at({ 0,1,0,1 }), 64);
				CompareFloats(tensor2c.at({ 0,2,0,0 }), 94);
				CompareFloats(tensor2c.at({ 0,2,0,1 }), 100);
				CompareFloats(tensor2c.at({ 1,0,0,0 }), 49);
				CompareFloats(tensor2c.at({ 1,0,0,1 }), 64);
				CompareFloats(tensor2c.at({ 1,1,0,0 }), 139);
				CompareFloats(tensor2c.at({ 1,1,0,1 }), 154);
				CompareFloats(tensor2c.at({ 1,2,0,0 }), 229);
				CompareFloats(tensor2c.at({ 1,2,0,1 }), 244);
			}
			Tensor tensor2d = Tensor::range({ 2,3,1,2 });
			{
				CompareFloats(tensor2d.at({ 0,0,0,0 }), 0);
				CompareFloats(tensor2d.at({ 0,0,0,1 }), 1);
				CompareFloats(tensor2d.at({ 0,1,0,0 }), 2);
				CompareFloats(tensor2d.at({ 0,1,0,1 }), 3);
				CompareFloats(tensor2d.at({ 0,2,0,0 }), 4);
				CompareFloats(tensor2d.at({ 0,2,0,1 }), 5);
				CompareFloats(tensor2d.at({ 1,0,0,0 }), 6);
				CompareFloats(tensor2d.at({ 1,0,0,1 }), 7);
				CompareFloats(tensor2d.at({ 1,1,0,0 }), 8);
				CompareFloats(tensor2d.at({ 1,1,0,1 }), 9);
				CompareFloats(tensor2d.at({ 1,2,0,0 }), 10);
				CompareFloats(tensor2d.at({ 1,2,0,1 }), 11);
			}
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				tensor2d
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			CompareFloats(gradient2.at({ 0,0,0,0 }), 24.0f);
			CompareFloats(gradient2.at({ 0,0,0,1 }), 29.0f);
			CompareFloats(gradient2.at({ 0,0,1,0 }), 30.0f);
			CompareFloats(gradient2.at({ 0,0,1,1 }), 37.0f);
			CompareFloats(gradient2.at({ 0,0,2,0 }), 36.0f);
			CompareFloats(gradient2.at({ 0,0,2,1 }), 45.0f);

			CompareFloats(gradient2.at({ 0,1,0,0 }), 34.0f);
			CompareFloats(gradient2.at({ 0,1,0,1 }), 39.0f);
			CompareFloats(gradient2.at({ 0,1,1,0 }), 44.0f);
			CompareFloats(gradient2.at({ 0,1,1,1 }), 51.0f);
			CompareFloats(gradient2.at({ 0,1,2,0 }), 54.0f);
			CompareFloats(gradient2.at({ 0,1,2,1 }), 63.0f);

			CompareFloats(gradient2.at({ 0,2,0,0 }), 44.0f);
			CompareFloats(gradient2.at({ 0,2,0,1 }), 49.0f);
			CompareFloats(gradient2.at({ 0,2,1,0 }), 58.0f);
			CompareFloats(gradient2.at({ 0,2,1,1 }), 65.0f);
			CompareFloats(gradient2.at({ 0,2,2,0 }), 72.0f);
			CompareFloats(gradient2.at({ 0,2,2,1 }), 81.0f);

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

	TEST_CLASS(MaxSingleFunction)
	{
	public:

	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,1,3 }).requireGradient();
			Tensor tensor1b = Tensor::max(tensor1a, 2.0f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::zeroes({ 2,1,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(gradient1.getShape()[0], tensor1a.getShape()[0]);
			Assert::AreEqual(gradient1.getShape()[1], tensor1a.getShape()[1]);
			Assert::AreEqual(gradient1.getShape()[2], tensor1a.getShape()[2]);

			Tensor tensor2a = Tensor::zeroes({ 10 }).requireGradient();
			Tensor tensor2b = Tensor::max(tensor2a, 2.0f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::zeroes({ 10 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			Assert::AreEqual(gradient2.getShape()[0], tensor2a.getShape()[0]);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::range({ 2,1,3 }).requireGradient();
			Tensor tensor1b = Tensor::max(tensor1a, 2.5f);
			gradientList gradients1 = tensor1b.getFunction()->calculateGradient(
				Tensor::range({ 2,1,3 }, 1)
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			CompareFloats(gradient1.at(0), 0.0f);
			CompareFloats(gradient1.at(1), 0.0f);
			CompareFloats(gradient1.at(2), 0.0f);
			CompareFloats(gradient1.at(3), 4.0f);
			CompareFloats(gradient1.at(4), 5.0f);
			CompareFloats(gradient1.at(5), 6.0f);

			Tensor tensor2a = Tensor::range({ 5 }).requireGradient();
			Tensor tensor2b = Tensor::max(tensor2a, 5.5f);
			gradientList gradients2 = tensor2b.getFunction()->calculateGradient(
				Tensor::range({ 5 }, 1)
			);
			Tensor& gradient2 = std::get<1>(gradients2[0]);
			CompareFloats(gradient2.at(0), 0.0f);
			CompareFloats(gradient2.at(1), 0.0f);
			CompareFloats(gradient2.at(2), 0.0f);
			CompareFloats(gradient2.at(3), 0.0f);
			CompareFloats(gradient2.at(4), 0.0f);
		}
	};

	TEST_CLASS(MaxTensorFunction)
	{
	public:
		TEST_METHOD(Shape1)
		{
			Tensor tensor1a = Tensor::zeroes({ 1, 3 });
			Tensor tensor1b = Tensor::zeroes({ 4, 1 }).requireGradient();
			Tensor tensor1c = Tensor::max(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[0]);
			Assert::AreEqual(1, gradient1.getShape()[0]);
			Assert::AreEqual(3, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::max(tensor2a, tensor2b);
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
			Tensor tensor1c = Tensor::max(tensor1a, tensor1b);
			gradientList gradients1 = tensor1c.getFunction()->calculateGradient(
				Tensor::zeroes({ 4,3 })
			);
			Tensor& gradient1 = std::get<1>(gradients1[1]);
			Assert::AreEqual(4, gradient1.getShape()[0]);
			Assert::AreEqual(1, gradient1.getShape()[1]);

			Tensor tensor2a = Tensor::zeroes({ 1, });
			Tensor tensor2b = Tensor::zeroes({ 2, 3, 1 }).requireGradient();
			Tensor tensor2c = Tensor::max(tensor2a, tensor2b);
			gradientList gradients2 = tensor2c.getFunction()->calculateGradient(
				Tensor::zeroes({ 2, 3, 1 })
			);
			Tensor& gradient2 = std::get<1>(gradients2[1]);
			Assert::AreEqual(2, gradient2.getShape()[0]);
			Assert::AreEqual(3, gradient2.getShape()[1]);
			Assert::AreEqual(1, gradient2.getShape()[2]);
		}
	};
}