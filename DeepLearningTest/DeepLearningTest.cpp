#include "pch.h"
#include "deep_learning.h"
#include "util.h"


namespace TensorTest
{
	TEST_CLASS(ZeroesTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Assert::AreEqual(1, Tensor::zeroes({ 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor::zeroes({ 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor::zeroes({ 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor::zeroes({ 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor::zeroes({ 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::zeroes({ 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor::zeroes({ 3,7,5 }).getShape()[2]);
		}

		TEST_METHOD(Size)
		{
			Assert::AreEqual(1, Tensor::zeroes({}).getSize());
			Assert::AreEqual(1, Tensor::zeroes({ 1 }).getSize());
			Assert::AreEqual(1, Tensor::zeroes({ 1, 1, 1 }).getSize());
			Assert::AreEqual(2, Tensor::zeroes({ 2 }).getSize());
			Assert::AreEqual(9, Tensor::zeroes({ 3, 3 }).getSize());
			Assert::AreEqual(105, Tensor::zeroes({ 3, 7, 5 }).getSize());
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1 = Tensor::zeroes({ 3 });
			CompareFloats(tensor1.get({ 0 }).item(), 0.0f);
			CompareFloats(tensor1.get({ 1 }).item(), 0.0f);
			CompareFloats(tensor1.get({ 2 }).item(), 0.0f);

			Tensor tensor2 = Tensor::zeroes({ 2, 3 });
			CompareFloats(tensor2.get({ 0, 0 }).item(), 0.0f);
			CompareFloats(tensor2.get({ 0, 1 }).item(), 0.0f);
			CompareFloats(tensor2.get({ 0, 2 }).item(), 0.0f);
			CompareFloats(tensor2.get({ 1, 0 }).item(), 0.0f);
			CompareFloats(tensor2.get({ 1, 1 }).item(), 0.0f);
			CompareFloats(tensor2.get({ 1, 2 }).item(), 0.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 1, 0, 3 }); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::zeroes({ 4, 3, 2 }).getGradient());
			Assert::IsFalse(Tensor::zeroes({  }).getGradient());
			Assert::IsFalse(Tensor::zeroes({ 10 }).getGradient());
		}
	};

	TEST_CLASS(OnesTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Assert::AreEqual(1, Tensor::ones({ 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor::ones({ 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor::ones({ 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor::ones({ 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor::ones({ 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::ones({ 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor::ones({ 3,7,5 }).getShape()[2]);
		}

		TEST_METHOD(Size)
		{
			Assert::AreEqual(1, Tensor::ones({}).getSize());
			Assert::AreEqual(1, Tensor::ones({ 1 }).getSize());
			Assert::AreEqual(1, Tensor::ones({ 1, 1, 1 }).getSize());
			Assert::AreEqual(2, Tensor::ones({ 2 }).getSize());
			Assert::AreEqual(9, Tensor::ones({ 3, 3 }).getSize());
			Assert::AreEqual(105, Tensor::ones({ 3, 7, 5 }).getSize());
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1 = Tensor::ones({ 3 });
			CompareFloats(tensor1.get({ 0 }).item(), 1.0f);
			CompareFloats(tensor1.get({ 1 }).item(), 1.0f);
			CompareFloats(tensor1.get({ 2 }).item(), 1.0f);

			Tensor tensor2 = Tensor::ones({ 2, 3 });
			CompareFloats(tensor2.get({ 0, 0 }).item(), 1.0f);
			CompareFloats(tensor2.get({ 0, 1 }).item(), 1.0f);
			CompareFloats(tensor2.get({ 0, 2 }).item(), 1.0f);
			CompareFloats(tensor2.get({ 1, 0 }).item(), 1.0f);
			CompareFloats(tensor2.get({ 1, 1 }).item(), 1.0f);
			CompareFloats(tensor2.get({ 1, 2 }).item(), 1.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::ones({ -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::ones({ 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::ones({ 1, 0, 3 }); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::ones({ 4, 3, 2 }).getGradient());
			Assert::IsFalse(Tensor::ones({  }).getGradient());
			Assert::IsFalse(Tensor::ones({ 10 }).getGradient());
		}
	};

	TEST_CLASS(FullTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Assert::AreEqual(1, Tensor::full({ 1 }, 0.0f).getShape()[0]);

			Assert::AreEqual(3, Tensor::full({ 3 }, 1.0f).getShape()[0]);

			Assert::AreEqual(5, Tensor::full({ 5, 2 }, -2.0f).getShape()[0]);
			Assert::AreEqual(2, Tensor::full({ 5, 2 }, -2.0f).getShape()[1]);

			Assert::AreEqual(3, Tensor::full({ 3,7,5 }, 1.0f).getShape()[0]);
			Assert::AreEqual(7, Tensor::full({ 3,7,5 }, 1.0f).getShape()[1]);
			Assert::AreEqual(5, Tensor::full({ 3,7,5 }, 1.0f).getShape()[2]);
		}

		TEST_METHOD(Size)
		{
			Assert::AreEqual(1, Tensor::full({}, -1.0f).getSize());
			Assert::AreEqual(1, Tensor::full({ 1 }, 2.0f).getSize());
			Assert::AreEqual(1, Tensor::full({ 1, 1, 1 }, 0.0f).getSize());
			Assert::AreEqual(2, Tensor::full({ 2 }, 1.0f).getSize());
			Assert::AreEqual(9, Tensor::full({ 3, 3 }, 5.0f).getSize());
			Assert::AreEqual(105, Tensor::full({ 3, 7, 5 }, 1.0f).getSize());
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1 = Tensor::full({ 3 }, 4.0f);
			CompareFloats(tensor1.get({ 0 }).item(), 4.0f);
			CompareFloats(tensor1.get({ 1 }).item(), 4.0f);
			CompareFloats(tensor1.get({ 2 }).item(), 4.0f);

			Tensor tensor2 = Tensor::full({ 2, 3 }, 2.0f);
			CompareFloats(tensor2.get({ 0, 0 }).item(), 2.0f);
			CompareFloats(tensor2.get({ 0, 1 }).item(), 2.0f);
			CompareFloats(tensor2.get({ 0, 2 }).item(), 2.0f);
			CompareFloats(tensor2.get({ 1, 0 }).item(), 2.0f);
			CompareFloats(tensor2.get({ 1, 1 }).item(), 2.0f);
			CompareFloats(tensor2.get({ 1, 2 }).item(), 2.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::full({ -1 }, 0.0f); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::full({ 0 }, 2.0f); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::full({ 1, 0, 3 }, 10.0f); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::full({ 4, 3, 2 }, 0.0f).getGradient());
			Assert::IsFalse(Tensor::full({  }, 5.0f).getGradient());
			Assert::IsFalse(Tensor::full({ 10 }, -3.0f).getGradient());
		}
	};

	TEST_CLASS(FromValuesTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Assert::AreEqual(1, Tensor::fromValues(new float[1], { 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor::fromValues(new float[3], { 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor::fromValues(new float[10], { 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor::fromValues(new float[10], { 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor::fromValues(new float[105], { 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::fromValues(new float[105], { 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor::fromValues(new float[105], { 3,7,5 }).getShape()[2]);
		}

		TEST_METHOD(Size)
		{
			Assert::AreEqual(1, Tensor::fromValues(new float[1], {}).getSize());
			Assert::AreEqual(1, Tensor::fromValues(new float[1], { 1 }).getSize());
			Assert::AreEqual(1, Tensor::fromValues(new float[1], { 1, 1, 1 }).getSize());
			Assert::AreEqual(2, Tensor::fromValues(new float[2], { 2 }).getSize());
			Assert::AreEqual(9, Tensor::fromValues(new float[9], { 3, 3 }).getSize());
			Assert::AreEqual(105, Tensor::fromValues(new float[105], { 3, 7, 5 }).getSize());
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1 = Tensor::fromValues(new float[3] { 1.0f, 2.0f, 3.0f }, { 3 });
			CompareFloats(tensor1.get({ 0 }).item(), 1.0f);
			CompareFloats(tensor1.get({ 1 }).item(), 2.0f);
			CompareFloats(tensor1.get({ 2 }).item(), 3.0f);

			Tensor tensor2 = Tensor::fromValues(new float[6] {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f}, { 2, 3 });
			CompareFloats(tensor2.get({ 0, 0 }).item(), -1.0f);
			CompareFloats(tensor2.get({ 0, 1 }).item(), -2.0f);
			CompareFloats(tensor2.get({ 0, 2 }).item(), -3.0f);
			CompareFloats(tensor2.get({ 1, 0 }).item(), -4.0f);
			CompareFloats(tensor2.get({ 1, 1 }).item(), -5.0f);
			CompareFloats(tensor2.get({ 1, 2 }).item(), -6.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::fromValues(new float[1], { -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::fromValues(new float[1], { 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::fromValues(new float[1], { 1, 0, 3 }); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::fromValues(new float[24], { 4, 3, 2 }).getGradient());
			Assert::IsFalse(Tensor::fromValues(new float[1], {  }).getGradient());
			Assert::IsFalse(Tensor::fromValues(new float[10], { 10 }).getGradient());
		}
	};

	TEST_CLASS(ItemTest)
	{
	public:
		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 3 }).item(); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 1, 1, 2 }).item(); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 1, 4, 1, 1 }).item(); });
		}

		TEST_METHOD(Value)
		{
			CompareFloats(Tensor::zeroes({ }).item(), 0.0f);
			CompareFloats(Tensor::zeroes({ 1, 1, 1 }).item(), 0.0f);

			Tensor tensor = Tensor::zeroes({ 2,3 }).set(-1.0f, { 1 });
			CompareFloats(tensor.get({ 1,2 }).item(), -1.0f);
		}
	};

	TEST_CLASS(AtTest)
	{
	public:
		TEST_METHOD(Value)
		{
			Tensor tensor1 = Tensor::zeroes({});
			CompareFloats(tensor1.at(0), 0.0f);

			Tensor tensor2 = Tensor::full({ 2,3 }, 2.0f).set(-2.0f, { 1 });
			CompareFloats(tensor2.at(0), 2.0f);
			CompareFloats(tensor2.at(1), 2.0f);
			CompareFloats(tensor2.at(2), 2.0f);
			CompareFloats(tensor2.at(3), -2.0f);
			CompareFloats(tensor2.at(4), -2.0f);
			CompareFloats(tensor2.at(5), -2.0f);
		}
	};

	TEST_CLASS(ReshapeTest)
	{
	public:
		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 1 }).reshape({ -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 1 }).reshape({ 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 1 }).reshape({ 1, 0, 3 }); });
		}

		TEST_METHOD(WrongSize)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 1 }).reshape({ 1, 2 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 3, 2 }).reshape({ 5, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 7, 1, 3 }).reshape({ 5, 4, 1 }); });
		}

		TEST_METHOD(NewShape)
		{
			Assert::AreEqual(1, Tensor::zeroes({}).reshape({ 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor::zeroes({ 1, 3, 1 }).reshape({ 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor::zeroes({ 10 }).reshape({ 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor::zeroes({ 10 }).reshape({ 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor::zeroes({ 5, 21 }).reshape({ 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::zeroes({ 5, 21 }).reshape({ 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor::zeroes({ 5, 21 }).reshape({ 3,7,5 }).getShape()[2]);
		}
	};

	TEST_CLASS(GetTest)
	{
	public:
		TEST_METHOD(TooManyIndices)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ }).get({ 0 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2 }).get({ 1, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2, 4, 6 }).get({ 1, 1, 2, 2 }); });
		}

		TEST_METHOD(IndexOutOfBounds)
		{
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 1, 3 }).get({ 2, 1 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 3 }).get({ 4 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 4, 2, 8, 1, 2 }).get({ 3, 1, 5, 4 }); });
		}

		TEST_METHOD(NewShape)
		{
			Assert::AreEqual(5, Tensor::zeroes({ 1, 4, 5, 7 }).get({ 0, 0 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::zeroes({ 1, 4, 5, 7 }).get({ 0, 0 }).getShape()[1]);

			Assert::AreEqual(2, Tensor::zeroes({ 3, 2 }).get({ 0 }).getShape()[0]);
		}

		TEST_METHOD(MatchingValue)
		{
			Tensor tensor1a = Tensor::full({ 2,3 }, 1.0f);
			Tensor tensor1b = tensor1a.get({ 1 });
			Assert::AreEqual(tensor1a.get({ 1,0 }).item(), tensor1b.get({ 0 }).item());

			Tensor tensor2a = Tensor::zeroes({ 5,4,3,7 });
			Tensor tensor2b = tensor2a.get({ 2, 3 });
			Assert::AreEqual(tensor2a.get({ 2, 3, 1, 5 }).item(), tensor2b.get({ 1, 5 }).item());
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1 = Tensor::zeroes({ 10, 3 });
			Tensor tensor2 = tensor1.get({});
			Assert::IsFalse(tensor2.getGradient());
			Assert::IsNull(tensor2.getFunction());

			tensor1.setGradient(true);
			Tensor tensor3 = tensor1.get({});
			Assert::IsTrue(tensor3.getGradient());
			Assert::IsNotNull((const GetFunction*)tensor3.getFunction());
		}
	};

	TEST_CLASS(SetSingleTest)
	{
	public:
		TEST_METHOD(TooManyIndices)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ }).set(0.0f, { 0 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2 }).set(0.0f, { 1, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2, 4, 6 }).set(0.0f, { 1, 1, 2, 2 }); });
		}

		TEST_METHOD(IndexOutOfBounds)
		{
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 1, 3 }).set(0.0f, { 2, 1 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 3 }).set(0.0f, { 4 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 4, 2, 8, 1, 2 }).set(0.0f, { 3, 1, 5, 4 }); });
		}

		TEST_METHOD(NewValue)
		{
			Tensor tensor1 = Tensor::zeroes({ 3 }).set(1.0f);
			CompareFloats(tensor1.get({ 0 }).item(), 1.0f);
			CompareFloats(tensor1.get({ 1 }).item(), 1.0f);
			CompareFloats(tensor1.get({ 2 }).item(), 1.0f);

			Tensor tensor2 = Tensor::zeroes({ 2, 3 }).set(1.0f, { 1 });
			CompareFloats(tensor2.get({ 0, 0 }).item(), 0.0f);
			CompareFloats(tensor2.get({ 0, 1 }).item(), 0.0f);
			CompareFloats(tensor2.get({ 0, 2 }).item(), 0.0f);
			CompareFloats(tensor2.get({ 1, 0 }).item(), 1.0f);
			CompareFloats(tensor2.get({ 1, 1 }).item(), 1.0f);
			CompareFloats(tensor2.get({ 1, 2 }).item(), 1.0f);
		}

		TEST_METHOD(IndependentValues)
		{
			Tensor tensor1 = Tensor::zeroes({ 3 });
			Tensor tensor2 = tensor1.set(1.0f, {});
			Assert::AreEqual(tensor1.at(0), 0.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1 = Tensor::zeroes({ 10, 3 });
			Tensor tensor2 = tensor1.set(0);
			Assert::IsFalse(tensor2.getGradient());
			Assert::IsNull(tensor2.getFunction());

			tensor1.setGradient(true);
			Tensor tensor3 = tensor1.set(0);
			Assert::IsTrue(tensor3.getGradient());
			Assert::IsNotNull((const SetSingleFunction*)tensor3.getFunction());
		}
	};

	TEST_CLASS(SetTensorTest)
	{
	public:
		TEST_METHOD(TooManyIndices)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ }).set(Tensor::zeroes({}), { 0 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2 }).set(Tensor::zeroes({}), { 1, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2, 4, 6 }).set(Tensor::zeroes({}), { 1, 1, 2, 2 }); });
		}

		TEST_METHOD(IndexOutOfBounds)
		{
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 1, 3 }).set(Tensor::zeroes({}), { 2, 1 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 3 }).set(Tensor::zeroes({}), { 4 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 4, 2, 8, 1, 2 }).set(Tensor::zeroes({}), { 3, 1, 5, 4 }); });
		}

		TEST_METHOD(UnbroadcastableTensor)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::zeroes({ 10, 3, 5 }).set(Tensor::zeroes({ 1, 2, 5 }), { 0 }); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::zeroes({ 10, 1, 3, 5 }).set(Tensor::zeroes({ 3, 10 }), { 0 }); }
			);
		}

		TEST_METHOD(NewValue)
		{
			Tensor tensor1 = Tensor::zeroes({ 3 }).set(Tensor::ones({ 1 }), { 2 });
			CompareFloats(tensor1.get({ 0 }).item(), 0.0f);
			CompareFloats(tensor1.get({ 1 }).item(), 0.0f);
			CompareFloats(tensor1.get({ 2 }).item(), 1.0f);

			Tensor tensor2 = Tensor::zeroes({ 2, 1, 3 }).set(Tensor::ones({ 1, 1 }), { 1 });
			CompareFloats(tensor2.at(0), 0.0f);
			CompareFloats(tensor2.at(1), 0.0f);
			CompareFloats(tensor2.at(2), 0.0f);
			CompareFloats(tensor2.at(3), 1.0f);
			CompareFloats(tensor2.at(4), 1.0f);
			CompareFloats(tensor2.at(5), 1.0f);
		}

		TEST_METHOD(Independentvalues)
		{
			Tensor tensor1 = Tensor::zeroes({ 3 });
			Tensor tensor2 = tensor1.set(Tensor::ones({ 1 }), { 2 });
			CompareFloats(tensor1.get({ 2 }).item(), 0.0f);
		}
	};

	TEST_CLASS(GradientTest)
	{
	public:
		TEST_METHOD(Value)
		{
			Tensor tensor1 = Tensor::zeroes({});
			tensor1.setGradient(true);
			Assert::IsTrue(tensor1.getGradient());

			Tensor tensor2 = Tensor::ones({ 4, 3 });
			tensor2.setGradient(false);
			Assert::IsFalse(tensor2.getGradient());

			Tensor tensor3 = Tensor::full({ 10 }, 1.0f);
			tensor3.setGradient(true);
			tensor3.setGradient(true);
			Assert::IsTrue(tensor3.getGradient());
			tensor3.setGradient(false);
			Assert::IsFalse(tensor3.getGradient());
		}
	};
}
