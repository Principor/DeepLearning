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
			CompareFloats(tensor1.at(0), 0.0f);
			CompareFloats(tensor1.at(1), 0.0f);
			CompareFloats(tensor1.at(2), 0.0f);

			Tensor tensor2 = Tensor::zeroes({ 2, 3 });
			CompareFloats(tensor2.at({ 0, 0 }), 0.0f);
			CompareFloats(tensor2.at({ 0, 1 }), 0.0f);
			CompareFloats(tensor2.at({ 0, 2 }), 0.0f);
			CompareFloats(tensor2.at({ 1, 0 }), 0.0f);
			CompareFloats(tensor2.at({ 1, 1 }), 0.0f);
			CompareFloats(tensor2.at({ 1, 2 }), 0.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 1, 0, 3 }); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::zeroes({ 4, 3, 2 }).requiresGradient());
			Assert::IsFalse(Tensor::zeroes({  }).requiresGradient());
			Assert::IsFalse(Tensor::zeroes({ 10 }).requiresGradient());
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
			CompareFloats(tensor1.at(0), 1.0f);
			CompareFloats(tensor1.at(1), 1.0f);
			CompareFloats(tensor1.at(2), 1.0f);

			Tensor tensor2 = Tensor::ones({ 2, 3 });
			CompareFloats(tensor2.at({ 0, 0 }), 1.0f);
			CompareFloats(tensor2.at({ 0, 1 }), 1.0f);
			CompareFloats(tensor2.at({ 0, 2 }), 1.0f);
			CompareFloats(tensor2.at({ 1, 0 }), 1.0f);
			CompareFloats(tensor2.at({ 1, 1 }), 1.0f);
			CompareFloats(tensor2.at({ 1, 2 }), 1.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::ones({ -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::ones({ 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::ones({ 1, 0, 3 }); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::ones({ 4, 3, 2 }).requiresGradient());
			Assert::IsFalse(Tensor::ones({  }).requiresGradient());
			Assert::IsFalse(Tensor::ones({ 10 }).requiresGradient());
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
			CompareFloats(tensor1.at(0), 4.0f);
			CompareFloats(tensor1.at(1), 4.0f);
			CompareFloats(tensor1.at(2), 4.0f);

			Tensor tensor2 = Tensor::full({ 2, 3 }, 2.0f);
			CompareFloats(tensor2.at({ 0, 0 }), 2.0f);
			CompareFloats(tensor2.at({ 0, 1 }), 2.0f);
			CompareFloats(tensor2.at({ 0, 2 }), 2.0f);
			CompareFloats(tensor2.at({ 1, 0 }), 2.0f);
			CompareFloats(tensor2.at({ 1, 1 }), 2.0f);
			CompareFloats(tensor2.at({ 1, 2 }), 2.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::full({ -1 }, 0.0f); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::full({ 0 }, 2.0f); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::full({ 1, 0, 3 }, 10.0f); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::full({ 4, 3, 2 }, 0.0f).requiresGradient());
			Assert::IsFalse(Tensor::full({  }, 5.0f).requiresGradient());
			Assert::IsFalse(Tensor::full({ 10 }, -3.0f).requiresGradient());
		}
	};

	TEST_CLASS(RangeTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Assert::AreEqual(1, Tensor::range({ 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor::range({ 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor::range({ 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor::range({ 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor::range({ 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::range({ 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor::range({ 3,7,5 }).getShape()[2]);
		}

		TEST_METHOD(Size)
		{
			Assert::AreEqual(1, Tensor::range({}).getSize());
			Assert::AreEqual(1, Tensor::range({ 1 }).getSize());
			Assert::AreEqual(1, Tensor::range({ 1, 1, 1 }).getSize());
			Assert::AreEqual(2, Tensor::range({ 2 }).getSize());
			Assert::AreEqual(9, Tensor::range({ 3, 3 }).getSize());
			Assert::AreEqual(105, Tensor::range({ 3, 7, 5 }).getSize());
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1 = Tensor::range({ 3 });
			CompareFloats(tensor1.at(0), 0.0f);
			CompareFloats(tensor1.at(1), 1.0f);
			CompareFloats(tensor1.at(2), 2.0f);

			Tensor tensor2 = Tensor::range({ 2, 3 }, 5, -1);
			CompareFloats(tensor2.at({ 0, 0 }), 5.0f);
			CompareFloats(tensor2.at({ 0, 1 }), 4.0f);
			CompareFloats(tensor2.at({ 0, 2 }), 3.0f);
			CompareFloats(tensor2.at({ 1, 0 }), 2.0f);
			CompareFloats(tensor2.at({ 1, 1 }), 1.0f);
			CompareFloats(tensor2.at({ 1, 2 }), 0.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::range({ -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::range({ 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::range({ 1, 0, 3 }); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::range({ 4, 3, 2 }).requiresGradient());
			Assert::IsFalse(Tensor::range({  }).requiresGradient());
			Assert::IsFalse(Tensor::range({ 10 }).requiresGradient());
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
			CompareFloats(tensor1.at(0), 1.0f);
			CompareFloats(tensor1.at(1), 2.0f);
			CompareFloats(tensor1.at(2), 3.0f);

			Tensor tensor2 = Tensor::fromValues(new float[6] {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f}, { 2, 3 });
			CompareFloats(tensor2.at({ 0, 0 }), -1.0f);
			CompareFloats(tensor2.at({ 0, 1 }), -2.0f);
			CompareFloats(tensor2.at({ 0, 2 }), -3.0f);
			CompareFloats(tensor2.at({ 1, 0 }), -4.0f);
			CompareFloats(tensor2.at({ 1, 1 }), -5.0f);
			CompareFloats(tensor2.at({ 1, 2 }), -6.0f);
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::fromValues(new float[1], { -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::fromValues(new float[1], { 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::fromValues(new float[1], { 1, 0, 3 }); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::fromValues(new float[24], { 4, 3, 2 }).requiresGradient());
			Assert::IsFalse(Tensor::fromValues(new float[1], {  }).requiresGradient());
			Assert::IsFalse(Tensor::fromValues(new float[10], { 10 }).requiresGradient());
		}
	};

	TEST_CLASS(UniformTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Assert::AreEqual(1, Tensor::uniform({ 1 }, 0, 1).getShape()[0]);

			Assert::AreEqual(3, Tensor::uniform({ 3 }, 0, 1).getShape()[0]);

			Assert::AreEqual(5, Tensor::uniform({ 5, 2 }, 0, 1).getShape()[0]);
			Assert::AreEqual(2, Tensor::uniform({ 5, 2 }, 0, 1).getShape()[1]);

			Assert::AreEqual(3, Tensor::uniform({ 3,7,5 }, 0, 1).getShape()[0]);
			Assert::AreEqual(7, Tensor::uniform({ 3,7,5 }, 0, 1).getShape()[1]);
			Assert::AreEqual(5, Tensor::uniform({ 3,7,5 }, 0, 1).getShape()[2]);
		}

		TEST_METHOD(Size)
		{
			Assert::AreEqual(1, Tensor::uniform({}, 0, 1).getSize());
			Assert::AreEqual(1, Tensor::uniform({ 1 }, 0, 1).getSize());
			Assert::AreEqual(1, Tensor::uniform({ 1, 1, 1 }, 0, 1).getSize());
			Assert::AreEqual(2, Tensor::uniform({ 2 }, 0, 1).getSize());
			Assert::AreEqual(9, Tensor::uniform({ 3, 3 }, 0, 1).getSize());
			Assert::AreEqual(105, Tensor::uniform({ 3, 7, 5 }, 0, 1).getSize());
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::uniform({ -1 }, 0, 1); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::uniform({ 0 }, 0, 1); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::uniform({ 1, 0, 3 }, 0, 1); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::uniform({ 4, 3, 2 }, 0, 1).requiresGradient());
			Assert::IsFalse(Tensor::uniform({  }, 0, 1).requiresGradient());
			Assert::IsFalse(Tensor::uniform({ 10 }, 0, 1).requiresGradient());
		}
	};

	TEST_CLASS(NormalTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Assert::AreEqual(1, Tensor::normal({ 1 }, 0, 1).getShape()[0]);

			Assert::AreEqual(3, Tensor::normal({ 3 }, 0, 1).getShape()[0]);

			Assert::AreEqual(5, Tensor::normal({ 5, 2 }, 0, 1).getShape()[0]);
			Assert::AreEqual(2, Tensor::normal({ 5, 2 }, 0, 1).getShape()[1]);

			Assert::AreEqual(3, Tensor::normal({ 3,7,5 }, 0, 1).getShape()[0]);
			Assert::AreEqual(7, Tensor::normal({ 3,7,5 }, 0, 1).getShape()[1]);
			Assert::AreEqual(5, Tensor::normal({ 3,7,5 }, 0, 1).getShape()[2]);
		}

		TEST_METHOD(Size)
		{
			Assert::AreEqual(1, Tensor::normal({}, 0, 1).getSize());
			Assert::AreEqual(1, Tensor::normal({ 1 }, 0, 1).getSize());
			Assert::AreEqual(1, Tensor::normal({ 1, 1, 1 }, 0, 1).getSize());
			Assert::AreEqual(2, Tensor::normal({ 2 }, 0, 1).getSize());
			Assert::AreEqual(9, Tensor::normal({ 3, 3 }, 0, 1).getSize());
			Assert::AreEqual(105, Tensor::normal({ 3, 7, 5 }, 0, 1).getSize());
		}

		TEST_METHOD(InvalidShape)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::normal({ -1 }, 0, 1); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::normal({ 0 }, 0, 1); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::normal({ 1, 0, 3 }, 0, 1); });
		}

		TEST_METHOD(RequiresGradient)
		{
			Assert::IsFalse(Tensor::normal({ 4, 3, 2 }, 0, 1).requiresGradient());
			Assert::IsFalse(Tensor::normal({  }, 0, 1).requiresGradient());
			Assert::IsFalse(Tensor::normal({ 10 }, 0, 1).requiresGradient());
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

	TEST_CLASS(AtIndexTest)
	{
	public:
		TEST_METHOD(IndexOutOfBounds)
		{
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 1, 3 }).at(3); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 3 }).at(-1); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 4, 2, 8, 1, 2 }).at(200); });
		}

		TEST_METHOD(Value)
		{
			Tensor tensor1 = Tensor::zeroes({});
			CompareFloats(tensor1.at(0), 0.0f);

			Tensor tensor2 = Tensor::range({ 2,3 });
			CompareFloats(tensor2.at(0), 0.0f);
			CompareFloats(tensor2.at(1), 1.0f);
			CompareFloats(tensor2.at(2), 2.0f);
			CompareFloats(tensor2.at(3), 3.0f);
			CompareFloats(tensor2.at(4), 4.0f);
			CompareFloats(tensor2.at(5), 5.0f);
		}
	};

	TEST_CLASS(AtIndicesTest)
	{
	public:
		TEST_METHOD(WrongNumberOfIndices)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ }).at({ 5,2 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2, 1, 1 }).at({ 1, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2, 4, 6 }).at({ 1, 1, 2, 2 }); });
		}

		TEST_METHOD(IndexOutOfBounds)
		{
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 1, 3 }).at({ 2, 1 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 5 }).at({ -1 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 3 }).at({ 4 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 4, 2, 8, 1, 2 }).at({ 3, 1, 5, 4, 1 }); });
		}

		TEST_METHOD(Value)
		{
			Tensor tensor1 = Tensor::zeroes({});
			CompareFloats(tensor1.at({ 0 }), 0.0f);

			Tensor tensor2 = Tensor::range({ 2,3 });
			CompareFloats(tensor2.at({ 0, 0 }), 0.0f);
			CompareFloats(tensor2.at({ 0, 1 }), 1.0f);
			CompareFloats(tensor2.at({ 0, 2 }), 2.0f);
			CompareFloats(tensor2.at({ 1, 0 }), 3.0f);
			CompareFloats(tensor2.at({ 1, 1 }), 4.0f);
			CompareFloats(tensor2.at({ 1, 2 }), 5.0f);
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
			Tensor tensor1a = Tensor::range({ 2,3 });
			Tensor tensor1b = tensor1a.get({ 1 });
			Assert::AreEqual(tensor1a.at({ 1,0 }), tensor1b.at(0));

			Tensor tensor2a = Tensor::range({ 5,4,3,7 });
			Tensor tensor2b = tensor2a.get({ 2, 3 });
			Assert::AreEqual(tensor2a.at({ 2, 3, 1, 5 }), tensor2b.at({ 1, 5 }));
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1 = Tensor::zeroes({ 10, 3 });
			Tensor tensor2 = tensor1.get({});
			Assert::IsFalse(tensor2.requiresGradient());
			Assert::IsNull(tensor2.getFunction());

			tensor1.requireGradient();
			Tensor tensor3 = tensor1.get({});
			Assert::IsTrue(tensor3.requiresGradient());
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
			CompareFloats(tensor1.at(0), 1.0f);
			CompareFloats(tensor1.at(1), 1.0f);
			CompareFloats(tensor1.at(2), 1.0f);

			Tensor tensor2 = Tensor::range({ 2, 3 }).set(-1.0f, { 1 });
			CompareFloats(tensor2.at({ 0, 0 }), 0.0f);
			CompareFloats(tensor2.at({ 0, 1 }), 1.0f);
			CompareFloats(tensor2.at({ 0, 2 }), 2.0f);
			CompareFloats(tensor2.at({ 1, 0 }), -1.0f);
			CompareFloats(tensor2.at({ 1, 1 }), -1.0f);
			CompareFloats(tensor2.at({ 1, 2 }), -1.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1 = Tensor::zeroes({ 10, 3 });
			Tensor tensor2 = tensor1.set(0);
			Assert::IsFalse(tensor2.requiresGradient());
			Assert::IsNull(tensor2.getFunction());

			tensor1.requireGradient();
			Tensor tensor3 = tensor1.set(0);
			Assert::IsTrue(tensor3.requiresGradient());
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

		TEST_METHOD(UnbroadcastableDims)
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
			CompareFloats(tensor1.at(0), 0.0f);
			CompareFloats(tensor1.at(1), 0.0f);
			CompareFloats(tensor1.at(2), 1.0f);

			Tensor tensor2 = Tensor::range({ 2, 1, 3 }, 0, -1).set(Tensor::ones({ 1, 1 }), { 1 });
			CompareFloats(tensor2.at({ 0,0,0 }), 0.0f);
			CompareFloats(tensor2.at({ 0,0,1 }), -1.0f);
			CompareFloats(tensor2.at({ 0,0,2 }), -2.0f);
			CompareFloats(tensor2.at({ 1,0,0 }), 1.0f);
			CompareFloats(tensor2.at({ 1,0,1 }), 1.0f);
			CompareFloats(tensor2.at({ 1,0,2 }), 1.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 10, 3 });
			Tensor tensor1b = tensor1a.set(Tensor::zeroes({}));
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 10,3 });
			Tensor tensor2b = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2c = tensor2a.set(tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((const SetTensorFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::zeroes({ 10, 3 }).requireGradient();
			Tensor tensor3b = tensor3a.set(Tensor::zeroes({}));
			Assert::IsTrue(tensor3b.requiresGradient());
			Assert::IsNotNull((const SetTensorFunction*)tensor3b.getFunction());
		}
	};

	TEST_CLASS(AddSingleTest)
	{
	public:
		TEST_METHOD(NewValue)
		{
			Tensor tensor1a = Tensor::range({ 3,1 });
			Tensor tensor1b = Tensor::add(tensor1a, 3.0f);
			Assert::AreEqual(tensor1b.at(0), 3.0f);
			Assert::AreEqual(tensor1b.at(1), 4.0f);
			Assert::AreEqual(tensor1b.at(2), 5.0f);

			Tensor tensor2a = Tensor::range({ 2 }, 1);
			Tensor tensor2b = Tensor::add(tensor2a, -1.0f);
			Assert::AreEqual(tensor2b.at(0), 0.0f);
			Assert::AreEqual(tensor2b.at(1), 1.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::add(tensor1a, 3.0f);
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::ones({ 2 }).set(-1, { 0 }).requireGradient();
			Tensor tensor2b = Tensor::add(tensor2a, -1.0f);
			Assert::IsTrue(tensor2b.requiresGradient());
			Assert::IsNotNull((AddSingleFunction*)tensor2b.getFunction());
		}
	};

	TEST_CLASS(AddTensorTest)
	{
	public:
		TEST_METHOD(UnbroadcastableDims)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::add(Tensor::zeroes({ 10, 3, 5 }), Tensor::zeroes({ 1, 2, 5 })); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::add(Tensor::zeroes({ 10, 1, 3, 5 }), Tensor::zeroes({ 3, 10 })); }
			);
		}

		TEST_METHOD(NewValue)
		{
			Tensor tensor1 = Tensor::add(Tensor::range({ 3 }), Tensor::ones({ 1 }));
			CompareFloats(tensor1.at(0), 1.0f);
			CompareFloats(tensor1.at(1), 2.0f);
			CompareFloats(tensor1.at(2), 3.0f);

			Tensor tensor2 = Tensor::add(Tensor::range({ 2, 1, 3 }), Tensor::range({ 1, 3 }));
			CompareFloats(tensor2.at({ 0,0,0 }), 0.0f);
			CompareFloats(tensor2.at({ 0,0,1 }), 2.0f);
			CompareFloats(tensor2.at({ 0,0,2 }), 4.0f);
			CompareFloats(tensor2.at({ 1,0,0 }), 3.0f);
			CompareFloats(tensor2.at({ 1,0,1 }), 5.0f);
			CompareFloats(tensor2.at({ 1,0,2 }), 7.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::add(tensor1a, Tensor::zeroes({ 1 }));
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 1 });
			Tensor tensor2b = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2c = Tensor::add(tensor2a, tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((AddTensorFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::ones({ 2, 1, 3 }).requireGradient();
			Tensor tensor3b = Tensor::add(tensor3a, Tensor::ones({ 1, 1 }));
			Assert::IsTrue(tensor3b.requiresGradient());
			Assert::IsNotNull((AddTensorFunction*)tensor3b.getFunction());
		}
	};

	TEST_CLASS(SubtractSingleTest)
	{
	public:
		TEST_METHOD(NewValue)
		{
			Tensor tensor1a = Tensor::range({ 3,1 });
			Tensor tensor1b = Tensor::subtract(tensor1a, 3.0f);
			Assert::AreEqual(tensor1b.at(0), -3.0f);
			Assert::AreEqual(tensor1b.at(1), -2.0f);
			Assert::AreEqual(tensor1b.at(2), -1.0f);

			Tensor tensor2a = Tensor::range({ 2 }, 1);
			Tensor tensor2b = Tensor::subtract(tensor2a, -1.0f);
			Assert::AreEqual(tensor2b.at(0), 2.0f);
			Assert::AreEqual(tensor2b.at(1), 3.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::subtract(tensor1a, 3.0f);
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::ones({ 2 }).set(-1, { 0 }).requireGradient();
			Tensor tensor2b = Tensor::subtract(tensor2a, -1.0f);
			Assert::IsTrue(tensor2b.requiresGradient());
			Assert::IsNotNull((AddSingleFunction*)tensor2b.getFunction());
		}
	};

	TEST_CLASS(SubtractTensorTest)
	{
	public:
		TEST_METHOD(UnbroadcastableDims)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::subtract(Tensor::zeroes({ 10, 3, 5 }), Tensor::zeroes({ 1, 2, 5 })); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::subtract(Tensor::zeroes({ 10, 1, 3, 5 }), Tensor::zeroes({ 3, 10 })); }
			);
		}

		TEST_METHOD(NewValue)
		{
			Tensor tensor1 = Tensor::subtract(Tensor::range({ 3 }), Tensor::ones({ 1 }));
			CompareFloats(tensor1.at(0), -1.0f);
			CompareFloats(tensor1.at(1), 0.0f);
			CompareFloats(tensor1.at(2), 1.0f);

			Tensor tensor2 = Tensor::subtract(Tensor::range({ 2, 1, 3 }), Tensor::range({ 1, 3 }));
			CompareFloats(tensor2.at({ 0,0,0 }), 0.0f);
			CompareFloats(tensor2.at({ 0,0,1 }), 0.0f);
			CompareFloats(tensor2.at({ 0,0,2 }), 0.0f);
			CompareFloats(tensor2.at({ 1,0,0 }), 3.0f);
			CompareFloats(tensor2.at({ 1,0,1 }), 3.0f);
			CompareFloats(tensor2.at({ 1,0,2 }), 3.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::subtract(tensor1a, Tensor::zeroes({ 1 }));
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 1 });
			Tensor tensor2b = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2c = Tensor::subtract(tensor2a, tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((AddTensorFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::ones({ 2, 1, 3 }).requireGradient();
			Tensor tensor3b = Tensor::subtract(tensor3a, Tensor::ones({ 1, 1 }));
			Assert::IsTrue(tensor3b.requiresGradient());
			Assert::IsNotNull((AddTensorFunction*)tensor3b.getFunction());
		}
	};

	TEST_CLASS(MultiplySingleTest)
	{
	public:
		TEST_METHOD(NewValue)
		{
			Tensor tensor1a = Tensor::range({ 3,1 });
			Tensor tensor1b = Tensor::multiply(tensor1a, 3.0f);
			Assert::AreEqual(tensor1b.at(0), 0.0f);
			Assert::AreEqual(tensor1b.at(1), 3.0f);
			Assert::AreEqual(tensor1b.at(2), 6.0f);

			Tensor tensor2a = Tensor::range({ 2 }, 1);
			Tensor tensor2b = Tensor::multiply(tensor2a, -1.0f);
			Assert::AreEqual(tensor2b.at(0), -1.0f);
			Assert::AreEqual(tensor2b.at(1), -2.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::multiply(tensor1a, 3.0f);
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::ones({ 2 }).set(-1, { 0 }).requireGradient();
			Tensor tensor2b = Tensor::multiply(tensor2a, -1.0f);
			Assert::IsTrue(tensor2b.requiresGradient());
			Assert::IsNotNull((AddSingleFunction*)tensor2b.getFunction());
		}
	};

	TEST_CLASS(MultiplyTensorTest)
	{
	public:
		TEST_METHOD(UnbroadcastableDims)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::multiply(Tensor::zeroes({ 10, 3, 5 }), Tensor::zeroes({ 1, 2, 5 })); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::multiply(Tensor::zeroes({ 10, 1, 3, 5 }), Tensor::zeroes({ 3, 10 })); }
			);
		}

		TEST_METHOD(NewValue)
		{
			Tensor tensor1 = Tensor::multiply(Tensor::range({ 3 }), Tensor::full({ 1 }, 2));
			CompareFloats(tensor1.at(0), 0.0f);
			CompareFloats(tensor1.at(1), 2.0f);
			CompareFloats(tensor1.at(2), 4.0f);

			Tensor tensor2 = Tensor::multiply(Tensor::range({ 2, 1, 3 }), Tensor::range({ 1, 3 }));
			CompareFloats(tensor2.at({ 0,0,0 }), 0.0f);
			CompareFloats(tensor2.at({ 0,0,1 }), 1.0f);
			CompareFloats(tensor2.at({ 0,0,2 }), 4.0f);
			CompareFloats(tensor2.at({ 1,0,0 }), 0.0f);
			CompareFloats(tensor2.at({ 1,0,1 }), 4.0f);
			CompareFloats(tensor2.at({ 1,0,2 }), 10.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::multiply(tensor1a, Tensor::zeroes({ 1 }));
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 1 });
			Tensor tensor2b = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2c = Tensor::multiply(tensor2a, tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((AddTensorFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::ones({ 2, 1, 3 }).requireGradient();
			Tensor tensor3b = Tensor::multiply(tensor3a, Tensor::ones({ 1, 1 }));
			Assert::IsTrue(tensor3b.requiresGradient());
			Assert::IsNotNull((AddTensorFunction*)tensor3b.getFunction());
		}
	};

	TEST_CLASS(DivideSingleTest)
	{
	public:
		TEST_METHOD(NewValue)
		{
			Tensor tensor1a = Tensor::range({ 3,1 }, 0, 3);
			Tensor tensor1b = Tensor::divide(tensor1a, 3.0f);
			Assert::AreEqual(tensor1b.at(0), 0.0f);
			Assert::AreEqual(tensor1b.at(1), 1.0f);
			Assert::AreEqual(tensor1b.at(2), 2.0f);

			Tensor tensor2a = Tensor::range({ 2 }, 1);
			Tensor tensor2b = Tensor::divide(tensor2a, -1.0f);
			Assert::AreEqual(tensor2b.at(0), -1.0f);
			Assert::AreEqual(tensor2b.at(1), -2.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::divide(tensor1a, 3.0f);
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::ones({ 2 }).set(-1, { 0 }).requireGradient();
			Tensor tensor2b = Tensor::divide(tensor2a, -1.0f);
			Assert::IsTrue(tensor2b.requiresGradient());
			Assert::IsNotNull((AddSingleFunction*)tensor2b.getFunction());
		}
	};

	TEST_CLASS(DivideTensorTest)
	{
	public:
		TEST_METHOD(UnbroadcastableDims)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::divide(Tensor::zeroes({ 10, 3, 5 }), Tensor::zeroes({ 1, 2, 5 })); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::divide(Tensor::zeroes({ 10, 1, 3, 5 }), Tensor::zeroes({ 3, 10 })); }
			);
		}

		TEST_METHOD(NewValue)
		{
			Tensor tensor1 = Tensor::divide(Tensor::range({ 3 }), Tensor::full({ 1 }, 2));
			CompareFloats(tensor1.at(0), 0.0f);
			CompareFloats(tensor1.at(1), 0.5f);
			CompareFloats(tensor1.at(2), 1.0f);

			Tensor tensor2 = Tensor::divide(Tensor::range({ 2, 1, 3 }, 1), Tensor::range({ 1, 3 }, 1));
			CompareFloats(tensor2.at({ 0,0,0 }), 1.0f);
			CompareFloats(tensor2.at({ 0,0,1 }), 1.0f);
			CompareFloats(tensor2.at({ 0,0,2 }), 1.0f);
			CompareFloats(tensor2.at({ 1,0,0 }), 4.0f);
			CompareFloats(tensor2.at({ 1,0,1 }), 2.5f);
			CompareFloats(tensor2.at({ 1,0,2 }), 2.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::divide(tensor1a, Tensor::zeroes({ 1 }));
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 1 });
			Tensor tensor2b = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2c = Tensor::divide(tensor2a, tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((DivideTensorFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::ones({ 2, 1, 3 }).requireGradient();
			Tensor tensor3b = Tensor::divide(tensor3a, Tensor::ones({ 1, 1 }));
			Assert::IsTrue(tensor3b.requiresGradient());
			Assert::IsNotNull((DivideTensorFunction*)tensor3b.getFunction());
		}
	};

	TEST_CLASS(TransposeTest)
	{
	public:
		TEST_METHOD(InsuffientDims)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 3 }).transpose(); });

			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 1 }).transpose(); });

			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({}).transpose(); });
		}

		TEST_METHOD(NewShape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,3 });
			Tensor tensor1b = tensor1a.transpose();
			Assert::AreEqual(tensor1b.getShape()[0], 3);
			Assert::AreEqual(tensor1b.getShape()[1], 2);

			Tensor tensor2a = Tensor::zeroes({ 10, 2, 1, 5 });
			Tensor tensor2b = tensor2a.transpose();
			Assert::AreEqual(tensor2b.getShape()[0], 10);
			Assert::AreEqual(tensor2b.getShape()[1], 2);
			Assert::AreEqual(tensor2b.getShape()[2], 5);
			Assert::AreEqual(tensor2b.getShape()[3], 1);
		}

		TEST_METHOD(NewValues) {
			Tensor tensor1a = Tensor::range({ 2,3 });
			Tensor tensor1b = tensor1a.transpose();
			CompareFloats(tensor1b.at({ 0,0 }), tensor1a.at({ 0,0 }));
			CompareFloats(tensor1b.at({ 1,0 }), tensor1a.at({ 0,1 }));
			CompareFloats(tensor1b.at({ 2,0 }), tensor1a.at({ 0,2 }));
			CompareFloats(tensor1b.at({ 0,1 }), tensor1a.at({ 1,0 }));
			CompareFloats(tensor1b.at({ 1,1 }), tensor1a.at({ 1,1 }));
			CompareFloats(tensor1b.at({ 2,1 }), tensor1a.at({ 1,2 }));

			Tensor tensor2a = Tensor::range({ 10, 2, 1, 5 });
			Tensor tensor2b = tensor2a.transpose();
			for (int i = 0; i < 10; i++) {
				for (int j = 0; j < 2; j++) {
					for (int k = 0; k < 1; k++) {
						for (int l = 0; l < 5; l++) {
							CompareFloats(tensor2a.at({ i,j,k,l }), tensor2b.at({ i,j,l,k }));
						}
					}
				}
			}

		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = tensor1a.transpose();
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::ones({ 2, 1, 3 }).requireGradient();
			Tensor tensor2b = tensor2a.transpose();
			Assert::IsTrue(tensor2b.requiresGradient());
			Assert::IsNotNull((TransposeFunction*)tensor2b.getFunction());
		}
	};

	TEST_CLASS(MatrixMultiplyTest)
	{
	public:
		TEST_METHOD(InsufficientDims)
		{
			Assert::ExpectException<std::length_error>(
				[]() {Tensor::matrixMultiply(Tensor::zeroes({ 1 }), Tensor::zeroes({ 10, 3 })); }
			);

			Assert::ExpectException<std::length_error>(
				[]() { Tensor::matrixMultiply(Tensor::zeroes({ 10, 2, 5 }), Tensor::zeroes({})); }
			);
		}

		TEST_METHOD(WrongInnerDimsTest)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::matrixMultiply(Tensor::zeroes({ 10, 3, 1 }), Tensor::zeroes({ 2, 5 })); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::matrixMultiply(Tensor::zeroes({ 7, 2 }), Tensor::zeroes({ 5, 1, 4 })); }
			);
		}

		TEST_METHOD(UnbroadcastableDims)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::matrixMultiply(Tensor::zeroes({ 10, 3, 1 }), Tensor::zeroes({ 2, 1, 3 })); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::matrixMultiply(Tensor::zeroes({ 5, 3, 3, 1 }), Tensor::zeroes({ 5, 1, 3 })); }
			);
		}

		TEST_METHOD(NewShape)
		{
			Tensor tensor1a = Tensor::zeroes({ 10, 3, 1 });
			Tensor tensor1b = Tensor::zeroes({ 1, 1, 2 });
			Tensor tensor1c = Tensor::matrixMultiply(tensor1a, tensor1b);
			Assert::AreEqual(tensor1c.getShape()[0], 10);
			Assert::AreEqual(tensor1c.getShape()[1], 3);
			Assert::AreEqual(tensor1c.getShape()[2], 2);

			Tensor tensor2a = Tensor::zeroes({ 7, 5, 4, 2 });
			Tensor tensor2b = Tensor::zeroes({ 5, 2, 3 });
			Tensor tensor2c = Tensor::matrixMultiply(tensor2a, tensor2b);
			Assert::AreEqual(tensor2c.getShape()[0], 7);
			Assert::AreEqual(tensor2c.getShape()[1], 5);
			Assert::AreEqual(tensor2c.getShape()[2], 4);
			Assert::AreEqual(tensor2c.getShape()[3], 3);
		}

		TEST_METHOD(NewValues)
		{
			Tensor tensor1a = Tensor::range({ 2, 3 }, 1);
			Tensor tensor1b = Tensor::range({ 3, 4 }, 1);
			Tensor tensor1c = Tensor::matrixMultiply(tensor1a, tensor1b);
			CompareFloats(tensor1c.at(0), 38);
			CompareFloats(tensor1c.at(1), 44);
			CompareFloats(tensor1c.at(2), 50);
			CompareFloats(tensor1c.at(3), 56);
			CompareFloats(tensor1c.at(4), 83);
			CompareFloats(tensor1c.at(5), 98);
			CompareFloats(tensor1c.at(6), 113);
			CompareFloats(tensor1c.at(7), 128);

			Tensor tensor2a = Tensor::range({ 2, 1, 1, 3 }, 1);
			Tensor tensor2b = Tensor::range({ 1, 3 ,3, 2 }, 1);
			Tensor tensor2c = Tensor::matrixMultiply(tensor2a, tensor2b);
			CompareFloats(tensor2c.at(0), 22);
			CompareFloats(tensor2c.at(1), 28);
			CompareFloats(tensor2c.at(2), 58);
			CompareFloats(tensor2c.at(3), 64);
			CompareFloats(tensor2c.at(4), 94);
			CompareFloats(tensor2c.at(5), 100);
			CompareFloats(tensor2c.at(6), 49);
			CompareFloats(tensor2c.at(7), 64);
			CompareFloats(tensor2c.at(8), 139);
			CompareFloats(tensor2c.at(9), 154);
			CompareFloats(tensor2c.at(10), 229);
			CompareFloats(tensor2c.at(11), 244);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 10, 3, 1 });
			Tensor tensor1b = Tensor::zeroes({ 1, 1, 2 });
			Tensor tensor1c = Tensor::matrixMultiply(tensor1a, tensor1b);
			Assert::IsFalse(tensor1c.requiresGradient());
			Assert::IsNull(tensor1c.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 7, 5, 4, 2 }).requireGradient();
			Tensor tensor2b = Tensor::zeroes({ 5, 2, 3 });
			Tensor tensor2c = Tensor::matrixMultiply(tensor2a, tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((MatrixMultiplicationFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::range({ 2, 1, 1, 3 }, 1);
			Tensor tensor3b = Tensor::range({ 1, 3 ,3, 2 }, 3).requireGradient();
			Tensor tensor3c = Tensor::matrixMultiply(tensor3a, tensor3b);
			Assert::IsTrue(tensor3c.requiresGradient());
			Assert::IsNotNull((MatrixMultiplicationFunction*)tensor3c.getFunction());
		}
	};

	TEST_CLASS(MaxSingleTest)
	{
	public:
		TEST_METHOD(NewValue)
		{
			Tensor tensor1a = Tensor::range({ 3,1 });
			Tensor tensor1b = Tensor::max(tensor1a, 1.0f);
			Assert::AreEqual(tensor1b.at(0), 1.0f);
			Assert::AreEqual(tensor1b.at(1), 1.0f);
			Assert::AreEqual(tensor1b.at(2), 2.0f);

			Tensor tensor2a = Tensor::range({ 2 }, 1);
			Tensor tensor2b = Tensor::max(tensor2a, 1.3f);
			Assert::AreEqual(tensor2b.at(0), 1.3f);
			Assert::AreEqual(tensor2b.at(1), 2.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::max(tensor1a, 3.0f);
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::ones({ 2 }).set(-1, { 0 }).requireGradient();
			Tensor tensor2b = Tensor::max(tensor2a, -1.0f);
			Assert::IsTrue(tensor2b.requiresGradient());
			Assert::IsNotNull((MaxSingleFunction*)tensor2b.getFunction());
		}
	};

	TEST_CLASS(MaxTensorTest)
	{
	public:
		TEST_METHOD(UnbroadcastableDims)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::max(Tensor::zeroes({ 10, 3, 5 }), Tensor::zeroes({ 1, 2, 5 })); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::max(Tensor::zeroes({ 10, 1, 3, 5 }), Tensor::zeroes({ 3, 10 })); }
			);
		}

		TEST_METHOD(NewValue)
		{
			Tensor tensor1 = Tensor::max(Tensor::range({ 3 }), Tensor::full({ 1 }, 1.5));
			CompareFloats(tensor1.at(0), 1.5f);
			CompareFloats(tensor1.at(1), 1.5f);
			CompareFloats(tensor1.at(2), 2.0f);

			Tensor tensor2 = Tensor::max(Tensor::range({ 2, 1, 3 }, 1), Tensor::full({ 1, 3 }, 3));
			CompareFloats(tensor2.at({ 0,0,0 }), 3.0f);
			CompareFloats(tensor2.at({ 0,0,1 }), 3.0f);
			CompareFloats(tensor2.at({ 0,0,2 }), 3.0f);
			CompareFloats(tensor2.at({ 1,0,0 }), 4.0f);
			CompareFloats(tensor2.at({ 1,0,1 }), 5.0f);
			CompareFloats(tensor2.at({ 1,0,2 }), 6.0f);
		}


		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::max(tensor1a, Tensor::zeroes({ 1 }));
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 1 });
			Tensor tensor2b = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2c = Tensor::max(tensor2a, tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((MaxTensorFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::ones({ 2, 1, 3 }).requireGradient();
			Tensor tensor3b = Tensor::max(tensor3a, Tensor::ones({ 1, 1 }));
			Assert::IsTrue(tensor3b.requiresGradient());
			Assert::IsNotNull((MaxTensorFunction*)tensor3b.getFunction());
		}
	};

	TEST_CLASS(MinSingleTest)
	{
	public:
		TEST_METHOD(NewValue)
		{
			Tensor tensor1a = Tensor::range({ 3,1 });
			Tensor tensor1b = Tensor::min(tensor1a, 1.0f);
			Assert::AreEqual(tensor1b.at(0), 0.0f);
			Assert::AreEqual(tensor1b.at(1), 1.0f);
			Assert::AreEqual(tensor1b.at(2), 1.0f);

			Tensor tensor2a = Tensor::range({ 2 }, 1);
			Tensor tensor2b = Tensor::min(tensor2a, 1.3f);
			Assert::AreEqual(tensor2b.at(0), 1.0f);
			Assert::AreEqual(tensor2b.at(1), 1.3f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::min(tensor1a, 3.0f);
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::ones({ 2 }).set(-1, { 0 }).requireGradient();
			Tensor tensor2b = Tensor::min(tensor2a, -1.0f);
			Assert::IsTrue(tensor2b.requiresGradient());
			Assert::IsNotNull((MinSingleFunction*)tensor2b.getFunction());
		}
	};

	TEST_CLASS(MinTensorTest)
	{
	public:
		TEST_METHOD(UnbroadcastableDims)
		{
			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::min(Tensor::zeroes({ 10, 3, 5 }), Tensor::zeroes({ 1, 2, 5 })); }
			);

			Assert::ExpectException<std::invalid_argument>(
				[]() {Tensor::min(Tensor::zeroes({ 10, 1, 3, 5 }), Tensor::zeroes({ 3, 10 })); }
			);
		}

		TEST_METHOD(NewValue)
		{
			Tensor tensor1 = Tensor::min(Tensor::range({ 3 }), Tensor::full({ 1 }, 1.5));
			CompareFloats(tensor1.at(0), 0.0f);
			CompareFloats(tensor1.at(1), 1.0f);
			CompareFloats(tensor1.at(2), 1.5f);

			Tensor tensor2 = Tensor::min(Tensor::range({ 2, 1, 3 }, 1), Tensor::full({ 1, 3 }, 3));
			CompareFloats(tensor2.at({ 0,0,0 }), 1.0f);
			CompareFloats(tensor2.at({ 0,0,1 }), 2.0f);
			CompareFloats(tensor2.at({ 0,0,2 }), 3.0f);
			CompareFloats(tensor2.at({ 1,0,0 }), 3.0f);
			CompareFloats(tensor2.at({ 1,0,1 }), 3.0f);
			CompareFloats(tensor2.at({ 1,0,2 }), 3.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::min(tensor1a, Tensor::zeroes({ 1 }));
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 1 });
			Tensor tensor2b = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2c = Tensor::min(tensor2a, tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((MinTensorFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::ones({ 2, 1, 3 }).requireGradient();
			Tensor tensor3b = Tensor::min(tensor3a, Tensor::ones({ 1, 1 }));
			Assert::IsTrue(tensor3b.requiresGradient());
			Assert::IsNotNull((MinTensorFunction*)tensor3b.getFunction());
		}
	};

	TEST_CLASS(ReLUTest)
	{
	public:
		TEST_METHOD(NewValues)
		{
			Tensor tensor1a = Tensor::range({ 3,1 }, -1.5);
			Tensor tensor1b = Tensor::ReLU(tensor1a);
			Assert::AreEqual(tensor1b.at(0), 0.0f);
			Assert::AreEqual(tensor1b.at(1), 0.0f);
			Assert::AreEqual(tensor1b.at(2), 0.5f);

			Tensor tensor2a = Tensor::range({ 2 }, 1, -2);
			Tensor tensor2b = Tensor::ReLU(tensor2a);
			Assert::AreEqual(tensor2b.at(0), 1.0f);
			Assert::AreEqual(tensor2b.at(1), 0.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::ReLU(tensor1a);
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::ones({ 2 }).set(-1, { 0 }).requireGradient();
			Tensor tensor2b = Tensor::ReLU(tensor2a);
			Assert::IsTrue(tensor2b.requiresGradient());
			Assert::IsNotNull((MaxSingleFunction*)tensor2b.getFunction());
		}
	};

	TEST_CLASS(MeanSquaredErrorLossTest)
	{
	public:
		TEST_METHOD(NewValue)
		{
			Tensor tensor1a = Tensor::full({ 1,3 }, 5.0f);
			Tensor tensor1b = Tensor::full({ 4,1 }, 1.0f);
			Tensor tensor1c = Tensor::meanSquaredErrorLoss(tensor1a, tensor1b);
			CompareFloats(tensor1c.item(), 16.0f);

			Tensor tensor2a = Tensor::range({ 3 }, 3.0f, -1.0f);
			Tensor tensor2b = Tensor::range({ 3 }, 1.0f, 2.0f);
			Tensor tensor2c = Tensor::meanSquaredErrorLoss(tensor2a, tensor2b);
			CompareFloats(tensor2c.item(), 7.0f);
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 3,1 });
			Tensor tensor1b = Tensor::meanSquaredErrorLoss(tensor1a, Tensor::zeroes({ 1 }));
			Assert::IsFalse(tensor1b.requiresGradient());
			Assert::IsNull(tensor1b.getFunction());

			Tensor tensor2a = Tensor::zeroes({ 1 });
			Tensor tensor2b = Tensor::ones({ 3 }).requireGradient();
			Tensor tensor2c = Tensor::meanSquaredErrorLoss(tensor2a, tensor2b);
			Assert::IsTrue(tensor2c.requiresGradient());
			Assert::IsNotNull((MeanSquaredErrorLossFunction*)tensor2c.getFunction());

			Tensor tensor3a = Tensor::ones({ 2, 1, 3 }).requireGradient();
			Tensor tensor3b = Tensor::meanSquaredErrorLoss(tensor3a, Tensor::ones({ 1, 1 }));
			Assert::IsTrue(tensor3b.requiresGradient());
			Assert::IsNotNull((MeanSquaredErrorLossFunction*)tensor3b.getFunction());
		}
	};

	TEST_CLASS(CategoricalCrossEntropyTest)
	{
		TEST_METHOD(NewValues)
		{
			Tensor tensor1a = Tensor::range({ 2,1,3 }, 1.0f);
			Tensor tensor1b = Tensor::range({ 3 }, 0.0f, 1.0f / 3.0f);
			Tensor tensor1c = Tensor::CategoricalCrossEntropyLoss(tensor1a, tensor1b);
			CompareFloats(tensor1c.item(), 0.741f);

			Tensor tensor2a = Tensor::full({ 2 }, 3.0f);
			Tensor tensor2b = Tensor::fromValues(new float[4] {0.2f, 0.8f, 0.8f, 0.2f}, { 2,1,2 });
			Tensor tensor2c = Tensor::CategoricalCrossEntropyLoss(tensor2a, tensor2b);
			CompareFloats(tensor2c.item(), 0.693f);
		}

		TEST_METHOD(Gradient)
		{

			Tensor tensor1a = Tensor::range({ 2,1,3 }, 1.0f);
			Tensor tensor1b = Tensor::range({ 3 }, 0.0f, 1.0f / 3.0f);
			Tensor tensor1c = Tensor::CategoricalCrossEntropyLoss(tensor1a, tensor1b);
			Assert::IsFalse(tensor1c.requiresGradient());
			Assert::IsNull(tensor1c.getFunction());

			Tensor tensor2a = Tensor::full({ 2 }, 3.0f);
			Tensor tensor2b = Tensor::fromValues(new float[4] {0.2f, 0.8f, 0.8f, 0.2f}, { 2,1,2 }).requireGradient();
			Tensor tensor2c = Tensor::CategoricalCrossEntropyLoss(tensor2a, tensor2b);
			Assert::IsFalse(tensor2c.requiresGradient());
			Assert::IsNull(tensor2c.getFunction());

			Tensor tensor3a = Tensor::full({ 2 }, 3.0f).requireGradient();
			Tensor tensor3b = Tensor::fromValues(new float[4] {0.2f, 0.8f, 0.8f, 0.2f}, { 2,1,2 });
			Tensor tensor3c = Tensor::CategoricalCrossEntropyLoss(tensor3a, tensor3b);
			Assert::IsTrue(tensor3c.requiresGradient());
			Assert::IsNotNull((CategoricalCrossEntropyLossFunction*)tensor3c.getFunction());
		}
	};

	TEST_CLASS(GradientTest)
	{
	public:
		TEST_METHOD(Value)
		{
			Tensor tensor1 = Tensor::zeroes({});
			Assert::IsFalse(tensor1.requiresGradient());
			tensor1.requireGradient();
			Assert::IsTrue(tensor1.requiresGradient());

			Tensor tensor2 = Tensor::ones({ 4, 3 });
			Assert::IsFalse(tensor2.requiresGradient());
			tensor2.requireGradient();
			Assert::IsTrue(tensor2.requiresGradient());
		}
	};

	TEST_CLASS(DetachedTest)
	{
	public:
		TEST_METHOD(Shape)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,4 });
			Tensor tensor1b = tensor1a.detached();
			Assert::AreEqual(tensor1b.getShape()[0], 2);
			Assert::AreEqual(tensor1b.getShape()[1], 4);

			Tensor tensor2a = Tensor::zeroes({ 1, 3, 2, 5 });
			Tensor tensor2b = tensor2a.detached();
			Assert::AreEqual(tensor2b.getShape()[0], 1);
			Assert::AreEqual(tensor2b.getShape()[1], 3);
			Assert::AreEqual(tensor2b.getShape()[2], 2);
			Assert::AreEqual(tensor2b.getShape()[3], 5);
		}

		TEST_METHOD(Values)
		{
			Tensor tensor1a = Tensor::range({ 2,4 });
			Tensor tensor1b = tensor1a.detached();
			for (int i = 0; i < 8; i++) {
				CompareFloats(tensor1b.at(i), i);
			}

			Tensor tensor2a = Tensor::range({ 1, 3, 2, 5 });
			Tensor tensor2b = tensor2a.detached();			
			for (int i = 0; i < 30; i++) {
				CompareFloats(tensor2b.at(i), i);
			}
		}

		TEST_METHOD(Gradient)
		{
			Tensor tensor1a = Tensor::zeroes({ 2,4 });
			Tensor tensor1b = tensor1a.detached();
			Assert::IsFalse(tensor1b.requiresGradient());

			Tensor tensor2a = Tensor::zeroes({ 1, 3, 2, 5 }).requireGradient();
			Tensor tensor2b = tensor2a.detached();
			Assert::IsFalse(tensor2b.requiresGradient());
		}
	};
}
