#include "pch.h"
#include "CppUnitTest.h"
#include "deep_learning.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace DeepLearningTest
{
	TEST_CLASS(DeepLearningTest)
	{
	public:

		TEST_METHOD(TestZeroesShapes)
		{
			Assert::AreEqual(1, Tensor::zeroes({ 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor::zeroes({ 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor::zeroes({ 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor::zeroes({ 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor::zeroes({ 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::zeroes({ 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor::zeroes({ 3,7,5 }).getShape()[2]);
		}

		TEST_METHOD(TestZeroesSizes)
		{
			Assert::AreEqual(1, Tensor::zeroes({}).getSize());
			Assert::AreEqual(1, Tensor::zeroes({ 1 }).getSize());
			Assert::AreEqual(1, Tensor::zeroes({ 1, 1, 1 }).getSize());
			Assert::AreEqual(2, Tensor::zeroes({ 2 }).getSize());
			Assert::AreEqual(9, Tensor::zeroes({ 3, 3 }).getSize());
			Assert::AreEqual(105, Tensor::zeroes({ 3, 7, 5 }).getSize());
		}

		TEST_METHOD(TestZeroesValues)
		{
			Tensor tensor1 = Tensor::zeroes({ 3 });
			Assert::IsTrue(std::abs(tensor1.get({ 0 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor1.get({ 1 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor1.get({ 2 }).item()) < 1e-10);

			Tensor tensor2 = Tensor::zeroes({ 2, 3 });
			Assert::IsTrue(std::abs(tensor2.get({ 0, 0 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 0, 1 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 0, 2 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 1, 0 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 1, 1 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 1, 2 }).item()) < 1e-10);
		}

		TEST_METHOD(TestInvalidShapes)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor::zeroes({ 1, 0, 3 }); });
		}


		TEST_METHOD(TestItemInvalidShapes)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 3 }).item(); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 1, 1, 2 }).item(); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 1, 4, 1, 1 }).item(); });
		}

		TEST_METHOD(TestInvalidReshapes) {
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 1 }).reshape({ 1, 2 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 3, 2 }).reshape({ 5, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 7, 1, 3 }).reshape({ 5, 4, 1 }); });
		}

		TEST_METHOD(TestReshapes) {
			Assert::AreEqual(1, Tensor::zeroes({}).reshape({ 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor::zeroes({ 1, 3, 1 }).reshape({ 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor::zeroes({ 10 }).reshape({ 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor::zeroes({ 10 }).reshape({ 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor::zeroes({ 5, 21 }).reshape({ 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::zeroes({ 5, 21 }).reshape({ 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor::zeroes({ 5, 21 }).reshape({ 3,7,5 }).getShape()[2]);
		}

		TEST_METHOD(TestTooManyIndices) {
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ }).get({ 0 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2 }).get({ 1, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor::zeroes({ 2, 4, 6 }).get({ 1, 1, 2, 2 }); });
		}

		TEST_METHOD(TestIndexOutOfBounds) {
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 1, 3 }).get({ 2, 1 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 3 }).get({ 4 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor::zeroes({ 4, 2, 8, 1, 2 }).get({ 3, 1, 5, 4 }); });
		}

		TEST_METHOD(TestGetterNewShape) {
			Assert::AreEqual(5, Tensor::zeroes({ 1, 4, 5, 7 }).get({ 0, 0 }).getShape()[0]);
			Assert::AreEqual(7, Tensor::zeroes({ 1, 4, 5, 7 }).get({ 0, 0 }).getShape()[1]);

			Assert::AreEqual(2, Tensor::zeroes({ 3, 2 }).get({ 0 }).getShape()[0]);
		}

		TEST_METHOD(TestAssigningSingle) {
			Tensor tensor1 = Tensor::zeroes({ 3 });
			tensor1.set(1.0f);
			Assert::IsTrue(std::abs(tensor1.get({ 0 }).item() - 1.0f) < 1e-10);
			Assert::IsTrue(std::abs(tensor1.get({ 1 }).item() - 1.0f) < 1e-10);
			Assert::IsTrue(std::abs(tensor1.get({ 2 }).item() - 1.0f) < 1e-10);

			Tensor tensor2 = Tensor::zeroes({ 2, 3 });
			tensor2.set({ 1 }, 1.0f);
			Assert::IsTrue(std::abs(tensor2.get({ 0, 0 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 0, 1 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 0, 2 }).item()) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 1, 0 }).item() - 1.0f) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 1, 1 }).item() - 1.0f) < 1e-10);
			Assert::IsTrue(std::abs(tensor2.get({ 1, 2 }).item() - 1.0f) < 1e-10);
		}
	};
}
