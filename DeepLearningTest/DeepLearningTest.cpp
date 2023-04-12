#include "pch.h"
#include "CppUnitTest.h"
#include "deep_learning.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace DeepLearningTest
{
	TEST_CLASS(DeepLearningTest)
	{
	public:

		TEST_METHOD(TestShapes)
		{
			Assert::AreEqual(1, Tensor({ 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor({ 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor({ 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor({ 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor({ 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor({ 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor({ 3,7,5 }).getShape()[2]);
		}

		TEST_METHOD(TestInvalidShapes)
		{
			Assert::ExpectException<std::invalid_argument>([]() { Tensor _({ -1 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor _({ 0 }); });
			Assert::ExpectException<std::invalid_argument>([]() { Tensor _({ 1, 0, 3 }); });
		}

		TEST_METHOD(TestSizes)
		{
			Assert::AreEqual(1, Tensor({}).getSize());
			Assert::AreEqual(1, Tensor({ 1 }).getSize());
			Assert::AreEqual(1, Tensor({ 1, 1, 1 }).getSize());
			Assert::AreEqual(2, Tensor({ 2 }).getSize());
			Assert::AreEqual(9, Tensor({ 3, 3 }).getSize());
			Assert::AreEqual(105, Tensor({ 3, 7, 5 }).getSize());
		}

		TEST_METHOD(TestItemInvalidShapes)
		{
			Assert::ExpectException<std::length_error>([]() {Tensor({ 3 }).item(); });
			Assert::ExpectException<std::length_error>([]() {Tensor({ 1, 1, 2 }).item(); });
			Assert::ExpectException<std::length_error>([]() {Tensor({ 1, 4, 1, 1 }).item(); });
		}

		TEST_METHOD(TestInvalidReshapes) {
			Assert::ExpectException<std::length_error>([]() {Tensor({ 1 }).reshape({ 1, 2 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor({ 3, 2 }).reshape({ 5, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor({ 7, 1, 3 }).reshape({ 5, 4, 1 }); });
		}

		TEST_METHOD(TestReshapes) {
			Assert::AreEqual(1, Tensor({}).reshape({ 1 }).getShape()[0]);

			Assert::AreEqual(3, Tensor({ 1, 3, 1 }).reshape({ 3 }).getShape()[0]);

			Assert::AreEqual(5, Tensor({ 10 }).reshape({ 5, 2 }).getShape()[0]);
			Assert::AreEqual(2, Tensor({ 10 }).reshape({ 5, 2 }).getShape()[1]);

			Assert::AreEqual(3, Tensor({ 5, 21 }).reshape({ 3,7,5 }).getShape()[0]);
			Assert::AreEqual(7, Tensor({ 5, 21 }).reshape({ 3,7,5 }).getShape()[1]);
			Assert::AreEqual(5, Tensor({ 5, 21 }).reshape({ 3,7,5 }).getShape()[2]);
		}

		TEST_METHOD(TestTooManyIndices) {
			Assert::ExpectException<std::length_error>([]() {Tensor({ }).get({ 0 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor({ 2 }).get({ 1, 1 }); });
			Assert::ExpectException<std::length_error>([]() {Tensor({ 2, 4, 6 }).get({ 1, 1, 2, 2 }); });
		}

		TEST_METHOD(TestIndexOutOfBounds) {
			Assert::ExpectException<std::out_of_range>([]() {Tensor({ 1, 3 }).get({ 2, 1 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor({ 3 }).get({ 4 }); });
			Assert::ExpectException<std::out_of_range>([]() {Tensor({ 4, 2, 8, 1, 2 }).get({ 3, 1, 5, 4 }); });
		}

		TEST_METHOD(TestGetterNewShape) {
			Assert::AreEqual(5, Tensor({ 1, 4, 5, 7 }).get({ 0, 0 }).getShape()[0]);
			Assert::AreEqual(7, Tensor({ 1, 4, 5, 7 }).get({ 0, 0 }).getShape()[1]);

			Assert::AreEqual(2, Tensor({ 3, 2 }).get({ 0 }).getShape()[0]);
		}
	};
}
