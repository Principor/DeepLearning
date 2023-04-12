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

		TEST_METHOD(TestSizes) {
			Assert::AreEqual(1, Tensor({}).getSize());
			Assert::AreEqual(1, Tensor({ 1 }).getSize());
			Assert::AreEqual(1, Tensor({ 1, 1, 1 }).getSize());
			Assert::AreEqual(2, Tensor({ 2 }).getSize());
			Assert::AreEqual(9, Tensor({ 3, 3 }).getSize());
			Assert::AreEqual(105, Tensor({ 3, 7, 5 }).getSize());
		}

		TEST_METHOD(TestItemInvalidShapes) {
			Assert::ExpectException<std::length_error>([]() {Tensor({ 3 }).item(); });
			Assert::ExpectException<std::length_error>([]() {Tensor({ 1, 1, 2 }).item(); });
			Assert::ExpectException<std::length_error>([]() {Tensor({ 1, 4, 1, 1 }).item(); });
		}
	};
}
