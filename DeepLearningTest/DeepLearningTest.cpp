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
	};
}
