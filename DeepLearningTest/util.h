#pragma once
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

void CompareFloats(float x, float y);
template<typename T> void ComparePointers(const T* p1, const T* p2) {
	if (p1 != p2)
	{
		std::wstringstream message;
		message << "Pointers were not equal.";
		Assert::Fail(message.str().c_str());
	}
}