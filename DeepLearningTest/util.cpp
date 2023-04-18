#include "pch.h"
#include "util.h"

void CompareFloats(float x, float y) {
	Assert::AreEqual(x, y);
	Assert::IsTrue(std::abs(x - y) < 1e-10);
}