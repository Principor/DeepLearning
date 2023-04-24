#include "pch.h"
#include "util.h"

void CompareFloats(float x, float y) {
	float difference = std::abs(x - y);
	if (difference > 1e-3)
	{
		std::wstringstream message;
		message << "Expected:<" << x << "> Actual:<" << y << ">";
		Assert::Fail(message.str().c_str());
	}
}