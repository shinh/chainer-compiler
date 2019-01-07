#pragma once

#include <map>
#include <string>

#include <chainerx/array.h>

namespace oniku {
namespace runtime {

// TODO(hamaji): Investigate xChainer's BatchNorm.
chainerx::Array BatchNormONNX(
        chainerx::Array x, chainerx::Array s, chainerx::Array bias, chainerx::Array mean, chainerx::Array var, float epsilon);

chainerx::Shape ArrayToShape(const chainerx::Array& a);

chainerx::Array ShapeToArray(const chainerx::Shape& s);

chainerx::Array MakeArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src);

chainerx::Array MakeScalarArray(float f);

chainerx::Array MakeHostArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src);

// This function was renamed from `Split` to clearly tell this is
// different from chainerx::Split.
std::vector<chainerx::Array> SplitByLengths(const chainerx::Array& input, int axis, const std::vector<int64_t>& split);

chainerx::Array PadSequence(const std::vector<chainerx::Array>& inputs, int64_t length, chainerx::Scalar padding);

chainerx::Array Sigmoid(chainerx::Array a);

chainerx::Array SlowRandom(chainerx::Shape shape);

}  // namespace runtime
}  // namespace oniku
