#include <gtest/gtest.h>

#include "../../../include/miniDL/shape.h"

using namespace miniDL;

TEST(ShapeTest, ConstructorAndElements) {
    Shape s({2, 3, 4});

    EXPECT_EQ(s.size(), 3);
    EXPECT_EQ(s.elements(), 24);
    EXPECT_EQ(s[2], 4);
}