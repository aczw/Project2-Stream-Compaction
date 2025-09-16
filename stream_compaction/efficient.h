#pragma once

#include "common.h"

namespace StreamCompaction {
namespace Efficient {

StreamCompaction::Common::PerformanceTimer& timer();

void scan(int n, int* odata, const int* idata, bool measure = true);

int compact(int n, int* odata, const int* idata);

}  // namespace Efficient
}  // namespace StreamCompaction
