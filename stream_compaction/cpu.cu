#include "common.h"
#include "cpu.h"

#include <memory>

namespace StreamCompaction {
namespace CPU {

using StreamCompaction::Common::PerformanceTimer;

namespace {

inline void scanImplementation(int n, int* odata, const int* idata) {
  int currentSum = 0;
  for (int i = 0; i < n; ++i) {
    odata[i] = currentSum;
    currentSum += idata[i];
  }
}

}  // namespace

PerformanceTimer& timer() {
  static PerformanceTimer timer;
  return timer;
}

/**
 * CPU scan (prefix sum).
 * For performance analysis, this is supposed to be a simple for loop.
 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function
 * first.
 */
void scan(int n, int* odata, const int* idata) {
  timer().startCpuTimer();

  if (n <= 0) return;

  scanImplementation(n, odata, idata);

  timer().endCpuTimer();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int* odata, const int* idata) {
  timer().startCpuTimer();

  if (n <= 0) return 0;

  int outputIndex = 0;
  for (int inputIndex = 0; inputIndex < n; ++inputIndex) {
    int element = idata[inputIndex];

    if (element > 0) {
      odata[outputIndex] = element;
      outputIndex++;
    }
  }

  timer().endCpuTimer();

  return outputIndex;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int* odata, const int* idata) {
  timer().startCpuTimer();

  if (n <= 0) return 0;

  std::unique_ptr<int[]> valid = std::make_unique<int[]>(n);
  for (int i = 0; i < n; ++i) {
    valid[i] = idata[i] > 0 ? 1 : 0;
  }

  std::unique_ptr<int[]> scanResult = std::make_unique<int[]>(n);
  scanImplementation(n, scanResult.get(), valid.get());

  for (int i = 0; i < n; ++i) {
    if (valid[i] > 0) {
      odata[scanResult[i]] = idata[i];
    }
  }

  timer().endCpuTimer();

  return scanResult[n - 1];
}

}  // namespace CPU
}  // namespace StreamCompaction
