#include "common.h"
#include "efficient.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>

namespace StreamCompaction {
namespace Efficient {

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer() {
  static PerformanceTimer timer;
  return timer;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int* odata, const int* idata) {
  int actualN = n;
  const int* actualInputData = idata;
  size_t numBytes = n * sizeof(int);

  // Input array size is not a power of two; we have to pad the left with zeroes
  if (int numLeaves = 1 << ilog2ceil(n); n < numLeaves) {
    int offset = numLeaves - n;

    // Pad to the next power of two
    std::unique_ptr<int[]> paddedInputData = std::make_unique<int[]>(numLeaves);
    std::memcpy(paddedInputData.get() + offset, idata, numBytes);

    actualN = numLeaves;
    actualInputData = paddedInputData.release();
    numBytes = numLeaves * sizeof(int);
  }

  int* dev_data = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&dev_data), numBytes);
  checkCUDAError("cudaMalloc: dev_data failed!");
  cudaMemcpy(dev_data, actualInputData, numBytes, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy: actualInputData -> dev_data failed!");

  timer().startGpuTimer();
  // TODO
  timer().endGpuTimer();

  cudaFree(dev_data);
  checkCUDAError("cudaFree: dev_data failed!");
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int* odata, const int* idata) {
  timer().startGpuTimer();
  // TODO
  timer().endGpuTimer();
  return -1;
}

}  // namespace Efficient
}  // namespace StreamCompaction
