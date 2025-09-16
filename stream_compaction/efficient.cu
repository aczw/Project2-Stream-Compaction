#include "common.h"
#include "efficient.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <optional>

namespace StreamCompaction {
namespace Efficient {

using StreamCompaction::Common::PerformanceTimer;

/// Number of threads per block.
constexpr int blockSize = 256;

/// Enable `checkCUDAError()` calls within the performance measuring fence.
constexpr bool checkErrorsDuringTimer = true;

PerformanceTimer& timer() {
  static PerformanceTimer timer;
  return timer;
}

__global__ void kernReduceForLayer(int n, int* data, int layer, int stride) {
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (k >= n) return;

  int offset = k * stride;
  int previousStride = 1 << layer;

  int rightChild = offset + stride - 1;
  int leftChild = offset + previousStride - 1;

  data[rightChild] += data[leftChild];
}

__global__ void kernTraverseDownLayer(int n, int* data, int layer, int stride) {
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (k >= n) return;

  int offset = k * stride;
  int previousStride = 1 << layer;

  int rightChild = offset + stride - 1;
  int leftChild = offset + previousStride - 1;

  int leftValue = data[leftChild];
  data[leftChild] = data[rightChild];
  data[rightChild] += leftValue;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int* odata, const int* idata) {
  int actualN = n;
  size_t numBytes = n * sizeof(int);

  std::unique_ptr<int[]> actualInputData = std::make_unique<int[]>(n);
  std::memcpy(actualInputData.get(), idata, numBytes);

  // Input array size is not a power of two; we have to pad the left with zeroes
  std::optional<int> paddingOpt;
  if (int numLeaves = 1 << ilog2ceil(n); n < numLeaves) {
    int offset = numLeaves - n;

    // Pad to the next power of two
    std::unique_ptr<int[]> paddedInputData = std::make_unique<int[]>(numLeaves);
    std::memcpy(paddedInputData.get() + offset, idata, numBytes);

    paddingOpt = offset;
    actualN = numLeaves;
    numBytes = numLeaves * sizeof(int);
    actualInputData.swap(paddedInputData);
  }

  int* dev_data = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&dev_data), numBytes);
  checkCUDAError("cudaMalloc: dev_data failed!");
  cudaMemcpy(dev_data, actualInputData.get(), numBytes, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy: actualInputData -> dev_data failed!");

  timer().startGpuTimer();

  // Perform up-sweep via parallel reduction
  for (int layer = 0; layer < ilog2(actualN); ++layer) {
    int stride = 1 << (layer + 1);
    int numDispatches = actualN / stride;
    int numBlocks = (numDispatches + blockSize - 1) / blockSize;

    kernReduceForLayer<<<numBlocks, blockSize>>>(numDispatches, dev_data, layer, stride);
  }

  // Zero out the root
  int zero = 0;
  cudaMemcpy(dev_data + (actualN - 1), &zero, sizeof(int), cudaMemcpyHostToDevice);
  if constexpr (checkErrorsDuringTimer) checkCUDAError("cudaMemcpy: 0 -> dev_data failed!");

  for (int layer = ilog2(actualN) - 1; layer >= 0; --layer) {
    int stride = 1 << (layer + 1);
    int numDispatches = actualN / stride;
    int numBlocks = (numDispatches + blockSize - 1) / blockSize;

    kernTraverseDownLayer<<<numBlocks, blockSize>>>(numDispatches, dev_data, layer, stride);
  }

  timer().endGpuTimer();

  if (paddingOpt) {
    // If previously padded, remove extra zeroes
    cudaMemcpy(actualInputData.get(), dev_data, numBytes, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy: dev_data -> actualInputData failed!");
    std::memcpy(odata, actualInputData.get() + paddingOpt.value(), n * sizeof(int));
  } else {
    cudaMemcpy(odata, dev_data, numBytes, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy: dev_data -> odata failed!");
  }

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
