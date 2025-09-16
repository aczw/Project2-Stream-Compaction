#include "common.h"
#include "naive.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
namespace Naive {

using StreamCompaction::Common::PerformanceTimer;

/// Number of threads per block.
constexpr int blockSize = 256;

/// Whether to return an inclusive or exclusive scan.
constexpr bool useExclusiveScan = true;

PerformanceTimer& timer() {
  static PerformanceTimer timer;
  return timer;
}

__global__ void kernSumPairsForIteration(int n, const int* in, int* out, int stride) {
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  // As the number of dispatches decrease with every iteration, we have to add the stride to
  // get the last index of the array
  if (k >= n + stride) return;

  if (k >= stride) {
    out[k] = in[k - stride] + in[k];
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int* odata, const int* idata) {
  int* dev_dataA = nullptr;
  int* dev_dataB = nullptr;

  size_t numBytes = n * sizeof(int);
  cudaMalloc(reinterpret_cast<void**>(&dev_dataA), numBytes);
  checkCUDAError("cudaMalloc: dev_dataA failed!");
  cudaMalloc(reinterpret_cast<void**>(&dev_dataB), numBytes);
  checkCUDAError("cudaMalloc: dev_dataB failed!");

  cudaMemcpy(dev_dataA, idata, numBytes, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy: idata -> dev_dataA failed!");
  cudaMemcpy(dev_dataB, dev_dataA, numBytes, cudaMemcpyDeviceToDevice);
  checkCUDAError("cudaMemcpy: dev_dataA -> dev_dataB failed!");

  timer().startGpuTimer();

  for (int iteration = 1; iteration <= ilog2ceil(n); ++iteration) {
    int stride = 1 << (iteration - 1);
    int numDispatches = n - stride;
    int numBlocks = (numDispatches + blockSize - 1) / blockSize;

    kernSumPairsForIteration<<<numBlocks, blockSize>>>(numDispatches, dev_dataA, dev_dataB, stride);

    // Write new results back into A to be read from
    cudaMemcpy(dev_dataA, dev_dataB, numBytes, cudaMemcpyDeviceToDevice);
  }

  timer().endGpuTimer();

  cudaMemcpy(odata, dev_dataA, numBytes, cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy: dev_dataA -> odata failed!");

  // Convert from inclusive scan to exclusive
  if constexpr (useExclusiveScan) {
    for (int i = n - 1; i > 0; --i) {
      odata[i] = odata[i - 1];
    }
    odata[0] = 0;
  }

  cudaFree(dev_dataA);
  checkCUDAError("cudaFree: dev_dataA failed!");
  cudaFree(dev_dataB);
  checkCUDAError("cudaFree: dev_dataB failed!");
}

}  // namespace Naive
}  // namespace StreamCompaction
