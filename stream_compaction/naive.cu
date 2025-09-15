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

/// Perform inner loop within the kernel. Results in only 1 invocation per "layer" versus one
/// kernel dispatch per iteration of the inner loop.
constexpr bool runInnerLoopOnGPU = true;

PerformanceTimer& timer() {
  static PerformanceTimer timer;
  return timer;
}

__global__ void kernSumStrided(int n, const int* in, int* out, int stride) {
  int tId = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (tId >= n) return;

  for (int k = stride; k <= n; ++k) {
    out[k] = in[k - stride] + in[k];
  }
}

__global__ void kernAddStridedPair(int n, const int* in, int* out, int k, int stride) {
  int tId = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (tId >= n) return;

  out[k] = in[k - stride] + in[k];
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int* odata, const int* idata) {
  int* dev_dataA = nullptr;
  int* dev_dataB = nullptr;

  size_t numBytes = n * sizeof(int);
  cudaMalloc((void**)&dev_dataA, numBytes);
  checkCUDAError("cudaMalloc: dev_dataA failed!");
  cudaMalloc((void**)&dev_dataB, numBytes);
  checkCUDAError("cudaMalloc: dev_dataB failed!");

  cudaMemcpy(dev_dataA, idata, numBytes, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy: idata -> dev_dataA failed!");
  cudaMemcpy(dev_dataB, dev_dataA, numBytes, cudaMemcpyDeviceToDevice);
  checkCUDAError("cudaMemcpy: dev_dataA -> dev_dataB failed!");

  timer().startGpuTimer();

  for (int iteration = 1; iteration <= ilog2ceil(n); ++iteration) {
    int stride = 1 << (iteration - 1);
    int numDispatches = n - stride;
    int numBlocks = (numDispatches + blockSize + 1) / blockSize;

    if constexpr (runInnerLoopOnGPU) {
      kernSumStrided<<<numBlocks, blockSize>>>(n, dev_dataA, dev_dataB, stride);
    } else {
      for (int k = stride; k < n; ++k) {
        kernAddStridedPair<<<numBlocks, blockSize>>>(numDispatches, dev_dataA, dev_dataB, k, stride);
      }
    }

    // Swap read and write buffers (output in B will be read next in A)
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
