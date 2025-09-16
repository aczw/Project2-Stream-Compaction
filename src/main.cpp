#include "testing_helpers.hpp"

#include <cstdio>
#include <functional>
#include <stream_compaction/cpu.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/thrust.h>

const int SIZE = 1 << 16;   // feel free to change the size of array
const int NPOT = SIZE - 3;  // Non-Power-Of-Two

/// If true, run additional simpler tests.
constexpr bool runDebugTests = false;

/// Print out resulting arrays from computation.
constexpr bool enablePrintingArrays = false;

constexpr bool enableCPUScan = false;
constexpr bool enableNaiveScan = true;
constexpr bool enableEfficientScan = true;
constexpr bool enableThrustScan = false;

constexpr bool enableCPUCompactWith = false;
constexpr bool enableEfficientCompact = true;
constexpr bool enableThrustCompact = false;

namespace Perf {

using ScanFn = std::function<void(int, int*, const int*)>;
using CompactionFn = std::function<int(int, int*, const int*)>;

constexpr int numIterations = 1'000;

}  // namespace Perf

int* a = new int[SIZE];
int* b = new int[SIZE];
int* c = new int[SIZE];

int main(int argc, char* argv[]) {
  // Scan tests

  printf("\n");
  printf("****************\n");
  printf("** SCAN TESTS **\n");
  printf("****************\n");

  genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
  a[SIZE - 1] = 0;
  printArray(SIZE, a, true);

  // initialize b using StreamCompaction::CPU::scan you implement
  // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
  // At first all cases passed because b && c are all zeroes.
  zeroArray(SIZE, b);
  printDesc("cpu scan, power-of-two");
  StreamCompaction::CPU::scan(SIZE, b, a);
  printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
  if constexpr (enablePrintingArrays) printArray(SIZE, b, true);

  if constexpr (enableCPUScan) {
    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    if constexpr (enablePrintingArrays) printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
  }

  if constexpr (enableNaiveScan) {
    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);
  }

  if constexpr (enableEfficientScan) {
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
  }

  if constexpr (enableThrustScan) {
    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
  }

  // For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
  if constexpr (runDebugTests) {
    printf("\n");
    printf("*************************\n");
    printf("** SCAN TESTS (ALL 1s) **\n");
    printf("*************************\n");

    onesArray(SIZE, a);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    if constexpr (enablePrintingArrays) printArray(SIZE, b, true);

    if constexpr (enableCPUScan) {
      zeroArray(SIZE, c);
      printDesc("cpu scan, non-power-of-two");
      StreamCompaction::CPU::scan(NPOT, c, a);
      printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                       "(std::chrono Measured)");
      if constexpr (enablePrintingArrays) printArray(NPOT, c, true);
      printCmpResult(NPOT, b, c);
    }

    if constexpr (enableNaiveScan) {
      zeroArray(SIZE, c);
      printDesc("naive scan, power-of-two");
      StreamCompaction::Naive::scan(SIZE, c, a);
      printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(SIZE, c, true);
      printCmpResult(SIZE, b, c);

      zeroArray(SIZE, c);
      printDesc("naive scan, non-power-of-two");
      StreamCompaction::Naive::scan(NPOT, c, a);
      printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(SIZE, c, true);
      printCmpResult(NPOT, b, c);
    }

    if constexpr (enableEfficientScan) {
      zeroArray(SIZE, c);
      printDesc("work-efficient scan, power-of-two");
      StreamCompaction::Efficient::scan(SIZE, c, a);
      printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(SIZE, c, true);
      printCmpResult(SIZE, b, c);

      zeroArray(SIZE, c);
      printDesc("work-efficient scan, non-power-of-two");
      StreamCompaction::Efficient::scan(NPOT, c, a);
      printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(NPOT, c, true);
      printCmpResult(NPOT, b, c);
    }
  }

  printf("\n");
  printf("*****************************\n");
  printf("** STREAM COMPACTION TESTS **\n");
  printf("*****************************\n");

  // Compaction tests

  genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
  a[SIZE - 1] = 0;
  printArray(SIZE, a, true);

  int count, expectedCount, expectedNPOT;

  // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
  // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
  zeroArray(SIZE, b);
  printDesc("cpu compact without scan, power-of-two");
  count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
  printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
  expectedCount = count;
  if constexpr (enablePrintingArrays) printArray(count, b, true);
  printCmpLenResult(count, expectedCount, b, b);

  zeroArray(SIZE, c);
  printDesc("cpu compact without scan, non-power-of-two");
  count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
  printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
  expectedNPOT = count;
  if constexpr (enablePrintingArrays) printArray(count, c, true);
  printCmpLenResult(count, expectedNPOT, b, c);

  if constexpr (enableCPUCompactWith) {
    zeroArray(SIZE, b);
    printDesc("cpu compact with scan, power-of-two");
    count = StreamCompaction::CPU::compactWithScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    if constexpr (enablePrintingArrays) printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    if constexpr (enablePrintingArrays) printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
  }

  if constexpr (enableEfficientCompact) {
    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
  }

  if constexpr (enableThrustCompact) {
    zeroArray(SIZE, c);
    printDesc("thrust compact, power-of-two");
    count = StreamCompaction::Thrust::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust compact, non-power-of-two");
    count = StreamCompaction::Thrust::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if constexpr (enablePrintingArrays) printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
  }

  delete[] a;
  delete[] b;
  delete[] c;
}
