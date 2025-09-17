#include "testing_helpers.hpp"

#include <array>
#include <cstdio>
#include <functional>
#include <stream_compaction/cpu.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/thrust.h>
#include <string_view>
#include <utility>

constexpr int sizePOT = 1 << 26;       // feel free to change the size of array
constexpr int sizeNPOT = sizePOT - 3;  // Non-Power-Of-Two

/// If true, run additional simpler tests.
constexpr bool runDebugTests = false;

/// Run benchmarks instead of tests.
constexpr bool runBenchmarks = false;

/// Print out resulting arrays from computation.
constexpr bool enablePrintingArrays = false;

constexpr bool enableCPUScan = true;
constexpr bool enableNaiveScan = true;
constexpr bool enableEfficientScan = true;
constexpr bool enableThrustScan = true;

constexpr bool enableCPUCompact = true;
constexpr bool enableEfficientCompact = true;
constexpr bool enableThrustCompact = true;

namespace Perf {

using TimerFn = std::function<float()>;
using ScanFn = std::function<void(int, int*, const int*)>;
using CompactionFn = std::function<int(int, int*, const int*)>;

enum class Implementation { CPU, Naive, Efficient, Thrust };

constexpr int numIterations = 10;
constexpr int maxValue = 50;

std::pair<ScanFn, TimerFn> getScanImplementation(Implementation implementation) {
  using namespace StreamCompaction;

  auto cpu = &Common::PerformanceTimer::getCpuElapsedTimeForPreviousOperation;
  auto gpu = &Common::PerformanceTimer::getGpuElapsedTimeForPreviousOperation;

  switch (implementation) {
    case Implementation::CPU:
      return std::make_pair<ScanFn, TimerFn>(CPU::scan, std::bind(cpu, &CPU::timer()));

    case Implementation::Naive:
      return std::make_pair<ScanFn, TimerFn>(Naive::scan, std::bind(gpu, &Naive::timer()));

    case Implementation::Efficient:
      return std::make_pair<ScanFn, TimerFn>(Efficient::scan, std::bind(gpu, &Efficient::timer()));

    case Implementation::Thrust:
      return std::make_pair<ScanFn, TimerFn>(Thrust::scan, std::bind(gpu, &Thrust::timer()));

    default:
      throw std::invalid_argument("invalid enum");
  }
}

void runScanBenchmark(Implementation implementation, int n, std::string_view benchmarkName) {
  std::string prefix = "[" + std::string(benchmarkName) + "]";
  const auto [scan, getTime] = getScanImplementation(implementation);

  std::vector<float> elapsedTimes;
  std::unique_ptr<int[]> out = std::make_unique<int[]>(sizePOT);
  std::unique_ptr<int[]> in = std::make_unique<int[]>(sizePOT);
  genArray(sizePOT, in.get(), maxValue);

  for (int i = 1; i <= numIterations; ++i) {
    std::cout << prefix << " Executing scan(): " << i << " of " << numIterations << "...\r";

    scan(n, out.get(), in.get());
    elapsedTimes.push_back(getTime());
  }

  float average = 0.f;
  for (const float& time : elapsedTimes) {
    average += time;
  }
  average /= elapsedTimes.size();

  std::cout << prefix << " Average scan() time: " << average << "                         " << std::endl;
}

}  // namespace Perf

int* a = new int[sizePOT];
int* b = new int[sizePOT];
int* c = new int[sizePOT];

int main(int argc, char* argv[]) {
  if constexpr (runBenchmarks) {
    printf("********************\n");
    printf("** SCAN BENCHMARK **\n");
    printf("********************\n\n");

    std::cout << "- Number of iterations: " << Perf::numIterations << std::endl;
    std::cout << "- Size of POT array: " << sizePOT << std::endl;
    std::cout << "- Size of NPOT array: " << sizeNPOT << "\n" << std::endl;

    if constexpr (enableCPUScan) {
      Perf::runScanBenchmark(Perf::Implementation::CPU, sizePOT, "CPU/POT");
      Perf::runScanBenchmark(Perf::Implementation::CPU, sizeNPOT, "CPU/NPOT");
    }

    if constexpr (enableNaiveScan) {
      Perf::runScanBenchmark(Perf::Implementation::Naive, sizePOT, "Naive/POT");
      Perf::runScanBenchmark(Perf::Implementation::Naive, sizeNPOT, "Naive/NPOT");
    }

    if constexpr (enableEfficientScan) {
      Perf::runScanBenchmark(Perf::Implementation::Efficient, sizePOT, "Efficient/POT");
      Perf::runScanBenchmark(Perf::Implementation::Efficient, sizeNPOT, "Efficient/NPOT");
    }

    if constexpr (enableThrustScan) {
      Perf::runScanBenchmark(Perf::Implementation::Thrust, sizePOT, "Thrust/POT");
      Perf::runScanBenchmark(Perf::Implementation::Thrust, sizeNPOT, "Thrust/NPOT");
    }
  } else {
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(sizePOT - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[sizePOT - 1] = 0;
    printArray(sizePOT, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(sizePOT, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(sizePOT, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    if constexpr (enablePrintingArrays) printArray(sizePOT, b, true);

    if constexpr (enableCPUScan) {
      zeroArray(sizePOT, c);
      printDesc("cpu scan, non-power-of-two");
      StreamCompaction::CPU::scan(sizeNPOT, c, a);
      printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                       "(std::chrono Measured)");
      if constexpr (enablePrintingArrays) printArray(sizeNPOT, c, true);
      printCmpResult(sizeNPOT, b, c);
    }

    if constexpr (enableNaiveScan) {
      zeroArray(sizePOT, c);
      printDesc("naive scan, power-of-two");
      StreamCompaction::Naive::scan(sizePOT, c, a);
      printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(sizePOT, c, true);
      printCmpResult(sizePOT, b, c);

      zeroArray(sizePOT, c);
      printDesc("naive scan, non-power-of-two");
      StreamCompaction::Naive::scan(sizeNPOT, c, a);
      printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(sizePOT, c, true);
      printCmpResult(sizeNPOT, b, c);
    }

    if constexpr (enableEfficientScan) {
      zeroArray(sizePOT, c);
      printDesc("work-efficient scan, power-of-two");
      StreamCompaction::Efficient::scan(sizePOT, c, a);
      printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(sizePOT, c, true);
      printCmpResult(sizePOT, b, c);

      zeroArray(sizePOT, c);
      printDesc("work-efficient scan, non-power-of-two");
      StreamCompaction::Efficient::scan(sizeNPOT, c, a);
      printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(sizeNPOT, c, true);
      printCmpResult(sizeNPOT, b, c);
    }

    if constexpr (enableThrustScan) {
      zeroArray(sizePOT, c);
      printDesc("thrust scan, power-of-two");
      StreamCompaction::Thrust::scan(sizePOT, c, a);
      printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(sizePOT, c, true);
      printCmpResult(sizePOT, b, c);

      zeroArray(sizePOT, c);
      printDesc("thrust scan, non-power-of-two");
      StreamCompaction::Thrust::scan(sizeNPOT, c, a);
      printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(sizeNPOT, c, true);
      printCmpResult(sizeNPOT, b, c);
    }

    // For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    if constexpr (runDebugTests) {
      printf("\n");
      printf("*************************\n");
      printf("** SCAN TESTS (ALL 1s) **\n");
      printf("*************************\n");

      onesArray(sizePOT, a);

      zeroArray(sizePOT, b);
      printDesc("cpu scan, power-of-two");
      StreamCompaction::CPU::scan(sizePOT, b, a);
      printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                       "(std::chrono Measured)");
      if constexpr (enablePrintingArrays) printArray(sizePOT, b, true);

      if constexpr (enableCPUScan) {
        zeroArray(sizePOT, c);
        printDesc("cpu scan, non-power-of-two");
        StreamCompaction::CPU::scan(sizeNPOT, c, a);
        printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                         "(std::chrono Measured)");
        if constexpr (enablePrintingArrays) printArray(sizeNPOT, c, true);
        printCmpResult(sizeNPOT, b, c);
      }

      if constexpr (enableNaiveScan) {
        zeroArray(sizePOT, c);
        printDesc("naive scan, power-of-two");
        StreamCompaction::Naive::scan(sizePOT, c, a);
        printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        if constexpr (enablePrintingArrays) printArray(sizePOT, c, true);
        printCmpResult(sizePOT, b, c);

        zeroArray(sizePOT, c);
        printDesc("naive scan, non-power-of-two");
        StreamCompaction::Naive::scan(sizeNPOT, c, a);
        printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        if constexpr (enablePrintingArrays) printArray(sizePOT, c, true);
        printCmpResult(sizeNPOT, b, c);
      }

      if constexpr (enableEfficientScan) {
        zeroArray(sizePOT, c);
        printDesc("work-efficient scan, power-of-two");
        StreamCompaction::Efficient::scan(sizePOT, c, a);
        printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                         "(CUDA Measured)");
        if constexpr (enablePrintingArrays) printArray(sizePOT, c, true);
        printCmpResult(sizePOT, b, c);

        zeroArray(sizePOT, c);
        printDesc("work-efficient scan, non-power-of-two");
        StreamCompaction::Efficient::scan(sizeNPOT, c, a);
        printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                         "(CUDA Measured)");
        if constexpr (enablePrintingArrays) printArray(sizeNPOT, c, true);
        printCmpResult(sizeNPOT, b, c);
      }
    }
  }

  if constexpr (runBenchmarks) {
  } else {
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    genArray(sizePOT - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[sizePOT - 1] = 0;
    printArray(sizePOT, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(sizePOT, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(sizePOT, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    if constexpr (enablePrintingArrays) printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(sizePOT, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(sizeNPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    if constexpr (enablePrintingArrays) printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    if constexpr (enableCPUCompact) {
      zeroArray(sizePOT, b);
      printDesc("cpu compact with scan, power-of-two");
      count = StreamCompaction::CPU::compactWithScan(sizePOT, b, a);
      printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                       "(std::chrono Measured)");
      expectedCount = count;
      if constexpr (enablePrintingArrays) printArray(count, b, true);
      printCmpLenResult(count, expectedCount, b, b);

      zeroArray(sizePOT, c);
      printDesc("cpu compact with scan, non-power-of-two");
      count = StreamCompaction::CPU::compactWithScan(sizeNPOT, c, a);
      printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                       "(std::chrono Measured)");
      expectedNPOT = count;
      if constexpr (enablePrintingArrays) printArray(count, c, true);
      printCmpLenResult(count, expectedNPOT, b, c);
    }

    if constexpr (enableEfficientCompact) {
      zeroArray(sizePOT, c);
      printDesc("work-efficient compact, power-of-two");
      count = StreamCompaction::Efficient::compact(sizePOT, c, a);
      printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(count, c, true);
      printCmpLenResult(count, expectedCount, b, c);

      zeroArray(sizePOT, c);
      printDesc("work-efficient compact, non-power-of-two");
      count = StreamCompaction::Efficient::compact(sizeNPOT, c, a);
      printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(count, c, true);
      printCmpLenResult(count, expectedNPOT, b, c);
    }

    if constexpr (enableThrustCompact) {
      zeroArray(sizePOT, c);
      printDesc("thrust compact, power-of-two");
      count = StreamCompaction::Thrust::compact(sizePOT, c, a);
      printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(count, c, true);
      printCmpLenResult(count, expectedCount, b, c);

      zeroArray(sizePOT, c);
      printDesc("thrust compact, non-power-of-two");
      count = StreamCompaction::Thrust::compact(sizeNPOT, c, a);
      printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
      if constexpr (enablePrintingArrays) printArray(count, c, true);
      printCmpLenResult(count, expectedNPOT, b, c);
    }
  }

  delete[] a;
  delete[] b;
  delete[] c;
}
