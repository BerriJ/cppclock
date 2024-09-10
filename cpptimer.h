#ifndef cpptimer_h
#define cpptimer_h

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <string>
#include <vector>
#include <map>

#ifndef _OPENMP
inline int omp_get_thread_num() { return 0; }
#endif

namespace sc = std::chrono;
using hr_clock = sc::high_resolution_clock;
using keypair = std::pair<std::string, unsigned int>;

class CppTimer
{

protected:
  std::map<keypair, hr_clock::time_point> tics; // Map of start times
  std::set<std::string> missing_tics;           // Set of missing tics
  // Data to be returned: Tag, Mean, SD, Count
  std::map<std::string, std::tuple<double, double, double, double, unsigned long int>> data;

public:
  std::vector<std::string> tags; // Vector of identifiers
  std::vector<double> durations; // Vector of durations
  bool verbose = true;           // Print warnings about not stopped timers

  // This ensures that there are no implicit conversions in the constructors
  // That means, the types must exactly match the constructor signature
  template <typename T>
  CppTimer(T &&) = delete;

  CppTimer() {}
  CppTimer(bool verbose) : verbose(verbose) {}

  // start a timer - save time
  void tic(std::string &&tag = "tictoc")
  {
    keypair key(std::move(tag), omp_get_thread_num());

#pragma omp critical
    tics[key] = hr_clock::now();
  }

  // stop a timer - calculate time difference and save key
  void toc(std::string &&tag = "tictoc")
  {

    keypair key(std::move(tag), omp_get_thread_num());

#pragma omp critical
    {
      if (auto tic{tics.find(key)}; tic != std::end(tics))
      {
        sc::nanoseconds duration = hr_clock::now() - std::move(tic->second);
        durations.push_back(duration.count());
        tics.erase(tic);
        tags.push_back(std::move(key.first));
      }
      else
      {
        missing_tics.insert(std::move(key.first));
      }
    }
  }

  class ScopedTimer
  {
  private:
    CppTimer &timer;
    std::string tag;

  public:
    ScopedTimer(CppTimer &timer, std::string tag = "scoped") : timer(timer), tag(tag)
    {
      timer.tic(std::string(tag));
    }
    ~ScopedTimer()
    {
      timer.toc(std::string(tag));
    }
  };

  std::map<std::string, std::tuple<double, double, double, double, unsigned long int>> aggregate()
  {

    // Calculate summary statistics
    for (unsigned long int i = 0; i < tags.size(); i++)
    {
      // sst = sum of squared total deviations
      double mean = 0, sst = 0, min = std::numeric_limits<double>::max(), max = 0;
      unsigned long int count = 0;

      if (data.count(tags[i]) > 0)
      {
        std::tie(mean, sst, min, max, count) = data[tags[i]];
      }

      // Welford's online algorithm for mean and variance
      count++;
      double delta = durations[i] - mean;
      mean += delta / count;
      sst += delta * (durations[i] - mean);
      min = std::min(min, durations[i]);
      max = std::max(max, durations[i]);

      // Save mean, variance and count
      data[tags[i]] = {mean, sst, min, max, count};
    }

    tags.clear(), durations.clear();
    return (data);
  }

  void reset()
  {
    tics.clear(), durations.clear(), tags.clear(), data.clear();
  }
};

#endif
