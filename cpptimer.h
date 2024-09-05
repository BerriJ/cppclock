#ifndef cpptimer_h
#define cpptimer_h

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <string>
#include <vector>
#include <map>

// Language specific functions (currently warnings)
#include <chameleon.h>

#ifndef _OPENMP
inline int omp_get_thread_num() { return 0; }
#endif

namespace sc = std::chrono;

class CppTimer
{
  using hr_clock = sc::high_resolution_clock;
  using keypair = std::pair<std::string, unsigned int>;

private:
  std::map<keypair, hr_clock::time_point> tics; // Map of start times

protected:
  // Data to be returned: Tag, Mean, SD, Count
  std::map<std::string, std::tuple<double, double, unsigned long int>> data;

public:
  std::vector<std::string> tags; // Vector of identifiers
  std::vector<double> durations; // Vector of durations

  bool verbose = true; // Print warnings about not stopped timers

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
  void
  toc(std::string &&tag = "tictoc")
  {

    keypair key(std::move(tag), omp_get_thread_num());

    // This construct is used to have a single lookup in the map
    // See https://stackoverflow.com/a/31806386/9551847
    auto it = tics.find(key);
    auto *address = it == tics.end() ? nullptr : std::addressof(it->second);

    if (address == nullptr)
    {
      if (verbose)
      {
        std::string msg;
        msg += "Timer \"" + key.first + "\" not started yet. \n";
        msg += "Use tic(\"" + key.first + "\") to start the timer.";
        warn(msg);
      }
      return;
    }
    else
    {
      sc::nanoseconds duration = hr_clock::now() - std::move(*address);
#pragma omp critical
      {
        durations.push_back(duration.count());
        tics.erase(key);
        tags.push_back(std::move(key.first));
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

  std::map<std::string, std::tuple<double, double, unsigned long int>> aggregate()
  {
    // Warn about all timers not being stopped
    if (verbose)
    {
      for (auto const &tic : tics)
      {
        std::string msg;
        msg += "Timer \"" + tic.first.first + "\" not stopped yet. \n";
        msg += "Use toc(\"" + tic.first.first + "\") to stop the timer.";
        warn(msg);
      }
    }

    // Calculate summary statistics
    for (unsigned long int i = 0; i < tags.size(); i++)
    {
      double mean = 0, sst = 0; // sst = sum of squared total deviations
      unsigned long int count = 0;

      if (data.count(tags[i]) > 0)
      {
        std::tie(mean, sst, count) = data[tags[i]];
      }

      // Welford's online algorithm for mean and variance
      count++;
      double delta = durations[i] - mean;
      mean += delta / count;
      sst += delta * (durations[i] - mean);

      // Save mean, variance and count
      data[tags[i]] = std::make_tuple(mean, sst, count);
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
