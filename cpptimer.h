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

#pragma omp critical
      {
        sc::nanoseconds duration = hr_clock::now() - std::move(*address);
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

  // Pass data to R / Python
  std::map<std::string, std::tuple<double, double, unsigned long int>> aggregate()
  {
    // Warn about all timers not being stopped
    if (verbose)
    {
      for (auto const &tic : tics)
      {
        std::string tic_tag = tic.first.first;
        std::string msg;
        msg += "Timer \"" + tic_tag + "\" not stopped yet. \n";
        msg += "Use toc(\"" + tic_tag + "\") to stop the timer.";
        warn(msg);
      }
    }

    // Get vector of unique tags

    std::vector<std::string> unique_tags = tags;
    std::sort(unique_tags.begin(), unique_tags.end());
    unique_tags.erase(
        std::unique(unique_tags.begin(), unique_tags.end()), unique_tags.end());

    for (unsigned int i = 0; i < unique_tags.size(); i++)
    {

      std::string tag = unique_tags[i];

      unsigned long int count;
      double mean, sst; // mean and sum of squared total deviations

      // Init
      if (data.count(tag) == 0)
      {
        count = 0;
        mean = 0;
        sst = 0;
      }
      else
      {
        mean = std::get<0>(data[tag]);
        sst = std::get<1>(data[tag]);
        count = std::get<2>(data[tag]);
      }

      // Update
      for (unsigned long int j = 0; j < tags.size(); j++)
      {
        if (tags[j] == tag)
        {
          // Welford's online algorithm for mean and variance
          count++;
          double delta = durations[j] - mean;
          mean += delta / count;
          sst += delta * (durations[j] - mean);
        }
      }

      // Save mean, variance and count
      unsigned long int one = 1;
      data[tag] = std::make_tuple(mean, sst / std::max(count - 1, one), count);
    }

    tags.clear();
    durations.clear();

    return (data);
  }

  void reset()
  {
    tics.clear();
    durations.clear();
    tags.clear();
    data.clear();
  }
};

#endif
