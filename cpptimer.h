#ifndef cpptimer_h
#define cpptimer_h

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <set>

#ifndef _OPENMP
inline int omp_get_thread_num() { return 0; }
#endif

using namespace std;
using namespace chrono;

using keypair = pair<string, unsigned int>;
using statistics = tuple<double, double, double, double, unsigned long int>;

class CppTimer
{
protected:
  map<keypair, high_resolution_clock::time_point> tics; // Map of start times
  set<string> missing_tics, needless_tocs;              // Set of missing tics
  // Data to be returned: Tag, Mean, SD, Min, Max, Count
  map<string, statistics> data;

public:
  vector<string> tags;      // Vector of identifiers
  vector<double> durations; // Vector of durations
  bool verbose = true;      // Print warnings about not stopped timers

  // This ensures that there are no implicit conversions in the constructors
  // That means, the types must exactly match the constructor signature
  template <typename T>
  CppTimer(T &&) = delete;

  CppTimer() {}
  CppTimer(bool verbose) : verbose(verbose) {}

  // start a timer - save time
  void tic(string &&tag = "tictoc")
  {
    keypair key(std::move(tag), omp_get_thread_num());

#pragma omp critical
    tics[key] = high_resolution_clock::now();
  }

  // stop a timer - calculate time difference and save key
  void toc(string &&tag = "tictoc")
  {

    keypair key(std::move(tag), omp_get_thread_num());

#pragma omp critical
    {
      if (auto tic{tics.find(key)}; tic != end(tics))
      {
        nanoseconds duration = high_resolution_clock::now() -
                               std::move(tic->second);
        durations.push_back(duration.count());
        tic->second = tic->second.min();
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
    string tag;

  public:
    ScopedTimer(CppTimer &timer, string tag = "scoped") : timer(timer), tag(tag)
    {
      timer.tic(string(tag));
    }
    ~ScopedTimer()
    {
      timer.toc(string(tag));
    }
  };

  map<string, statistics> aggregate()
  {
    // Calculate summary statistics
    for (unsigned long int i = 0; i < tags.size(); i++)
    {
      // Welford's online algorithm for mean and sst
      // sst = sum of squared total deviations
      double mean = 0, sst = 0, min = numeric_limits<double>::max(), max = 0;
      unsigned long int count = 0;
      if (auto entry{data.find(tags[i])}; entry != end(data))
      {
        tie(mean, sst, min, max, count) = entry->second;
      }
      count++;
      double duration = durations[i];
      if (duration < 0)
      {
        needless_tocs.insert(tags[i]);
        continue;
      }
      double delta = duration - mean;
      mean += delta / count;
      sst += delta * (duration - mean);
      min = std::min(min, duration);
      max = std::max(max, duration);
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
