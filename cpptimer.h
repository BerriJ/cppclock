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

// Helper functions

// Get all unique elements of a std::vector
template <typename T>
std::vector<T> remove_duplicates(std::vector<T> vec)
{
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    return (vec);
}

// Retrieve adress of value from a std::map or retreive nullptr if not found
template <typename Map, typename Key>
auto value(Map &m, const Key &key)
{
    auto it = m.find(key);

    return it == m.end() ? nullptr : std::addressof(it->second);
}

namespace sc = std::chrono;

class CppTimer
{
    using hr_clock = sc::high_resolution_clock;
    using keypair = std::pair<std::string, unsigned int>;

private:
    std::map<keypair, hr_clock::time_point> tics;  // Map of start times
    std::vector<std::string> tags;                 // Vector of identifiers
    std::vector<unsigned long long int> durations; // Vector of durations

protected:
    // Data to be returned: Tag, Mean, SD, Count
    std::map<std::string, std::tuple<double, double, unsigned long int>> data;

public:
    std::string name = "times"; // Name of R object to return
    bool verbose = true;        // Print warnings about not stopped timers

    template <typename T>
    CppTimer(T &&) = delete;

    CppTimer() {}
    CppTimer(const char *name) : name(name) {}
    CppTimer(bool verbose) : verbose(verbose) {}
    CppTimer(std::string name, bool verbose) : name(name), verbose(verbose) {}

    // start a timer - save time
    void tic(std::string &&tag)
    {
        keypair key(std::move(tag), omp_get_thread_num());

#pragma omp critical
        tics[key] = hr_clock::now();
    }

    // stop a timer - calculate time difference and save key
    void
    toc(std::string &&tag)
    {

        keypair key(std::move(tag), omp_get_thread_num());

        // This construct is used to have a single lookup in the map
        // See https://stackoverflow.com/a/31806386/9551847
        auto *command = value(tics, key);

        if (command == nullptr)
        {
            if (verbose)
            {
                std::string msg;
                msg += "Timer \"" + key.first + "\" not started yet. \n";
                msg += "Use tic(\"" + key.first + "\") to start the timer.";
                chameleon::warn(msg);
            }
            return;
        }
        else
        {

#pragma omp critical
            {
                durations.push_back(
                    sc::duration_cast<sc::microseconds>(
                        hr_clock::now() - std::move(*command))
                        .count());
                tics.erase(key);
                tags.push_back(std::move(key.first));
            }
        }
    }

    class ScopedTimer
    {
    private:
        CppTimer &clock;
        std::string tag = "ScopedTimer";

    public:
        ScopedTimer(CppTimer &clock, std::string tag) : clock(clock), tag(tag)
        {
            clock.tic(std::string(tag));
        }
        ~ScopedTimer()
        {
            clock.toc(std::string(tag));
        }
    };

    // Pass data to R / Python
    void aggregate()
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
                chameleon::warn(msg);
            }
        }

        // Vector of unique identifiers
        std::vector<std::string> unique_tags = remove_duplicates(tags);

        for (unsigned int i = 0; i < unique_tags.size(); i++)
        {

            std::string tag = unique_tags[i];

            unsigned long int count;
            double mean, M2;

            // Init
            if (data.count(tag) == 0)
            {
                count = 0;
                mean = 0;
                M2 = 0;
            }
            else
            {
                mean = std::get<0>(data[tag]);
                M2 = std::get<1>(data[tag]);
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
                    M2 += delta * (durations[j] - mean);
                }
            }

            // Save
            data[tag] = std::make_tuple(mean, M2, count);
        }

        tags.clear();
        durations.clear();
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