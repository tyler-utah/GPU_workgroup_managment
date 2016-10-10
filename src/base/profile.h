// Profiler for getting high resolution times for marked regions of code. This
// is a simple implementation, which does no corrections on overhead or any
// other fixes to the times read, so there will be some error. All times are in
// nanoseconds. Uses std::chrono::high_precision_clock, and reports duration
// (not cpu_time).
//
// Provides two means of mmeasuring time:
//   - profile::Region: Gives the start and end times for a scoped region.
//   - profile::Point: Gives the time at a point.
//
// Example:
//
//   void MyFunc() {
//     profile::Region region("MyFunc", profile::VERBOSITY_MEDIUM);
//     ...
//     profile::Point("MyFunc_midpoint", profile::VERBOSITY_HIGH);
//     ...
//   }
//
// The names given to each region must be a string literal, that will exist for
// the duration of the program, not mutable data that has been marked const.

#ifndef BASE_PROFILE_H_
#define BASE_PROFILE_H_

#include <ostream>

namespace profile {

// Profiling verbosity for printing/recording time points.
// Events are only printed if its verbosity is less or equal to that of
// FLAGS_profile_verbosity.
enum Verbosity {
  VERBOSITY_NONE = 0,
  VERBOSITY_LOW = 1,
  VERBOSITY_MEDIUM = 2,
  VERBOSITY_HIGH = 3
};

// Resolution at which to print the output.
// Must update the translations in the cc if these are changed.
enum Resolution {
  Nanoseconds = 0,
  Microseconds = 1,
  Milliseconds = 2,
  Seconds = 3
};

// Scoped region. Put at the start of a region for the begin and end times.
class Region {
 public:
  Region(const char *name, Verbosity verbosity);
  ~Region();
 private:
  const char *name;
  Verbosity verbosity;
};

// Point. Gives a single time at the point.
class Point {
 public:
  Point(const char *name, Verbosity verbosity);
};

// Prints the trace of profiling events.
// Period gives the number of nanoseconds per unit to print.
void PrintProfileTrace(std::ostream *out);
void PrintProfileTraceAtResolution(std::ostream *out, Resolution resolution);

}  // namespace profile

#endif  // BASE_PROFILE_H_
