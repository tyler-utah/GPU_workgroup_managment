#include "base/profile.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <ostream>
#include <string>
#include <vector>

#include "base/commandlineflags.h"

DEFINE_int32(profile_verbosity, 2, "Profiling trace verbosity");

namespace profile {

// Type of timed event. Invisible to users.
// Be sure to update the array in TimedTypeToString if changed.
enum TimedType {
  START = 0,
  EXCLUSION_BEGIN = 1,
  EXCLUSION_END = 2,
  REGION_BEGIN = 3,
  REGION_END = 4,
  POINT = 5
};

const char *TimedTypeToString(const TimedType type) {
  static const char *type_strings[6] =
      { "Start", "ExclusionBegin", "ExclusionEnd",
        "RegionBegin", "RegionEnd", "Point" };
  return type_strings[type];
}

// Resolution translations.
const char *ResolutionToUnit(const Resolution resolution) {
  static const char *unit_strings[4] = { "ns", "us", "ms", "s" };
  return unit_strings[resolution];
}

unsigned ResolutionToPeriod(const Resolution resolution) {
  static const unsigned ratios[4] = {1, 1000, 1000000, 1000000000};
  return ratios[resolution];
}

// Start point of the profiler. The timer used does not start at 0, giving large
// and unreadable clock values. The start point will therefore be excluded from
// all times.
class StartPoint {
 public:
  StartPoint();
};

// Sections of code inside the profiler that are too significant to ignore.
// These are not printed in the output, but are subtracted from future times.
class Exclusion {
 public:
  Exclusion();
  ~Exclusion();
};

// Collects all the time points into a single stream. Users should not refer to
// this directly, rather, use the timed event types in the header.
class Profiler {
 public:
  // Get or create the global instance.
  static Profiler *GetProfiler();

  // Appends a time to the stream, regardless of type. scope_change should be 1
  // if we increase or -1 if decrease.
  void AddTimePoint(const char *name, int scope_change, TimedType type);

  // Format and print the stream of time points.
  // Exclusion entries will propagate as they are encountered, and subtracted
  // from events further in the stream.
  void PrintEntries(std::ostream *out, Resolution resolution);

 private:
  // Individual timed points.
  struct TimeEntry {
    TimeEntry(const char *name, unsigned scope, TimedType type,
              uint64_t duration)
        : name(name), scope(scope), type(type), duration(duration) {
    }
    const char *name;
    unsigned scope;
    TimedType type;
    uint64_t duration;
  };

  // Global instance.
  static Profiler *profiler_;

  // Current scope.
  int scope_ = 0;
  // Entry stream.
  std::vector<TimeEntry> entries_;
};

Profiler *Profiler::profiler_ = nullptr;

// Start point entries.
StartPoint::StartPoint() {
  Profiler::GetProfiler()->AddTimePoint("Start", 0, START);
}

// Exclusion entries.
Exclusion::Exclusion() {
  Profiler::GetProfiler()->AddTimePoint("Exclude", 1, EXCLUSION_BEGIN);
}

Exclusion::~Exclusion() {
  Profiler::GetProfiler()->AddTimePoint("Exclude", -1, EXCLUSION_END);
}

// Region entries.
Region::Region(const char *name, Verbosity verbosity)
    : name(name), verbosity(verbosity) {
  if (FLAGS_profile_verbosity >= verbosity) {
    Profiler::GetProfiler()->AddTimePoint(name, 1, REGION_BEGIN);
  }
}

Region::~Region() {
  if (FLAGS_profile_verbosity >= verbosity) {
    Profiler::GetProfiler()->AddTimePoint(name, -1, REGION_END);
  }
}

// Point entries.
Point::Point(const char *name, Verbosity verbosity) {
  if (FLAGS_profile_verbosity >= verbosity) {
    Profiler::GetProfiler()->AddTimePoint(name, 0, POINT);
  }
}

// Profiler functions.
Profiler *Profiler::GetProfiler() {
  if (profiler_ == nullptr) {
    profiler_ = new Profiler;
    StartPoint();
  }
  return profiler_;
}

void Profiler::AddTimePoint(
    const char *name, int scope_change, TimedType type) {
  if (entries_.size() >= (entries_.capacity() - 5) &&
      (type != EXCLUSION_BEGIN && type != EXCLUSION_END)) {
    Exclusion exclusion;
    entries_.reserve(entries_.capacity() + 10000);
  }
  uint64_t time = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
  int recorded_scope = (scope_change < 0) ? (scope_ + scope_change) : scope_;
  scope_ += scope_change;
  entries_.emplace_back(name, recorded_scope, type, time);
}

void Profiler::PrintEntries(std::ostream *out, Resolution resolution) {
  auto flags = out->flags();
  uint64_t exclusion = 0;
  const char *unit = ResolutionToUnit(resolution);
  unsigned period = ResolutionToPeriod(resolution);
  *out << std::fixed << std::setprecision(2);
  if (entries_.size() != 0 || entries_[0].type == START) {
    exclusion += entries_[0].duration;
  }
  for (size_t idx = 1; idx < entries_.size(); ++idx) {
    const TimeEntry& entry = entries_[idx];
    // Exclude EXCLUSION events and add to exclusion duration.
    if (entry.type == EXCLUSION_BEGIN &&
        idx + 1 != entries_.size() &&
        entries_[idx + 1].type == EXCLUSION_END) {
      uint64_t exc_duration = entries_[idx + 1].duration - entry.duration;
      exclusion += exc_duration;
      ++idx;
      continue;
    }
    // Scale by the provided period duration.
    double duration = static_cast<double>(entry.duration - exclusion) /
                      static_cast<double>(period);
    *out << std::string(2 * entry.scope, '.')
         << TimedTypeToString(entry.type) << " " << entry.name << " "
         << duration << unit << std::endl;
  }
  out->flags(flags);
}

// Interface for printing.
void PrintProfileTrace(std::ostream *out) {
  PrintProfileTraceAtResolution(out, Nanoseconds);
}

void PrintProfileTraceAtResolution(std::ostream *out, Resolution resolution) {
  Profiler::GetProfiler()->PrintEntries(out, resolution);
}

}  // namespace profile
