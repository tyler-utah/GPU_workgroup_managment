// This is the file that should be included by any file which declares
// or defines a command line flag or wants to parse command line flags
// or print a program usage message (which will include information about
// flags).  Executive summary, in the form of an example foo.cc file:
//
//    #include "foo.h"         // foo.h has a line "DECLARE_int32(start);"
//    #include "validators.h"  // hypothetical file defining ValidateIsFile()
//
//    DEFINE_int32(end, 1000, "The last record to read");
//
//    DEFINE_string(filename, "my_file.txt", "The file to read");
//    // Crash if the specified file does not exist.
//    static bool dummy = RegisterFlagValidator(&FLAGS_filename,
//                                              &ValidateIsFile);
//
//    DECLARE_bool(verbose); // some other file has a DEFINE_bool(verbose, ...)
//
//    void MyFunc() {
//      if (FLAGS_verbose) printf("Records %d-%d\n", FLAGS_start, FLAGS_end);
//    }
//
//    Then, at the command-line:
//       ./foo --noverbose --start=5 --end=100
//
// For more details, see
//    doc/gflags.html
//
// --- A note about thread-safety:
//
// We describe many functions in this routine as being thread-hostile,
// thread-compatible, or thread-safe.  Here are the meanings we use:
//
// thread-safe: it is safe for multiple threads to call this routine
//   (or, when referring to a class, methods of this class)
//   concurrently.
// thread-hostile: it is not safe for multiple threads to call this
//   routine (or methods of this class) concurrently.  In gflags,
//   most thread-hostile routines are intended to be called early in,
//   or even before, main() -- that is, before threads are spawned.
// thread-compatible: it is safe for multiple threads to read from
//   this variable (when applied to variables), or to call const
//   methods of this class (when applied to classes), as long as no
//   other thread is writing to the variable or calling non-const
//   methods of this class.

#ifndef BASE_COMMANDLINEFLAGS_H_
#define BASE_COMMANDLINEFLAGS_H_

#include "third_party/gflags/gflags.h"

namespace flags = gflags;

#endif  // BASE_COMMANDLINEFLAGS_H_
