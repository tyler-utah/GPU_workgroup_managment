// Utilities for handling files and platform specific paths.

#ifndef BASE_FILE_H_
#define BASE_FILE_H_

#include <string>

namespace file {

// Path converter to convert between OSes and append project root if necessary.
//
// Three basic conversions will be made to the path:
//   - Replace root (e.g. 'C:' -> '/').
//   - Append project path if there is no root
//     (e.g. 'single_kernel\kernel.cl' -> 'C:\project\single_kernel\kernel.cl').
//
// This struct is intended to be used inline, and will automatically convert to
// string or char * when necessary, for example:
//   context.CreateProgramFromFile(file::Path("single_kernel/kernel.cl"), "");
// will become:
//   context.CreateProgramFromFile(
//       std::string("/path/to/project/single_kernel/kernel.cl"), "");
//
// The conversion does not currently translate path separator, as both Unix and
// Windows support '/', including other platforms may rquire this. All paths
// must therefore use '/' for separators.
struct Path {
  Path(const char *path);
  Path(const std::string& path);
  operator std::string() const { return path_; }
  operator const char *() const { return path_.c_str(); }
  const std::string path_;
};

}  // namespace file

#endif  // BASE_FILE_H_
