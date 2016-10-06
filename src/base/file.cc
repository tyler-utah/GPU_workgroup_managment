#include "base/file.h"

#include <string>

#include "base/platform.h"

namespace file {
namespace {

// Replace root (e.g. / -> C:) based on the current OS.
void ConvertPathRoot(std::string *path) {
  platform::PlatformType type = platform::IsPathRooted(path->c_str());
  if (type == platform::UNSUPPORTED ||
      type == platform::Platform()) {
    return;
  }
  *path = std::string(platform::PathRoot()) +
          &path->at(std::string(platform::PathRootForPlatform(type)).size());
}

// Append the path to the project if it is not already there.
void AppendProjectDirectory(std::string *path) {
  if (platform::IsPathRooted(path->c_str()) == platform::UNSUPPORTED) {
    *path = platform::ProjectRoot() + *path;
  }
}

// Applies all of the path conversions, allowing the user to specify a single
// path that will work on all useful machines.
std::string ConvertPath(const std::string& path) {
  std::string new_path = path;
  ConvertPathRoot(&new_path);
  AppendProjectDirectory(&new_path);
  return new_path;
}

}  // namespace

Path::Path(const char *path) : path_(ConvertPath(std::string(path))) {}
Path::Path(const std::string& path) : path_(ConvertPath(path)) {}

}  // namespace file
