// Platform specific stuff goes here.

#ifndef BASE_PLATFORM_H_
#define BASE_PLATFORM_H_

namespace platform {

// Platforms supported by this project.
enum PlatformType {
  POSIX = 0,
  WINDOWS = 1,
  UNSUPPORTED = 2
};

inline PlatformType& operator++(PlatformType& type) {
  type = static_cast<PlatformType>(type + 1);
  return type;
}

// The current platform.
const PlatformType Platform();

// The absolute path to the top level of this project.
const char *ProjectRoot();

// Root path.
const char *PathRoot();
const char *PathRootForPlatform(const PlatformType type);

// Checks if the path is rooted for any platform, and returns the first platform
// the path is rooted for if so.
PlatformType IsPathRooted(const char *path);

}  // namespace platform

#endif  // BASE_PLATFORM_H_
