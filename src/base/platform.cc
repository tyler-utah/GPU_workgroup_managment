#include "base/platform.h"

#include <cstring>

namespace platform {

// The current platform, used throughout.
const PlatformType g_platform =
#if defined __unix__
POSIX;
#elif defined __WINDOWS__
WINDOWS;
#else
UNSUPPORTED;
#endif

const PlatformType Platform() {
  return g_platform;
}

// The absolute path to the top level of this project.
#define PROJECT_ROOT_ PROJECT_ROOT_EXPAND(PROJECT_ROOT)
#define PROJECT_ROOT_EXPAND(P) PROJECT_ROOT_STRINGIFY(P)
#define PROJECT_ROOT_STRINGIFY(P) #P
const char *g_project_root = PROJECT_ROOT_;

const char *ProjectRoot() {
  return g_project_root;
}

// Root path.
// For Windows, this requires getting the letter from the project. Provide a
// root for Windows for size info.
#ifdef __WINDOWS__
char g_win_drive[4];
#define WIN_DRIVE strncpy(g_win_drive, PROJECT_ROOT_, 3)
#else
#define WIN_DRIVE "C:\\"
#endif

const char *g_path_roots[UNSUPPORTED + 1] = { "/", WIN_DRIVE, "" };
const char *g_path_root = g_path_roots[g_platform];

const char *PathRoot() {
  return g_path_root;
}

const char *PathRootForPlatform(const PlatformType type) {
  return g_path_roots[type];
}

// It is not enough to check is each path root is a substring, as Windows can
// have any letter as root. Returns the first platform the path is rooted for.
PlatformType IsPathRooted(const char *path) {
  if (strlen(path) >= 3 &&
      path[0] >= 'A' && path[0] <= 'Z' &&
      path[1] == ':' && path[2] == '\\') {
    return WINDOWS;
  }
  const size_t path_size = strlen(path);
  for (PlatformType type = POSIX; type != UNSUPPORTED; ++type) {
    const size_t path_root_size = strlen(g_path_roots[type]);
    if (path_root_size > 0 &&
        strncmp(path, g_path_roots[type], path_root_size) == 0) {
      return type;
    }
  }
  return UNSUPPORTED;
}

}  // namespace platform
