import argparse
import findtools
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Kernel merge")

# Required arguments

parser.add_argument("--non_persistent", type=str, action="store",
                    help="Path to non-persistent kernel; must be used in combination with --persistent.")
parser.add_argument("--persistent", type=str, action="store",
                    help="Path to persistent kernel; if used in combination with --non_persistent, causes the kernels to be merged, otherwise causes a stand-alone persistent kernel to be generated.")

args = parser.parse_args()

if args.persistent is None:
  sys.stderr.write("A persistent kernel must be specified using --persistent")
  sys.exit(1)

standalone = args.non_persistent is None
toolname = "standalone_persistent" if standalone else "kernel_merge"

cmd = [ findtools.kernel_merge_bin + os.sep + toolname ]

if not standalone:
  cmd += [ args.non_persistent ]

cmd += [ args.persistent,
         "--",
         "-I",
         findtools.clang_built_includes,
         "-I",
         os.sep.join([findtools.GPU_workgroup_management_root, "kernel_merge", "stubs"]),
         "-include",
         "opencl-c.h",
         "-include",
         "stubs.h",
         "-cl-std=CL2.0" ]

print("Running " + toolname + " tool: " + " ".join(cmd))

proc = subprocess.Popen(cmd)
proc.communicate()
if proc.returncode != 0:
  sys.exit(1)

outputfile = ("standalone" if standalone else "merged") + ".cl"

cmd = [ findtools.llvm_bin_dir + os.sep + "clang",
        "-cc1",
        "-finclude-default-header",
        "-I",
        findtools.clang_built_includes,
        "-cl-std=CL2.0",
        "-I",
        ".",
        "-I",
        os.sep.join([findtools.GPU_workgroup_management_root, "src", "scheduler_rt", "rt_device"]),
        "-include",
        os.sep.join([findtools.GPU_workgroup_management_root, "src", "scheduler_rt", "rt_common", "cl_types.h"]),
        outputfile ]

print("Checking result using clang: " + " ".join(cmd))

proc = subprocess.Popen(cmd)
proc.communicate()
if proc.returncode != 0:
  sys.exit(1)

cmd = [ findtools.llvm_bin_dir + os.sep + "clang-format",
        '-style={SortIncludes: false}',
        "-i",
        outputfile ]

print("Tidying up using clang-format: " + " ".join(cmd))

proc = subprocess.Popen(cmd)
proc.communicate()
if proc.returncode != 0:
  sys.exit(1)

