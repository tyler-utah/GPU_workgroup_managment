from __future__ import print_function
import sys
import os
import subprocess

def fake_placeholder(x):
    return

EXE_PATH     = ""
DATA_PATH    = ""
ITERS        = 1
PRINT        = fake_placeholder
DATA_PRINT   = fake_placeholder
NAME_OF_CHIP = ""

PROGRAMS = {
    "pannotia_color"
}

PROGRAM_DATA = {
    "pannotia_color" : [ {
        "input" : os.path.join("inputs", "color", "ecology1.graph"),
        "solution" : os.path.join("solutions", "color_ecology.txt") } ]
}

def run_suite():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            exe = os.path.join(EXE_PATH, p)
            graph_in = os.path.join(DATA_PATH, d["input"])
            graph_sol = os.path.join(DATA_PATH, d["solution"])
            cmd = [exe, "--graph_file", graph_in, "--graph_solution_file", graph_sol]
            print("WOULD RUN: ", cmd)
            # p_obj = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # ret_code = p_obj.wait()
            # sout, serr = p_obj.communicate()
            # print("* stdout:")
            # print(sout)
            # print("* stderr:")
            # print(serr)
            # if (ret_code != 0):
            #     print("Error running " + " ".join(cmd))
            #     exit(ret_code)

def main():

    global EXE_PATH
    global DATA_PATH
    global PRINT
    global DATA_PRINT
    global NAME_OF_CHIP

    # if len(sys.argv) != 5:
    #     print "Please provide the following arguments:"
    #     print "path to executables, path to data, name of run, name of chip"
    #     return 1

    if len(sys.argv) != 3:
        print("Please provide the following arguments:")
        print("path to executables, path to data")
        return 1

    EXE_PATH = sys.argv[1]
    DATA_PATH = sys.argv[2]

    run_suite()

    #NAME_OF_CHIP = sys.argv[4]
    # log_file = sys.argv[3] + "_exec.log"
    # print "recording all to " + log_file
    # log_file_handle = open(log_file, "w")
    # PRINT = lambda x : my_print(log_file_handle,x)

    # data_file = sys.argv[3] + "_data.txt"
    # print "recording data to " + data_file
    # data_file_handle = open(data_file, "w")
    # DATA_PRINT = lambda x : my_print(data_file_handle,x)

    # log_file_handle.close()
    # data_file_handle.close()

# cmd = [os.path.join("pannotia", BUILD_MODE, "pannotia_color")]
# p_obj = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# ret_code = p_obj.wait()
# sout, serr = p_obj.communicate()
# print("STDOUT:")
# print(sout)
# print("STDERR:")
# print(serr)

if __name__ == '__main__':
    sys.exit(main())
