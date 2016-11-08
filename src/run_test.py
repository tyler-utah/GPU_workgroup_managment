from __future__ import print_function
import sys
import os
import subprocess
import shutil
import glob

def fake_placeholder(x):
    return

EXE_PATH      = ""
DATA_PATH     = ""
STAT_PATH     = ""
ITERS         = 1
PRINT         = fake_placeholder
DATA_PRINT    = fake_placeholder
NAME_OF_CHIP  = ""
NON_PRST_FREQ = ""

PROGRAMS = {
    "pannotia_color",
    "pannotia_mis",
    "pannotia_bc",
    "pannotia_sssp"
}

PROGRAM_DATA = {

    "pannotia_color" : [ { "input" : os.path.join("inputs", "color", "ecology1.graph"),
                           "solution" : os.path.join("solutions", "color_ecology.txt"),
                           "stat" : "color_ecology" },
                         { "input" : os.path.join("inputs", "color", "G3_circuit.graph"),
                           "solution" : os.path.join("solutions", "color_G3_circuit.txt"),
                           "stat" : "color_G3_circuit" }
    ],

    "pannotia_mis" : [ { "input" : os.path.join("inputs", "color", "ecology1.graph"),
                         "solution" : os.path.join("solutions", "color_ecology.txt"),
                         "stat" : "mis_ecology" },
                       { "input" : os.path.join("inputs", "color", "G3_circuit.graph"),
                         "solution" : os.path.join("solutions", "color_G3_circuit.txt"),
                         "stat" : "mis_G3_circuit" } ],

    "pannotia_bc" : [ { "input" : os.path.join("inputs", "bc", "1k_128k.gr"),
                        "solution" : os.path.join("solutions", "bc_1k_128k.txt"),
                        "stat" : "bc_1k_128k" },
                      { "input" : os.path.join("inputs", "bc", "2k_1M.gr"),
                        "solution" : os.path.join("solutions", "bc_2k_1M.txt"),
                        "stat" : "bc_2k_1M" }
    ],

    "pannotia_sssp" : [ { "input" : os.path.join("inputs", "sssp", "USA-road-d.NW.gr"),
                          "solution" : os.path.join("solutions", "sssp_usa.txt"),
                          "stat" : "sssp_usa_road" } ]
}

def my_print(file_handle, data):
    print(data)
    file_handle.write(data + os.linesep)

def exec_cmd(cmd, extr_args=[], flag=""):
    finalcmd = cmd + extr_args
    PRINT("====================== RUN COMMAND =======================")
    PRINT(flag)
    for s in finalcmd:
        PRINT(s)
    PRINT("----------------------------------------------------------")
    p_obj = subprocess.Popen(finalcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret_code = p_obj.wait()
    sout, serr = p_obj.communicate()
    PRINT("* stdout:")
    PRINT(sout.decode())
    PRINT("* stderr:")
    PRINT(serr.decode())
    PRINT("----------------------------------------------------------")
    str_ret_code = "Return code: " + str(ret_code)
    PRINT(str_ret_code)
    PRINT("==========================================================")
    if (ret_code != 0):
        print("Error running " + " ".join(cmd))
        exit(ret_code)

def mv_wildcard(path, dest):
    for f in glob.glob(path):
        shutil.move(f, dest)

def collect_stats(d):
    stat_dir = os.path.join(STAT_PATH, d["stat"])
    try:
        os.makedirs(stat_dir)
    except OSError:
        PRINT("Note: stats dir already exists: ")
        PRINT(stat_dir)
    PRINT("Collect stats of run in directory:")
    PRINT(stat_dir)
    # move all potential stat files to the stats dir
    mv_wildcard("summary_*.txt", stat_dir)
    mv_wildcard("non_persistent_duration_*.txt", stat_dir)
    mv_wildcard("timestamp_executing_groups_*.txt", stat_dir)
    mv_wildcard("timestamp_non_persistent_*.txt", stat_dir)

def run_suite():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            exe = os.path.join(EXE_PATH, p)
            graph_in = os.path.join(DATA_PATH, d["input"])
            graph_sol = os.path.join(DATA_PATH, d["solution"])
            cmd = [exe, "--non_persistent_frequency", NON_PRST_FREQ, "--graph_file", graph_in, "--graph_solution_file", graph_sol, "--threads_per_wg", "128"]
            # standalone, which does not produce stat files, so nothing to collect here
            # exec_cmd(cmd, ["--run_persistent", "2"], "== standalone")
            # merged without tasks, stats to collect afterwards
            # exec_cmd(cmd, ["--skip_tasks", "1", "--merged_iterations", "2"], "== merged without task")
            # collect_stats(d)
            # merged
            exec_cmd(cmd, ["--merged_iterations", "1"], "== merged")
            collect_stats(d)

def main():

    global EXE_PATH
    global DATA_PATH
    global STAT_PATH
    global PRINT
    global DATA_PRINT
    global NAME_OF_CHIP
    global NON_PRST_FREQ

    if len(sys.argv) != 7:
        print("Please provide the following arguments:")
        print("path to executables, path to data, path to result (where to store them), name of run, name of chip, frequency (ms) of non-persistent kernel")
        return 1

    EXE_PATH = sys.argv[1]
    DATA_PATH = sys.argv[2]
    STAT_PATH = sys.argv[3]

    NAME_OF_CHIP = sys.argv[5]
    log_file = sys.argv[4] + ".log"
    print("recording all to " + log_file)
    log_file_handle = open(log_file, "w")
    PRINT = lambda x : my_print(log_file_handle,x)

    NON_PRST_FREQ = sys.argv[6]

    PRINT("Name of chip:")
    PRINT(NAME_OF_CHIP)

    run_suite()

    # data_file = sys.argv[3] + "_data.txt"
    # print "recording data to " + data_file
    # data_file_handle = open(data_file, "w")
    # DATA_PRINT = lambda x : my_print(data_file_handle,x)

    log_file_handle.close()
    # data_file_handle.close()

if __name__ == '__main__':
    sys.exit(main())
