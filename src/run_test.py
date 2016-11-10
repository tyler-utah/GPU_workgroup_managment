from __future__ import print_function
import sys
import os
import subprocess
import shutil
import glob
import re

def fake_placeholder(x):
    return

EXE_PATH      = ""
DATA_PATH     = ""
STAT_PATH     = ""
ITERS         = 1
PRINT         = fake_placeholder
DATA_PRINT    = fake_placeholder
NAME_OF_CHIP  = ""

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
                         "stat" : "mis_G3_circuit" }
    ],

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

# TODO: make it per machine
MATMULT_CONFIG = [
    # This is for Hugues laptop
    { "freq" : "70", "matdim" : "160", "name" : "light" },
    { "freq" : "40", "matdim" : "160", "name" : "medium" },
    { "freq" : "40", "matdim" : "260", "name" : "high" },
]

def my_print(file_handle, data):
    print(data)
    file_handle.write(data + os.linesep)

def exec_cmd(cmd, prefix="", record_file=""):
    cmd = cmd + ["--output_summary", prefix + "_summary"]
    cmd = cmd + ["--output_non_persistent_duration", prefix + "_non_persistent_duration"]
    cmd = cmd + ["--output_timestamp_executing_groups", prefix + "_timestamp_executing_groups"]
    cmd = cmd + ["--output_timestamp_non_persistent", prefix + "_timestamp_non_persistent"]
    PRINT("====================== RUN START =========================")
    PRINT(prefix)
    for s in cmd:
        PRINT(s)
    PRINT("----------------------------------------------------------")
    p_obj = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret_code = p_obj.wait()
    sout, serr = p_obj.communicate()
    PRINT("* stdout:")
    PRINT(sout.decode())
    if record_file != "":
        f = open(record_file, "w")
        f.write(sout.decode())
        f.close()
    PRINT("* stderr:")
    PRINT(serr.decode())
    PRINT("----------------------------------------------------------")
    str_ret_code = "Return code: " + str(ret_code)
    PRINT(str_ret_code)
    PRINT("====================== RUN END ===========================")
    if (ret_code != 0):
        print("Error running " + " ".join(cmd))
        exit(ret_code)

def mv_wildcard(path, dest):
    for f in glob.glob(path):
        shutil.move(f, dest)

def collect_stats(d, prefix):
    stat_dir = os.path.join(STAT_PATH, d["stat"])
    if not os.path.isdir(stat_dir):
        os.makedirs(stat_dir)
    PRINT("Collect stats for run: " + prefix + " in " + stat_dir)
    # move all potential stat files to the stats dir
    mv_wildcard(prefix + "*", stat_dir)

def extract_finalsize(filename):
    finalsize = -1
    re_finalsize = re.compile(', [0-9]+ occupancy, [0-9]+ final size')
    with open(filename) as f:
        for line in f:
            match = re_finalsize.search(line)
            if match != None:
                finalsize = match.group().split()[3]
                break
    return finalsize

def run_suite():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            exe = os.path.join(EXE_PATH, p)
            graph_in = os.path.join(DATA_PATH, d["input"])
            graph_sol = os.path.join(DATA_PATH, d["solution"])

            # RUN: merged skip tasks
            cmd = [exe]
            cmd = cmd + ["--graph_file", graph_in]
            cmd = cmd + ["--graph_solution_file", graph_sol]
            cmd = cmd + ["--threads_per_wg", "128"]
            # indicate very high number of workgroups to finally obtain occupancy
            cmd = cmd + ["--num_wgs", "1000"]
            cmd = cmd + ["--skip_tasks", "1"]
            prefix = d["stat"] + "_skiptask"
            record_stdout = prefix + "_stdout.txt"
            exec_cmd(cmd, prefix, record_stdout)
            # grab finalsize (min(occupancy, nb of workgroups))
            finalsize = extract_finalsize(record_stdout)
            collect_stats(d, prefix)
            if finalsize == -1:
                PRINT("Could not find finalsize after run of merged skiptask")
                exit(-1)

            # RUN: standalone
            cmd = [exe]
            cmd = cmd + ["--graph_file", graph_in]
            cmd = cmd + ["--graph_solution_file", graph_sol]
            cmd = cmd + ["--threads_per_wg", "128"]
            cmd = cmd + ["--run_persistent", "1"]
            cmd = cmd + ["--num_wgs", finalsize]
            prefix = d["stat"] + "_standalone"
            record_stdout = prefix + "_stdout.txt"
            exec_cmd(cmd, prefix, record_stdout)
            collect_stats(d, prefix)

            for c in MATMULT_CONFIG:
                for npconfig in ["npwg_one", "npwg_all_but_one", "npwg_half", "npwg_quarter"]:
                    cmd = [exe]
                    cmd = cmd + ["--graph_file", graph_in]
                    cmd = cmd + ["--graph_solution_file", graph_sol]
                    cmd = cmd + ["--threads_per_wg", "128"]
                    # indicate very high number of workgroups to finally obtain occupancy
                    cmd = cmd + ["--num_wgs", "1000"]
                    cmd = cmd + ["--non_persistent_frequency", c["freq"]]
                    npwg = "0"
                    if npconfig == "npwg_one":
                        npwg = "-2"
                    elif npconfig == "npwg_all_but_one":
                        npwg = "-1"
                    elif npconfig == "npwg_half":
                        npwg = "2"
                    elif npconfig == "npwg_quarter":
                        npwg = "4"
                    cmd = cmd + ["--non_persistent_wgs", npwg]
                    cmd = cmd + ["--matdim", c["matdim"]]
                    prefix = d["stat"] + "_" + c["name"] + "_" + npconfig + "_merged"
                    exec_cmd(cmd, prefix)
                    collect_stats(d, prefix)

def main():

    global EXE_PATH
    global DATA_PATH
    global STAT_PATH
    global PRINT
    global DATA_PRINT
    global NAME_OF_CHIP
    global MATMULT_CONFIG

    if len(sys.argv) != 6:
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
