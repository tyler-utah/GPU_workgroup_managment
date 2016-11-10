from __future__ import print_function
import sys
import os
import subprocess
import shutil
import glob
import re
import platform

def fake_placeholder(x):
    return

EXE_PATH      = ""
DATA_PATH     = ""
STAT_PATH     = ""
ITERATIONS    = "1"
PRINT         = fake_placeholder
DATA_PRINT    = fake_placeholder
NAME_OF_CHIP  = ""
PLATFORM_ID   = "0"
IS_AMD        = "0"

PROGRAMS = {
    "lonestar_sssp"
#    "lonestar_bfs",
#    "pannotia_color",
#    "pannotia_mis",
#    "pannotia_bc",
#    "pannotia_sssp" 
}

PROGRAM_DATA = {

    "pannotia_color" : [ { "input" : os.path.join("pannotia","inputs", "color", "ecology1.graph"),
                           "solution" : os.path.join("pannotia","solutions", "color_ecology.txt"),
                           "stat" : "color_ecology",
                           "suite" : "pannotia"},
                         { "input" : os.path.join("pannotia","inputs", "color", "G3_circuit.graph"),
                           "solution" : os.path.join("pannotia","solutions", "color_G3_circuit.txt"),
                           "stat" : "color_G3_circuit",
                           "suite" : "pannotia"}
    ],

    "pannotia_mis" : [ { "input" : os.path.join("pannotia","inputs", "color", "ecology1.graph"),
                         "solution" : os.path.join("pannotia","solutions", "color_ecology.txt"),
                         "stat" : "mis_ecology",
                         "suite" : "pannotia"},
                       { "input" : os.path.join("pannotia","inputs", "color", "G3_circuit.graph"),
                         "solution" : os.path.join("pannotia","solutions", "color_G3_circuit.txt"),
                         "stat" : "mis_G3_circuit",
                         "suite" : "pannotia"}
    ],

    "pannotia_bc" : [ { "input" : os.path.join("pannotia","inputs", "bc", "1k_128k.gr"),
                        "solution" : os.path.join("pannotia","solutions", "bc_1k_128k.txt"),
                        "stat" : "bc_1k_128k",
                        "suite" : "pannotia"},
                      { "input" : os.path.join("pannotia","inputs", "bc", "2k_1M.gr"),
                        "solution" : os.path.join("pannotia","solutions", "bc_2k_1M.txt"),
                        "stat" : "bc_2k_1M",
                        "suite" : "pannotia"}
    ],

    "pannotia_sssp" : [ { "input" : os.path.join("pannotia","inputs", "sssp", "USA-road-d.NW.gr"),
                          "solution" : os.path.join("pannotia","solutions", "sssp_usa.txt"),
                          "stat" : "sssp_usa_road",
                          "suite" : "pannotia"}
    ],
    
    "lonestar_sssp" : [ { "input" : os.path.join("lonestar", "inputs","rmat22.gr"),
                        "solution" : os.path.join("lonestar","solutions", "sssp_rmat22.txt"),
                        "stat" : "sssp_rmat22",
                        "suite" : "lonestar"},
                      { "input" : os.path.join( "lonestar", "inputs", "r4-2e23.gr"),
                        "solution" : os.path.join("lonestar","solutions", "sssp_r4-2e23.txt"),
                        "stat" : "sssp_r4-2e23",
                        "suite" : "lonestar"},
                   #   { "input" : os.path.join( "lonestar", "inputs", "USA-road-d.W.gr"),
                   #     "solution" : os.path.join("lonestar","solutions", "sssp_USA_W.txt"),
                   #     "stat" : "sssp_USA-road-d.W",
                   #     "suite" : "lonestar"}
    ],
    "lonestar_bfs" : [ { "input" : os.path.join( "lonestar", "inputs", "rmat22.gr"),
                        "solution" : os.path.join("lonestar","solutions", "bfs_rmat22.txt"),
                        "stat" : "bfs_rmat22",
                        "suite" : "lonestar"},
                      { "input" : os.path.join( "lonestar", "inputs", "r4-2e23.gr"),
                        "solution" : os.path.join("lonestar","solutions", "bfs_r4-2e23.txt"),
                        "stat" : "bfs_r4-2e23",
                        "suite" : "lonestar"},
                      { "input" : os.path.join( "lonestar", "inputs", "USA-road-d.W.gr"),
                        "solution" : os.path.join("lonestar","solutions", "bfs_USA_W.txt"),
                        "stat" : "bfs_USA-road-d.W",
                        "suite" : "lonestar"}
                        
    ],
}

MATMULT_CONFIG = []

MATMULT_CONFIG_TEMPLATE = [
    # This is for Hugues laptop
    { "freq" : "70", "matdim" : "160", "name" : "light" },
    { "freq" : "40", "matdim" : "160", "name" : "medium" },
    { "freq" : "40", "matdim" : "260", "name" : "high" },
]

MATMULT_CONFIG_HD5500 = [
    # This is for Tylers laptop
    { "freq" : "70", "matdim" : "250", "name" : "light" }, #light is 3 ms
    { "freq" : "40", "matdim" : "250", "name" : "medium" }, #medium is 3ms
    { "freq" : "40", "matdim" : "375", "name" : "high" },   #heavy is 10ms
]

MATMULT_CONFIG_HD520 = [
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
        return ret_code
	return 0

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

def optional_debug():
    if platform.system() == "Windows":
        return "Debug"
    return ""
    

def run_suite():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            exe = os.path.join(EXE_PATH, d["suite"], optional_debug(),  p)
            graph_in = os.path.join(DATA_PATH, d["input"])
            graph_sol = os.path.join(DATA_PATH, d["solution"])

            # RUN: merged skip tasks
            cmd = [exe]
            cmd = cmd + ["--graph_file", graph_in]
            cmd = cmd + ["--graph_solution_file", graph_sol]
            cmd = cmd + ["--threads_per_wg", "128"]
            # indicate very high number of workgroups to finally obtain occupancy
            cmd = cmd + ["--num_wgs", "256"]
            cmd = cmd + ["--skip_tasks", "1"]
            cmd = cmd + ["--merged_iterations", ITERATIONS]
            cmd = cmd + ["--platform_id", PLATFORM_ID]
            cmd = cmd + ["--is_AMD", IS_AMD]

            prefix = d["stat"] + "_skiptask"
            record_stdout = prefix + "_stdout.txt"
            err_code = exec_cmd(cmd, prefix, record_stdout)
            if err_code != 0:
                continue
            # grab finalsize (min(occupancy, nb of workgroups))
            finalsize = extract_finalsize(record_stdout)
            collect_stats(d, prefix)
            if finalsize == -1:
                PRINT("Could not find finalsize after run of merged skiptask")
                continue

            # RUN: standalone
            cmd = [exe]
            cmd = cmd + ["--graph_file", graph_in]
            cmd = cmd + ["--graph_solution_file", graph_sol]
            cmd = cmd + ["--num_wgs", "256"]

            cmd = cmd + ["--threads_per_wg", "128"]
            cmd = cmd + ["--run_persistent", ITERATIONS]
            cmd = cmd + ["--num_wgs", finalsize]
            cmd = cmd + ["--platform_id", PLATFORM_ID]
            cmd = cmd + ["--is_AMD", IS_AMD]
            prefix = d["stat"] + "_standalone"
            record_stdout = prefix + "_stdout.txt"
            err_code = exec_cmd(cmd, prefix, record_stdout)
            if err_code != 0:
                continue
            collect_stats(d, prefix)

            for c in MATMULT_CONFIG:
                for npconfig in ["npwg_one", "npwg_all_but_one", "npwg_half", "npwg_quarter"]:
                    cmd = [exe]
                    cmd = cmd + ["--graph_file", graph_in]
                    cmd = cmd + ["--graph_solution_file", graph_sol]
                    cmd = cmd + ["--threads_per_wg", "128"]
                    cmd = cmd + ["--merged_iterations", ITERATIONS]
                    # indicate very high number of workgroups to finally obtain occupancy
                    cmd = cmd + ["--num_wgs", "256"]
                    cmd = cmd + ["--non_persistent_frequency", c["freq"]]
                    cmd = cmd + ["--platform_id", PLATFORM_ID]
                    cmd = cmd + ["--is_AMD", IS_AMD]
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
                    err_code = exec_cmd(cmd, prefix)
                    if err_code != 0:
                        continue
                    collect_stats(d, prefix)

def main():

    global EXE_PATH
    global DATA_PATH
    global STAT_PATH
    global PRINT
    global ITERATIONS
    global DATA_PRINT
    global NAME_OF_CHIP
    global MATMULT_CONFIG
    global PLATFORM_ID
    global IS_AMD

    if len(sys.argv) != 8:
        print("Please provide the following arguments:")
        print("path to build, path to src, path to result (where to store them), name of run, name of chip, platform_id, is_AMD")
        return 1

    EXE_PATH = sys.argv[1]
    DATA_PATH = sys.argv[2]
    STAT_PATH = sys.argv[3]

    NAME_OF_CHIP = sys.argv[5]
    if NAME_OF_CHIP == "HD5500":
        MATMULT_CONFIG = MATMULT_CONFIG_HD5500
    elif NAME_OF_CHIP == "HD520":
	    MATMULT_CONFIG = MATMULT_CONFIG_HD5500
	else:
	    print("Cannot find a matmult for your chip! Exiting")
		exit(0)
    PLATFORM_ID = sys.argv[6]
    IS_AMD = sys.argv[7]
    log_file = sys.argv[4] + ".log"
    print("recording all to " + log_file)
    log_file_handle = open(log_file, "w")
    PRINT = lambda x : my_print(log_file_handle,x)

    PRINT("Name of chip:")
    PRINT(NAME_OF_CHIP)

    run_suite()

    log_file_handle.close()

if __name__ == '__main__':
    sys.exit(main())
