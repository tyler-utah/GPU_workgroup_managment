from __future__ import print_function
import sys
import os
import subprocess
import shutil
import glob
import re
import platform
import time
from threading import Thread
import signal
import pdb

def fake_placeholder(x):
    return

EXE_PATH      = ""
DATA_PATH     = ""
STAT_PATH     = ""
ITERATIONS    = "10"
PRINT         = fake_placeholder
DATA_PRINT    = fake_placeholder
NAME_OF_CHIP  = ""
PLATFORM_ID   = "0"
IS_AMD        = "0"
CHECK_POINT_FILE = "checkpoint.txt"
CHECK_POINT_DATA = []
TIME_BEGIN       = 0.0
EXIT_THREAD      = 0

PROGRAMS = [
#    "lonestar_sssp"
#    "lonestar_bfs",
#    "pannotia_color",
#    "pannotia_mis",
#    "pannotia_bc",
#    "pannotia_sssp",
    "final_octree",
    "connect_four"
]

PROGRAM_DATA = {

    "pannotia_color" : [ { "input" : os.path.join("pannotia","inputs", "color", "ecology1.graph"),
                           "solution" : os.path.join("pannotia","solutions", "color_ecology.txt"),
                           "stat" : "color_ecology",
                           "suite" : "pannotia",
                           "query_barrier" : [0,1]},
                         { "input" : os.path.join("pannotia","inputs", "color", "G3_circuit.graph"),
                           "solution" : os.path.join("pannotia","solutions", "color_G3_circuit.txt"),
                           "stat" : "color_G3_circuit",
                           "suite" : "pannotia",
                           "query_barrier" : [0,1]}
    ],

    "pannotia_mis" : [ { "input" : os.path.join("pannotia","inputs", "color", "ecology1.graph"),
                         "solution" : os.path.join("pannotia","solutions", "mis_ecology.txt"),
                         "stat" : "mis_ecology",
                         "suite" : "pannotia",
                         "query_barrier" : [0,1]},
                       { "input" : os.path.join("pannotia","inputs", "color", "G3_circuit.graph"),
                         "solution" : os.path.join("pannotia","solutions", "mis_G3_circuit.txt"),
                         "stat" : "mis_G3_circuit",
                         "suite" : "pannotia",
                         "query_barrier" : [0,1]}
    ],

    "pannotia_bc" : [ { "input" : os.path.join("pannotia","inputs", "bc", "1k_128k.gr"),
                        "solution" : os.path.join("pannotia","solutions", "bc_1k_128k.txt"),
                        "stat" : "bc_1k_128k",
                        "suite" : "pannotia",
                        "query_barrier" : [0,1]},
                      { "input" : os.path.join("pannotia","inputs", "bc", "2k_1M.gr"),
                        "solution" : os.path.join("pannotia","solutions", "bc_2k_1M.txt"),
                        "stat" : "bc_2k_1M",
                        "suite" : "pannotia",
                        "query_barrier" : [0,1]}
    ],

    "pannotia_sssp" : [ { "input" : os.path.join("pannotia","inputs", "sssp", "USA-road-d.NW.gr"),
                          "solution" : os.path.join("pannotia","solutions", "sssp_usa.txt"),
                          "stat" : "sssp_usa_road",
                          "suite" : "pannotia",
                          "query_barrier" : [0,1]}
    ],

    "lonestar_sssp" : [ { "input" : os.path.join("lonestar", "inputs","rmat22.gr"),
                        "solution" : os.path.join("lonestar","solutions", "sssp_rmat22.txt"),
                        "stat" : "sssp_rmat22",
                        "suite" : "lonestar",
                        "query_barrier" : [0,1]},
                      { "input" : os.path.join( "lonestar", "inputs", "r4-2e23.gr"),
                        "solution" : os.path.join("lonestar","solutions", "sssp_r4-2e23.txt"),
                        "stat" : "sssp_r4-2e23",
                        "suite" : "lonestar",
                        "query_barrier" : [0,1]},
                   #   { "input" : os.path.join( "lonestar", "inputs", "USA-road-d.W.gr"),
                   #     "solution" : os.path.join("lonestar","solutions", "sssp_USA_W.txt"),
                   #     "stat" : "sssp_USA-road-d.W",
                   #     "suite" : "lonestar",
                   #     "query_barrier" : [0,1]}
    ],
    "lonestar_bfs" : [ { "input" : os.path.join( "lonestar", "inputs", "rmat22.gr"),
                        "solution" : os.path.join("lonestar","solutions", "bfs_rmat22.txt"),
                        "stat" : "bfs_rmat22",
                        "suite" : "lonestar",
                        "query_barrier" : [0,1]},
                      { "input" : os.path.join( "lonestar", "inputs", "r4-2e23.gr"),
                        "solution" : os.path.join("lonestar","solutions", "bfs_r4-2e23.txt"),
                        "stat" : "bfs_r4-2e23",
                        "suite" : "lonestar",
                        "query_barrier" : [0,1]},
                      { "input" : os.path.join( "lonestar", "inputs", "USA-road-d.W.gr"),
                        "solution" : os.path.join("lonestar","solutions", "bfs_USA_W.txt"),
                        "stat" : "bfs_USA-road-d.W",
                        "suite" : "lonestar",
                        "query_barrier" : [0,1]}

    ],
    "final_octree" : [ { "stat" : "octree",
                   "suite" : os.path.join("test_suite","final_octree"),
                   "query_barrier" : [0]},
                 ],
    "connect_four" : [ { "stat" : "connect_four",
                   "suite" : os.path.join("test_suite","connect_four"),
                   "query_barrier" : [0]},
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
    { "freq" : "70", "matdim" : "200", "name" : "light" },
    { "freq" : "40", "matdim" : "200", "name" : "medium" },
    { "freq" : "40", "matdim" : "322", "name" : "high" },
]

MATMULT_CONFIG_IRIS = [
    # This is for purple laptop
    { "freq" : "70", "matdim" : "290", "name" : "light" },
    { "freq" : "40", "matdim" : "290", "name" : "medium" },
    { "freq" : "40", "matdim" : "445", "name" : "high" },
]

MATMULT_CONFIG_RADEON_R7 = [
    # This is for carrot
    { "freq" : "70", "matdim" : "277", "name" : "light" },
    { "freq" : "40", "matdim" : "277", "name" : "medium" },
    { "freq" : "40", "matdim" : "431", "name" : "high" },
]

def my_print(file_handle, data):
    print(data)
    file_handle.write(data + os.linesep)

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    global EXIT_THREAD
    EXIT_THREAD = 1
    sys.exit(0)

def write_to_checkpoint(data):
    f = open(CHECK_POINT_FILE, "a")
    f.write(data + "\n")
    f.write(str(time.time() - TIME_BEGIN) + "\n")
    f.close()

def get_check_point_data():
    global CHECK_POINT_DATA
    global TIME_BEGIN
    #pdb.set_trace()
    if (os.path.isfile(CHECK_POINT_FILE)):
        f = open(CHECK_POINT_FILE, "r")
        CHECK_POINT_DATA = f.readlines()
        CHECK_POINT_DATA = [x.replace("\n", "") for x in CHECK_POINT_DATA]
        checked_time = float(CHECK_POINT_DATA[len(CHECK_POINT_DATA) - 1])
        TIME_BEGIN = TIME_BEGIN - checked_time
        f.close()

def const_print_time():
    global EXIT_THREAD
    i = 0
    local_time_begin = time.time()
    while (EXIT_THREAD != 1) and (time.time() - local_time_begin < 1200):
        time.sleep(10)
        print("time update: " + str(i) + " - " + str(time.time() - local_time_begin))
        i = i + 1
    if (EXIT_THREAD == 1):
          return
    print("time has been longer than 20 minutes, probably you should restart!")
    return

def exec_cmd(cmd, prefix="", record_file=""):
    global EXIT_THREAD
    
    cmd = cmd + ["--output_summary", prefix + "_summary"]
    cmd = cmd + ["--output_non_persistent_duration", prefix + "_non_persistent_duration"]
    cmd = cmd + ["--output_timestamp_executing_groups", prefix + "_timestamp_executing_groups"]
    cmd = cmd + ["--output_timestamp_non_persistent", prefix + "_timestamp_non_persistent"]
    cmd = cmd + ["--output_summary2", record_file]

    if prefix in CHECK_POINT_DATA:
        print(prefix + " found in checkpoint")
        return 0;

    #time.sleep(30)
    EXIT_THREAD = 0
    local_time_begin = time.time()
    #thread = Thread(target = const_print_time)
    #thread.start()

    PRINT("====================== RUN START =========================")
    PRINT(prefix)
    for s in cmd:
        PRINT(s)
    PRINT("----------------------------------------------------------")
    p_obj = subprocess.Popen(cmd)
    ret_code = p_obj.wait()
    #sout, serr = p_obj.communicate()
    #PRINT("* stdout:")
    #PRINT(sout.decode())
    #if record_file != "":
    #    f = open(record_file, "w")
    #    f.write(sout.decode())
    #    f.close()
    #PRINT("* stderr:")
    #PRINT(serr.decode())
    local_time_end = time.time()
    PRINT("----------------------------------------------------------")
    str_ret_code = "Return code: " + str(ret_code)
    PRINT(str_ret_code)
    PRINT("time for run: " + str(local_time_end - local_time_begin) + " seconds")
    PRINT("time since start: " + str(local_time_end - TIME_BEGIN) + " seconds")
    PRINT("====================== RUN END ===========================")
    EXIT_THREAD = 1
    #thread.join();
    if (ret_code != 0):
        print("Error running " + " ".join(cmd))
        return ret_code
    
    write_to_checkpoint(prefix)
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

def optional_query(q):
    if q == 1:
        return "_query"
    return ""

def optional_graph(d, graph_in):
    if "input" in d:
        return ["--graph_file", graph_in]
    return []

def optional_solution(d, graph_sol):
    if "solution" in d:
        return ["--graph_solution_file", graph_sol]
    return []

def run_suite():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            for q in d["query_barrier"]:
                exe = os.path.join(EXE_PATH, d["suite"], optional_debug(),  p)
                graph_in = ""
                if "input" in d:
                     graph_in = os.path.join(DATA_PATH, d["input"])
                graph_sol = ""
                if "solution" in d:
                    graph_sol = os.path.join(DATA_PATH, d["solution"])

                # RUN: merged skip tasks
                cmd = [exe]

               
                cmd = cmd + optional_graph(d, graph_in)
                cmd = cmd + optional_solution(d, graph_sol)
                cmd = cmd + ["--threads_per_wg", "128"]
                # indicate very high number of workgroups to finally obtain occupancy
                cmd = cmd + ["--num_wgs", "256"]
                cmd = cmd + ["--skip_tasks", "1"]
                cmd = cmd + ["--merged_iterations", ITERATIONS]
                cmd = cmd + ["--platform_id", PLATFORM_ID]
                cmd = cmd + ["--is_AMD", IS_AMD]
                cmd = cmd + ["--use_query_barrier", str(q)]


                prefix = d["stat"] + "_skiptask" + optional_query(q)
                record_stdout = prefix + "_stdout" + optional_query(q) + ".txt"
                err_code = exec_cmd(cmd, prefix, record_stdout)

                #dummy value for check pointing
                finalsize = 256
                if err_code != 0:
                    continue
                # grab finalsize (min(occupancy, nb of workgroups))
                if (prefix not in CHECK_POINT_DATA):
                    finalsize = extract_finalsize(record_stdout)
                    collect_stats(d, prefix)
                    if finalsize == -1:
                        PRINT("Could not find finalsize after run of merged skiptask")
                        continue
            

                # RUN: standalone
                if (q == 0):
                    cmd = [exe]
                    cmd = cmd + optional_graph(d, graph_in)
                    cmd = cmd + optional_solution(d, graph_sol)

                    cmd = cmd + ["--threads_per_wg", "128"]
                    cmd = cmd + ["--run_persistent", ITERATIONS]
                    cmd = cmd + ["--num_wgs", str(int(finalsize) - 1)]
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
                        cmd = cmd + optional_graph(d, graph_in)
                        cmd = cmd + optional_solution(d, graph_sol)
                        cmd = cmd + ["--threads_per_wg", "128"]
                        cmd = cmd + ["--merged_iterations", ITERATIONS]
                        # indicate very high number of workgroups to finally obtain occupancy
                        cmd = cmd + ["--num_wgs", "256"]
                        cmd = cmd + ["--non_persistent_frequency", c["freq"]]
                        cmd = cmd + ["--platform_id", PLATFORM_ID]
                        cmd = cmd + ["--is_AMD", IS_AMD]
                        cmd = cmd + ["--use_query_barrier", str(q)]
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
                        prefix = d["stat"] + "_" + c["name"] + "_" + npconfig + "_merged" + optional_query(q)
                        record_stdout = prefix + "_stdout.txt"
                        err_code = exec_cmd(cmd, prefix, record_stdout)
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
    global TIME_BEGIN

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
        MATMULT_CONFIG = MATMULT_CONFIG_HD520
    elif NAME_OF_CHIP == "IRIS":
        MATMULT_CONFIG = MATMULT_CONFIG_IRIS
    else:
        print("Cannot find a matmult for your chip! Exiting")
        exit(0)
    PLATFORM_ID = sys.argv[6]
    IS_AMD = sys.argv[7]
    log_file = sys.argv[4] + ".log"
    TIME_BEGIN = time.time()
    get_check_point_data()
    print("recording all to " + log_file)
    log_file_handle = ""
    if (os.path.isfile(log_file)):
        log_file_handle = open(log_file, "a")
        
    log_file_handle = open(log_file, "w")
    PRINT = lambda x : my_print(log_file_handle,x)

    PRINT("Name of chip:")
    PRINT(NAME_OF_CHIP)

    run_suite()

    time_end = time.time()

    PRINT("-------------------------------")
    PRINT("total time: " + str(time_end - TIME_BEGIN) + " seconds")

    log_file_handle.close()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    sys.exit(main())
