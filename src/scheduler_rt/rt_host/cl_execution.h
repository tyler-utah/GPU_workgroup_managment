#pragma once

#include <CL/cl2.hpp>
#include <map>
#include <string>
#include "ggc_ocl.h"
#include <iostream>
#include <sstream>
#include <fstream>

//From IWOCL tutorial (needs attribution)
#ifndef CL_DEVICE_BOARD_NAME_AMD
#define CL_DEVICE_BOARD_NAME_AMD 0x4038
#endif

class CL_Execution {
 public:
  cl::Device exec_device;
  cl::Context exec_context;
  cl::CommandQueue exec_queue;
  cl::Program exec_program;
  std::map<std::string, cl::Kernel> exec_kernels;
  
  //From IWOCL tutorial (needs attribution)
  static std::string getDeviceName(const cl::Device& device) {
    std::string name;
    cl_device_info info = CL_DEVICE_NAME;
    
    // Special case for AMD
#ifdef CL_DEVICE_BOARD_NAME_AMD
    device.getInfo(CL_DEVICE_VENDOR, &name);
    if (strstr(name.c_str(), "Advanced Micro Devices"))
      info = CL_DEVICE_BOARD_NAME_AMD;
#endif
    
    device.getInfo(info, &name);
    return name;
  }
  
  std::string getExecDeviceName() {
    return getDeviceName(exec_device);
  }
  
  int get_SMs() {
    cl_uint ret;
    cl_device_info info = CL_DEVICE_MAX_COMPUTE_UNITS;
    
    // Maybe special case it for Intel?
    //#ifdef CL_DEVICE_BOARD_NAME_AMD
    //device.getInfo(CL_DEVICE_VENDOR, &name);
    //if (strstr(name.c_str(), "Advanced Micro Devices"))
    //  info = CL_DEVICE_BOARD_NAME_AMD;
    //#endif
    
    exec_device.getInfo(info, &ret);
    return ret;
  }
  
  bool is_Nvidia() {
    std::string buffer;   
    cl_device_info info = CL_DEVICE_VENDOR;
    int err = 0;
    err = exec_device.getInfo(info, &buffer);
    check_ocl(err);
    if (buffer.find("NVIDIA Corporation") == std::string::npos) {
      return false;
    }
    return true;
  }

  bool is_AMD() {
	  std::string buffer;
	  cl_device_info info = CL_DEVICE_VENDOR;
	  int err = 0;
	  err = exec_device.getInfo(info, &buffer);
	  check_ocl(err);
	  if (buffer.find("Advanced Micro Devices") == std::string::npos) {
		  return false;
	  }
	  return true;
  }
  
  bool is_ocl2() {
    std::string buffer;   
    cl_device_info info = CL_DEVICE_VERSION;
    int err = 0;
    err = exec_device.getInfo(info, &buffer);
    check_ocl(err);
    
    if (buffer.find("OpenCL 2.") == std::string::npos) {
      return false;
    }
    return true;
  }
  
  std::string check_atomics() {
    if (is_Nvidia() && (!is_ocl2())) {
      return " -DNVIDIA_ATOMICS ";
    }
    return "";    
  }
  
  std::string check_ocl2x() {
    if (is_ocl2()) {
      return " -cl-std=CL2.0 ";
    }
    return "";
  }
  
  //From the IWOCL tutorial (needs attribution)
  std::string loadProgram(const char* input) {
    std::ifstream stream(input);
    if (!stream.is_open()) {
      std::cout << "Cannot open file: " << input << std::endl;
#if defined(_WIN32) && !defined(__MINGW32__)
      system("pause");
#endif
      exit(1);
    }
    
    return std::string(
		       std::istreambuf_iterator<char>(stream),
		       (std::istreambuf_iterator<char>()));
  }
  
  
  //roughly from IWOCL tutorial (needs attribution)
  int compile_kernel(const char* kernel_file, const char * kernel_include, const char * extra_include = "") {
    int ret = CL_SUCCESS;
    exec_program = cl::Program(exec_context, loadProgram(kernel_file));
    
    std::stringstream options;
    options.setf(std::ios::fixed, std::ios::floatfield);

	
    
    //set compiler options here, example below 
    //options << " -cl-fast-relaxed-math";
    
    //Include the rt_device sources
    options << "-I" << kernel_include << " ";

	if (strcmp(extra_include, "") != 0) {
		options << "-I" << extra_include << " ";
	}

	//Define the int and atomic int type
	options << "-D" << "CL_INT_TYPE=int" << " ";

	options << "-D" << "CL_UCHAR_TYPE=uchar" << " ";

	options << "-D" << "ATOMIC_CL_INT_TYPE=atomic_int" << " ";

	options << "-D" << "MY_CL_GLOBAL=__global" << " ";

	if (is_AMD()) {
	  options << "-D" << "AMD_MEM_ORDERS" << " ";
	}
  
    //Needed so we know to include Nvidia atomics
    options << check_atomics();
    
    //Check to see if we're OpenCL 2.0
    options << check_ocl2x();

	//std::cout << options.str() << std::endl;
    
    //build the program
    ret = exec_program.build(options.str().c_str());

    if (ret != CL_SUCCESS) {
      std::string log = exec_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(exec_device);
      std::cerr << log << std::endl;
    }
    return ret;        
  }
};
