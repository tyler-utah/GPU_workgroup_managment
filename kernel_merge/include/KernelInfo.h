#ifndef KERNELINFO_H
#define KERNELINFO_H

#include <string>

namespace clang {
  class FunctionDecl;
}

struct KernelInfo {
  clang::FunctionDecl *KernelFunction;
  std::string OriginalParameterText;

  KernelInfo() {
    this->KernelFunction = 0;
  }

  KernelInfo(KernelInfo &KI) {
    this->KernelFunction = KI.KernelFunction;
    this->OriginalParameterText = KI.OriginalParameterText;
  }

};

#endif