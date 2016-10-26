#ifndef KERNEL_MERGE_PROCESSNONPERSISTENTKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSNONPERSISTENTKERNELVISITOR_H

#include "ProcessKernelVisitor.h"

using namespace clang;
using namespace llvm;

class ProcessNonPersistentKernelVisitor
  : public ProcessKernelVisitor<ProcessNonPersistentKernelVisitor> {
public:
  ProcessNonPersistentKernelVisitor(ASTUnit * AU) : ProcessKernelVisitor(AU) {
    TraverseTranslationUnitDecl(AU->getASTContext().getTranslationUnitDecl());
    if (!GetKI().KernelFunction) {
      errs() << "Non-persistent kernel file must declare a kernel function.\n";
      exit(1);
    }
  }

  bool VisitCallExpr(CallExpr *CE);

  virtual void ProcessKernelFunction(FunctionDecl *D);

  virtual void AddArgumentsForIdCalls(FunctionDecl *D, SourceLocation StartOfParams);

};

#endif