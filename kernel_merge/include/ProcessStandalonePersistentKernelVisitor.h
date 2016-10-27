#ifndef KERNEL_MERGE_PROCESSSTANDALONEPERSISTENTKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSSTANDALONEPERSISTENTKERNELVISITOR_H

#include "ProcessKernelVisitor.h"

using namespace clang;
using namespace llvm;

class ProcessStandalonePersistentKernelVisitor
  : public ProcessKernelVisitor<ProcessStandalonePersistentKernelVisitor> {
public:
  ProcessStandalonePersistentKernelVisitor(ASTUnit * AU) : ProcessKernelVisitor(AU) {
    TraverseTranslationUnitDecl(AU->getASTContext().getTranslationUnitDecl());
    if (!GetKI().KernelFunction) {
      errs() << "Persistent kernel file must declare a kernel function.\n";
      exit(1);
    }
  }

  bool VisitCallExpr(CallExpr *CE);

  virtual void ProcessKernelFunction(FunctionDecl *D);

  virtual void AddArgumentsForIdCalls(FunctionDecl *D, SourceLocation StartOfParams);

};

#endif