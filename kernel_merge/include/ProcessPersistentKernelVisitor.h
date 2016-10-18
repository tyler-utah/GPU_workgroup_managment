#ifndef KERNEL_MERGE_PROCESSPERSISTENTKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSPERSISTENTKERNELVISITOR_H

#include "ProcessKernelVisitor.h"

using namespace clang;
using namespace llvm;

class ProcessPersistentKernelVisitor
  : public ProcessKernelVisitor<ProcessPersistentKernelVisitor> {
public:
  ProcessPersistentKernelVisitor(ASTUnit * AU) : ProcessKernelVisitor(AU) {
    this->RestorationCtx = "";
    this->ForkPointCounter = 0;
    TraverseTranslationUnitDecl(AU->getASTContext().getTranslationUnitDecl());
    if (!GetKI().KernelFunction) {
      errs() << "Persistent kernel file must declare a kernel function.\n";
      exit(1);
    }
  }

  bool VisitFunctionDecl(FunctionDecl *D);

  bool VisitCallExpr(CallExpr *CE);

  std::string GetRestorationCtx() {
    return RestorationCtx;
  }

private:

  void ProcessKernelFunction(FunctionDecl *D);
  void ProcessWhileStmt(WhileStmt *S);

  std::vector<DeclStmt*> DeclsToRestore;
  std::string RestorationCtx;

  unsigned ForkPointCounter;

};

#endif