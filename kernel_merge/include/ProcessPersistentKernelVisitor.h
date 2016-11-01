#ifndef KERNEL_MERGE_PROCESSPERSISTENTKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSPERSISTENTKERNELVISITOR_H

#include "ProcessKernelVisitor.h"

using namespace clang;
using namespace llvm;

bool ASTUsesOfferFunctions(ASTUnit * AU);

class ProcessPersistentKernelVisitor
  : public ProcessKernelVisitor<ProcessPersistentKernelVisitor> {
public:
  ProcessPersistentKernelVisitor(ASTUnit * AU) : ProcessKernelVisitor(AU), UsesOfferFunctions(ASTUsesOfferFunctions(AU)) {
    this->RestorationCtx = "";
    this->ForkPointCounter = 0;
    this->VisitedFunctionCallsGlobalBarrierRobustToResizingFunction = false;
    TraverseTranslationUnitDecl(AU->getASTContext().getTranslationUnitDecl());
    if (!GetKI().KernelFunction) {
      errs() << "Persistent kernel file must declare a kernel function.\n";
      exit(1);
    }
  }

  bool VisitCallExpr(CallExpr *CE);

  std::string GetRestorationCtx() {
    return RestorationCtx;
  }

  bool TraverseFunctionDecl(FunctionDecl *D);

  bool UsesGlobalBarrierRobustToResizing();

  virtual void ProcessKernelFunction(FunctionDecl *D);

  virtual void AddArgumentsForIdCalls(FunctionDecl *D, SourceLocation endOfParams);

private:
  const bool UsesOfferFunctions;
  void ProcessWhileStmt(WhileStmt *S);
  std::string ConvertType(QualType type);
  bool CallsGlobalBarrierRobustToResizing(std::string name);

  bool VisitedFunctionCallsGlobalBarrierRobustToResizingFunction;
  std::set<std::string> FunctionsThatCallGlobalBarrierRobustToResizingFunction;

  std::vector<DeclStmt*> DeclsToRestore;
  std::string RestorationCtx;

  unsigned ForkPointCounter;

};

#endif