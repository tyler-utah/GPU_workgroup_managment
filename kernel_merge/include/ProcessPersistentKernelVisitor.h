#ifndef KERNEL_MERGE_PROCESSPERSISTENTKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSPERSISTENTKERNELVISITOR_H

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "KernelInfo.h"

using namespace clang;
using namespace llvm;

class ProcessPersistentKernelVisitor
  : public RecursiveASTVisitor<ProcessPersistentKernelVisitor> {
public:
  ProcessPersistentKernelVisitor(ASTUnit * AU) {
    this->AU = AU;
    this->RW = Rewriter(AU->getSourceManager(),
      AU->getLangOpts());
    this->RestorationCtx = "";
    this->ForkPointCounter = 0;
    TraverseTranslationUnitDecl(AU->getASTContext().getTranslationUnitDecl());
  }

  bool VisitFunctionDecl(FunctionDecl *D);

  bool VisitCallExpr(CallExpr *CE);

  void EmitRewrittenText(std::ostream & out);

  KernelInfo GetKI() {
    return this->KI;
  }

  std::string GetRestorationCtx() {
    return RestorationCtx;
  }

private:

  void ProcessKernelFunction(FunctionDecl *D);
  void ProcessWhileStmt(WhileStmt *S);

  ASTUnit *AU;
  Rewriter RW;
  KernelInfo KI;

  std::vector<DeclStmt*> DeclsToRestore;
  std::string RestorationCtx;

  unsigned ForkPointCounter;

};

#endif