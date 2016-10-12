#ifndef KERNEL_MERGE_PROCESSNONPERSISTENTKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSNONPERSISTENTKERNELVISITOR_H

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "KernelInfo.h"

using namespace clang;
using namespace llvm;

class ProcessNonPersistentKernelVisitor
  : public RecursiveASTVisitor<ProcessNonPersistentKernelVisitor> {
public:
  ProcessNonPersistentKernelVisitor(ASTUnit * AU) {
    this->AU = AU;
    this->RW = Rewriter(AU->getSourceManager(),
      AU->getLangOpts());
    TraverseTranslationUnitDecl(AU->getASTContext().getTranslationUnitDecl());
  }

  bool VisitFunctionDecl(FunctionDecl *D);

  bool VisitCallExpr(CallExpr *CE);

  void EmitRewrittenText(std::ostream & out);

  KernelInfo GetKI() {
    return this->KI;
  }

private:

  void ProcessKernelFunction(FunctionDecl * D);

  ASTUnit *AU;
  Rewriter RW;
  KernelInfo KI;

};

#endif