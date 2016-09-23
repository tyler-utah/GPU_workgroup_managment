#ifndef KERNEL_MERGE_PROCESSKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSKERNELVISITOR_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace llvm;

class ProcessKernelVisitor
  : public RecursiveASTVisitor<ProcessKernelVisitor> {
public:
  ProcessKernelVisitor(TranslationUnitDecl *TU, Rewriter &RW, ASTContext &ASTC) : RW(RW), ASTC(ASTC), KernelFunction(0) {
    TraverseTranslationUnitDecl(TU);
  }

  bool VisitFunctionDecl(FunctionDecl *D);

  bool processedKernel();

  std::string getOriginalKernelParameterText();

  FunctionDecl *getKernelFunctionDecl();

private:
  Rewriter &RW;
  ASTContext &ASTC;
  FunctionDecl *KernelFunction;
  std::string OriginalParameterText;
  std::vector<DeclStmt*> DeclsToRestore;

  void ProcessWhileStmt(WhileStmt *S);
};

#endif