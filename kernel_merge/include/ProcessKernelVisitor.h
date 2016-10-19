#ifndef KERNEL_MERGE_PROCESSKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSKERNELVISITOR_H

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "KernelInfo.h"
#include "LocalStorageDetector.h"

using namespace clang;
using namespace llvm;

template<class T>
class ProcessKernelVisitor
  : public RecursiveASTVisitor<T> {
public:
  ProcessKernelVisitor(ASTUnit * AU) {
    this->AU = AU;
    this->RW = Rewriter(AU->getSourceManager(),
      AU->getLangOpts());
  }

  KernelInfo& GetKI() {
    return this->KI;
  }

  Rewriter& GetRW() {
    return this->RW;
  }

  void EmitRewrittenText(std::ostream & out) {
    const RewriteBuffer *RewriteBuf =
      RW.getRewriteBufferFor(AU->getSourceManager().getMainFileID());
    if (!RewriteBuf) {
      errs() << "Nothing was re-written\n";
      exit(1);
    }
    out << std::string(RewriteBuf->begin(), RewriteBuf->end());
  }

  virtual void ProcessKernelFunction(FunctionDecl *D) = 0;

  void DetectLocalStorage(Stmt *S) {
    LocalStorageDetector LSD(S);
    LocalArrays = LSD.GetLocalArrays();
  }

  std::vector<VarDecl*> GetLocalArrays() {
    return LocalArrays;
  }

  bool VisitFunctionDecl(FunctionDecl *D)
  {

    if (D->hasAttr<OpenCLKernelAttr>() && D->hasBody()) {
      ProcessKernelFunction(D);
    }
    return RecursiveASTVisitor::VisitFunctionDecl(D);
  }

private:

  KernelInfo KI;

protected:
  ASTUnit *AU;
  Rewriter RW;
  std::vector<VarDecl*> LocalArrays;

};

#endif