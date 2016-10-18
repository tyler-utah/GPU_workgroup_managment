#ifndef KERNEL_MERGE_PROCESSKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSKERNELVISITOR_H

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "KernelInfo.h"

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

  void EmitRewrittenText(std::ostream & out) {
    const RewriteBuffer *RewriteBuf =
      RW.getRewriteBufferFor(AU->getSourceManager().getMainFileID());
    if (!RewriteBuf) {
      errs() << "Nothing was re-written\n";
      exit(1);
    }
    out << std::string(RewriteBuf->begin(), RewriteBuf->end());
  }

private:

  ASTUnit *AU;
  KernelInfo KI;

protected:
  Rewriter RW;

};

#endif