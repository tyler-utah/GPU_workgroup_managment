#ifndef KERNEL_MERGE_PROCESSKERNELVISITOR_H
#define KERNEL_MERGE_PROCESSKERNELVISITOR_H

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Lex/Lexer.h"

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
    this->VisitedFunctionCallsIdFunction = false;
    this->RW = Rewriter(AU->getSourceManager(),
      AU->getLangOpts());
  }

  virtual ~ProcessKernelVisitor() { }

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

  virtual void AddArgumentsForIdCalls(FunctionDecl *D, SourceLocation StartOfParams) = 0;

  void DetectLocalStorage(Stmt *S) {
    LocalStorageDetector LSD(S);
    LocalArrays = LSD.GetLocalArrays();
  }

  std::vector<VarDecl*> GetLocalArrays() {
    return LocalArrays;
  }

  bool TraverseFunctionDecl(FunctionDecl *D)
  {

    if (D->hasAttr<OpenCLKernelAttr>() && D->hasBody()) {
      ProcessKernelFunction(D);
    }
    assert(!VisitedFunctionCallsIdFunction);
    bool result = RecursiveASTVisitor::TraverseFunctionDecl(D);
    if (!D->hasAttr<OpenCLKernelAttr>() && VisitedFunctionCallsIdFunction) {
      FunctionsThatCallIdFunctions.insert(D->getNameAsString());
      SourceLocation StartOfParams = Lexer::findLocationAfterToken(D->getLocation(),
        tok::l_paren,
        AU->getSourceManager(),
        AU->getLangOpts(),
        /*SkipTrailingWhitespaceAndNewLine=*/true);
      this->AddArgumentsForIdCalls(D, StartOfParams);
    }
    VisitedFunctionCallsIdFunction = false;
    return result;
  }

private:

  KernelInfo KI;

protected:
  bool VisitedFunctionCallsIdFunction;
  std::set<std::string> FunctionsThatCallIdFunctions;
  ASTUnit *AU;
  Rewriter RW;
  std::vector<VarDecl*> LocalArrays;

};

#endif