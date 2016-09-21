#include <sstream>

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;



class KernelMergeVisitor
  : public RecursiveASTVisitor<KernelMergeVisitor> {
public:
  KernelMergeVisitor(TranslationUnitDecl *TU, Rewriter &RW) : RW(RW) {
    TraverseTranslationUnitDecl(TU);
  }

  bool VisitFunctionDecl(FunctionDecl *D);

private:
  Rewriter &RW;

  void ProcessWhileStmt(WhileStmt *S);
};

bool KernelMergeVisitor::VisitFunctionDecl(FunctionDecl *D)
{
  if (!D->hasBody()) {
    return true;
  }
  CompoundStmt *CS = dyn_cast<CompoundStmt>(D->getBody());
  if (!CS) {
    return true;
  }
  if (CS->size() == 0) {
    return true;
  }
  WhileStmt *WS = dyn_cast<WhileStmt>(*(CS->body_begin()));
  if (!WS) {
    return true;
  }
  ProcessWhileStmt(WS);
  std::stringstream strstr;
  if (D->getNumParams() > 0) {
    strstr << ", ";
  }
  strstr << "__target";
  RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), 
    strstr.str());
}

void KernelMergeVisitor::ProcessWhileStmt(WhileStmt *S) {

  CompoundStmt *CS = dyn_cast<CompoundStmt>(S->getBody());
  if (!CS) {
    return;
  }

  auto condition = RW.getRewrittenText(S->getCond()->getSourceRange());
  RW.ReplaceText(S->getCond()->getSourceRange(), "true");
  RW.InsertTextAfterToken(CS->getLBracLoc(), "\nswitch(__target) {\ncase 0: if(!(" + condition + ") { break; }\n");
  unsigned counter = 1;

  for (auto s : CS->body()) {
    CallExpr *CE = dyn_cast<CallExpr>(s);
    if (!CE) {
      continue;
    }
    if ("global_barrier" == CE->getCalleeDecl()->getAsFunction()->getNameAsString()) {
      std::stringstream strstr;
      strstr << "\ncase " << counter << ":\n";
      RW.InsertTextBefore(CE->getSourceRange().getBegin(), strstr.str());
      counter++;
    }

  }

  RW.InsertTextBefore(CS->getRBracLoc(), "}");
}


static cl::OptionCategory TheTool("");

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, TheTool);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter consumer(llvm::errs(), &*DiagOpts);
  Tool.setDiagnosticConsumer(&consumer);

  std::vector<std::unique_ptr<ASTUnit>> AUs;
  Tool.buildASTs(AUs);

  if (consumer.getNumErrors() > 0) {
    return 1;
  }

  assert(AUs.size() == 1);
  ASTUnit *TheASTUnit = AUs[0].get();

  TranslationUnitDecl *TheTU =
    TheASTUnit->getASTContext().getTranslationUnitDecl();

  Rewriter TheRewriter(TheASTUnit->getSourceManager(),
    TheASTUnit->getLangOpts());

  KernelMergeVisitor KMV(TheTU, TheRewriter);

  const RewriteBuffer *RewriteBuf =
    TheRewriter.getRewriteBufferFor(TheASTUnit->getSourceManager().getMainFileID());
  llvm::outs() << std::string(RewriteBuf->begin(), RewriteBuf->end());

  return 0;

}
