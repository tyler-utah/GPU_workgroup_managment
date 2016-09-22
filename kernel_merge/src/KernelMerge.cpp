#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"

#include "ProcessKernelVisitor.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

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

  std::string MergedKernelParams = "";
  std::vector<FunctionDecl*> KernelFunctions;

  for(unsigned i = 0; i < AUs.size(); ++i) {
    ASTUnit *ASTUnit = AUs[i].get();

    TranslationUnitDecl *TheTU =
      ASTUnit->getASTContext().getTranslationUnitDecl();

    Rewriter TheRewriter(ASTUnit->getSourceManager(),
      ASTUnit->getLangOpts());

    ProcessKernelVisitor KMV(TheTU, TheRewriter, ASTUnit->getASTContext());

    if(!KMV.processedKernel()) {
      errs() << "Did not find a suitable kernel function in source file, stopping.\n";
      exit(1);
    }

    if (i > 0) {
      MergedKernelParams += ", ";
    }
    MergedKernelParams += KMV.getOriginalKernelParameterText();

    KernelFunctions.push_back(KMV.getKernelFunctionDecl());

    const RewriteBuffer *RewriteBuf =
      TheRewriter.getRewriteBufferFor(ASTUnit->getSourceManager().getMainFileID());
    if (!RewriteBuf) {
      errs() << "Nothing was re-written\n";
    } else {
      llvm::outs() << std::string(RewriteBuf->begin(), RewriteBuf->end());
    }
  }

  llvm::outs() << "kernel void mega_kernel("
               << MergedKernelParams << ") {\n";

  for (auto KernelFunction : KernelFunctions) {
    llvm::outs() << "  " << KernelFunction->getNameAsString() << "(";
    bool AtLeastOneParameter = false;
    for (unsigned i = 0; i < KernelFunction->getNumParams(); ++i) {
      if (AtLeastOneParameter) {
        llvm::outs() << ", ";
      }
      AtLeastOneParameter = true;
      llvm::outs() << KernelFunction->getParamDecl(i)->getNameAsString();
    }
    if (AtLeastOneParameter) {
      llvm::outs() << ", ";
    }
    llvm::outs() << "0";
    llvm::outs() << ");\n";
  }

  llvm::outs() << "}\n";

  return 0;

}
