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

  llvm::outs() << "#include \"cl_scheduler.h\"\n";

  std::string MegaKernelParams = "";
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

    MegaKernelParams += KMV.getOriginalKernelParameterText();

    MegaKernelParams += ", kernel_exec_ctx_t " + KMV.getKernelFunctionDecl()->getNameAsString() + "_ctx, ";

    KernelFunctions.push_back(KMV.getKernelFunctionDecl());

    const RewriteBuffer *RewriteBuf =
      TheRewriter.getRewriteBufferFor(ASTUnit->getSourceManager().getMainFileID());
    if (!RewriteBuf) {
      errs() << "Nothing was re-written\n";
      exit(1);
    }
    llvm::outs() << std::string(RewriteBuf->begin(), RewriteBuf->end());
  }

  MegaKernelParams += "discovery_ctx_t discovery_ctx, scheduling_ctx_t scheduling_ctx";

  llvm::outs() << "kernel void mega_kernel("
               << MegaKernelParams << ") {\n";

  llvm::outs() << "  discovery_protocol(&discovery_ctx);\n";
  llvm::outs() << "  if (participating_group_id(&discovery_ctx) == 0) {\n";
  llvm::outs() << "    run_as_scheduler(&discovery_ctx, &scheduling_ctx, &foo_ctx, &bar_ctx);\n";
  llvm::outs() << "  } else {\n";
  llvm::outs() << "    while (true) {\n";
  llvm::outs() << "      task_t current_task = get_task_from_scheduler(&scheduling_ctx, participating_group_id(&discovery_ctx));\n";
  llvm::outs() << "      switch (current_task.TYPE) {\n";
  llvm::outs() << "      case QUIT:\n";
  llvm::outs() << "        return;\n";

  unsigned count = 0;
  for (auto KernelFunction : KernelFunctions) {
    count += 1;
    llvm::outs() << "      case KERNEL_" << count << ":\n";
    llvm::outs() << "        " << KernelFunction->getNameAsString() << "(";
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
    llvm::outs() << "        break;\n";
  }

  llvm::outs() << "      }\n";
  llvm::outs() << "    }\n";
  llvm::outs() << "  }\n";
  llvm::outs() << "}\n";

  return 0;

}
