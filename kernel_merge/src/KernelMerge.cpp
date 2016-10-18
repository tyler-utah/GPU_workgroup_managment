#include <fstream>

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"

#include "ProcessNonPersistentKernelVisitor.h"
#include "ProcessPersistentKernelVisitor.h"

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


  if (AUs.size() != 2) {
    errs() << "Usage: " << argv[0] << " <non_persistent>.cl <persistent>.cl\n";
    exit(1);
  }

  ProcessNonPersistentKernelVisitor NonPersistentVisitor(AUs[0].get());
  ProcessPersistentKernelVisitor PersistentVisitor(AUs[1].get());

  std::ofstream RestorationContextH;
  RestorationContextH.open("restoration_ctx.h");
  RestorationContextH << "#pragma once\n";
  RestorationContextH << PersistentVisitor.GetRestorationCtx();
  RestorationContextH.close();

  std::ofstream Merged;
  Merged.open("merged.cl");

  Merged << "#include \"discovery.cl\"\n";
  Merged << "#include \"kernel_ctx.cl\"\n";
  Merged << "#include \"cl_scheduler.cl\"\n";
  Merged << "#include \"iw_barrier.cl\"\n";
  Merged << "\n";
  NonPersistentVisitor.EmitRewrittenText(Merged);
  Merged << "\n";
  PersistentVisitor.EmitRewrittenText(Merged);
  Merged << "\n";
  Merged << "\n";
  Merged << "kernel void mega_kernel("
               << NonPersistentVisitor.GetKI().OriginalParameterText << ", "
               << PersistentVisitor.GetKI().OriginalParameterText << ", "
               << "__global IW_barrier * __bar, "
               << "__global Discovery_ctx * d_ctx, "
               << "__global Kernel_ctx * non_persistent_kernel_ctx, "
               << "__global Kernel_ctx * persistent_kernel_ctx, "
               << "SCHEDULER_ARGS) {\n";

  Merged << "  #define NON_PERSISTENT_KERNEL "
         << NonPersistentVisitor.GetKI().KernelFunction->getNameAsString()
         << "(";
  for (auto param : NonPersistentVisitor.GetKI().KernelFunction->parameters()) {
    Merged << param->getNameAsString() << ", ";
  }
  Merged << "non_persistent_kernel_ctx)\n";

  Merged << "  #define PERSISTENT_KERNEL "
         << PersistentVisitor.GetKI().KernelFunction->getNameAsString()
         << "(";
  for (auto param : PersistentVisitor.GetKI().KernelFunction->parameters()) {
    Merged << param->getNameAsString() << ", ";
  }
  Merged << "__bar, persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local)\n";

  Merged << "  #include \"main_device_body.cl\"\n";
  Merged << "}\n";

  Merged.close();

  return 0;

}
