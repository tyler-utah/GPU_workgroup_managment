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

  llvm::outs() << "#include \"cl_scheduler.h\"\n";
  NonPersistentVisitor.EmitRewrittenText();
  PersistentVisitor.EmitRewrittenText();

  llvm::outs() << "kernel void mega_kernel("
               << NonPersistentVisitor.GetKI().OriginalParameterText << ", "
               << PersistentVisitor.GetKI().OriginalParameterText << ") {\n";

  llvm::outs() << "  discovery_protocol(&discovery_ctx);\n";
  llvm::outs() << "  if (participating_group_id(&discovery_ctx) == 0) {\n";
  llvm::outs() << "    run_as_scheduler(&discovery_ctx, &scheduling_ctx, &foo_ctx, &bar_ctx);\n";
  llvm::outs() << "  } else {\n";
  llvm::outs() << "    while (true) {\n";
  llvm::outs() << "      task_t current_task = get_task_from_scheduler(&scheduling_ctx, participating_group_id(&discovery_ctx));\n";
  llvm::outs() << "      switch (current_task.TYPE) {\n";
  llvm::outs() << "      case QUIT:\n";
  llvm::outs() << "        return;\n";

  llvm::outs() << "      case NON_PERSISTENT_KERNEL:\n";
  FunctionDecl * nonPersistent = NonPersistentVisitor.GetKI().KernelFunction;
  llvm::outs() << "        " << nonPersistent->getNameAsString() << "(";
  for (unsigned i = 0; i < nonPersistent->getNumParams(); ++i) {
    llvm::outs() << nonPersistent->getParamDecl(i)->getNameAsString() << ", ";
  }
  llvm::outs() << "0);\n";
  llvm::outs() << "        break;\n";

  llvm::outs() << "      case PERSISTENT_KERNEL:\n";
  FunctionDecl * persistent = PersistentVisitor.GetKI().KernelFunction;
  llvm::outs() << "        " << persistent->getNameAsString() << "(";
  for (unsigned i = 0; i < persistent->getNumParams(); ++i) {
    llvm::outs() << persistent->getParamDecl(i)->getNameAsString() << ", ";
  }
  llvm::outs() << "0);\n";
  llvm::outs() << "        break;\n";

  llvm::outs() << "      }\n";
  llvm::outs() << "    }\n";
  llvm::outs() << "  }\n";
  llvm::outs() << "}\n";

  return 0;

}
