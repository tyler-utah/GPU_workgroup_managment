#include <fstream>
#include <sstream>

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"

#include "ProcessStandalonePersistentKernelVisitor.h"

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


  if (AUs.size() != 1) {
    errs() << "Usage: " << argv[0] << " <persistent>.cl\n";
    exit(1);
  }

  ProcessStandalonePersistentKernelVisitor StandalonePersistentVisitor(AUs[0].get());

  std::ofstream Standalone;
  Standalone.open("standalone.cl");

  Standalone << "#include \"../rt_common/cl_types.h\"\n";
  Standalone << "#include \"restoration_ctx.h\"\n";
  Standalone << "#include \"discovery.cl\"\n";
  Standalone << "#include \"kernel_ctx.cl\"\n";
  Standalone << "#include \"cl_scheduler.cl\"\n";
  Standalone << "#include \"iw_barrier.cl\"\n";
  Standalone << "\n";
  StandalonePersistentVisitor.EmitRewrittenText(Standalone);
  Standalone << "//";

  Standalone.close();

  return 0;

}
