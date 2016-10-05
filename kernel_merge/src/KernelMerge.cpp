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

  llvm::outs() << "#include \"discovery.cl\"\n";
  llvm::outs() << "#include \"kernel_ctx.cl\"\n";
  llvm::outs() << "#include \"cl_scheduler.cl\"\n";
  llvm::outs() << "#include \"iw_barrier.cl\"\n";
  llvm::outs() << "\n";
  NonPersistentVisitor.EmitRewrittenText();
  llvm::outs() << "\n";
  PersistentVisitor.EmitRewrittenText();
  llvm::outs() << "\n";
  llvm::outs() << "\n";
  llvm::outs() << "kernel void mega_kernel("
               << NonPersistentVisitor.GetKI().OriginalParameterText << ", "
               << PersistentVisitor.GetKI().OriginalParameterText << ", "
               << "__global IW_barrier * __bar, "
               << "__global Discovery_ctx * d_ctx, "
               << "__global Kernel_ctx * non_persistent_kernel_ctx, "
               << "__global Kernel_ctx * persistent_kernel_ctx, "
               << "SCHEDULER_ARGS) {\n";

  llvm::outs() << "  __local int __scratch;\n";

  llvm::outs() << "  DISCOVERY_PROTOCOL(d_ctx, &__scratch);\n";
  llvm::outs() << "\n";
  llvm::outs() << "  // Scheduler init (makes a variable named s_ctx)\n";
  llvm::outs() << "  INIT_SCHEDULER;\n";
  llvm::outs() << "\n";
  llvm::outs() << "  int group_id = p_get_group_id(d_ctx);\n";
  llvm::outs() << "\n";
  llvm::outs() << "  // Scheduler workgroup\n";
  llvm::outs() << "  if (group_id == 0) {\n";
  llvm::outs() << "    if (get_local_id(0) == 0) {\n";
  llvm::outs() << "\n";
  llvm::outs() << "      // Do any initialisation here before the main loop.\n";
  llvm::outs() << "      scheduler_init(s_ctx, d_ctx, non_persistent_kernel_ctx, persistent_kernel_ctx);\n";
  llvm::outs() << "\n";
  llvm::outs() << "      // Loops forever waiting for signals from the host. Host can issue a quit signal though.\n";
  llvm::outs() << "      scheduler_loop(s_ctx, d_ctx, non_persistent_kernel_ctx, persistent_kernel_ctx);\n";
  llvm::outs() << "\n";
  llvm::outs() << "    }\n";
  llvm::outs() << "    BARRIER;\n";
  llvm::outs() << "    return;\n";
  llvm::outs() << "  }\n";
  llvm::outs() << "\n";
  llvm::outs() << "  // All other workgroups\n";
  llvm::outs() << "  while (true) {\n";
  llvm::outs() << "    // Workgroups are initially available\n";
  llvm::outs() << "    if (get_local_id(0) == 0) {\n";
  llvm::outs() << "      atomic_fetch_add(s_ctx.available_workgroups, 1);\n";
  llvm::outs() << "    }\n";
  llvm::outs() << "\n";
  llvm::outs() << "    // This is synchronous\n";
  llvm::outs() << "    int task = get_task(s_ctx, group_id, &__scratch);\n";
  llvm::outs() << "\n";
  llvm::outs() << "\n";
  llvm::outs() << "    if (task == TASK_QUIT) {\n";
  llvm::outs() << "      break;\n";
  llvm::outs() << "    }\n";
  llvm::outs() << "    if (task == TASK_MULT) {\n";
  llvm::outs() << "      " << NonPersistentVisitor.GetKI().KernelFunction->getNameAsString() << "(";
  for (auto param : NonPersistentVisitor.GetKI().KernelFunction->parameters()) {
    llvm::outs() << param->getNameAsString() << ", ";
  }
  llvm::outs() << "non_persistent_kernel_ctx);\n";
  llvm::outs() << "      BARRIER;\n";
  llvm::outs() << "      if (get_local_id(0) == 0) {\n";
  llvm::outs() << "        atomic_fetch_add(&(non_persistent_kernel_ctx->completed), 1);\n";
  llvm::outs() << "        atomic_store_explicit(&(s_ctx.task_array[group_id]), TASK_WAIT, memory_order_relaxed, memory_scope_device);\n";
  llvm::outs() << "      }\n";
  llvm::outs() << "    }\n";
  llvm::outs() << "    if (task == TASK_PERSIST) {\n";
  llvm::outs() << "      " << PersistentVisitor.GetKI().KernelFunction->getNameAsString() << "(";
  for (auto param : PersistentVisitor.GetKI().KernelFunction->parameters()) {
    llvm::outs() << param->getNameAsString() << ", ";
  }
  llvm::outs() << "/* TODO: restoration context */ 0, __bar, persistent_kernel_ctx);\n";
  llvm::outs() << "      BARRIER;\n";
  llvm::outs() << "      if (get_local_id(0) == 0) {\n";
  llvm::outs() << "        atomic_fetch_add(&(persistent_kernel_ctx->completed), 1);\n";
  llvm::outs() << "        atomic_store_explicit(&(s_ctx.task_array[group_id]), TASK_WAIT, memory_order_relaxed, memory_scope_device);\n";
  llvm::outs() << "      }\n";
  llvm::outs() << "    }\n";
  llvm::outs() << "  }\n";
  llvm::outs() << "}\n";

  return 0;

}
