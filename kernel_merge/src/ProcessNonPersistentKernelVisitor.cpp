#include "ProcessNonPersistentKernelVisitor.h"

bool ProcessNonPersistentKernelVisitor::VisitFunctionDecl(FunctionDecl *D)
{

  if (D->hasAttr<OpenCLKernelAttr>() && D->hasBody()) {
    ProcessKernelFunction(D);
  }
  return RecursiveASTVisitor::VisitFunctionDecl(D);
}

bool ProcessNonPersistentKernelVisitor::VisitCallExpr(CallExpr *CE) {
  auto name = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if(name == "get_global_id" ||
     name == "get_group_id" ||
     name == "get_num_groups" || 
     name == "get_global_size") {
     assert(CE->getNumArgs() == 1);
     // TODO: Abort unless the argument has the literal value 0
     RW.ReplaceText(CE->getArg(0)->getSourceRange(), "__k_ctx");
     RW.InsertTextBefore(CE->getSourceRange().getBegin(), "k_");
  }
  if(name == "global_barrier") {
    errs() << "global_barrier should not appear in non-persistent kernel\n";
    exit(1);
  }
  return true;
}


void ProcessNonPersistentKernelVisitor::EmitRewrittenText() {
  const RewriteBuffer *RewriteBuf =
    RW.getRewriteBufferFor(AU->getSourceManager().getMainFileID());
  if (!RewriteBuf) {
    errs() << "Nothing was re-written\n";
    exit(1);
  }
  llvm::outs() << std::string(RewriteBuf->begin(), RewriteBuf->end());
}

void ProcessNonPersistentKernelVisitor::ProcessKernelFunction(FunctionDecl *D) {
  if (KI.KernelFunction) {
    errs() << "Multiple kernel functions in source file not supported, stopping.\n";
    exit(1);
  }
  KI.KernelFunction = D;

  if (D->getNumParams() == 0) {
    KI.OriginalParameterText = "";
  }
  else {
    KI.OriginalParameterText = RW.getRewrittenText(
      SourceRange(D->getParamDecl(0)->getSourceRange().getBegin(),
        D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd()));
  }

  // Add "__k_ctx" parameter to kernel
  std::string newParam = "";
  if (D->getNumParams() > 0) {
    newParam = ", ";
  }
  newParam += "__global Kernel_ctx * __k_ctx";
  RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), newParam);

  // Remove the "kernel" attribute
  RW.RemoveText(D->getAttr<OpenCLKernelAttr>()->getRange());
}