#include "ProcessStandalonePersistentKernelVisitor.h"

#include <sstream>

#include "clang/Lex/Lexer.h"

bool ProcessStandalonePersistentKernelVisitor::VisitCallExpr(CallExpr *CE) {
  auto name = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (name == "get_global_id" ||
    name == "get_group_id" ||
    name == "get_num_groups" ||
    name == "get_global_size") {
    assert(CE->getNumArgs() == 1);
    VisitedFunctionCallsIdFunction = true;
    // TODO: Abort unless the argument has the literal value 0
    RW.ReplaceText(CE->getArg(0)->getSourceRange(), "__d_ctx");
    RW.InsertTextBefore(CE->getSourceRange().getBegin(), "p_");
  }
  if (name == "resizing_global_barrier" || name == "global_barrier") {
    RW.ReplaceText(CE->getSourceRange(), "global_barrier_disc(__bar, __d_ctx)");
  }
  if (name == "offer_fork" || name == "offer_kill") {
    // Remove these calls
    RW.ReplaceText(CE->getSourceRange(), "");
  }
  if (FunctionsThatCallIdFunctions.find(name) != FunctionsThatCallIdFunctions.end()) {
    VisitedFunctionCallsIdFunction = true;
    SourceLocation StartOfParams = Lexer::findLocationAfterToken(CE->getCallee()->getSourceRange().getEnd(),
      tok::l_paren,
      AU->getSourceManager(),
      AU->getLangOpts(),
      /*SkipTrailingWhitespaceAndNewLine=*/true);
    RW.InsertTextAfter(StartOfParams, "__d_ctx");
    if (CE->getNumArgs() > 0) {
      RW.InsertTextAfter(StartOfParams, ", ");
    }
  }
  return true;
}

void ProcessStandalonePersistentKernelVisitor::AddArgumentsForIdCalls(FunctionDecl *D, SourceLocation StartOfParams) {
  RW.InsertTextAfter(StartOfParams, "__global Discovery_ctx *__d_ctx");
  if (D->getNumParams() > 0) {
    RW.InsertTextAfter(StartOfParams, ", ");
  }
}

void ProcessStandalonePersistentKernelVisitor::ProcessKernelFunction(FunctionDecl *D) {
  if (GetKI().KernelFunction) {
    errs() << "Multiple kernel functions in source file not supported, stopping.\n";
    exit(1);
  }
  GetKI().KernelFunction = D;

  SourceLocation endOfParams;

  bool paramAlreadyExists;
  if (D->getNumParams() == 0) {
    GetKI().OriginalParameterText = "";
    endOfParams = Lexer::findLocationAfterToken(D->getLocation(),
      tok::l_paren,
      AU->getSourceManager(),
      AU->getLangOpts(),
      /*SkipTrailingWhitespaceAndNewLine=*/true);
    paramAlreadyExists = false;
  }
  else {
    GetKI().OriginalParameterText = RW.getRewrittenText(
      SourceRange(D->getParamDecl(0)->getSourceRange().getBegin(),
        D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd()));
    endOfParams = Lexer::getLocForEndOfToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), 0, AU->getSourceManager(), AU->getLangOpts());
    paramAlreadyExists = true;
  }

  if (paramAlreadyExists) {
    RW.InsertTextAfter(endOfParams, ", ");
  }
  RW.InsertTextAfter(endOfParams, "__global IW_barrier * __bar, __global Discovery_ctx * __d_ctx, SCHEDULER_ARGS");

  // Now process the body, if it has the right form
  CompoundStmt *CS = dyn_cast<CompoundStmt>(D->getBody());
  if (!CS) {
    errs() << "Kernel function has unexpected body, stopping.\n";
    exit(1);
  }

  if (CS->size() == 0) {
    return;
  }

  // Expected form of body is:
  // <Decl>
  // <Decl>
  // ...
  // <Decl>
  // <NonDecl>
  // <Stmt>
  // ...
  // <Stmt> }
  //

  Stmt *FirstStmt = 0;
  Stmt *FirstNonDecl = 0;
  Stmt *LastStmt = 0;

  for (auto S : CS->body()) {

    if (!FirstStmt) {
      FirstStmt = S;
    }

    if(!FirstNonDecl && dyn_cast<DeclStmt>(S)) {
      // It's a leading decl - skip it.
      continue;
    }
    if(!FirstNonDecl) {
      FirstNonDecl = S;
    }
    LastStmt = S;
  }

  if (!(FirstStmt && FirstNonDecl && LastStmt)) {
    errs() << "Persistent kernel should have at least one non-declaration statement, stopping.\n";
    exit(1);
  }

  RW.InsertTextBefore(FirstStmt->getSourceRange().getBegin(),
    std::string("__local int __scratchpad[2];\n DISCOVERY_PROTOCOL(__d_ctx, __scratchpad);\n INIT_SCHEDULER;"));

  RW.InsertTextBefore(FirstNonDecl->getSourceRange().getBegin(),
    std::string("if (p_get_group_id(__d_ctx) == 0) {\n"
                "    if (get_local_id(0) == 0) {\n"
                "      atomic_store_explicit(s_ctx.persistent_flag, __d_ctx->count, memory_order_release, memory_scope_all_svm_devices);\n"
                "      atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, memory_order_release, memory_scope_all_svm_devices);\n"
                "      while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_acquire, memory_scope_all_svm_devices) != DEVICE_TO_PERSISTENT_TASK)\n"
                "      ;\n"
                "    }\n"
                "    BARRIER;\n"
                "}\n"
                "global_barrier_disc(__bar, __d_ctx);\n"));

  RW.InsertTextAfterToken(LastStmt->getSourceRange().getEnd(),
    std::string("if (get_local_id(0) == 0) {\n"
                "  atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_acq_rel, memory_scope_all_svm_devices);\n"
                "}\n"));

}