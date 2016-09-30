#include "ProcessPersistentKernelVisitor.h"

#include <sstream>

#include <ARCMigrate\Transforms.h>

bool ProcessPersistentKernelVisitor::VisitFunctionDecl(FunctionDecl *D)
{

  if (D->hasAttr<OpenCLKernelAttr>() && D->hasBody()) {
    ProcessKernelFunction(D);
  }

  return RecursiveASTVisitor::VisitFunctionDecl(D);

}

void ProcessPersistentKernelVisitor::ProcessWhileStmt(WhileStmt *S) {

  CompoundStmt *CS = dyn_cast<CompoundStmt>(S->getBody());
  if (!CS) {
    errs() << "Expected while loop with compound body, stopping.\n";
    exit(1);
  }

  auto condition = RW.getRewrittenText(S->getCond()->getSourceRange());
  RW.ReplaceText(S->getCond()->getSourceRange(), "true");
  RW.InsertTextAfterToken(CS->getLBracLoc(), "\nswitch(restoration_ctx->target) {\ncase 0:\nif(!(" + condition + ")) { break; }\n");
  unsigned counter = 1;

  for (auto s : CS->body()) {
    CallExpr *CE = dyn_cast<CallExpr>(s);
    if (!CE) {
      continue;
    }
    if ("global_barrier" == CE->getCalleeDecl()->getAsFunction()->getNameAsString()) {
      std::stringstream strstr;
      strstr << "\ncase " << counter << ":\n";
      strstr << "restoration_ctx->target = 0;\n";

      SourceLocation SemiLoc = clang::arcmt::trans::findSemiAfterLocation(CE->getLocEnd(), AU->getASTContext());
      RW.InsertTextAfterToken(SemiLoc, strstr.str());
      counter++;
    }
  }

  RW.InsertTextBefore(CS->getRBracLoc(), "}");
}

bool ProcessPersistentKernelVisitor::VisitCallExpr(CallExpr *CE) {
  auto name = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (name == "get_global_id" ||
    name == "get_group_id" ||
    name == "get_num_groups" ||
    name == "get_global_size") {
    assert(CE->getNumArgs() == 1);
    // TODO: Abort unless the argument has the literal value 0
    RW.ReplaceText(CE->getArg(0)->getSourceRange(), "kernel_ctx");
    RW.InsertTextBefore(CE->getSourceRange().getBegin(), "k_");
  }
  if (name == "global_barrier") {
    assert(CE->getNumArgs() == 1);
    RW.ReplaceText(CE->getSourceRange(), "global_barrier(kernel_ctx)");
  }
  return true;
}

void ProcessPersistentKernelVisitor::ProcessKernelFunction(FunctionDecl *D) {
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

  // Remove the "kernel" attribute
  RW.RemoveText(D->getAttr<OpenCLKernelAttr>()->getRange());

  // Add "restoration_ctx" parameter to kernel
  {
    std::string newParam = "";
    if (D->getNumParams() > 0) {
      newParam = ", ";
    }
    newParam += "RestorationCtx * restoration_ctx";
    RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(),
      newParam);
  }

  // Add "kernel_ctx" parameter to kernel
  {
    std::string newParam = "";
    if (D->getNumParams() > 0) {
      newParam = ", ";
    }
    newParam += "KernelCtx * kernel_ctx";
    RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), newParam);
  }

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
  // <Decl to be restored>
  // <Decl to be restored>
  // ...
  // <Decl to be restored>
  // while(c) {
  //    ...
  // }
  //
  // We reject anything else

  WhileStmt *WhileLoop = 0;

  for (auto S : CS->body()) {
    if (dyn_cast<DeclStmt>(S)) {
      if (WhileLoop) {
        errs() << "Declaration found after while loop, stopping.\n";
        exit(1);
      }
      DeclsToRestore.push_back(dyn_cast<DeclStmt>(S));
      continue;
    }
    if (dyn_cast<WhileStmt>(S)) {
      if (WhileLoop) {
        errs() << "Multiple loops found, stopping.\n";
        exit(1);
      }
      WhileLoop = dyn_cast<WhileStmt>(S);
      continue;
    }
    errs() << "Non declaration or loop statement found, stopping.\n";
    exit(1);
  }

  for (auto DS : DeclsToRestore) {
    for (auto D = DS->decl_rbegin(); D != DS->decl_rend(); D++) {
      auto VD = dyn_cast<VarDecl>(*D);
      if (!VD) {
        errs() << "Found non-variable declaration, stopping.\n";
        exit(1);
      }
    }
  }

  std::string restorationCtx;
  restorationCtx = "typedef struct {\n";
  restorationCtx += "  uchar target;\n";

  std::string restorationCode;
  restorationCode += "if(restoration_ctx->target != 0) {\n";
  for (auto DS : DeclsToRestore) {
    for (auto D : DS->decls()) {
      VarDecl *VD = dyn_cast<VarDecl>(D);
      restorationCode += VD->getNameAsString() + " = restoration_ctx->" + VD->getNameAsString() + ";\n";
      restorationCtx += "  " + VD->getType().getAsString() + " " + VD->getNameAsString() + ";\n";
    }
  }
  restorationCode += "}\n";
  restorationCtx += "} RestorationCtx;\n\n";

  RW.InsertTextBefore(WhileLoop->getLocStart(), restorationCode);

  RW.InsertTextBefore(D->getLocStart(), restorationCtx);

  ProcessWhileStmt(WhileLoop);

}

void ProcessPersistentKernelVisitor::EmitRewrittenText() {
  const RewriteBuffer *RewriteBuf =
    RW.getRewriteBufferFor(AU->getSourceManager().getMainFileID());
  if (!RewriteBuf) {
    errs() << "Nothing was re-written\n";
    exit(1);
  }
  llvm::outs() << std::string(RewriteBuf->begin(), RewriteBuf->end());
}
