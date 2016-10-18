#include "ProcessPersistentKernelVisitor.h"

#include <sstream>

#include <ARCMigrate\Transforms.h>

template <class T>
class RecordLoopAndSwitchDepth : public RecursiveASTVisitor<T> {

public:

  explicit RecordLoopAndSwitchDepth() {
    this->count = 0;
  }

  bool TraverseWhileStmt(WhileStmt *S) {
    this->count++;
    bool result = RecursiveASTVisitor::TraverseWhileStmt(S);
    this->count--;
    return result;
  }

  bool TraverseForStmt(ForStmt *S) {
    this->count++;
    bool result = RecursiveASTVisitor::TraverseForStmt(S);
    this->count--;
    return result;
  }

  bool TraverseDoStmt(DoStmt *S) {
    this->count++;
    bool result = RecursiveASTVisitor::TraverseDoStmt(S);
    this->count--;
    return result;
  }

  bool TraverseSwitchStmt(SwitchStmt *S) {
    this->count++;
    bool result = RecursiveASTVisitor::TraverseSwitchStmt(S);
    this->count--;
    return result;
  }

protected:
  int count;

};

class ReplaceTopLevelBreakWithReturn : public RecordLoopAndSwitchDepth<ReplaceTopLevelBreakWithReturn> {

public:

  explicit ReplaceTopLevelBreakWithReturn(WhileStmt *S, Rewriter &RW) : RW(RW) {
    TraverseStmt(S->getBody());
  }

  bool VisitBreakStmt(BreakStmt *S) {
    if (this->count == 0) {
      // This is a top-level break, so replace it with a return
      RW.ReplaceText(S->getSourceRange(), "return");
    }
    return true;
  }

private:
  Rewriter &RW;

};

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
  RW.InsertTextAfterToken(CS->getLBracLoc(), "\nswitch(__restoration_ctx->target) {\ncase 0:\nif(!(" + condition + ")) { return; }\n");
  RW.InsertTextBefore(CS->getRBracLoc(), "}");

  ReplaceTopLevelBreakWithReturn RTLBWR(S, RW);

}

bool ProcessPersistentKernelVisitor::VisitCallExpr(CallExpr *CE) {
  auto name = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (name == "get_global_id" ||
    name == "get_group_id" ||
    name == "get_num_groups" ||
    name == "get_global_size") {
    assert(CE->getNumArgs() == 1);
    // TODO: Abort unless the argument has the literal value 0
    RW.ReplaceText(CE->getArg(0)->getSourceRange(), "__k_ctx");
    RW.InsertTextBefore(CE->getSourceRange().getBegin(), "k_");
  }
  if (name == "resizing_global_barrier") {
    ForkPointCounter++;
    assert(CE->getNumArgs() == 1);
    std::stringstream sstr;
    sstr << "{ Restoration_ctx __to_fork; __to_fork.target = " << ForkPointCounter << "; ";
    for (auto DS : DeclsToRestore) {
      for (auto D : DS->decls()) {
        VarDecl *VD = dyn_cast<VarDecl>(D);
        sstr << "__to_fork." << VD->getNameAsString() << " = " << VD->getNameAsString() << "; ";
      }
    }
    sstr << "global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad, &__to_fork); } ";
    sstr << "case " << ForkPointCounter << ": __restoration_ctx->target = 0";
    RW.ReplaceText(CE->getSourceRange(), sstr.str());
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
    errs() << "The kernel must have at least one parameter, for technical reasons; please add a dummy parameter if necessary.  Stopping.\n";
    exit(1);
  }
  KI.OriginalParameterText = RW.getRewrittenText(
    SourceRange(D->getParamDecl(0)->getSourceRange().getBegin(),
      D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd()));

  // Remove the "kernel" attribute
  RW.RemoveText(D->getAttr<OpenCLKernelAttr>()->getRange());

  RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), ", __global IW_barrier * __bar");
  RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), ", __global Kernel_ctx * __k_ctx");
  RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), ", CL_Scheduler_ctx __s_ctx");
  RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), ", __local int * __scratchpad");
  RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), ", Restoration_ctx * __restoration_ctx");

  // Add "__k_ctx" parameter to kernel

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

  this->RestorationCtx = "typedef struct {\n";
  this->RestorationCtx += "  uchar target;\n";

  std::string preLoopCode;
  preLoopCode += "if(__restoration_ctx->target != 0) {\n";
  for (auto DS : DeclsToRestore) {
    for (auto D : DS->decls()) {
      VarDecl *VD = dyn_cast<VarDecl>(D);
      preLoopCode += VD->getNameAsString() + " = __restoration_ctx->" + VD->getNameAsString() + ";\n";
      this->RestorationCtx += "  " + VD->getType().getAsString() + " " + VD->getNameAsString() + ";\n";
    }
  }
  preLoopCode += "}\n";
  this->RestorationCtx += "} Restoration_ctx;\n\n";

  RW.InsertTextBefore(WhileLoop->getLocStart(), preLoopCode);

  ProcessWhileStmt(WhileLoop);

}

void ProcessPersistentKernelVisitor::EmitRewrittenText(std::ostream & out) {
  const RewriteBuffer *RewriteBuf =
    RW.getRewriteBufferFor(AU->getSourceManager().getMainFileID());
  if (!RewriteBuf) {
    errs() << "Nothing was re-written\n";
    exit(1);
  }
  out << std::string(RewriteBuf->begin(), RewriteBuf->end());
}
