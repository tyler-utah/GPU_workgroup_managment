#include "ProcessKernelVisitor.h"

#include <sstream>

#include <ARCMigrate\Transforms.h>

bool ProcessKernelVisitor::VisitFunctionDecl(FunctionDecl *D)
{

  if (!D->hasAttr<OpenCLKernelAttr>() || !D->hasBody()) {
    return true;
  }

  if (KernelFunction) {
    errs() << "Multiple kernel functions in source file not supported, stopping.\n";
    exit(1);
  }
  KernelFunction = D;

  if (D->getNumParams() == 0) {
    OriginalParameterText = "";
  } else {
    OriginalParameterText = RW.getRewrittenText(
      SourceRange(D->getParamDecl(0)->getSourceRange().getBegin(),
                  D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd()));
  }

  // Remove the "kernel" attribute
  RW.RemoveText(D->getAttr<OpenCLKernelAttr>()->getRange());

  // Add "__target" parameter to kernel
  {
    std::stringstream strstr;
    if (D->getNumParams() > 0) {
      strstr << ", ";
    }
    strstr << D->getNameAsString() + "_restoration_ctx_t * restoration_ctx";
    RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(),
      strstr.str());
  }

  // Now process the body, if it has the right form
  CompoundStmt *CS = dyn_cast<CompoundStmt>(D->getBody());
  if (!CS) {
    errs() << "Kernel function has unexpected body, stopping.\n";
    exit(1);
  }

  if(CS->size() == 0) {
    return true;
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
      if (!VD->hasInit()) {
        errs() << "Persistent declaration " << VD->getNameAsString() << " has no initialiser, stopping.\n";
        exit(1);
      }
    }
  }
  
  {
    std::stringstream restorationstruct;
    restorationstruct << "typedef struct {\n";

    std::stringstream restorationcode;
    restorationcode << "if(restoration_ctx->target != 0) {\n";
    for (auto DS : DeclsToRestore) {
      for (auto D : DS->decls()) {
        VarDecl *VD = dyn_cast<VarDecl>(D);
        restorationcode << VD->getNameAsString() << " = restoration_ctx->" << VD->getNameAsString() << ";\n";
        restorationstruct << "  " << VD->getType().getAsString() << " " << VD->getNameAsString() << ";\n";
      }
    }
    restorationcode << "}\n";

    restorationstruct << "} " << D->getNameAsString() << "_restoration_ctx_t;\n\n";

    RW.InsertTextBefore(WhileLoop->getLocStart(), restorationcode.str());

    RW.InsertTextBefore(D->getLocStart(), restorationstruct.str());
  }

  ProcessWhileStmt(WhileLoop);
  return true;
}

void ProcessKernelVisitor::ProcessWhileStmt(WhileStmt *S) {

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

      SourceLocation SemiLoc = clang::arcmt::trans::findSemiAfterLocation(CE->getLocEnd(), ASTC);
      RW.InsertTextAfterToken(SemiLoc, strstr.str());
      counter++;
    }
  }

  RW.InsertTextBefore(CS->getRBracLoc(), "}");
}

bool ProcessKernelVisitor::processedKernel() {
  return KernelFunction;
}

std::string ProcessKernelVisitor::getOriginalKernelParameterText() {
  return OriginalParameterText;
}

FunctionDecl *ProcessKernelVisitor::getKernelFunctionDecl() {
  return KernelFunction;
}
