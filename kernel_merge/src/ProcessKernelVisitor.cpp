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

  CompoundStmt *CS = dyn_cast<CompoundStmt>(D->getBody());
  if (!CS) {
    errs() << "Kernel function has unexpected body, stopping.\n";
    exit(1);
  }

  // Add "__target" parameter to kernel
  std::stringstream strstr;
  if (D->getNumParams() > 0) {
    strstr << ", ";
  }
  strstr << "__target";
  RW.InsertTextAfterToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(),
    strstr.str());

  // Now process the body, if it has the right form
  if(CS->size() == 0) {
    return true;
  }

  WhileStmt *WS = dyn_cast<WhileStmt>(*(CS->body_begin()));
  if (!WS) {
    return true;
  }
  ProcessWhileStmt(WS);
  return true;
}

void ProcessKernelVisitor::ProcessWhileStmt(WhileStmt *S) {

  CompoundStmt *CS = dyn_cast<CompoundStmt>(S->getBody());
  if (!CS) {
    return;
  }

  auto condition = RW.getRewrittenText(S->getCond()->getSourceRange());
  RW.ReplaceText(S->getCond()->getSourceRange(), "true");
  RW.InsertTextAfterToken(CS->getLBracLoc(), "\nswitch(__target) {\ncase 0:\nif(!(" + condition + ")) { break; }\n");
  unsigned counter = 1;

  for (auto s : CS->body()) {
    CallExpr *CE = dyn_cast<CallExpr>(s);
    if (!CE) {
      continue;
    }
    if ("global_barrier" == CE->getCalleeDecl()->getAsFunction()->getNameAsString()) {
      std::stringstream strstr;
      strstr << "\ncase " << counter << ":\n";
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
