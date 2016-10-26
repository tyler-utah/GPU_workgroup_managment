#include "ProcessNonPersistentKernelVisitor.h"

#include <sstream>

#include "clang/Lex/Lexer.h"

bool ProcessNonPersistentKernelVisitor::VisitCallExpr(CallExpr *CE) {
  auto name = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if(name == "get_global_id" ||
     name == "get_group_id" ||
     name == "get_num_groups" || 
     name == "get_global_size") {
     assert(CE->getNumArgs() == 1);
     VisitedFunctionCallsIdFunction = true;
     // TODO: Abort unless the argument has the literal value 0
     RW.ReplaceText(CE->getArg(0)->getSourceRange(), "__k_ctx");
     RW.InsertTextBefore(CE->getSourceRange().getBegin(), "k_");
  }
  if(name == "resizing_global_barrier" || name == "global_barrier" || name == "offer_fork" || name == "offer_kill") {
    errs() << "Persistent kernel functions should not appear in non-persistent kernel\n";
    exit(1);
  }
  if (FunctionsThatCallIdFunctions.find(name) != FunctionsThatCallIdFunctions.end()) {
    VisitedFunctionCallsIdFunction = true;
    SourceLocation StartOfParams = Lexer::findLocationAfterToken(CE->getCallee()->getSourceRange().getEnd(),
      tok::l_paren,
      AU->getSourceManager(),
      AU->getLangOpts(),
      /*SkipTrailingWhitespaceAndNewLine=*/true);
    RW.InsertTextAfter(StartOfParams, "__k_ctx");
    if (CE->getNumArgs() > 0) {
      RW.InsertTextAfter(StartOfParams, ", ");
    }
  }
  return true;
}

void ProcessNonPersistentKernelVisitor::AddArgumentsForIdCalls(FunctionDecl *D, SourceLocation StartOfParams) {
  RW.InsertTextAfter(StartOfParams, "__global Kernel_ctx *__k_ctx");
  if (D->getNumParams() > 0) {
    RW.InsertTextAfter(StartOfParams, ", ");
  }
}

void ProcessNonPersistentKernelVisitor::ProcessKernelFunction(FunctionDecl *D) {
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
  } else {
    GetKI().OriginalParameterText = RW.getRewrittenText(
      SourceRange(D->getParamDecl(0)->getSourceRange().getBegin(),
        D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd()));
    endOfParams = Lexer::getLocForEndOfToken(D->getParamDecl(D->getNumParams() - 1)->getSourceRange().getEnd(), 0, AU->getSourceManager(), AU->getLangOpts());
    paramAlreadyExists = true;
  }

  DetectLocalStorage(D->getBody());

  for (auto VD : LocalArrays) {
    std::stringstream strstr;
    if (paramAlreadyExists) {
      strstr << ", ";
    }
    paramAlreadyExists = true;
    strstr << "__local ";
    std::string typeName = dyn_cast<BuiltinType>(dyn_cast<ConstantArrayType>(VD->getType())->getElementType())->getName(PrintingPolicy(AU->getLangOpts()));
    strstr << typeName;
    strstr << " * " << VD->getNameAsString();
    RW.InsertTextAfter(endOfParams, strstr.str());
  }

  // Add "__k_ctx" parameter to kernel
  if(paramAlreadyExists) {
    RW.InsertTextAfter(endOfParams, ", ");
  }
  RW.InsertTextAfter(endOfParams, "__global Kernel_ctx * __k_ctx");

  // Remove the "kernel" attribute
  RW.RemoveText(D->getAttr<OpenCLKernelAttr>()->getRange());

}