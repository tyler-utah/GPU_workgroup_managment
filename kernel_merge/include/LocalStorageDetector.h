#ifndef KERNEL_MERGE_LOCALSTORAGEDETECTOR_H
#define KERNEL_MERGE_LOCALSTORAGEDETECTOR_H

#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang;
using namespace llvm;

class LocalStorageDetector : public RecursiveASTVisitor<LocalStorageDetector> {

public:
  LocalStorageDetector(Stmt *S) {
    TraverseStmt(S);
  }

  bool VisitDeclStmt(DeclStmt *DS);

  std::vector<VarDecl*> GetLocalArrays();

private:
  std::vector<VarDecl*> LocalArrays;

};


#endif