#include "LocalStorageDetector.h"

bool LocalStorageDetector::VisitDeclStmt(DeclStmt *DS) {
  VarDecl* LocalArrayDecl = 0;
  int count = 0;
  for (auto D = DS->decl_rbegin(); D != DS->decl_rend(); D++) {
    count++;
    auto VD = dyn_cast<VarDecl>(*D);
    if (!VD) {
      errs() << "Found non-variable declaration, stopping.\n";
      exit(1);
    }
    if (VD->getType()->isConstantArrayType()) {
      QualType QT = dyn_cast<ConstantArrayType>(VD->getType())->getElementType();
      if (QT.getQualifiers().hasAddressSpace() && QT.getAddressSpace() == LangAS::opencl_local) {
        LocalArrayDecl = VD;
      }
    }
  }
  if (LocalArrayDecl) {
    if (count > 1) {
      errs() << "Cannot handle multiple local declarations in a single declaration group; please separate into multiple singleton groups.\n";
      exit(1);
    }
    LocalArrays.push_back(LocalArrayDecl);
  }
  return true;
}

std::vector<VarDecl*> LocalStorageDetector::GetLocalArrays() {
  return LocalArrays;
}
