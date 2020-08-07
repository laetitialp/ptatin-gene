#if !defined(__PTATIN_PETSC_EX43_SOLCX_H__)
#define __PTATIN_PETSC_EX43_SOLCX_H__

/* version check */
#include "petscversion.h"

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 9) && (PETSC_VERSION_MINOR <= 13))
  #include "src/ksp/ksp/examples/tutorials/ex43-solcx.h"
#else
  #error "pTatin provided private petsc header for src/ksp/ksp/examples/tutorials/ex43-solcx.h is only valid for PETSc v3.9, v3.10, v3.11, v3.12, v3.13"
#endif

#endif
