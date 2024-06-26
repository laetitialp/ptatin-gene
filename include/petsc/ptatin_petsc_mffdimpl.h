#if !defined(__PTATIN_PETSC_MFFD_H__)
#define __PTATIN_PETSC_MFFD_H__

/* version check */
#include "petscversion.h"

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR == 13))
  #include "src/mat/impls/mffd/mffdimpl.h"
#else
  #error "pTatin provided private petsc header for src/mat/impls/mffd/mffdimpl.h is only valid for PETSc v3.13"
#endif

#endif

