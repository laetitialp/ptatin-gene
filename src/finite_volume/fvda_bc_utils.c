
#include <petsc.h>
#include <fvda.h>


/*
 Usage:
 FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,time, FVDABCMethod_SetNatural ,NULL);
 Can use with any face, with/without normals.
*/
PetscErrorCode FVDABCMethod_SetNatural(FVDA fv,
                             DACellFace face,
                             PetscInt nfaces,
                             const PetscReal coor[],
                             const PetscReal normal[],
                             const PetscInt cell[],
                             PetscReal time,
                             FVFluxType flux[],
                             PetscReal bcvalue[],
                             void *ctx)
{
  PetscInt f;
  
  for (f=0; f<nfaces; f++) {
    flux[f] = FVFLUX_NEUMANN_CONSTRAINT;
    bcvalue[f] = 0.0;
  }
  PetscFunctionReturn(0);
}

/*
 Usage:
 FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,time, FVDABCMethod_SetDirichlet ,(void*)&user_dirichlet_bc);
 Can use with any face, with/without normals.
*/
PetscErrorCode FVDABCMethod_SetDirichlet(FVDA fv,
                             DACellFace face,
                             PetscInt nfaces,
                             const PetscReal coor[],
                             const PetscReal normal[],
                             const PetscInt cell[],
                             PetscReal time,
                             FVFluxType flux[],
                             PetscReal bcvalue[],
                             void *ctx)
{
  PetscInt f;
  PetscReal *dirichlet_value = (PetscReal*)ctx;
  
  if (!dirichlet_value) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Expected non-NULL context (Arg 10)");
  
  for (f=0; f<nfaces; f++) {
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    bcvalue[f] = *dirichlet_value;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FVDABCMethod_SetNeumann(FVDA fv,
                                       DACellFace face,
                                       PetscInt nfaces,
                                       const PetscReal coor[],
                                       const PetscReal normal[],
                                       const PetscInt cell[],
                                       PetscReal time,
                                       FVFluxType flux[],
                                       PetscReal bcvalue[],
                                       void *ctx)
{
  PetscInt f;
  PetscReal *neumann_value = (PetscReal*)ctx;
  
  for (f=0; f<nfaces; f++) {
    flux[f] = FVFLUX_NEUMANN_CONSTRAINT;
    bcvalue[f] = *neumann_value;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FVDABCMethod_SetNeumannWithVector(FVDA fv,
                                       DACellFace face,
                                       PetscInt nfaces,
                                       const PetscReal coor[],
                                       const PetscReal normal[],
                                       const PetscInt cell[],
                                       PetscReal time,
                                       FVFluxType flux[],
                                       PetscReal bcvalue[],
                                       void *ctx)
{
  PetscInt f;
  PetscReal *neumann_value = (PetscReal*)ctx;
  
  if (!normal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Method requires normal vectors. Must call iterator FVDAFaceIterator() with require_normals = PETSC_TRUE");
  for (f=0; f<nfaces; f++) {
    flux[f] = FVFLUX_NEUMANN_CONSTRAINT;
    bcvalue[f] = neumann_value[0]*normal[0] + neumann_value[1]*normal[1] + neumann_value[2]*normal[2];
  }
  PetscFunctionReturn(0);
}
