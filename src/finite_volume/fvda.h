
#ifndef __fvda_h__
#define __fvda_h__

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>

typedef struct _p_FVDA *FVDA;
typedef struct _p_FVALE *FVALE;
typedef struct _p_FVTD *FVTD;


#include "fvda_impl.h"



PetscErrorCode DMGlobalToLocal(DM dm,Vec g,InsertMode mode,Vec l);
PetscErrorCode DMLocalToGlobal(DM dm,Vec l,InsertMode mode,Vec g);

PetscErrorCode FVDACreate(MPI_Comm comm,FVDA *_fv);
PetscErrorCode FVDASetDimension(FVDA fv,PetscInt dim);
PetscErrorCode FVDASetSizes(FVDA fv,const PetscInt mi[],const PetscInt Mi[]);
PetscErrorCode FVDASetUp(FVDA fv);
PetscErrorCode FVDACreate2d(MPI_Comm comm,PetscInt Mi[],FVDA *_fv);
PetscErrorCode FVDACreate3d(MPI_Comm comm,PetscInt Mi[],FVDA *_fv);
PetscErrorCode FVDACreateFromDMDA(DM vertex_layout,FVDA *_fv);
PetscErrorCode FVDASetGeometryDM(FVDA fv,DM dm);
PetscErrorCode FVDAUpdateGeometry(FVDA fv);
PetscErrorCode FVDASetProblemType(FVDA fv,PetscBool Qdot,FVDAPDEType equation_type,PetscInt numerical_flux,PetscInt reconstruction);
PetscErrorCode FVDADestroy(FVDA *_fv);

PetscErrorCode FVDAGetBoundaryFaceIndicesRead(FVDA fv,DACellFace face,PetscInt *len,const PetscInt *indices[]);
PetscErrorCode FVDAGetBoundaryFaceIndicesOwnershipRange(FVDA fv,DACellFace face,PetscInt *start,PetscInt *end);

PetscErrorCode FVDAGetFaceInfo(FVDA fv,PetscInt *nfaces,const DACellFaceLocation *l[],const DACellFace *f[],const PetscReal *n[],const PetscReal *c[]);

PetscErrorCode DACellGeometry2d_GetCoordinates(const PetscInt element[],const PetscReal mesh_coor[],PetscReal coor[]);
PetscErrorCode DACellGeometry3d_GetCoordinates(const PetscInt element[],const PetscReal mesh_coor[],PetscReal coor[]);

PetscErrorCode _EvaluateFaceNormal3d(DACellFace face,const PetscReal coor[],const PetscReal xi0[],PetscReal n0[]);
PetscErrorCode _EvaluateFaceCoord3d(DACellFace face,const PetscReal coor[],const PetscReal xi0[],PetscReal c0[]);

PetscErrorCode FVDAFaceIterator(FVDA fv,DACellFace face,PetscBool require_normals,PetscReal time,
                                PetscErrorCode (*user_setter)(FVDA,
                                                              DACellFace,
                                                              PetscInt,
                                                              const PetscReal*,
                                                              const PetscReal*,
                                                              const PetscInt*,
                                                              PetscReal,FVFluxType*,PetscReal*,void*),
                                void *data);

PetscErrorCode eval_F(SNES snes,Vec X,Vec F,void *data);
PetscErrorCode eval_J(SNES snes,Vec X,Mat Ja,Mat Jb,void *data);

PetscErrorCode eval_F_hr(SNES snes,Vec X,Vec F,void *data);

PetscErrorCode FVDAGetValidElement(FVDA fv,PetscInt faceid,PetscInt *cellid);

PetscErrorCode _EvaluateCellVolume3d(const PetscReal coor[],PetscReal *v);
PetscErrorCode _EvaluateFaceArea3d(DACellFace face,const PetscReal coor[],PetscReal *a);

PetscErrorCode FVDAVecTraverse(FVDA fv,Vec X,PetscReal time,PetscInt dof,
                               PetscBool user_setter(PetscScalar*,PetscScalar*,void*),
                               void *data);

#endif
