

#ifndef __ptatin3d_surfbclist_h__
#define __ptatin3d_surfbclist_h__

#include <petsc.h>
#include <petscdm.h>
#include <ptatin3d.h>
#include <quadrature.h>
#include <mesh_entity.h>
#include <surface_constraint.h>

typedef struct _p_SurfBCList *SurfBCList;
struct _p_SurfBCList {
  DM dm;
  MeshFacetInfo mfi;
  SurfaceQuadrature surfQ;
  PetscInt sc_nreg;
  SurfaceConstraint *sc_list;
};

PetscErrorCode SurfBCListDestroy(SurfBCList *_sl);
PetscErrorCode SurfBCListCreate(DM dm, SurfaceQuadrature surfQ, MeshFacetInfo mfi, SurfBCList *_sl);
PetscErrorCode SurfBCListAddConstraint(SurfBCList sl, const char name[], SurfaceConstraint *_sc);
PetscErrorCode SurfBCListGetConstraint(SurfBCList sl, const char name[], SurfaceConstraint *_sc);
PetscErrorCode SurfBCListInsertConstraint(SurfBCList sl, SurfaceConstraint sc, PetscBool *inserted);
PetscErrorCode SurfBCListViewer(SurfBCList sl,PetscViewer v);

#endif

