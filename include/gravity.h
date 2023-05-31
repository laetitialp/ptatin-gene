#ifndef __ptatin_gravity_h__
#define __ptatin_gravity_h__

#include "gravity/gravity_constant.h"
#include "gravity/gravity_radialconstant.h"

typedef enum {
  CONSTANT=0,
  RADIAL_CONSTANT,
  RADIAL_VAR,
  ARBITRARY,
  POISSON
} GravityType;

struct _p_GravityModel {
  GravityType gravity_type;
  void        *data; /* Gravity type data structure */
  PetscErrorCode (*destroy)(GravityModel);
  PetscErrorCode (*scale)(GravityModel,PetscReal);
  PetscErrorCode (*set)(GravityModel,void*);
  PetscErrorCode (*quadrature_set)(PhysCompStokes,GravityModel);
  PetscErrorCode (*update)(PhysCompStokes,GravityModel);
};

PetscErrorCode GravityModelCreateCtx(GravityModel *gravity);
PetscErrorCode GravitySetType(GravityModel gravity, GravityType gtype);
PetscErrorCode GravityScale(GravityModel gravity, PetscReal scaling_factor);
PetscErrorCode GravitySet(GravityModel gravity, void *data);
PetscErrorCode GravityModelDestroyCtx(GravityModel *gravity);
PetscErrorCode pTatinDestroyGravityModelCtx(pTatinCtx ptatin);
PetscErrorCode pTatinGetGravityModelCtx(pTatinCtx ptatin, GravityModel *ctx);
PetscErrorCode GravityCreateTypeCtx(GravityModel gravity);
PetscErrorCode pTatinCreateGravityModel(pTatinCtx ptatin, GravityType gtype);
PetscErrorCode QuadratureSetBodyForcesOnPoint(QPntVolCoefStokes *cell_gausspoints, PetscInt qp_idx);
PetscErrorCode pTatinQuadratureSetGravityModel(pTatinCtx ptatin);
PetscErrorCode pTatinQuadratureUpdateGravityModel(pTatinCtx ptatin);

#endif