#ifndef __ptatin_gravity_h__
#define __ptatin_gravity_h__

#include "gravity/gravity_constant.h"
#include "gravity/gravity_radialconstant.h"

typedef enum {
  GRAVITY_CONSTANT=0,
  GRAVITY_RADIAL_CONSTANT,
  GRAVITY_RADIAL_VAR,
  GRAVITY_ARBITRARY,
  GRAVITY_POISSON
} GravityType;

extern const char *GravityTypeNames[];

extern PetscErrorCode (*create_types[])(Gravity);

struct _p_Gravity {
  GravityType gravity_type;   /* Type of gravity used */
  PetscReal   gvec_ptwise[3]; /* stores a 3 entries gravity vec (Ease memory management) */
  void        *data;          /* Gravity type data structure */
  /* Function pointers to the different types implementations */
  PetscErrorCode (*destroy)(Gravity);                                             /* Destroy data structure and free memory */
  PetscErrorCode (*scale)(Gravity,PetscReal);                                     /* Scale values */
  PetscErrorCode (*quadrature_set)(PhysCompStokes,Gravity);                       /* Set gravity vector on quadrature points */
  PetscErrorCode (*update)(PhysCompStokes,Gravity);                               /* Update gravity vector on quadrature points */
  PetscErrorCode (*get_gvec)(Gravity,PetscInt,PetscReal*,PetscReal*,PetscReal**); /* Evaluate pointwise gravity vector */
};

PetscErrorCode GravityCreate(Gravity *gravity);
PetscErrorCode GravitySetType(Gravity gravity, GravityType gtype);
PetscErrorCode GravityScale(Gravity gravity, PetscReal scaling_factor);
PetscErrorCode GravityDestroyCtx(Gravity *gravity);
PetscErrorCode pTatinDestroyGravityCtx(pTatinCtx ptatin);
PetscErrorCode pTatinGetGravityCtx(pTatinCtx ptatin, Gravity *ctx);
PetscErrorCode pTatinContextValid_Gravity(pTatinCtx ptatin, PetscBool *exists);
PetscErrorCode pTatinCreateGravity(pTatinCtx ptatin, GravityType gtype);
PetscErrorCode QuadratureSetBodyForcesOnPoint(QPntVolCoefStokes *cell_gausspoints, PetscInt qp_idx);
PetscErrorCode pTatinQuadratureSetGravity(pTatinCtx ptatin);
PetscErrorCode pTatinQuadratureUpdateGravity(pTatinCtx ptatin);
PetscErrorCode pTatinGetGravityPointWiseVector(pTatinCtx ptatin, PetscInt eidx, PetscReal global_coords[], PetscReal local_coords[], PetscReal *gvec[]);

#endif