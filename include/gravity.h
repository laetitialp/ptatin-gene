#ifndef __ptatin_gravity_h__
#define __ptatin_gravity_h__

typedef enum {
  CONSTANT=0,
  RADIAL_CONSTANT,
  RADIAL_VAR,
  ARBITRARY,
  POISSON
} GravityType;

typedef struct _p_GravityConstant *GravityConstant;
typedef struct _p_GravityRadialConst *GravityRadialConst;

struct _p_GravityModel {
  GravityType gravity_type; 
  void        *data;
};

struct _p_GravityConstant {
  PetscReal gravity_const[3]; /* Constant gravity vector */
  PetscReal magnitude;        /* Magnitude of the gravity vector */
  PetscReal scaling_factor;
};


struct _p_GravityRadialConst {
  PetscReal gravity_mag;
  PetscReal scaling_factor;
};


PetscErrorCode GravityCreateCtx(GravityModel gravity);
PetscErrorCode pTatinCreateGravityModel(pTatinCtx ptatin, GravityType gtype, void *data);
PetscErrorCode GravityGetConstantCtx(GravityModel gravity, GravityConstant *ctx);
/*
PetscErrorCode GravityModelDestroyCtx(GravityModel *gravity);
PetscErrorCode pTatinGetContext_GravityModel(pTatinCtx ptatin, GravityModel *gravity);

PetscErrorCode GravitySetType(GravityModel gravity, GravityType gtype);
PetscErrorCode GravitySet_GravityConst(GravityModel gravity, PetscReal gvec[]);
PetscErrorCode GravitySet_GravityMag(GravityModel gravity, PetscReal magnitude);
PetscErrorCode GravitySetValues_ConstantVector(GravityModel gravity, void *data);
PetscErrorCode GravitySetValues_Magnitude(GravityModel gravity, void *data);
PetscErrorCode GravitySetValues(GravityModel gravity, void *data);

PetscErrorCode GravityScale_GravityConst(GravityModel gravity, PetscReal fac);
PetscErrorCode GravityScale_GravityMag(GravityModel gravity, PetscReal fac);
PetscErrorCode GravityScale(GravityModel gravity, PetscReal fac);
PetscErrorCode pTatinCreateGravityModel(pTatinCtx ptatin, GravityType gtype, PetscReal scaling_factor, void *data);

PetscErrorCode QuadratureSetGravityModel(PhysCompStokes stokes, QPntVolCoefStokes *all_gausspoints, GravityModel gravity, PetscBool body_forces);
PetscErrorCode GravityModelUpdateQuadraturePoints(PhysCompStokes stokes, QPntVolCoefStokes *all_gausspoints, GravityModel gravity, PetscBool body_forces);
PetscErrorCode pTatin_ApplyInitialStokesBodyForcesModel(pTatinCtx ptatin);
PetscErrorCode pTatin_ApplyInitialStokesGravityModel(pTatinCtx ptatin);
PetscErrorCode pTatin_UpdateStokesBodyForcesModel(pTatinCtx ptatin);
PetscErrorCode pTatin_UpdateStokesGravityModel(pTatinCtx ptatin);
*/
#endif