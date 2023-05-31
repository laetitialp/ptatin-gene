#ifndef __gravity_constant_h__
#define __gravity_constant_h__

typedef struct _p_GravityConstant *GravityConstant;

struct _p_GravityConstant {
  PetscReal gravity_const[3]; /* Constant gravity vector */
  PetscReal magnitude;        /* Magnitude of the gravity vector */
};

PetscErrorCode GravityDestroyCtx_Constant(GravityModel gravity);
PetscErrorCode GravityScale_Constant(GravityModel gravity, PetscReal scaling_factor);
PetscErrorCode GravitySet_ConstantVector(GravityConstant gravity, PetscReal gvec[]);
PetscErrorCode GravitySet_ConstantMagnitude(GravityConstant gravity, PetscReal magnitude);
PetscErrorCode GravitySet_Constant(GravityModel gravity, void *data);
PetscErrorCode GravityConstantCreateCtx(GravityModel gravity);
PetscErrorCode GravityGetConstantCtx(GravityModel gravity, GravityConstant *ctx);
PetscErrorCode QuadratureSetGravityModel_Constant(PhysCompStokes stokes, GravityModel gravity);
PetscErrorCode QuadratureUpdateGravityModel_Constant(PhysCompStokes stokes, GravityModel gravity);

#endif