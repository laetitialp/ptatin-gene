#ifndef __gravity_radialconstant_h__
#define __gravity_radialconstant_h__

typedef struct _p_GravityRadialConstant *GravityRadialConstant;

PetscErrorCode GravityDestroyCtx_RadialConstant(Gravity gravity);
PetscErrorCode GravityScale_RadialConstant(Gravity gravity, PetscReal scaling_factor);
PetscErrorCode GravitySet_RadialConstantMagnitude(GravityRadialConstant gravity, PetscReal magnitude);
PetscErrorCode GravitySet_RadialConstant(Gravity gravity, PetscReal magnitude);
PetscErrorCode GravityGetPointWiseVector_RadialConstant(Gravity gravity, PetscInt eidx, PetscReal global_coords[], PetscReal local_coords[], PetscReal gvec[]);
PetscErrorCode QuadratureSetGravity_RadialConstant(PhysCompStokes stokes, Gravity gravity);
PetscErrorCode GravityGetRadialConstantCtx(Gravity gravity, GravityRadialConstant *ctx);
PetscErrorCode GravityRadialConstantCreateCtx(Gravity gravity);

#endif