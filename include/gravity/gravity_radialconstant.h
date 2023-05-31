#ifndef __gravity_radialconstant_h__
#define __gravity_radialconstant_h__

typedef struct _p_GravityRadialConstant *GravityRadialConstant;

struct _p_GravityRadialConstant {
  PetscReal magnitude;
};

PetscErrorCode GravityDestroyCtx_RadialConstant(GravityModel gravity);
PetscErrorCode GravityScale_RadialConstant(GravityModel gravity, PetscReal scaling_factor);
PetscErrorCode GravitySet_RadialConstantMagnitude(GravityRadialConstant gravity, PetscReal magnitude);
PetscErrorCode GravitySet_RadialConstant(GravityModel gravity, void *data);
PetscErrorCode QuadratureSetGravityModel_RadialConstant(PhysCompStokes stokes, GravityModel gravity);
PetscErrorCode GravityGetRadialConstantCtx(GravityModel gravity, GravityRadialConstant *ctx);
PetscErrorCode GravityRadialConstantCreateCtx(GravityModel gravity);

#endif