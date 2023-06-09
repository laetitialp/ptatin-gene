#ifndef __gravity_constant_h__
#define __gravity_constant_h__

typedef struct _p_GravityConstant *GravityConstant;

PetscErrorCode GravityDestroyCtx_Constant(Gravity gravity);
PetscErrorCode GravityScale_Constant(Gravity gravity, PetscReal scaling_factor);
PetscErrorCode GravitySet_ConstantVector(GravityConstant gravity, PetscReal gvec[]);
PetscErrorCode GravitySet_ConstantMagnitude(GravityConstant gravity, PetscReal magnitude);
PetscErrorCode GravityGet_ConstantVector(Gravity gravity, PetscReal gvec[]);
PetscErrorCode GravityGet_ConstantMagnitude(Gravity gravity, PetscReal *magnitude);
PetscErrorCode GravitySet_Constant(Gravity gravity, PetscReal gvec[]);
PetscErrorCode GravityGetPointWiseVector_Constant(Gravity gravity, PetscInt eidx, PetscReal global_coords[], PetscReal local_coords[], PetscReal gvec[]);
PetscErrorCode GravityConstantCreateCtx(Gravity gravity);
PetscErrorCode GravityGetConstantCtx(Gravity gravity, GravityConstant *ctx);
PetscErrorCode QuadratureSetGravity_Constant(PhysCompStokes stokes, Gravity gravity);
PetscErrorCode QuadratureUpdateGravity_Constant(PhysCompStokes stokes, Gravity gravity);

#endif