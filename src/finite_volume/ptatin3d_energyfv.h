
#ifndef __ptatin3d_energyfv_h__
#define __ptatin3d_energyfv_h__


#include <finite_volume/fvda.h>
#include <finite_volume/fvda_utils.h>

typedef struct _p_PhysCompEnergyFV *PhysCompEnergyFV;


PetscErrorCode PhysCompEnergyFVDestroy(PhysCompEnergyFV *energy);
PetscErrorCode PhysCompEnergyFVCreate(MPI_Comm comm,PhysCompEnergyFV *energy);
PetscErrorCode PhysCompEnergyFVSetParams(PhysCompEnergyFV energy,PetscReal time,PetscReal dt,PetscInt nsub[]);
PetscErrorCode PhysCompEnergyFVSetFromOptions(PhysCompEnergyFV energy);
PetscErrorCode PhysCompEnergyFVSetUp(PhysCompEnergyFV energy,pTatinCtx pctx);
PetscErrorCode PhysCompEnergyFVUpdateGeometry(PhysCompEnergyFV energy,PhysCompStokes stokes);
PetscErrorCode PhysCompEnergyFVInterpolateMacroQ2ToSubQ1(DM dmv,Vec X,PhysCompEnergyFV energy,DM dmv_fv,Vec X_fv);
PetscErrorCode PhysCompEnergyFVInterpolateNormalVectorToFace(PhysCompEnergyFV energy,Vec X,const char face_field_name[]);
PetscErrorCode PhysCompEnergyFVInterpolateVectorToFace(PhysCompEnergyFV energy,Vec X,const char face_field_name[]);

PetscErrorCode pTatinPhysCompActivate_EnergyFV(pTatinCtx user,PetscBool load);
PetscErrorCode pTatinGetContext_EnergyFV(pTatinCtx ctx,PhysCompEnergyFV *e);
PetscErrorCode pTatinContextValid_EnergyFV(pTatinCtx ctx,PetscBool *exists);
PetscErrorCode pTatinPhysCompEnergyFV_Initialise(PhysCompEnergyFV e,Vec T);

PetscErrorCode ptatin_macro_point_location_sub(
                     PetscInt q2_cell,const PetscInt q2_m[],
                     const PetscInt sub_m[],const PetscReal xi[],
                     PetscInt *sub_fv_cell);

PetscErrorCode ptatin_macro_get_nested_fv_rank_local(
                     PetscInt q2_cell,const PetscInt q2_m[],const PetscInt sub_m[],
                     const PetscInt fv_m[],PetscInt fv_cell[]);

PetscErrorCode EnergyFVEvaluateCoefficients(pTatinCtx user,PetscReal time,PhysCompEnergyFV efv,PetscScalar LA_T[],PetscScalar LA_U[]);

PetscErrorCode MaterialPointOrderingCreate_Cellwise(int nkeys,
                                                    int L,const MPntStd point[],
                                                    int offset[],int order[]);

PetscErrorCode pTatinPhysCompEnergyFV_CreateGetCornersCoefficient(PhysCompEnergyFV efv,PetscInt *n,PetscInt **_c);
PetscErrorCode pTatinPhysCompEnergyFV_CreateGetCornersFVCell(PhysCompEnergyFV efv,PetscInt *n,PetscInt **_c);
PetscErrorCode pTatinPhysCompEnergyFV_MPProjection(PhysCompEnergyFV efv,pTatinCtx pctx);

PetscErrorCode pTatinPhysCompEnergyFV_ComputeALESource(FVDA fv,Vec xk,Vec xk1,PetscReal dt,Vec S,PetscBool forward);
PetscErrorCode pTatinPhysCompEnergyFV_ComputeALEVelocity(DM dmg,Vec x0,Vec x1,PetscReal dt,Vec v);

PetscErrorCode pTatinPhysCompEnergyFV_ComputeAdvectiveTimestep(PhysCompEnergyFV energy,Vec X,PetscReal *_dt);

PetscErrorCode fvgeometry_dmda3d_create_from_element_partition(MPI_Comm comm,PetscInt target_decomp[],const PetscInt m[],DM *dm);

#endif

