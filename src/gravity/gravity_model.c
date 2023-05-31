#include "petsc.h"

#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "dmda_element_q2p1.h"
#include "quadrature.h"
#include "dmda_checkpoint.h"
#include "element_type_Q2.h"
#include "data_bucket.h"
//#include "mesh_entity.h"
#include "QPntVolCoefStokes_def.h"
#include "gravity.h"

PetscErrorCode GravityModelCreateCtx(GravityModel *gravity)
{
  PetscErrorCode ierr;
  GravityModel   grav;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_GravityModel),&grav);CHKERRQ(ierr);
  ierr = PetscMemzero(grav,sizeof(struct _p_GravityModel));CHKERRQ(ierr);
  *gravity = grav;
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySetType(GravityModel gravity, GravityType gtype)
{
  PetscFunctionBegin;
  gravity->gravity_type = gtype;
  PetscFunctionReturn(0);
}

PetscErrorCode GravityScale(GravityModel gravity, PetscReal scaling_factor)
{
  PetscFunctionBegin;
  gravity->scale(gravity,scaling_factor);
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySet(GravityModel gravity, void *data)
{
  PetscFunctionBegin;
  gravity->set(gravity,data);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetGravityModel(PhysCompStokes stokes, GravityModel gravity)
{
  PetscFunctionBegin;
  gravity->quadrature_set(stokes,gravity);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureUpdateGravityModel(PhysCompStokes stokes, GravityModel gravity)
{
  PetscFunctionBegin;
  gravity->update(stokes,gravity);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinQuadratureSetGravityModel(pTatinCtx ptatin)
{
  PhysCompStokes stokes;
  GravityModel   gravity;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinGetGravityModelCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = QuadratureSetGravityModel(stokes,gravity);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinQuadratureUpdateGravityModel(pTatinCtx ptatin)
{
  PhysCompStokes stokes;
  GravityModel   gravity;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinGetGravityModelCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = QuadratureUpdateGravityModel(stokes,gravity);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GravityModelDestroyCtx(GravityModel *gravity)
{
  PetscFunctionBegin;
  if (!*gravity) { PetscFunctionReturn(0); }
  (*gravity)->destroy(*gravity);
  *gravity = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinDestroyGravityModelCtx(pTatinCtx ptatin)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!ptatin->gravity_ctx) { PetscFunctionReturn(0); }
  ierr = GravityModelDestroyCtx(&ptatin->gravity_ctx);CHKERRQ(ierr);
  ptatin->gravity_ctx = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinGetGravityModelCtx(pTatinCtx ptatin, GravityModel *ctx)
{
  GravityModel gravity;

  PetscFunctionBegin;
  gravity = ptatin->gravity_ctx;
  *ctx = gravity;
  PetscFunctionReturn(0);
}

PetscErrorCode GravityCreateTypeCtx(GravityModel gravity)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (gravity->gravity_type)
  {
    case CONSTANT:
      ierr = GravityConstantCreateCtx(gravity);CHKERRQ(ierr);
      break;

    case RADIAL_CONSTANT:
      ierr = GravityRadialConstantCreateCtx(gravity);CHKERRQ(ierr);
      break;
      
    case RADIAL_VAR:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"GravityType RADIAL_VAR is not implemented. Use CONSTANT or RADIAL_CONSTANT instead.");
      break;

    case ARBITRARY:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"GravityType ARBITRARY is not implemented. Use CONSTANT or RADIAL_CONSTANT instead.");
      break;

    case POISSON:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"GravityType POISSON is not implemented. Use CONSTANT or RADIAL_CONSTANT instead.");
      break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No GravityType provided. Possible choices: CONSTANT, RADIAL_CONSTANT.");
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinCreateGravityModel(pTatinCtx ptatin, GravityType gtype)
{
  GravityModel   gravity;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Create GravityModel data structure */
  ierr = GravityModelCreateCtx(&gravity);CHKERRQ(ierr);
  ierr = GravitySetType(gravity,gtype);CHKERRQ(ierr);
  ierr = GravityCreateTypeCtx(gravity);CHKERRQ(ierr);
  
  ptatin->gravity_ctx = gravity;

  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetBodyForcesOnPoint(QPntVolCoefStokes *cell_gausspoints, PetscInt qp_idx)
{
  PetscInt d;
  double   density,Fu[3];
  double   *gravity;
  PetscFunctionBegin;
  /* Get density on quadrature point */
  QPntVolCoefStokesGetField_rho_effective(&cell_gausspoints[qp_idx],&density);
  /* Get gravity on quadrature point */
  QPntVolCoefStokesGetField_gravity_vector(&cell_gausspoints[qp_idx],&gravity); 
  /* Compute rho*g on the quadrature point and attach it to the body force vector */
  for (d=0; d<NSD; d++) {
    //cell_gausspoints[qp_idx].Fu[d] = cell_gausspoints[qp_idx].gravity_vector[d] * cell_gausspoints[qp_idx].rho;
    Fu[d] = density * gravity[d];
  }
  QPntVolCoefStokesSetField_momentum_rhs(&cell_gausspoints[qp_idx],Fu);
  PetscFunctionReturn(0);
}