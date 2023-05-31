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

PetscErrorCode GravityDestroyCtx_Constant(GravityModel gravity)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(gravity->data);CHKERRQ(ierr);
  ierr = PetscFree(gravity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravityScale_Constant(GravityModel gravity, PetscReal scaling_factor)
{
  GravityConstant ctx = (GravityConstant)gravity->data;
  PetscInt        d;
  PetscFunctionBegin;
  ctx->magnitude *= scaling_factor;
  for (d=0; d<NSD; d++) {
    ctx->gravity_const[d] *= scaling_factor; 
  }
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySet_ConstantVector(GravityConstant gravity, PetscReal gvec[])
{
  PetscInt d;
  PetscFunctionBegin;
  for (d=0; d<NSD; d++) {
    gravity->gravity_const[d] = gvec[d];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySet_ConstantMagnitude(GravityConstant gravity, PetscReal magnitude)
{
  PetscFunctionBegin;
  gravity->magnitude = magnitude;
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySet_Constant(GravityModel gravity, void *data)
{
  GravityConstant gc;
  PetscReal       *gvec = (PetscReal*)data;
  PetscReal       grav[3],magnitude;
  PetscInt        d;
  PetscErrorCode  ierr;
  PetscFunctionBegin;

  gc = (GravityConstant)gravity->data;

  magnitude = 0.0;
  for (d=0; d<NSD; d++) {
    grav[d] = gvec[d];
    magnitude += grav[d]*grav[d];
  }
  magnitude = PetscSqrtReal(magnitude);
  ierr = GravitySet_ConstantVector(gc,grav);CHKERRQ(ierr);
  ierr = GravitySet_ConstantMagnitude(gc,magnitude);CHKERRQ(ierr);
  PetscFunctionReturn(0);

  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetGravityModel_Constant(PhysCompStokes stokes, GravityModel gravity)
{
  GravityConstant   gc;
  QPntVolCoefStokes *all_gausspoints,*cell_gausspoints;
  PetscInt          e,q,d,nel,nqp;
  double            gvec[3];
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  nel = stokes->volQ->n_elements;
  nqp = stokes->volQ->npoints;

  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);
  gc = (GravityConstant)gravity->data;

  for (d=0; d<NSD; d++) {
    gvec[d] = (double)gc->gravity_const[d];
  }

  /* Loop over elements */
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /* Loop over quadrature points */
    for (q=0; q<nqp; q++) {
      /* Set gvec[] on quadrature points */
      QPntVolCoefStokesSetField_gravity_vector(&cell_gausspoints[q],gvec);
      /* Set rho*g on quadrature points */
      ierr = QuadratureSetBodyForcesOnPoint(cell_gausspoints,q);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* We only need to modify the body forces because of the density, but the gravity is constant */
PetscErrorCode QuadratureUpdateGravityModel_Constant(PhysCompStokes stokes, GravityModel gravity)
{
  QPntVolCoefStokes *all_gausspoints,*cell_gausspoints;
  PetscInt          e,q,nel,nqp;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  nel = stokes->volQ->n_elements;
  nqp = stokes->volQ->npoints;

  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);

  /* Loop over elements */
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /* Loop over quadrature points */
    for (q=0; q<nqp; q++) {
      /* Set rho*g on quadrature points */
      ierr = QuadratureSetBodyForcesOnPoint(cell_gausspoints,q);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode GravityGetConstantCtx(GravityModel gravity, GravityConstant *ctx)
{
  GravityConstant gc;
  PetscFunctionBegin;
  gc = (GravityConstant)gravity->data;
  *ctx = gc;
  PetscFunctionReturn(0);
}

PetscErrorCode GravityConstantCreateCtx(GravityModel gravity)
{
  PetscErrorCode  ierr;
  GravityConstant gc;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_GravityConstant),&gc);CHKERRQ(ierr);
  ierr = PetscMemzero(gc,sizeof(struct _p_GravityConstant));CHKERRQ(ierr);
  
  gravity->data = (void*)gc;
  gravity->destroy        = GravityDestroyCtx_Constant;
  gravity->scale          = GravityScale_Constant;
  gravity->set            = GravitySet_Constant;
  gravity->quadrature_set = QuadratureSetGravityModel_Constant;
  gravity->update         = QuadratureUpdateGravityModel_Constant;

  PetscFunctionReturn(0);
}