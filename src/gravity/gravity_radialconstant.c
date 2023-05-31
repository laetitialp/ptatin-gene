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

PetscErrorCode GravityDestroyCtx_RadialConstant(GravityModel gravity)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(gravity->data);CHKERRQ(ierr);
  ierr = PetscFree(gravity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravityScale_RadialConstant(GravityModel gravity, PetscReal scaling_factor)
{
  GravityRadialConstant ctx = (GravityRadialConstant)gravity->data;
  PetscFunctionBegin;
  ctx->magnitude *= scaling_factor;
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySet_RadialConstantMagnitude(GravityRadialConstant gravity, PetscReal magnitude)
{
  PetscFunctionBegin;
  gravity->magnitude = magnitude;
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySet_RadialConstant(GravityModel gravity, void *data)
{
  GravityRadialConstant gc;
  PetscReal             magnitude = *((PetscReal*)data);
  PetscErrorCode        ierr;
  PetscFunctionBegin;

  gc = (GravityRadialConstant)gravity->data;
  ierr = GravitySet_RadialConstantMagnitude(gc,magnitude);CHKERRQ(ierr);
  PetscFunctionReturn(0);

  PetscFunctionReturn(0);
}

static PetscErrorCode GravitySetOnPoint_RadialConstant(PhysCompStokes stokes,
                                                       PetscReal magnitude,
                                                       QPntVolCoefStokes *cell_gausspoints,
                                                       PetscInt qp_idx,
                                                       PetscReal elcoords[])
{
  PetscInt  d,k;
  PetscReal Ni[Q2_NODES_PER_EL_3D],qp_coor[3],position[3],norm;
  double    gvec[3];

  PetscFunctionBegin;
  /* Get quadrature point coordinates */
  for (d=0; d<NSD; d++) {
    qp_coor[d] = stokes->volQ->q_xi_coor[3*qp_idx + d];
  }
  /* Construct Q2 interpolation function */
  pTatin_ConstructNi_Q2_3D( qp_coor, Ni );

  /* Interpolate quadrature point global coords */
  position[0] = position[1] = position[2] = 0.0;
  for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
    for (d=0; d<NSD; d++) {
      position[d] += Ni[k] * elcoords[3*k + d];
    }
  }

  /* Compute the coordinates vector norm */
  norm = 0.0;
  for (d=0; d<NSD; d++) {
    norm += position[d]*position[d];
  }
  norm = PetscSqrtReal(norm);
  for (d=0; d<NSD; d++) {
    if (norm > 1.0e-20) {
      gvec[d] = magnitude * position[d]/norm;
    } else { 
      gvec[d] = 0.0;
    }
  }
  /* Set gvec[] on quadrature points */
  QPntVolCoefStokesSetField_gravity_vector(&cell_gausspoints[qp_idx],gvec);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetGravityModel_RadialConstant(PhysCompStokes stokes, GravityModel gravity)
{
  GravityRadialConstant gc;
  QPntVolCoefStokes     *all_gausspoints,*cell_gausspoints;
  DM                    stokes_pack,dau,dap,cda;
  Vec                   gcoords;
  PetscReal             *LA_gcoords;
  PetscReal             elcoords[3*Q2_NODES_PER_EL_3D],magnitude;
  const PetscInt        *elnidx_u;
  PetscInt              e,q,nel,nqp,nen_u;
  PetscErrorCode        ierr;
  
  PetscFunctionBegin;
  nqp = stokes->volQ->npoints;

  /* Get Stokes DMs */
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  /* Element-nodes connectivity */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);

  gc = (GravityRadialConstant)gravity->data;
  magnitude = gc->magnitude;

  /* Loop over elements */
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /* Get element coordinates */
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);
    /* Loop over quadrature points */
    for (q=0; q<nqp; q++) {
      ierr = GravitySetOnPoint_RadialConstant(stokes,magnitude,cell_gausspoints,q,elcoords);CHKERRQ(ierr);
      /* Set rho*g on quadrature points */
      ierr = QuadratureSetBodyForcesOnPoint(cell_gausspoints,q);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravityGetRadialConstantCtx(GravityModel gravity, GravityRadialConstant *ctx)
{
  GravityRadialConstant gc;
  PetscFunctionBegin;
  gc = (GravityRadialConstant)gravity->data;
  *ctx = gc;
  PetscFunctionReturn(0);
}

PetscErrorCode GravityRadialConstantCreateCtx(GravityModel gravity)
{
  GravityRadialConstant gc;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_GravityRadialConstant),&gc);CHKERRQ(ierr);
  ierr = PetscMemzero(gc,sizeof(struct _p_GravityRadialConstant));CHKERRQ(ierr);
  
  gravity->data = (void*)gc;
  gravity->destroy        = GravityDestroyCtx_RadialConstant;
  gravity->scale          = GravityScale_RadialConstant;
  gravity->set            = GravitySet_RadialConstant;
  gravity->quadrature_set = QuadratureSetGravityModel_RadialConstant;
  gravity->update         = QuadratureSetGravityModel_RadialConstant;

  PetscFunctionReturn(0);
}