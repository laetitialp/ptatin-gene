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

struct _p_GravityRadialConstant {
  PetscReal magnitude;
};

PetscErrorCode GravityDestroyCtx_RadialConstant(Gravity gravity)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(gravity->data);CHKERRQ(ierr);
  ierr = PetscFree(gravity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravityScale_RadialConstant(Gravity gravity, PetscReal scaling_factor)
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

PetscErrorCode GravitySet_RadialConstant(Gravity gravity, PetscReal magnitude)
{
  GravityRadialConstant gc = NULL;
  PetscErrorCode        ierr;
  PetscFunctionBegin;

  if (gravity->gravity_type != GRAVITY_RADIAL_CONSTANT) { PetscFunctionReturn(0); }

  gc = (GravityRadialConstant)gravity->data;
  ierr = GravitySet_RadialConstantMagnitude(gc,magnitude);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GravityGetPointWiseVector_RadialConstant(Gravity gravity, PetscInt eidx, PetscReal global_coords[], PetscReal local_coords[], PetscReal *gvec[])
{
  GravityRadialConstant gc = NULL;
  PetscInt              d;
  PetscReal             coor_norm;
  PetscErrorCode        ierr;
  PetscFunctionBegin;

  /* Check the type */
  if (gravity->gravity_type != GRAVITY_RADIAL_CONSTANT) { PetscFunctionReturn(0); }

  gc = (GravityRadialConstant)gravity->data;

  coor_norm = 0.0;
  for (d=0; d<NSD; d++) {
    coor_norm += global_coords[d]*global_coords[d];
  }
  coor_norm = PetscSqrtReal(coor_norm);
  for (d=0; d<NSD; d++) {
    if (coor_norm > 1.0e-20) {
      gravity->gvec_ptwise[d] = gc->magnitude * global_coords[d]/coor_norm;
    } else { 
      gravity->gvec_ptwise[d] = 0.0;
    }
  }
  *gvec = gravity->gvec_ptwise;
  PetscFunctionReturn(0);
}

static PetscErrorCode GravitySetOnPoint_RadialConstant(PhysCompStokes stokes,
                                                       Gravity gravity,
                                                       QPntVolCoefStokes *cell_gausspoints,
                                                       PetscInt qp_idx,
                                                       PetscReal elcoords[])
{
  PetscInt       d,k;
  PetscReal      Ni[Q2_NODES_PER_EL_3D],qp_coor[3],position[3];
  PetscReal      *gvec=NULL;
  PetscErrorCode ierr;

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

  ierr = GravityGetPointWiseVector_RadialConstant(gravity,0,position,qp_coor,&gvec);CHKERRQ(ierr);

  /* Set grav on quadrature points */
  QPntVolCoefStokesSetField_gravity_vector(&cell_gausspoints[qp_idx],gvec);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetGravity_RadialConstant(PhysCompStokes stokes, Gravity gravity)
{
  QPntVolCoefStokes     *all_gausspoints,*cell_gausspoints;
  DM                    stokes_pack,dau,dap,cda;
  Vec                   gcoords;
  PetscReal             *LA_gcoords;
  PetscReal             elcoords[3*Q2_NODES_PER_EL_3D];
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

  /* Loop over elements */
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /* Get element coordinates */
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);
    /* Loop over quadrature points */
    for (q=0; q<nqp; q++) {
      ierr = GravitySetOnPoint_RadialConstant(stokes,gravity,cell_gausspoints,q,elcoords);CHKERRQ(ierr);
      /* Set rho*g on quadrature points */
      ierr = QuadratureSetBodyForcesOnPoint(cell_gausspoints,q);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravityGetRadialConstantCtx(Gravity gravity, GravityRadialConstant *ctx)
{
  GravityRadialConstant gc;
  PetscFunctionBegin;
  gc = (GravityRadialConstant)gravity->data;
  *ctx = gc;
  PetscFunctionReturn(0);
}

PetscErrorCode GravityRadialConstantCreateCtx(Gravity gravity)
{
  GravityRadialConstant gc;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_GravityRadialConstant),&gc);CHKERRQ(ierr);
  ierr = PetscMemzero(gc,sizeof(struct _p_GravityRadialConstant));CHKERRQ(ierr);
  
  gravity->data = (void*)gc;
  gravity->destroy        = GravityDestroyCtx_RadialConstant;
  gravity->scale          = GravityScale_RadialConstant;
  gravity->quadrature_set = QuadratureSetGravity_RadialConstant;
  gravity->update         = QuadratureSetGravity_RadialConstant;
  gravity->get_gvec       = GravityGetPointWiseVector_RadialConstant;

  PetscFunctionReturn(0);
}