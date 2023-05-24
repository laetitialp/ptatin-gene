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

PetscErrorCode GravityStokesDMCreateGlobalVector(PhysCompStokes stokes, GravityModel gravity)
{
  DM             stokes_pack,dau,dap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Get Stokes DMs */
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dau,&gravity->gravity_vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravityCreateGlobalVector(PhysCompStokes stokes, GravityModel gravity)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  switch (gravity->gravity_type)
  {
    case CONSTANT:
      gravity->gravity_vec = NULL;
      break;

    case RADIAL_CONSTANT:
      gravity->gravity_vec = NULL;
      break;

    case RADIAL_VAR:
      gravity->gravity_vec = NULL;
      break;

    case ARBITRARY:
      ierr = GravityStokesDMCreateGlobalVector(stokes,gravity);CHKERRQ(ierr);
      break;

    case POISSON:
      ierr = GravityStokesDMCreateGlobalVector(stokes,gravity);CHKERRQ(ierr);
      break;

    default:
      gravity->gravity_vec = NULL;
      break;

  }
  PetscFunctionReturn(0);
}

PetscErrorCode GravityModelDestroyCtx(GravityModel *gravity)
{
  GravityModel   user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!gravity) { PetscFunctionReturn(0); }
  user = *gravity;
  if (!user)    { PetscFunctionReturn(0); }

  if (user->gravity_vec) { ierr = VecDestroy(&user->gravity_vec);CHKERRQ(ierr); }
  if (user)              { ierr = PetscFree(user);CHKERRQ(ierr); }

  *gravity = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinGetContext_GravityModel(pTatinCtx ptatin, GravityModel *gravity)
{
  PetscFunctionBegin;
  if (gravity) { *gravity = ptatin->gravity_ctx; }
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySetType(GravityModel gravity, GravityType gtype)
{
  PetscFunctionBegin;
  gravity->gravity_type = gtype;
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySet_GravityConst(GravityModel gravity, PetscReal gvec[])
{
  PetscInt d;
  PetscFunctionBegin;
  for (d=0; d<NSD; d++) {
    gravity->gravity_const[d] = gvec[d];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySet_GravityMag(GravityModel gravity, PetscReal magnitude)
{
  PetscFunctionBegin;
  gravity->gravity_mag = magnitude;
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySetValues_ConstantVector(GravityModel gravity, void *data)
{
  PetscReal      *gvec = (PetscReal*)data;
  PetscReal      grav[3],magnitude;
  PetscInt       d;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  magnitude = 0.0;
  for (d=0; d<NSD; d++) {
    grav[d] = gvec[d];
    magnitude += grav[d]*grav[d];
  }
  magnitude = PetscSqrtReal(magnitude);
  ierr = GravitySet_GravityConst(gravity,grav);CHKERRQ(ierr);
  ierr = GravitySet_GravityMag(gravity,magnitude);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySetValues_Magnitude(GravityModel gravity, void *data)
{
  PetscReal      *magnitude = (PetscReal*)data;
  PetscReal      mag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mag = magnitude[0];
  ierr = GravitySet_GravityMag(gravity,mag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySetValues(GravityModel gravity, void *data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (gravity->gravity_type)
  {
    case CONSTANT:
      ierr = GravitySetValues_ConstantVector(gravity,data);CHKERRQ(ierr);
      break;

    case RADIAL_CONSTANT:
      ierr = GravitySetValues_Magnitude(gravity,data);CHKERRQ(ierr);
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

/*
Note: 
A choice needs to be done for the scaling factor.
Either:
1 - for consistency with the VecScale() function and the previous implementation
    the components of the gravity vector or its magnitude are MULTIPLIED by the scaling factor.
2 - for consistency with the scaling done in users' models the components of the gravity vector
    or its magnitude are DIVIDED by the scaling factor.
Inluence in users' models:
  Case 1: fac should be passed as: 1.0/fac
  Case 2: fac is passed as it is
To be discussed with Dave. 
*/
PetscErrorCode GravityScale_GravityConst(GravityModel gravity, PetscReal fac)
{
  PetscInt d;
  PetscFunctionBegin;
  for (d=0; d<NSD; d++) {
    gravity->gravity_const[d] = gravity->gravity_const[d] * fac;  
  }
  PetscFunctionReturn(0);
}

PetscErrorCode GravityScale_GravityMag(GravityModel gravity, PetscReal fac)
{
  PetscFunctionBegin;
  gravity->gravity_mag = gravity->gravity_mag * fac;
  PetscFunctionReturn(0);
}

PetscErrorCode GravityScale_GravityVec(GravityModel gravity, PetscReal fac)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScale(gravity->gravity_vec,fac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GravityScale(GravityModel gravity, PetscReal fac)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (gravity->gravity_type)
  {
    case CONSTANT:
      ierr = GravityScale_GravityConst(gravity,fac);CHKERRQ(ierr);
      ierr = GravityScale_GravityMag(gravity,fac);CHKERRQ(ierr);
      break;

    case RADIAL_CONSTANT:
      ierr = GravityScale_GravityMag(gravity,fac);CHKERRQ(ierr);
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

/*
pTatinCreateGravityModel() only creates the data structure and assign values depending on the gravity_type
It DOES NOT assign values to quadrature points
Use pTatin_ApplyInitialStokesBodyForcesModel() to set gravity and body forces on quadrature points
Use pTatin_UpdateStokesBodyForcesModel() to update gravity and body forces on quadrature
*/
PetscErrorCode pTatinCreateGravityModel(pTatinCtx ptatin, GravityType gtype, PetscReal scaling_factor, void *data)
{
  GravityModel   gravity;
  PhysCompStokes stokes;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Get Stokes data structure */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  /* Create GravityModel data structure */
  ierr = GravityModelCreateCtx(&gravity);CHKERRQ(ierr);
  ierr = GravitySetType(gravity,gtype);CHKERRQ(ierr);
  /* Create the Vec gravity_vec, set to NULL for case CONSTANT and default */
  ierr = GravityCreateGlobalVector(stokes,gravity);CHKERRQ(ierr);
  ierr = GravitySetValues(gravity,data);CHKERRQ(ierr);
  ierr = GravityScale(gravity,scaling_factor);CHKERRQ(ierr);

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

PetscErrorCode QuadratureSetGravityModel_GravityConstant(PhysCompStokes stokes, QPntVolCoefStokes *all_gausspoints, GravityModel gravity, PetscBool body_forces)
{
  QPntVolCoefStokes *cell_gausspoints;
  PetscInt          e,q,d,nel,nqp;
  double            gvec[3];
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  nel = stokes->volQ->n_elements;
  nqp = stokes->volQ->npoints;

  for (d=0; d<NSD; d++) {
    gvec[d] = (double)gravity->gravity_const[d];
  }

  /* Loop over elements */
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /* Loop over quadrature points */
    for (q=0; q<nqp; q++) {
      /* Set grav[] on quadrature points */
      QPntVolCoefStokesSetField_gravity_vector(&cell_gausspoints[q],gvec);
      if (body_forces) {
        /* Set rho*g on quadrature points */
        ierr = QuadratureSetBodyForcesOnPoint(cell_gausspoints,q);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySetOnPoint_RadialConstant(PhysCompStokes stokes,
                                                GravityModel gravity,
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
      gvec[d] = gravity->gravity_mag * position[d]/norm;
    } else { 
      gvec[d] = 0.0;
    }
  }
  /* Set gvec[] on quadrature points */
  QPntVolCoefStokesSetField_gravity_vector(&cell_gausspoints[qp_idx],gvec);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetGravityModel_GravityRadialConstant(PhysCompStokes stokes, QPntVolCoefStokes *all_gausspoints, GravityModel gravity, PetscBool body_forces)
{
  QPntVolCoefStokes *cell_gausspoints;
  DM                stokes_pack,dau,dap,cda;
  Vec               gcoords;
  PetscReal         *LA_gcoords;
  PetscReal         elcoords[3*Q2_NODES_PER_EL_3D];
  const PetscInt    *elnidx_u;
  PetscInt          e,q,nel,nqp,nen_u;
  PetscErrorCode    ierr;

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

  /* Loop over elements */
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /* Get element coordinates */
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);
    /* Loop over quadrature points */
    for (q=0; q<nqp; q++) {
      ierr = GravitySetOnPoint_RadialConstant(stokes,gravity,cell_gausspoints,q,elcoords);CHKERRQ(ierr);
      if (body_forces) {
        /* Set rho*g on quadrature points */
        ierr = QuadratureSetBodyForcesOnPoint(cell_gausspoints,q);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
The 2 following functions:
GravityVecInterpolateToQuadraturePoint()
QuadratureSetGravityModel_FromPetscVec()
are currently not used but are a support for a gravity stored on a petsc Vec object
like if this was the solution of a PDE 
*/
PetscErrorCode GravityVecInterpolateToQuadraturePoint(PhysCompStokes stokes,
                                                      QPntVolCoefStokes *cell_gausspoints,
                                                      PetscInt nqp,
                                                      PetscInt cell_idx,
                                                      const PetscInt *elnidx_u,
                                                      PetscInt nen,
                                                      PetscScalar gvec[],
                                                      PetscBool body_forces)
{
  PetscInt       q,k,d;
  PetscReal      Ni[Q2_NODES_PER_EL_3D];
  PetscScalar    el_grav[3*Q2_NODES_PER_EL_3D];
  double         qp_coor[3],grav[3];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMDAGetVectorElementFieldQ2_3D(el_grav,(PetscInt*)&elnidx_u[nen*cell_idx],gvec);CHKERRQ(ierr);
  for (q=0; q<nqp; q++) {
    /* Get quadrature point coordinates */
    for (d=0; d<NSD; d++) {
      qp_coor[d] = stokes->volQ->q_xi_coor[3*q + d];
    }
    /* Construct Q2 interpolation function */
    pTatin_ConstructNi_Q2_3D( qp_coor, Ni );
    /* Initialize to 0.0 */
    grav[0] = grav[1] = grav[2] = 0.0;
    /* Inteprolate to quadrature point */
    for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
      for (d=0; d<NSD; d++) {
        grav[d] += Ni[k] * el_grav[3*k + d];
      }
    }
    /* Set gravity on quadrature point */
    QPntVolCoefStokesSetField_gravity_vector(&cell_gausspoints[q],grav);
    if (body_forces) {
      /* Set rho*g on quadrature point */
      ierr = QuadratureSetBodyForcesOnPoint(cell_gausspoints,q);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetGravityModel_FromPetscVec(PhysCompStokes stokes, QPntVolCoefStokes *all_gausspoints, GravityModel gravity, PetscBool body_forces)
{
  QPntVolCoefStokes *cell_gausspoints;
  DM                stokes_pack,dau,dap;
  const PetscInt    *elnidx_u;
  PetscInt          e,nel,nen_u;
  PetscScalar       *gravity_vector;
  PetscErrorCode    ierr;

  /* Get Stokes DMs */
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);

  /* Element-nodes connectivity */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

  /* Get gravity vector */
  ierr = VecGetArray(gravity->gravity_vec,&gravity_vector);CHKERRQ(ierr);
  for (e=0; e<nel; e++) {
    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    ierr = GravityVecInterpolateToQuadraturePoint(stokes,cell_gausspoints,stokes->volQ->npoints,e,elnidx_u,nen_u,gravity_vector,body_forces);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(gravity->gravity_vec,&gravity_vector);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetGravityModel(PhysCompStokes stokes, QPntVolCoefStokes *all_gausspoints, GravityModel gravity, PetscBool body_forces)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (gravity->gravity_type) {
    case CONSTANT:
      ierr = QuadratureSetGravityModel_GravityConstant(stokes,all_gausspoints,gravity,body_forces);CHKERRQ(ierr);
      break;

    case RADIAL_CONSTANT:
      ierr = QuadratureSetGravityModel_GravityRadialConstant(stokes,all_gausspoints,gravity,body_forces);CHKERRQ(ierr);
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

PetscErrorCode GravityModelUpdateQuadraturePoints(PhysCompStokes stokes, QPntVolCoefStokes *all_gausspoints, GravityModel gravity, PetscBool body_forces)
{
  PetscErrorCode ierr;

  switch (gravity->gravity_type) {
    case CONSTANT:
      /* Nothing to do, it is constant */
      break;

    case RADIAL_CONSTANT:
      /* Re-evaluate the radial gravity after mesh nodes advection */
      ierr = QuadratureSetGravityModel_GravityRadialConstant(stokes,all_gausspoints,gravity,body_forces);CHKERRQ(ierr);
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

/* 
Set the body forces = gravity * density on quadrature points.
Because the same procedure needs to be applied for both the gravity and the body forces 
I set a boolean to control wether we want to only set the gravity or both gravity and body forces.

pTatin_ApplyInitialStokesBodyForcesModel() ==> computes both gravity and body forces on quadrature points
pTatin_ApplyInitialStokesGravityModel()    ==> computes only gravity on quadrature points
*/
PetscErrorCode pTatin_ApplyInitialStokesBodyForcesModel(pTatinCtx ptatin)
{
  PhysCompStokes    stokes;
  GravityModel      gravity;
  QPntVolCoefStokes *all_gausspoints;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Get Stokes data structure */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  /* Get Gravity data structure */
  ierr = pTatinGetContext_GravityModel(ptatin,&gravity);CHKERRQ(ierr);
  /* Get quadrature points data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);
  /* Apply rho*g on quadrature points */
  ierr = QuadratureSetGravityModel(stokes,all_gausspoints,gravity,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin_ApplyInitialStokesGravityModel(pTatinCtx ptatin)
{
  PhysCompStokes    stokes;
  GravityModel      gravity;
  QPntVolCoefStokes *all_gausspoints;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Get Stokes data structure */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  /* Get Gravity data structure */
  ierr = pTatinGetContext_GravityModel(ptatin,&gravity);CHKERRQ(ierr);
  /* Get quadrature points data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);
  /* Apply rho*g on quadrature points */
  ierr = QuadratureSetGravityModel(stokes,all_gausspoints,gravity,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin_UpdateStokesBodyForcesModel(pTatinCtx ptatin)
{
  PhysCompStokes    stokes;
  GravityModel      gravity;
  QPntVolCoefStokes *all_gausspoints;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Get Stokes data structure */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  /* Get Gravity data structure */
  ierr = pTatinGetContext_GravityModel(ptatin,&gravity);CHKERRQ(ierr);
  /* Get quadrature points data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);
  /* update rho*g on quadrature points */
  ierr = GravityModelUpdateQuadraturePoints(stokes,all_gausspoints,gravity,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin_UpdateStokesGravityModel(pTatinCtx ptatin)
{
  PhysCompStokes    stokes;
  GravityModel      gravity;
  QPntVolCoefStokes *all_gausspoints;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Get Stokes data structure */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  /* Get Gravity data structure */
  ierr = pTatinGetContext_GravityModel(ptatin,&gravity);CHKERRQ(ierr);
  /* Get quadrature points data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);
  /* Apply rho*g on quadrature points */
  ierr = GravityModelUpdateQuadraturePoints(stokes,all_gausspoints,gravity,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}