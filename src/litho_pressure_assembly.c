/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2012
 **        Dave A. May [dave.may@erdw.ethz.ch]
 **        Institute of Geophysics
 **        ETH Zürich
 **        Sonneggstrasse 5
 **        CH-8092 Zürich
 **        Switzerland
 **
 **    project:    pTatin3d
 **    filename:   litho_pressure_assembly.c
 **
 **
 **    pTatin3d is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published
 **    by the Free Software Foundation, either version 3 of the License,
 **    or (at your option) any later version.
 **
 **    pTatin3d is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 **    See the GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with pTatin3d. If not, see <http://www.gnu.org/licenses/>.
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ @*/
#include "petsc.h"
#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "dmda_bcs.h"
#include "element_utils_q1.h"
#include "dmda_element_q1.h"
#include "quadrature.h"
#include "dmda_checkpoint.h"
#include "data_bucket.h"
#include "dmdae.h"

#include "QPntVolCoefEnergy_def.h"
#include "phys_comp_energy.h"
#include "ptatin3d_energy.h"
#include "litho_pressure_assembly.h"


PetscErrorCode PhysCompCreate_LithoP(PDESolveLithoP *LP)
{
  PetscErrorCode ierr;
  PDESolveLithoP litho_p;

  PetscFunctionBegin;
  *LP = NULL;
  ierr = PetscMalloc(sizeof(struct _p_PDESolveLithoP),&litho_p);CHKERRQ(ierr);
  ierr = PetscMemzero(litho_p,sizeof(struct _p_PDESolveLithoP));CHKERRQ(ierr);
  *LP = litho_p;
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompDestroy_LithoP(PDESolveLithoP *LP)
{
  PetscErrorCode ierr;
  PDESolveLithoP ctx;

  PetscFunctionBegin;

  if (!LP) {PetscFunctionReturn(0);}
  ctx = *LP;

  //  for (e=0; e<HEX_FACES; e++) {
  //    if (ctx->surfQ[e]) { ierr = SurfaceQuadratureDestroy(&ctx->surfQ[e]);CHKERRQ(ierr); }
  //  }
  if (ctx->volQ) { ierr = QuadratureDestroy(&ctx->volQ);CHKERRQ(ierr); }
  if (ctx->LP_bclist) { ierr = BCListDestroy(&ctx->LP_bclist);CHKERRQ(ierr); }
  if (ctx->daLP) {
    ierr = DMDestroyDMDAE(ctx->daLP);CHKERRQ(ierr);
    ierr = DMDestroy(&ctx->daLP);CHKERRQ(ierr);
  }
  if (ctx->F) {      ierr = VecDestroy(&ctx->F);CHKERRQ(ierr); }
  if (ctx->X) {      ierr = VecDestroy(&ctx->X);CHKERRQ(ierr); }
  if (ctx) { ierr = PetscFree(ctx);CHKERRQ(ierr); }

  *LP = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompCreateMesh_LithoP(PDESolveLithoP LP,DM dav,PetscInt mx,PetscInt my, PetscInt mz,PetscInt mesh_generator_type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  LP->LP_mesh_type = mesh_generator_type;

  switch (mesh_generator_type) {
    DMDAE dae;

    case 0:
      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_Energy: Generating standard Q1 DMDA\n");
      LP->mx = mx;
      LP->my = my;
      LP->mz = mz;
      ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, LP->mx+1,LP->my+1,LP->mz+1, PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&dav);CHKERRQ(ierr);
      ierr = DMSetUp(dav);CHKERRQ(ierr);
      LP->daLP = dav;
      ierr = DMAttachDMDAE(LP->daLP);CHKERRQ(ierr);
      ierr = DMDASetElementType_Q1(LP->daLP);CHKERRQ(ierr);
      
      //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only overlapping and nested supported {1,2} ");
      break;

    case 1:
      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_Energy: Generating overlapping Q1 DMDA\n");
      if (!dav) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require valid DM dav");
      }
      ierr = DMDACreateOverlappingQ1FromQ2(dav,1,&LP->daLP);CHKERRQ(ierr);
      ierr = DMGetDMDAE(LP->daLP,&dae);CHKERRQ(ierr);
      LP->mx = dae->mx;
      LP->my = dae->my;
      LP->mz = dae->mz;
      break;

    case 2:
      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_Energy: Generating nested Q1 DMDA\n");
      if (!dav) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require valid DM dav");
      }
      ierr = DMDACreateNestedQ1FromQ2(dav,1,&LP->daLP);CHKERRQ(ierr);
      ierr = DMGetDMDAE(LP->daLP,&dae);CHKERRQ(ierr);
      LP->mx = dae->mx;
      LP->my = dae->my;
      LP->mz = dae->mz;
      break;

    default:
      LP->LP_mesh_type = 1;

      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_Energy: Generating overlapping Q1 DMDA\n");
      if (!dav) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require valid DM dav");
      }
      ierr = DMDACreateOverlappingQ1FromQ2(dav,1,&LP->daLP);CHKERRQ(ierr);
      ierr = DMGetDMDAE(LP->daLP,&dae);CHKERRQ(ierr);
      LP->mx = dae->mx;
      LP->my = dae->my;
      LP->mz = dae->mz;
      break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompCreateBoundaryList_LithoP(PDESolveLithoP LP)
{
  DM daLP;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  daLP = LP->daLP;
  if (!daLP) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"daLP must be set"); }
  ierr = DMDABCListCreate(daLP,&LP->LP_bclist);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompCreateVolumeQuadrature_LithoP(PDESolveLithoP LP)
{
  PetscInt dim, np_per_dim, ncells;
  DMDAE dae;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  np_per_dim = 2;
  dim = 3;

  ierr = DMGetDMDAE(LP->daLP,&dae);CHKERRQ(ierr);
  ncells = dae->lmx * dae->lmy * dae->lmz;
  // For now I am not sure if I need to store the value of rho on qp to use elsewhere so I keep the gauss quadrature
  // from energy since everything is similar except for diffusivity and heat source
  ierr = VolumeQuadratureCreate_GaussLegendreEnergy(dim,np_per_dim,ncells,&LP->volQ);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompNew_LithoP(DM dav,PetscInt mx,PetscInt my, PetscInt mz,PetscInt mesh_generator_type,PDESolveLithoP *LP)
{
  PetscErrorCode  ierr;
  PDESolveLithoP  litho_p;

  PetscFunctionBegin;

  ierr = PhysCompCreate_LithoP(&litho_p);CHKERRQ(ierr);

  ierr = PhysCompCreateMesh_LithoP(litho_p,dav,mx,my,mz,mesh_generator_type);CHKERRQ(ierr);
  ierr = PhysCompCreateBoundaryList_LithoP(litho_p);CHKERRQ(ierr);
  ierr = PhysCompCreateVolumeQuadrature_LithoP(litho_p);CHKERRQ(ierr);

  /* create vectors */
  ierr = DMCreateGlobalVector(litho_p->daLP,&litho_p->F);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(litho_p->daLP,&litho_p->X);CHKERRQ(ierr);

  *LP = litho_p;

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinGetContext_LithoP(pTatinCtx ctx,PDESolveLithoP *LP)
{
  PetscFunctionBegin;
  if (LP) { *LP = ctx->litho_p_ctx; } //TODO: add litho_p in ptatin ctx
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinContextValid_LithoP(pTatinCtx ctx,PetscBool *exists)
{
  PetscFunctionBegin;
  *exists = PETSC_FALSE;
  if (ctx->litho_p_ctx) {
    *exists = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinPhysCompCreate_LithoP(pTatinCtx user)
{
  PetscErrorCode ierr;
  PhysCompStokes stokes_ctx;
  PetscInt lithoP_mesh_type;

  PetscFunctionBegin;
  stokes_ctx = user->stokes_ctx;
  /* create from data */
  lithoP_mesh_type = 1; // default is Q1 overlapping Q2
  ierr = PetscOptionsGetInt(NULL,NULL,"-lithoP_mesh_type",&lithoP_mesh_type,0);CHKERRQ(ierr);
  ierr = PhysCompNew_LithoP(stokes_ctx->dav,-1,-1,-1,lithoP_mesh_type,&user->litho_p_ctx);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinPhysCompActivate_LithoP(pTatinCtx user,PetscBool load)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (load && (user->litho_p_ctx == NULL)) {
    ierr = pTatinPhysCompCreate_LithoP(user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



PetscErrorCode Element_FormFunction_LithoPressure(PetscScalar Re[],
                                                  PetscScalar el_coords[],
                                                  PetscScalar el_phi[],
                                                  PetscScalar gp_rho[],
                                                  PetscInt ngp,
                                                  PetscScalar gp_xi[],
                                                  PetscScalar gp_weight[])
{
  PetscInt    p,i,j,k;
  PetscScalar GNi_p[NSD][NODES_PER_EL_Q1_3D];
  PetscScalar GNx_p[NSD][NODES_PER_EL_Q1_3D];
  PetscScalar gradphi_p[NSD];
  PetscScalar rho_g_qp[3],grav[3];
  PetscScalar J_p,fac;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  // Will be removed in the final version for stokes->gravity_vector
  grav[0] = 0.0; grav[1] = -9.8; grav[2] = 0;
  /* Evaluate integral */
  for (p=0; p<ngp; p++){
    /* Evaluate shape functions local derivatives at current point */
    P3D_ConstructGNi_Q1_3D(&gp_xi[NSD*p],GNi_p);
    /* Evaluate shape functions global derivatives at current point */
    P3D_evaluate_geometry_elementQ1(1,el_coords,&GNi_p,&J_p,&GNx_p[0],&GNx_p[1],&GNx_p[2]);
    /* Numerical integration factor */
    fac = gp_weight[p] * J_p;
    
    /* Compute rho*g */
    for (k=0; k<3; k++){
      rho_g_qp[k] = grav[k]*gp_rho[p];//stokes->gravity_vector[k]*gp_rho[p];
    }
    
    /* We do: Re = A.phi - b 
       with:  A = dN^T.dN 
       and:   b = dN^T.(rho*g)
          A.phi = dN^T.dN.phi
       We first compute dN.phi
    */
    gradphi_p[0] = gradphi_p[1] = gradphi_p[2] = 0.0;
    for (j=0; j<NODES_PER_EL_Q1_3D; j++){
      gradphi_p[0] += GNx_p[0][j] * el_phi[j];
      gradphi_p[1] += GNx_p[1][j] * el_phi[j];
      gradphi_p[2] += GNx_p[2][j] * el_phi[j];
    }
    
    /* Form element stifness matrix */
    for (i=0; i<NODES_PER_EL_Q1_3D; i++){
      Re[i] += fac * (
        // - b: dN^T . rho*g
        -(GNx_p[0][i]*rho_g_qp[0] + GNx_p[1][i]*rho_g_qp[1] + GNx_p[2][i]*rho_g_qp[2])
        // + A: dN^T . dN.phi
        +(GNx_p[0][i]*gradphi_p[0] + GNx_p[1][i]*gradphi_p[1] + GNx_p[2][i]*gradphi_p[2]) );
    }  
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionLocal_LithoPressure(PDESolveLithoP data,DM da,PetscScalar *LA_phi,PetscScalar *LA_R)
{
  DM                cda;
  Vec               gcoords;
  PetscScalar       *LA_gcoords;
  PetscScalar       Re[NODES_PER_EL_Q1_3D];
  PetscScalar       el_coords[NSD*NODES_PER_EL_Q1_3D];
  PetscScalar       el_phi[NODES_PER_EL_Q1_3D];

  PetscInt          nel,nen,e,n;
  const PetscInt    *elnidx;
  PetscInt          ge_eqnums[NODES_PER_EL_Q1_3D];
  PetscInt          nqp;
  PetscScalar       *qp_xi,*qp_weight;
  Quadrature        volQ;
  QPntVolCoefEnergy *all_quadpoints,*cell_quadpoints;
  PetscScalar       qp_rho[NSD];
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  /* Quadrature */
  volQ      = data->volQ;
  nqp       = volQ->npoints;
  qp_xi     = volQ->q_xi_coor;
  qp_weight = volQ->q_weight;

  //ierr = VolumeQuadratureGetAllCellData_Energy(volQ,&all_quadpoints);CHKERRQ(ierr);

  /* Setup for coords */
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* Connectivity table */
  ierr = DMDAGetElementsQ1(da,&nel,&nen,&elnidx);CHKERRQ(ierr);

  for (e=0;e<nel;e++) {
    ierr = DMDAEQ1_GetElementLocalIndicesDOF(ge_eqnums,1,(PetscInt*)&elnidx[nen*e]);CHKERRQ(ierr);
    /* get coords for the element */
    ierr = DMDAEQ1_GetVectorElementField_3D(el_coords,(PetscInt*)&elnidx[nen*e],LA_gcoords);CHKERRQ(ierr);
    /* get value at the element */
    ierr = DMDAEQ1_GetScalarElementField_3D(el_phi,(PetscInt*)&elnidx[nen*e],LA_phi);CHKERRQ(ierr);
    
    // TODO: get rho from stokes qp for lithoP
    //ierr = VolumeQuadratureGetCellData_Energy(volQ,all_quadpoints,e,&cell_quadpoints);CHKERRQ(ierr);

    /* copy the density */
    for (n=0; n<nqp; n++) {
      qp_rho[n] = 1.0; //data->stokes->volQ->...rho           cell_quadpoints[n].rho;
    }

    /* initialise element stiffness matrix */
    ierr = PetscMemzero(Re,sizeof(PetscScalar)*NODES_PER_EL_Q1_3D);CHKERRQ(ierr);
    /* form element stiffness matrix */
    ierr = Element_FormFunction_LithoPressure(Re,el_coords,el_phi,qp_rho,nqp,qp_xi,qp_weight);CHKERRQ(ierr);
    /* Add value to the residue vector */
    ierr = DMDAEQ1_SetValuesLocalStencil_AddValues_DOF(LA_R,1,ge_eqnums,Re);CHKERRQ(ierr);
  }

  /* tidy up local arrays (input) */
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TS_FormFunctionLithoPressure(Vec X,Vec F,void *ctx)
{
  pTatinCtx      ptatin = (pTatinCtx)ctx;
  PDESolveLithoP data;
  PhysCompStokes stokes;
  DM             da;
  Vec            philoc, Fphiloc;
  PetscScalar    *LA_philoc, *LA_Fphiloc;
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  ierr = pTatinGetContext_LithoP(ptatin,&data);CHKERRQ(ierr);
  //ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  da = data->daLP;

  ierr = VecZeroEntries(F);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da,&philoc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Fphiloc);CHKERRQ(ierr);

  /* get local solution */
  ierr = VecZeroEntries(philoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,philoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,philoc);CHKERRQ(ierr);

  /* insert boundary conditions into local vectors */
  ierr = BCListInsertLocal(data->LP_bclist,philoc);CHKERRQ(ierr);

  /* initialise residual */
  ierr = VecZeroEntries(Fphiloc);CHKERRQ(ierr);

  /* get arrays */
  ierr = VecGetArray(philoc,  &LA_philoc);CHKERRQ(ierr);
  ierr = VecGetArray(Fphiloc, &LA_Fphiloc);CHKERRQ(ierr);

  /* ============= */
  /* FORM_FUNCTION */
  ierr = FormFunctionLocal_LithoPressure(data,da,LA_philoc,LA_Fphiloc);CHKERRQ(ierr);
  /* ============= */

  ierr = VecRestoreArray(Fphiloc, &LA_Fphiloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(philoc,  &LA_philoc);CHKERRQ(ierr);

  /* do global fem summation */
  ierr = VecZeroEntries(F);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(da,Fphiloc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (da,Fphiloc,ADD_VALUES,F);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(da,&Fphiloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&philoc);CHKERRQ(ierr);

  /* modify F for the boundary conditions, F_k = scale_k(x_k - phi_k) */
  ierr = BCListResidualDirichlet(data->LP_bclist,X,F);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SNES_FormFunctionLithoPressure(SNES snes,Vec X,Vec F,void *ctx)
{
  pTatinCtx      ptatin = (pTatinCtx)ctx;
  PhysCompStokes stokes;
  PDESolveLithoP lithoP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatinGetContext_LithoP(ptatin,&lithoP);CHKERRQ(ierr);
  //ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = TS_FormFunctionLithoPressure(X,F,ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode Test_Assembly(void)
{
  PetscInt       mx,my,mz;
  PetscInt       mesh_generator_type;
  DM             da = NULL;
  pTatinCtx      pctx = NULL;
  PDESolveLithoP LP = NULL;
  Vec            X = NULL, F = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  ierr = pTatin3dCreateContext(&pctx);CHKERRQ(ierr);
  
  mesh_generator_type = 0;
  mx = my = mz = 4;
  
  ierr = PhysCompNew_LithoP(da,mx,my,mz,mesh_generator_type,&LP);
  pctx->litho_p_ctx = LP;
  ierr = DMDASetUniformCoordinates(LP->daLP,-2.0,2.0,-2.0,2.0,-2.0,2.0);CHKERRQ(ierr);
  
  X = LP->X;
  F = LP->F;  
  ierr = TS_FormFunctionLithoPressure(X,F,pctx);CHKERRQ(ierr);
  
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"X.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    
    sprintf(fname,"F.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(F,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = PhysCompDestroy_LithoP(&LP);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = Test_Assembly();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}