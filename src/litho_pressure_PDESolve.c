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
 **    filename:   litho_pressure_PDESolve.c
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
#include "element_utils_q2.h"
#include "dmda_element_q1.h"
#include "dmda_element_q2p1.h"
#include "quadrature.h"
#include "dmda_checkpoint.h"
#include "data_bucket.h"
#include "dmdae.h"
#include "fvda_private.h"
#include "fvda.h"
#include "ptatin_log.h"
#include "QPntVolCoefEnergy_def.h"
#include "phys_comp_energy.h"
#include "ptatin3d_energy.h"
#include "litho_pressure_PDESolve.h"

/* defined in ptatin3d_energyfv.c */
PetscErrorCode dmda3d_create_q1_from_element_partition(MPI_Comm comm,PetscInt bs,PetscInt target_decomp[],const PetscInt m[],DM *dm);

void evaluate_cell_geometry_pointwise_3d(const PetscReal el_coords[3*DACELL3D_Q1_SIZE],
                                         PetscReal GNI[3][DACELL3D_Q1_SIZE],
                                         PetscReal *detJ,
                                         PetscReal dNudx[DACELL3D_Q1_SIZE],
                                         PetscReal dNudy[DACELL3D_Q1_SIZE],
                                         PetscReal dNudz[DACELL3D_Q1_SIZE])
{
  PetscInt  k;
  PetscReal J[3][3];
  PetscReal iJ[3][3];
  PetscReal t4, t6, t8, t10, t12, t14, t17;
  
  J[0][0] = J[0][1] = J[0][2] = 0.0;
  J[1][0] = J[1][1] = J[1][2] = 0.0;
  J[2][0] = J[2][1] = J[2][2] = 0.0;
  
  for (k=0; k<NODES_PER_EL_Q1_3D; k++) {
    PetscReal xc = el_coords[3*k+0];
    PetscReal yc = el_coords[3*k+1];
    PetscReal zc = el_coords[3*k+2];
    
    J[0][0] += GNI[0][k] * xc;
    J[0][1] += GNI[0][k] * yc;
    J[0][2] += GNI[0][k] * zc;
    
    J[1][0] += GNI[1][k] * xc;
    J[1][1] += GNI[1][k] * yc;
    J[1][2] += GNI[1][k] * zc;
    
    J[2][0] += GNI[2][k] * xc;
    J[2][1] += GNI[2][k] * yc;
    J[2][2] += GNI[2][k] * zc;
  }
  
  *detJ = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
        - J[0][1]*(J[1][0]*J[2][2] + J[1][2]*J[2][0])
        + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);
        
  t4  = J[2][0] * J[0][1];
  t6  = J[2][0] * J[0][2];
  t8  = J[1][0] * J[0][1];
  t10 = J[1][0] * J[0][2];
  t12 = J[0][0] * J[1][1];
  t14 = J[0][0] * J[1][2]; // 6
  t17 = 0.1e1 / (t4 * J[1][2] - t6 * J[1][1] - t8 * J[2][2] + t10 * J[2][1] + t12 * J[2][2] - t14 * J[2][1]);  // 12

  iJ[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * t17;  // 4
  iJ[0][1] = -(J[0][1] * J[2][2] - J[0][2] * J[2][1]) * t17; // 5
  iJ[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * t17;  // 4
  iJ[1][0] = -(-J[2][0] * J[1][2] + J[1][0] * J[2][2]) * t17;// 6
  iJ[1][1] = (-t6 + J[0][0] * J[2][2]) * t17;                // 4
  iJ[1][2] = -(-t10 + t14) * t17;                            // 4
  iJ[2][0] = (-J[2][0] * J[1][1] + J[1][0] * J[2][1]) * t17; // 5
  iJ[2][1] = -(-t4 + J[0][0] * J[2][1]) * t17;               // 5
  iJ[2][2] = (-t8 + t12) * t17;                              // 3
  /* flops = [NQP] * 58 */

  /* shape function derivatives */
  for (k=0; k<NODES_PER_EL_Q1_3D; k++) {
    dNudx[k] = iJ[0][0]*GNI[0][k] + iJ[0][1]*GNI[1][k] + iJ[0][2]*GNI[2][k];

    dNudy[k] = iJ[1][0]*GNI[0][k] + iJ[1][1]*GNI[1][k] + iJ[1][2]*GNI[2][k];

    dNudz[k] = iJ[2][0]*GNI[0][k] + iJ[2][1]*GNI[1][k] + iJ[2][2]*GNI[2][k];
  }
}

PetscErrorCode PhysCompCreate_LithoP(PDESolveLithoP *LP)
{
  PetscErrorCode ierr;
  PDESolveLithoP litho_p;

  PetscFunctionBegin;
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
  if (!LP) { PetscFunctionReturn(0); }
  ctx = *LP;

  if (ctx->volQ) { ierr = QuadratureDestroy(&ctx->volQ);CHKERRQ(ierr); }
  if (ctx->bclist) { ierr = BCListDestroy(&ctx->bclist);CHKERRQ(ierr); }
  ierr = DMDestroy(&ctx->da);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->F);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->X);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);

  *LP = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompCreateMesh_LithoP(PDESolveLithoP LP,DM dav,PetscInt mx,PetscInt my, PetscInt mz,PetscInt mesh_generator_type)
{
  PetscInt       MX,MY,MZ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  LP->mesh_type = mesh_generator_type;

  switch (mesh_generator_type) {

    case 0:
      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_LithoP: Generating standard Q1 DMDA\n");
      LP->mx = mx;
      LP->my = my;
      LP->mz = mz;
      ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, LP->mx+1,LP->my+1,LP->mz+1, PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&dav);CHKERRQ(ierr);
      ierr = DMSetUp(dav);CHKERRQ(ierr);
      LP->da = dav;
      ierr = DMDASetElementType(LP->da,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
      
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only overlapping Q1 From Q2 mesh supported {1} ");
      break;

    case 1:
      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_LithoP: Generating overlapping Q1 DMDA\n");
      if (!dav) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require valid DM dav");
      }

      {
        PetscInt q2_mi[]={0,0,0},decomp[]={0,0,0};
        
        ierr = DMDAGetLocalSizeElementQ2(dav,&q2_mi[0],&q2_mi[1],&q2_mi[2]);CHKERRQ(ierr);
        ierr = DMDAGetInfo(dav,NULL,NULL,NULL,NULL,&decomp[0],&decomp[1],&decomp[2],NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

        ierr = dmda3d_create_q1_from_element_partition(PETSC_COMM_WORLD,1,decomp,q2_mi,&LP->da);CHKERRQ(ierr);
      }
      ierr = DMDAProjectCoordinatesQ2toOverlappingQ1_3d(dav,LP->da);CHKERRQ(ierr);
      
      /* Push back the element number info into LP struct */
      ierr = DMDAGetInfo(LP->da,0, &MX,&MY,&MZ,0,0,0,0,0, 0,0,0, 0);CHKERRQ(ierr);
      LP->mx = MX-1;
      LP->my = MY-1;
      LP->mz = MZ-1;
      break;

    default:
      LP->mesh_type = 1;

      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_LithoP: Generating overlapping Q1 DMDA\n");
      if (!dav) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require valid DM dav");
      }

      {
        PetscInt q2_mi[]={0,0,0},decomp[]={0,0,0};
        
        ierr = DMDAGetLocalSizeElementQ2(dav,&q2_mi[0],&q2_mi[1],&q2_mi[2]);CHKERRQ(ierr);
        ierr = DMDAGetInfo(dav,NULL,NULL,NULL,NULL,&decomp[0],&decomp[1],&decomp[2],NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
        
        ierr = dmda3d_create_q1_from_element_partition(PETSC_COMM_WORLD,1,decomp,q2_mi,&LP->da);CHKERRQ(ierr);
      }
      ierr = DMDAProjectCoordinatesQ2toOverlappingQ1_3d(dav,LP->da);CHKERRQ(ierr);
      
      /* Push back the element number info into LP struct */
      ierr = DMDAGetInfo(LP->da,0, &MX,&MY,&MZ,0,0,0,0,0, 0,0,0, 0);CHKERRQ(ierr);
      LP->mx = MX-1;
      LP->my = MY-1;
      LP->mz = MZ-1;
      break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompCreateBoundaryList_LithoP(PDESolveLithoP LP)
{
  DM             da;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  da = LP->da;
  if (!da) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"da must be set"); }
  ierr = DMDABCListCreate(da,&LP->bclist);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode VolumeQuadratureCreate_GaussLegendreLithoP(PetscInt nsd,PetscInt np_per_dim,Quadrature *quadrature)
{
  Quadrature     Q;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = QuadratureCreate(&Q);CHKERRQ(ierr);
  Q->dim  = nsd;
  Q->type = VOLUME_QUAD;

  switch (np_per_dim) {
    case 1:
      /*PetscPrintf(PETSC_COMM_WORLD,"\tUsing 1 pnt Gauss Legendre quadrature\n");*/
      //QuadratureCreateGauss_1pnt_3D(&ngp,gp_xi,gp_weight);
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"This will result in a rank-deficient operator");
      break;

    case 2:
      /*PetscPrintf(PETSC_COMM_WORLD,"\tUsing 2x2 pnt Gauss Legendre quadrature\n");*/
      QuadratureCreateGauss_2pnt_3D(&Q->npoints,&Q->q_xi_coor,&Q->q_weight);
      break;

    case 3:
      /*PetscPrintf(PETSC_COMM_WORLD,"\tUsing 3x3 pnt Gauss Legendre quadrature\n");*/
      QuadratureCreateGauss_3pnt_3D(&Q->npoints,&Q->q_xi_coor,&Q->q_weight);
      break;

    default:
      /*PetscPrintf(PETSC_COMM_WORLD,"\tUsing 3x3 pnt Gauss Legendre quadrature\n");*/
      QuadratureCreateGauss_3pnt_3D(&Q->npoints,&Q->q_xi_coor,&Q->q_weight);
      break;
  }

  *quadrature = Q;
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompCreateVolumeQuadrature_LithoP(PDESolveLithoP LP)
{
  PetscInt dim, np_per_dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  np_per_dim = 3;
  dim = 3;

  ierr = VolumeQuadratureCreate_GaussLegendreLithoP(dim,np_per_dim,&LP->volQ);CHKERRQ(ierr);

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
  ierr = DMCreateGlobalVector(litho_p->da,&litho_p->F);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(litho_p->da,&litho_p->X);CHKERRQ(ierr);

  *LP = litho_p;

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinGetContext_LithoP(pTatinCtx ctx,PDESolveLithoP *LP)
{
  PetscFunctionBegin;
  if (LP) { *LP = ctx->litho_p_ctx; }
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
  PDESolveLithoP LP;
  PetscInt       lithoP_mesh_type;

  PetscFunctionBegin;
  stokes_ctx = user->stokes_ctx;
  /* create from data */
  lithoP_mesh_type = 1; // default is Q1 overlapping Q2
  ierr = PetscOptionsGetInt(NULL,NULL,"-lithoP_mesh_type",&lithoP_mesh_type,0);CHKERRQ(ierr);
  ierr = PhysCompNew_LithoP(stokes_ctx->dav,-1,-1,-1,lithoP_mesh_type,&LP);CHKERRQ(ierr);
  LP->stokes = stokes_ctx;
  
  if (stokes_ctx->volQ->npoints != LP->volQ->npoints) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Integration scheme is inconsistent");
  
  {
    PetscInt       nelQ2,nenQ2,nel,nen;
    const PetscInt *elnidxQ2,*elnidx;
    
    ierr = DMDAGetElements_pTatinQ2P1(LP->stokes->dav,&nelQ2,&nenQ2,&elnidxQ2);CHKERRQ(ierr);
    ierr = DMDAGetElements(LP->da,&nel,&nen,&elnidx);CHKERRQ(ierr);
    if (nelQ2 != nel) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nelQ2 != nel");
  }
  
  user->litho_p_ctx = LP;
  
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

PetscErrorCode Element_FormJacobian_LithoPressure(PetscScalar Re[],
                                                  PetscScalar el_coords[],
                                                  PetscInt ngp,
                                                  PetscScalar gp_xi[],
                                                  PetscScalar gp_weight[])
{
  PetscInt    p,i,j;
  PetscReal   GNi_p[NSD][NODES_PER_EL_Q1_3D];
  PetscReal   GNx_p[NSD][NODES_PER_EL_Q1_3D];
  PetscScalar J_p,fac;
  
  PetscFunctionBegin;
  /* Evaluate integral */
  for (p=0; p<ngp; p++){
    /* Evaluate shape functions local derivatives at current point */
    EvaluateBasisDerivative_Q1_3D(&gp_xi[NSD*p],GNi_p);
    /* Evaluate shape functions global derivatives at current point */
    evaluate_cell_geometry_pointwise_3d(el_coords,GNi_p,&J_p,GNx_p[0],GNx_p[1],GNx_p[2]);
    /* Numerical integration factor */
    fac = gp_weight[p] * J_p;

    /* We have: F = A(phi^(k+1)).P^(k+1) - b(phi^(k+1)) 
       with:    A = dN^T.dN 
       and:     b = dN^T.(rho*g)
       The Jacobian matrix is:
       J = dF/dP^(k+1) = A(phi^(k+1))
    */
    /* Form element stifness matrix */
    for (i=0; i<NODES_PER_EL_Q1_3D; i++){
      for (j=0; j<NODES_PER_EL_Q1_3D; j++){
        Re[j+i*NODES_PER_EL_Q1_3D] += fac * ((GNx_p[0][i]*GNx_p[0][j] + GNx_p[1][i]*GNx_p[1][j] + GNx_p[2][i]*GNx_p[2][j]));
      }
    }  
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianLithoPressure(Vec X,Mat A,Mat B,void *ctx)
{
  pTatinCtx              ptatin = (pTatinCtx)ctx;
  PDESolveLithoP         data;
  PetscInt               nqp;
  PetscScalar            *qp_xi,*qp_weight;
  Quadrature             volQ;
  DM                     da,cda;
  Vec                    gcoords;
  PetscScalar            *LA_gcoords;
  PetscScalar            ADe[NODES_PER_EL_Q1_3D*NODES_PER_EL_Q1_3D];
  PetscScalar            el_coords[NSD*NODES_PER_EL_Q1_3D];
  PetscInt               nel,nen,e,ii;
  const PetscInt         *elnidx;
  BCList                 bclist;
  ISLocalToGlobalMapping ltog;
  PetscInt               NUM_GINDICES,el_lidx[Q1_NODES_PER_EL_3D],ge_eqnums[Q1_NODES_PER_EL_3D];
  const PetscInt         *GINDICES;
  Vec                    local_X;
  PetscScalar            *LA_X;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = pTatinGetContext_LithoP(ptatin,&data);CHKERRQ(ierr);
  da     = data->da;
  bclist = data->bclist;
  volQ   = data->volQ;
    
  /* setup for coords */
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* get solution */
  ierr = DMGetLocalVector(da,&local_X);CHKERRQ(ierr);
  ierr = VecZeroEntries(local_X);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,local_X);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,local_X);CHKERRQ(ierr);
  ierr = VecGetArray(local_X,&LA_X);CHKERRQ(ierr);

  /* trash old entries */
  ierr = MatZeroEntries(B);CHKERRQ(ierr);

  /* quadrature */
  volQ      = data->volQ;
  nqp       = volQ->npoints;
  qp_xi     = volQ->q_xi_coor;
  qp_weight = volQ->q_weight;

  /* stuff for eqnums */
  ierr = DMGetLocalToGlobalMapping(da, &ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES);CHKERRQ(ierr);
  ierr = BCListApplyDirichletMask(NUM_GINDICES,(PetscInt*)GINDICES,bclist);CHKERRQ(ierr);

  ierr = DMDAGetElements(da,&nel,&nen,&elnidx);CHKERRQ(ierr);

  /* Volume integral */
  for (e=0; e<nel; e++) {
    /* get coords for the element */
    ierr = DMDAEQ1_GetVectorElementField_3D(el_coords,(PetscInt*)&elnidx[nen*e],LA_gcoords);CHKERRQ(ierr);

    /* initialise element stiffness matrix */
    ierr = PetscMemzero(ADe,sizeof(PetscScalar)*NODES_PER_EL_Q1_3D*NODES_PER_EL_Q1_3D);CHKERRQ(ierr);
    /* form element stiffness matrix */
    ierr = Element_FormJacobian_LithoPressure(ADe,el_coords,nqp,qp_xi,qp_weight);CHKERRQ(ierr);

    /* get indices */
    ierr = DMDAEQ1_GetElementLocalIndicesDOF(el_lidx,1,(PetscInt*)&elnidx[nen*e]);CHKERRQ(ierr);
    for (ii=0; ii<NODES_PER_EL_Q1_3D; ii++) {
      const PetscInt NID = elnidx[ NODES_PER_EL_Q1_3D * e + ii ];
      ge_eqnums[ii] = GINDICES[ NID ];
    }

    /* insert element matrix into global matrix */
    ierr = MatSetValues(B,NODES_PER_EL_Q1_3D,ge_eqnums, NODES_PER_EL_Q1_3D,ge_eqnums, ADe, ADD_VALUES );CHKERRQ(ierr);
  }
  
  /* Surface integral */
  /* not implemented, can work around this using -snes_fd_color */
  
  /* tidy up */
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = VecRestoreArray(local_X,&LA_X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&local_X);CHKERRQ(ierr);

  /* partial assembly */
  ierr = MatAssemblyBegin(B, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);

  /* boundary conditions */
  ierr = BCListRemoveDirichletMask(NUM_GINDICES,(PetscInt*)GINDICES,bclist);CHKERRQ(ierr);
  ierr = BCListInsertScaling(B,NUM_GINDICES,(PetscInt*)GINDICES,bclist);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog, &GINDICES);CHKERRQ(ierr);

  /* assemble */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNES_FormJacobianLithoPressure(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  pTatinCtx      ptatin  = (pTatinCtx)ctx;
  PDESolveLithoP litho_p = (PDESolveLithoP)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatinGetContext_LithoP(ptatin,&litho_p);CHKERRQ(ierr);
  ierr = FormJacobianLithoPressure(X,A,B,ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode Element_FormFunction_LithoPressure(PetscScalar Re[],
                                                  PetscScalar el_coords[],
                                                  PetscScalar el_phi[],
                                                  PetscScalar gp_rho[],
                                                  PetscInt ngp,
                                                  PetscScalar gp_xi[],
                                                  PetscScalar gp_weight[],
                                                  PDESolveLithoP data)
{
  PetscInt    p,i,j,k;
  PetscReal   GNi_p[NSD][NODES_PER_EL_Q1_3D];
  PetscReal   GNx_p[NSD][NODES_PER_EL_Q1_3D];
  PetscScalar gradphi_p[NSD];
  PetscScalar rho_g_qp[NSD];
  PetscScalar J_p,fac;
  
  PetscFunctionBegin;
  /* Evaluate integral */
  for (p=0; p<ngp; p++){
    /* Evaluate shape functions local derivatives at current point */
    EvaluateBasisDerivative_Q1_3D(&gp_xi[NSD*p],GNi_p);
    /* Evaluate shape functions global derivatives at current point */
    evaluate_cell_geometry_pointwise_3d(el_coords,GNi_p,&J_p,GNx_p[0],GNx_p[1],GNx_p[2]);
    /* Numerical integration factor */
    fac = gp_weight[p] * J_p;
    
    /* Compute rho*g */
    for (k=0; k<NSD; k++){
      rho_g_qp[k] = data->stokes->gravity_vector[k]*gp_rho[p];
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

PetscErrorCode FormFunctionLocal_LithoPressure_dV(PDESolveLithoP data,DM da,PetscScalar *LA_phi,PetscScalar *LA_R)
{
  DM                cda;
  Vec               gcoords;
  PetscScalar       *LA_gcoords;
  PetscScalar       Re[NODES_PER_EL_Q1_3D];
  PetscScalar       el_coords[NSD*NODES_PER_EL_Q1_3D];
  PetscScalar       el_phi[NODES_PER_EL_Q1_3D];

  PetscInt          nel,nen,e,q;
  const PetscInt    *elnidx;
  PetscInt          ge_eqnums[NODES_PER_EL_Q1_3D];
  PetscInt          nqp;
  PetscScalar       *qp_xi,*qp_weight;
  Quadrature        volQ;
  QPntVolCoefStokes *all_quadpoints = NULL,*cell_quadpoints = NULL;
  PetscScalar       *qp_rho;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Quadrature */
  volQ      = data->volQ;
  nqp       = volQ->npoints;
  qp_xi     = volQ->q_xi_coor;
  qp_weight = volQ->q_weight;

  ierr = PetscCalloc1(nqp,&qp_rho);CHKERRQ(ierr); /* buffer to store cell quad point densities */
  
  /* Get stokes volume quadrature data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(data->stokes->volQ,&all_quadpoints);CHKERRQ(ierr);
  
  /* Setup for coords */
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* Connectivity table */
  ierr = DMDAGetElements(da,&nel,&nen,&elnidx);CHKERRQ(ierr);

  for (e=0; e<nel; e++) {
    
    /* get coords for the element */
    ierr = DMDAEQ1_GetVectorElementField_3D(el_coords,(PetscInt*)&elnidx[nen*e],LA_gcoords);CHKERRQ(ierr);
    /* get value at the element */
    ierr = DMDAEQ1_GetScalarElementField_3D(el_phi,(PetscInt*)&elnidx[nen*e],LA_phi);CHKERRQ(ierr);
    
    /* copy the density from stokes volume quadrature points */
    ierr = VolumeQuadratureGetCellData_Stokes(data->stokes->volQ,all_quadpoints,e,&cell_quadpoints);CHKERRQ(ierr);
    for (q=0; q<nqp; q++) {
      qp_rho[q] = cell_quadpoints[q].rho;
    }

    /* initialise element stiffness matrix */
    ierr = PetscMemzero(Re,sizeof(PetscScalar)*NODES_PER_EL_Q1_3D);CHKERRQ(ierr);
    /* form element stiffness matrix */
    ierr = Element_FormFunction_LithoPressure(Re,el_coords,el_phi,qp_rho,nqp,qp_xi,qp_weight,data);CHKERRQ(ierr);
    /* Add value to the residue vector */
    ierr = DMDAEQ1_GetElementLocalIndicesDOF(ge_eqnums,1,(PetscInt*)&elnidx[nen*e]);CHKERRQ(ierr);
    ierr = DMDAEQ1_SetValuesLocalStencil_AddValues_DOF(LA_R,1,ge_eqnums,Re);CHKERRQ(ierr);
  }

  /* tidy up local arrays (input) */
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = PetscFree(qp_rho);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionLocal_LithoPressure_dS(PDESolveLithoP data,DM da,PetscScalar *LA_phi,PetscScalar *LA_R)
{
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionLithoPressure(Vec X,Vec F,void *ctx)
{
  pTatinCtx      ptatin = (pTatinCtx)ctx;
  PDESolveLithoP data;
  DM             da;
  Vec            philoc, Fphiloc;
  PetscScalar    *LA_philoc, *LA_Fphiloc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = pTatinGetContext_LithoP(ptatin,&data);CHKERRQ(ierr);
  da = data->da;

  ierr = VecZeroEntries(F);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da,&philoc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Fphiloc);CHKERRQ(ierr);

  /* get local solution */
  ierr = VecZeroEntries(philoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,philoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,philoc);CHKERRQ(ierr);

  /* insert boundary conditions into local vectors */
  ierr = BCListInsertLocal(data->bclist,philoc);CHKERRQ(ierr);

  /* initialise residual */
  ierr = VecZeroEntries(Fphiloc);CHKERRQ(ierr);

  /* get arrays */
  ierr = VecGetArray(philoc,  &LA_philoc);CHKERRQ(ierr);
  ierr = VecGetArray(Fphiloc, &LA_Fphiloc);CHKERRQ(ierr);

  /* ============= */
  /* FORM_FUNCTION */
  ierr = FormFunctionLocal_LithoPressure_dV(data,da,LA_philoc,LA_Fphiloc);CHKERRQ(ierr);
  ierr = FormFunctionLocal_LithoPressure_dS(data,da,LA_philoc,LA_Fphiloc);CHKERRQ(ierr); /* surface integral not implemented */
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
  ierr = BCListResidualDirichlet(data->bclist,X,F);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SNES_FormFunctionLithoPressure(SNES snes,Vec X,Vec F,void *ctx)
{
  pTatinCtx      ptatin = (pTatinCtx)ctx;
  PDESolveLithoP lithoP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatinGetContext_LithoP(ptatin,&lithoP);CHKERRQ(ierr);
  ierr = FormFunctionLithoPressure(X,F,ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SNESSolve_LithoPressure(PDESolveLithoP LP,Mat J,Vec X, Vec F, pTatinCtx pctx)
{
  SNES           snes;
  PetscLogDouble time[2];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,"LP_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,  SNES_FormFunctionLithoPressure,(void*)pctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,SNES_FormJacobianLithoPressure,(void*)pctx);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"   [[ COMPUTING LITHOSTATIC PRESSURE ]]\n");

  // insert boundary conditions into solution vector
  ierr = BCListInsert(LP->bclist,X);CHKERRQ(ierr);

  PetscTime(&time[0]);
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  PetscTime(&time[1]);
  ierr = pTatinLogBasicSNES(pctx,"LithoPressure",snes);CHKERRQ(ierr);
  ierr = pTatinLogBasicCPUtime(pctx,"LithoPressure",time[1]-time[0]);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode AverageVolumeQuadraturePointsToSurfaceQuadraturePointsStokes(pTatinCtx user)
{
  QPntVolCoefStokes  *all_gausspoints,*cell_gausspoints;
  SurfaceQuadrature  surfQ_face;
  QPntSurfCoefStokes *surfQ_coeff,*surfQ_cell_coeff;
  QPoint3d           *qp3d;
  Quadrature         volQ;
  PetscInt           VolQ_ngp,SurfQ_nqp,p,q,f,c,nfaces;
  PetscInt           *element_list,face_list_n;
  HexElementFace     face_location[] = {HEX_FACE_Nxi, HEX_FACE_Pxi, HEX_FACE_Neta, HEX_FACE_Peta, HEX_FACE_Nzeta, HEX_FACE_Pzeta};
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  volQ = user->stokes_ctx->volQ;
  /* Number of Volume quadrature points */
  VolQ_ngp = volQ->npoints;
  /* Get Volume Quadrature Stokes data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);
  face_list_n = 6;
  
  for (f=0; f<face_list_n; f++) {
    ierr = PhysCompStokesGetSurfaceQuadrature(user->stokes_ctx,face_location[f],&surfQ_face);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetQuadratureInfo(surfQ_face,&SurfQ_nqp,NULL,&qp3d);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetFaceInfo(surfQ_face,NULL,&nfaces,&element_list);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetAllCellData_Stokes(surfQ_face,&surfQ_coeff);CHKERRQ(ierr);

    for (c=0; c<nfaces; c++) {
      PetscInt  eidx;
      PetscReal avg_el_eta,avg_el_rho;
      /* Get the element index */
      eidx = element_list[c];
      /* Get the volume quadrature data of the element */
      ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,eidx,&cell_gausspoints);CHKERRQ(ierr);
      /* Get the surface quadrature data of the element */
      ierr = SurfaceQuadratureGetCellData_Stokes(surfQ_face,surfQ_coeff,c,&surfQ_cell_coeff);CHKERRQ(ierr);

      /* Average the viscosity laying on the volume quadrature points */
      avg_el_eta = 0.0;
      avg_el_rho = 0.0;
      for (p=0; p<VolQ_ngp; p++) {
        avg_el_eta += cell_gausspoints[p].eta;
        avg_el_rho += cell_gausspoints[p].rho;
      }
      avg_el_eta = avg_el_eta/VolQ_ngp;
      avg_el_rho = avg_el_rho/VolQ_ngp;
      /* Loop over surface quadrature points */
      for (q=0; q<SurfQ_nqp; q++) {
        PetscReal viscosity,rho;
        /* Get viscosity and density at surface quadrature point */
        QPntSurfCoefStokesGetField_viscosity(&surfQ_cell_coeff[q],&viscosity);
        QPntSurfCoefStokesGetField_density(&surfQ_cell_coeff[q],&rho);
        /* Apply the averaged value */
        viscosity = avg_el_eta;
        rho = avg_el_rho;
        /* Set viscosity and density at surface quadrature point */
        QPntSurfCoefStokesSetField_viscosity(&surfQ_cell_coeff[q],viscosity);
        QPntSurfCoefStokesSetField_density(&surfQ_cell_coeff[q],rho);
      }
    }
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode ApplyLithostaticPressure_SurfQuadratureStokes_FullFace(PhysCompStokes stokes,
                                                                      DM da,
                                                                      Vec X,
                                                                      HexElementFace face_location[],
                                                                      PetscInt face_list_n)
{
  SurfaceQuadrature  surfQ_face;
  QPntSurfCoefStokes *surfQ_coeff,*surfQ_cell_coeff;
  PetscInt           c,q,n,f,nqp,nfaces,*element_list;
  QPoint3d           *qp3d;
  Vec                LP_local;
  PetscReal          *LA_LP_local;
  PetscInt           nel_lp,nen_lp;
  const PetscInt     *elnidx_lp;
  PetscReal          el_lithop[Q1_NODES_PER_EL_3D];
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  
  /* Get Q1 elements connectivity table */
  ierr = DMDAGetElements(da,&nel_lp,&nen_lp,&elnidx_lp);CHKERRQ(ierr);

  /* Get the values of the lithostatic pressure solution vector at local rank */
  ierr = DMGetLocalVector(da,&LP_local);CHKERRQ(ierr);
  ierr = VecZeroEntries(LP_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,LP_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,LP_local);CHKERRQ(ierr);
  ierr = VecGetArray(LP_local,&LA_LP_local);CHKERRQ(ierr);
  
  for (f=0; f<face_list_n; f++) {
    ierr = PhysCompStokesGetSurfaceQuadrature(stokes,face_location[f],&surfQ_face);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetQuadratureInfo(surfQ_face,&nqp,NULL,&qp3d);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetFaceInfo(surfQ_face,NULL,&nfaces,&element_list);CHKERRQ(ierr);
    
    ierr = SurfaceQuadratureGetAllCellData_Stokes(surfQ_face,&surfQ_coeff);CHKERRQ(ierr);

    for (c=0; c<nfaces; c++) {
      PetscInt eidx;
      /* Get the element index */
      eidx = element_list[c];
      /* Get the lithostatic pressure in the element */
      ierr = DMDAEQ1_GetScalarElementField_3D(el_lithop,(PetscInt*)&elnidx_lp[nen_lp*eidx],LA_LP_local);CHKERRQ(ierr);
      /* Get the surface quadrature data of the element */
      ierr = SurfaceQuadratureGetCellData_Stokes(surfQ_face,surfQ_coeff,c,&surfQ_cell_coeff);CHKERRQ(ierr);
      
      /* Loop over quadrature points */
      for (q=0; q<nqp; q++) {
        PetscReal litho_pressure_qp;
        PetscReal NiQ1[Q1_NODES_PER_EL_3D];
        PetscReal qp_xi[NSD];
        double    *normal,*traction;
        
        /* Get normal and traction arrays at quadrature point (3 components) */
        QPntSurfCoefStokesGetField_surface_normal(&surfQ_cell_coeff[q],&normal);
        QPntSurfCoefStokesGetField_surface_traction(&surfQ_cell_coeff[q],&traction);
        
        /* Get local coords of quadrature point */
        qp_xi[0] = qp3d[q].xi;
        qp_xi[1] = qp3d[q].eta;
        qp_xi[2] = qp3d[q].zeta;
        /* Evaluate NiQ1 for interpolation on quadrature point */
        EvaluateBasis_Q1_3D(qp_xi,NiQ1);
        /* Interpolate lithostatic pressure at quadrature point */
        litho_pressure_qp = 0.0;
        for (n=0; n<Q1_NODES_PER_EL_3D; n++) {
          litho_pressure_qp += NiQ1[n]*el_lithop[n];
        }
        
        /* Set traction as t = -lp.n (colinear vector of opposite direction and scale lp) */
        traction[0] = (- litho_pressure_qp) * (normal[0]);
        traction[1] = (- litho_pressure_qp) * (normal[1]);
        traction[2] = (- litho_pressure_qp) * (normal[2]);
        
      }
    }
  }
  ierr = VecRestoreArray(LP_local,&LA_LP_local);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&LP_local);CHKERRQ(ierr);
  ierr = DMDARestoreElements(da,&nel_lp,&nen_lp,&elnidx_lp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ApplyDeviatoricValuePlusLithostaticPressure_SurfQuadratureStokes_FullFace
                (PhysCompStokes stokes,
                 DM da,
                 Vec X,
                 HexElementFace face_location[],
                 PetscInt face_list_n)
{
  SurfaceQuadrature  surfQ_face;
  QPntSurfCoefStokes *surfQ_coeff,*surfQ_cell_coeff;
  DM                 cda;
  Vec                gcoords;
  PetscInt           c,q,n,f,k,nqp,nfaces,*element_list;
  QPoint3d           *qp3d;
  Vec                LP_local;
  PetscReal          *LA_LP_local;
  PetscScalar        *LA_gcoords;
  PetscInt           nel_lp,nen_lp;
  const PetscInt     *elnidx_lp;
  PetscScalar        el_coords[NSD*NODES_PER_EL_Q1_3D];
  PetscReal          el_lithop[Q1_NODES_PER_EL_3D];
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* Get Q1 elements connectivity table */
  ierr = DMDAGetElements(da,&nel_lp,&nen_lp,&elnidx_lp);CHKERRQ(ierr);

  /* Get the values of the lithostatic pressure solution vector at local rank */
  ierr = DMGetLocalVector(da,&LP_local);CHKERRQ(ierr);
  ierr = VecZeroEntries(LP_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,LP_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,LP_local);CHKERRQ(ierr);
  ierr = VecGetArray(LP_local,&LA_LP_local);CHKERRQ(ierr);
  
  for (f=0; f<face_list_n; f++) {
    ierr = PhysCompStokesGetSurfaceQuadrature(stokes,face_location[f],&surfQ_face);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetQuadratureInfo(surfQ_face,&nqp,NULL,&qp3d);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetFaceInfo(surfQ_face,NULL,&nfaces,&element_list);CHKERRQ(ierr);
    
    ierr = SurfaceQuadratureGetAllCellData_Stokes(surfQ_face,&surfQ_coeff);CHKERRQ(ierr);

    for (c=0; c<nfaces; c++) {
      PetscInt  eidx;
      PetscReal tau;
      /* Get the element index */
      eidx = element_list[c];
      /* get coords for the element */
      ierr = DMDAEQ1_GetVectorElementField_3D(el_coords,(PetscInt*)&elnidx_lp[nen_lp*eidx],LA_gcoords);CHKERRQ(ierr);
      /* Get the lithostatic pressure in the element */
      ierr = DMDAEQ1_GetScalarElementField_3D(el_lithop,(PetscInt*)&elnidx_lp[nen_lp*eidx],LA_LP_local);CHKERRQ(ierr);
      /* Get the surface quadrature data of the element */
      ierr = SurfaceQuadratureGetCellData_Stokes(surfQ_face,surfQ_coeff,c,&surfQ_cell_coeff);CHKERRQ(ierr);
      
      tau = 0.0;
      if (face_location[f] == HEX_FACE_Pzeta) {
        for (k=0;k<Q1_NODES_PER_EL_3D;k++) {
          if (el_coords[3*k + 1] > -0.5) {
            tau = 15.0;
          }
        }
      }

      /* Loop over quadrature points */
      for (q=0; q<nqp; q++) {
        PetscReal litho_pressure_qp;
        PetscReal NiQ1[Q1_NODES_PER_EL_3D];
        PetscReal qp_xi[NSD];
        double    *normal,*traction;
        
        /* Get normal and traction arrays at quadrature point (3 components) */
        QPntSurfCoefStokesGetField_surface_normal(&surfQ_cell_coeff[q],&normal);
        QPntSurfCoefStokesGetField_surface_traction(&surfQ_cell_coeff[q],&traction);
        
        /* Get local coords of quadrature point */
        qp_xi[0] = qp3d[q].xi;
        qp_xi[1] = qp3d[q].eta;
        qp_xi[2] = qp3d[q].zeta;
        /* Evaluate NiQ1 for interpolation on quadrature point */
        EvaluateBasis_Q1_3D(qp_xi,NiQ1);
        /* Interpolate lithostatic pressure at quadrature point */
        litho_pressure_qp = 0.0;
        for (n=0; n<Q1_NODES_PER_EL_3D; n++) {
          litho_pressure_qp += NiQ1[n]*el_lithop[n];
        }

        /* Set traction as t = -lp.n (colinear vector of opposite direction and scale lp) */
        traction[0] = (tau - litho_pressure_qp) * (normal[0]);
        traction[1] = (tau - litho_pressure_qp) * (normal[1]);
        traction[2] = (tau - litho_pressure_qp) * (normal[2]);
        
      }
    }
  }

  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = VecRestoreArray(LP_local,&LA_LP_local);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&LP_local);CHKERRQ(ierr);
  ierr = DMDARestoreElements(da,&nel_lp,&nen_lp,&elnidx_lp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ApplyNormalStress_SurfQuadratureStokes_FullFace(PhysCompStokes stokes,
                                                               DM da,
                                                               Vec X,
                                                               Vec X_stokes,
                                                               HexElementFace face_location[],
                                                               PetscInt face_list_n)
{
  SurfaceQuadrature  surfQ_face;
  QPntSurfCoefStokes *surfQ_coeff,*surfQ_cell_coeff;
  PetscInt           c,q,n,f,k,i,j,nqp,nfaces,*element_list;
  QPoint3d           *qp3d;
  DM                 stokes_pack,dau,dap,cda;
  Vec                LP_local,Uloc,Ploc,gcoords;
  PetscReal          *LA_LP_local;
  PetscReal          *LA_Uloc,*LA_Ploc;
  PetscReal          *LA_gcoords;
  PetscInt           nel_lp,nen_lp,nel_u,nen_u;
  const PetscInt     *elnidx_lp;
  const PetscInt     *elnidx_u;
  PetscReal          el_lithop[Q1_NODES_PER_EL_3D];
  PetscReal          elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal          elu[3*Q2_NODES_PER_EL_3D];
  PetscReal          ux[Q2_NODES_PER_EL_3D],uy[Q2_NODES_PER_EL_3D],uz[Q2_NODES_PER_EL_3D];
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  
  /* Get Q1 elements connectivity table */
  ierr = DMDAGetElements(da,&nel_lp,&nen_lp,&elnidx_lp);CHKERRQ(ierr);

  /* Get the values of the lithostatic pressure solution vector at local rank */
  ierr = DMGetLocalVector(da,&LP_local);CHKERRQ(ierr);
  ierr = VecZeroEntries(LP_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,LP_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,LP_local);CHKERRQ(ierr);
  ierr = VecGetArray(LP_local,&LA_LP_local);CHKERRQ(ierr);
  
  /* Get the values of the velocity solution vector at local rank */
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);
  ierr = DMCompositeScatter(stokes_pack,X_stokes,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  /* Get Q2 elements connectivity table */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel_u,&nen_u,&elnidx_u);CHKERRQ(ierr);
  
  for (f=0; f<face_list_n; f++) {
    ierr = PhysCompStokesGetSurfaceQuadrature(stokes,face_location[f],&surfQ_face);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetQuadratureInfo(surfQ_face,&nqp,NULL,&qp3d);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetFaceInfo(surfQ_face,NULL,&nfaces,&element_list);CHKERRQ(ierr);
    
    ierr = SurfaceQuadratureGetAllCellData_Stokes(surfQ_face,&surfQ_coeff);CHKERRQ(ierr);

    for (c=0; c<nfaces; c++) {
      PetscInt  eidx;
      
      /* Get the element index */
      eidx = element_list[c];
      /* Get coordinates of the element */
      ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*eidx],LA_gcoords);CHKERRQ(ierr);
      /* Get the lithostatic pressure in the element */
      ierr = DMDAEQ1_GetScalarElementField_3D(el_lithop,(PetscInt*)&elnidx_lp[nen_lp*eidx],LA_LP_local);CHKERRQ(ierr);
      /* Get the velocity in the element */
      ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*eidx],(PetscScalar*)LA_Uloc);CHKERRQ(ierr);
    
      /* get velocity components */
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        ux[k] = elu[3*k  ];
        uy[k] = elu[3*k+1];
        uz[k] = elu[3*k+2];
      }
      /* Get the surface quadrature data of the element */
      ierr = SurfaceQuadratureGetCellData_Stokes(surfQ_face,surfQ_coeff,c,&surfQ_cell_coeff);CHKERRQ(ierr);
      
      /* Loop over quadrature points */
      for (q=0; q<nqp; q++) {
        PetscReal litho_pressure_qp;
        PetscReal NiQ1[Q1_NODES_PER_EL_3D];
        PetscReal qp_xi[NSD];
        PetscReal GNiQ2[NSD][Q2_NODES_PER_EL_3D];
        PetscReal dNudx[Q2_NODES_PER_EL_3D],dNudy[Q2_NODES_PER_EL_3D],dNudz[Q2_NODES_PER_EL_3D];
        PetscReal exx,exy,exz,eyy,eyz,ezz;
        PetscReal E[NSD][NSD],E_n,tau_n;
        double    *normal,*traction,viscosity,rho;
        
        /* Get normal and traction arrays at quadrature point (3 components) */
        QPntSurfCoefStokesGetField_surface_normal(&surfQ_cell_coeff[q],&normal);
        QPntSurfCoefStokesGetField_surface_traction(&surfQ_cell_coeff[q],&traction);
        /* Get viscosity at quadrature point */
        QPntSurfCoefStokesGetField_viscosity(&surfQ_cell_coeff[q],&viscosity);
        QPntSurfCoefStokesGetField_density(&surfQ_cell_coeff[q],&rho);
        /* Get local coords of quadrature point */
        qp_xi[0] = qp3d[q].xi;
        qp_xi[1] = qp3d[q].eta;
        qp_xi[2] = qp3d[q].zeta;
        /* Evaluate NiQ1 for interpolation on quadrature point */
        EvaluateBasis_Q1_3D(qp_xi,NiQ1);
        /* Interpolate lithostatic pressure at quadrature point */
        litho_pressure_qp = 0.0;
        for (n=0; n<Q1_NODES_PER_EL_3D; n++) {
          litho_pressure_qp += NiQ1[n]*el_lithop[n];
        }
        
        /* Evaluate grad.NiQ2 to compute strain rate */
        P3D_ConstructGNi_Q2_3D(qp_xi,GNiQ2);
        /* Get shape function global derivatives */
        P3D_evaluate_global_derivatives_Q2(elcoords,GNiQ2,dNudx,dNudy,dNudz);
        /* Compute strain rate */
        exx=0.0;  eyy=0.0;  ezz=0.0;
        exy=0.0;  exz=0.0;  eyz=0.0;
        for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
          exx += (dNudx[k] * ux[k]);
          eyy += (dNudy[k] * uy[k]);
          ezz += (dNudz[k] * uz[k]);

          exy += dNudy[k] * ux[k] + dNudx[k] * uy[k];
          exz += dNudz[k] * ux[k] + dNudx[k] * uz[k];
          eyz += dNudz[k] * uy[k] + dNudy[k] * uz[k];
        }
        exy = 0.5*exy;
        exz = 0.5*exz;
        eyz = 0.5*eyz;
        /* Build the tensor */
        E[0][0] = exx; E[0][1] = exy; E[0][2] = exz;
        E[1][0] = exy; E[1][1] = eyy; E[1][2] = eyz;
        E[2][0] = exz; E[2][1] = eyz; E[2][2] = ezz;
        /* Compute the normal strain rate: n[i]*E[i][j]*n[j] */
        E_n = 0.0;
        for (i=0;i<3;i++) {
          for (j=0;j<3;j++) {
            E_n += normal[i]*E[i][j]*normal[j];
          }
        }
        /* Compute the normal stress */
        tau_n = 2.0*viscosity*E_n;
        /* Set traction as t = (tau-lp).n (colinear vector of opposite direction and scale tau - lp) */
        traction[0] = (-tau_n - litho_pressure_qp) * (normal[0]);
        traction[1] = (-tau_n - litho_pressure_qp) * (normal[1]);
        traction[2] = (-tau_n - litho_pressure_qp) * (normal[2]);
      }
    }
  }
  ierr = VecRestoreArray(LP_local,&LA_LP_local);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&LP_local);CHKERRQ(ierr);
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = DMDARestoreElements(da,&nel_lp,&nen_lp,&elnidx_lp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeLambdaFromStep(pTatinCtx user,
                                     PetscInt start_step, 
                                     PetscInt end_step,
                                     PetscReal lambda_min,
                                     PetscReal lambda_max, 
                                     PetscReal *lambda)
{
  PetscFunctionBegin;
  *lambda = lambda_min;
  if (user->step > start_step && user->step <= end_step) {
    *lambda = (user->step - start_step) * (lambda_max - lambda_min)/(end_step-start_step) + lambda_min;
  } else if (user->step > end_step) {
    *lambda = lambda_max;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyTractionFromLithoPressureStokesVolumeQuadrature(pTatinCtx user,
                                                                         PhysCompStokes stokes,
                                                                         DM da,
                                                                         Vec X,
                                                                         Vec X_stokes,
                                                                         HexElementFace face_location[],
                                                                         PetscInt face_list_n)
{
  DM                 stokes_pack,dau,dap,cda;
  Vec                LP_local,Uloc,Ploc,gcoords;
  PetscReal          *LA_Uloc,*LA_Ploc;
  PetscReal          *LA_gcoords;
  PetscReal          *LA_LP_local;
  QPntVolCoefStokes  *all_gausspoints,*cell_gausspoints;
  SurfaceQuadrature  surfQ_face;
  QPntSurfCoefStokes *surfQ_coeff,*surfQ_cell_coeff;
  QPoint3d           *qp3d;
  Quadrature         volQ;
  PetscInt           VolQ_ngp,SurfQ_nqp,p,q,f,k,c,n,i,j,nfaces,nqp;
  PetscReal          VolQ_weight[NQP],VolQ_xi[NQP][3],VolQ_NiQ2[NQP][NPE],VolQ_GNiQ2[NQP][3][NPE];
  PetscReal          detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];
  PetscInt           *element_list;
  PetscInt           nel_u,nen_u,nen_p,nel_lp,nen_lp;
  const PetscInt     *elnidx_u;
  const PetscInt     *elnidx_p;
  const PetscInt     *elnidx_lp;
  PetscReal          elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal          elu[3*Q2_NODES_PER_EL_3D],elp[P_BASIS_FUNCTIONS];
  PetscReal          el_lithop[Q1_NODES_PER_EL_3D];
  PetscReal          el_eta[MAX_QUAD_PNTS];
  PetscReal          int_sxx,int_syy,int_szz,int_sxy,int_sxz,int_syz;
  PetscReal          el_volume;
  PetscInt           start_step,end_step;
  PetscReal          lambda,lambda_min,lambda_max;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* Get volume quadrature structure */
  volQ = stokes->volQ;
  /* Number of Volume quadrature points */
  VolQ_ngp = volQ->npoints;
  /* Get Volume Quadrature Stokes data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);
  
  P3D_prepare_elementQ2(VolQ_ngp,VolQ_weight,VolQ_xi,VolQ_NiQ2,VolQ_GNiQ2);
  /* Get the values of the velocity solution vector at local rank */
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);
  ierr = DMCompositeScatter(stokes_pack,X_stokes,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  /* Get Q2 elements connectivity table for u,p */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel_u,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel_u,&nen_p,&elnidx_p);CHKERRQ(ierr);
  
  /* Get Q1 elements connectivity table */
  ierr = DMDAGetElements(da,&nel_lp,&nen_lp,&elnidx_lp);CHKERRQ(ierr);
  /* Get the values of the lithostatic pressure solution vector at local rank */
  ierr = DMGetLocalVector(da,&LP_local);CHKERRQ(ierr);
  ierr = VecZeroEntries(LP_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,LP_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,LP_local);CHKERRQ(ierr);
  ierr = VecGetArray(LP_local,&LA_LP_local);CHKERRQ(ierr);
  
  start_step = 10;
  end_step   = 20;
  lambda_min = 0.0;
  lambda_max = 1.0;
  ierr = ComputeLambdaFromStep(user,start_step,end_step,lambda_min,lambda_max,&lambda);CHKERRQ(ierr);
  
  for (f=0; f<face_list_n; f++) {
    ierr = PhysCompStokesGetSurfaceQuadrature(stokes,face_location[f],&surfQ_face);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetQuadratureInfo(surfQ_face,&nqp,NULL,&qp3d);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetFaceInfo(surfQ_face,NULL,&nfaces,&element_list);CHKERRQ(ierr);
    
    ierr = SurfaceQuadratureGetAllCellData_Stokes(surfQ_face,&surfQ_coeff);CHKERRQ(ierr);

    for (c=0; c<nfaces; c++) {
      PetscInt  eidx;
      PetscReal sigma[NSD][NSD];
      PetscReal ux[Q2_NODES_PER_EL_3D],uy[Q2_NODES_PER_EL_3D],uz[Q2_NODES_PER_EL_3D];
      
      /* Get the element index */
      eidx = element_list[c];
      /* Get coordinates of the element */
      ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*eidx],LA_gcoords);CHKERRQ(ierr);
      /* Get the velocity in the element */
      ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*eidx],(PetscScalar*)LA_Uloc);CHKERRQ(ierr);
      /* Get the lithostatic pressure in the element */
      ierr = DMDAEQ1_GetScalarElementField_3D(el_lithop,(PetscInt*)&elnidx_lp[nen_lp*eidx],LA_LP_local);CHKERRQ(ierr);
      /* Get the volume quadrature data of the element */
      ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,eidx,&cell_gausspoints);CHKERRQ(ierr);
      
      /* get velocity components */
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        ux[k] = elu[3*k  ];
        uy[k] = elu[3*k+1];
        uz[k] = elu[3*k+2];
      }
      /* Evaluate geometry of the Q2 element */
      P3D_evaluate_geometry_elementQ2(VolQ_ngp,elcoords,VolQ_GNiQ2,detJ,dNudx,dNudy,dNudz);
      /* Get the viscosity at volume quadrature points */
      for (p=0; p<VolQ_ngp; p++) {
        el_eta[p] = cell_gausspoints[p].eta;
      }
      
      int_sxx = int_syy = int_szz = 0.0;
      int_sxy = int_sxz = int_syz = 0.0;
      el_volume = 0.0;
      /* Loop over volume quadrature points */
      for (p=0; p<VolQ_ngp; p++) {
        PetscReal fac;
        PetscReal exx,eyy,ezz,exy,exz,eyz;
        PetscReal sxx,syy,szz,sxy,sxz,syz;
        
        fac = VolQ_weight[p] * detJ[p];
        
        /* Compute the strain rate tensor */
        exx=0.0;  eyy=0.0;  ezz=0.0;
        exy=0.0;  exz=0.0;  eyz=0.0;
        for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
          exx += (dNudx[p][k] * ux[k]);
          eyy += (dNudy[p][k] * uy[k]);
          ezz += (dNudz[p][k] * uz[k]);

          exy += dNudy[p][k] * ux[k] + dNudx[p][k] * uy[k];
          exz += dNudz[p][k] * ux[k] + dNudx[p][k] * uz[k];
          eyz += dNudz[p][k] * uy[k] + dNudy[p][k] * uz[k];
        }
        exy = 0.5*exy;
        exz = 0.5*exz;
        eyz = 0.5*eyz;
        /* Compute the Stress tensor components */
        sxx = 2.0*el_eta[p]*exx;
        syy = 2.0*el_eta[p]*eyy;
        szz = 2.0*el_eta[p]*ezz;
        
        sxy = 2.0*el_eta[p]*exy;
        sxz = 2.0*el_eta[p]*exz;
        syz = 2.0*el_eta[p]*eyz;
        /* Integrate the Stress tensor components over the volume */
        int_sxx += sxx * fac;
        int_syy += syy * fac;
        int_szz += szz * fac;
        int_sxy += sxy * fac;
        int_sxz += sxz * fac;
        int_syz += syz * fac;
        /* Compute the volume of the element */
        el_volume += 1.0 * fac;
      }
      /* Construct the stress tensor */
      sigma[0][0] = int_sxx / el_volume;
      sigma[1][1] = int_syy / el_volume;
      sigma[2][2] = int_szz / el_volume;
      
      sigma[0][1] = int_sxy / el_volume;
      sigma[0][2] = int_sxz / el_volume;
      sigma[1][2] = int_syz / el_volume;
      
      sigma[1][0] = sigma[0][1];
      sigma[2][0] = sigma[0][2];
      sigma[2][1] = sigma[1][2];
      
      /* Get the surface quadrature data of the element */
      ierr = SurfaceQuadratureGetCellData_Stokes(surfQ_face,surfQ_coeff,c,&surfQ_cell_coeff);CHKERRQ(ierr);

      /* Loop over surface quadrature points */
      for (q=0; q<nqp; q++) {
        PetscReal litho_pressure_qp,tau_n;
        PetscReal NiQ1[Q1_NODES_PER_EL_3D];
        PetscReal qp_xi[NSD];
        double    *normal,*traction;
        
        /* Get normal and traction arrays at quadrature point (3 components) */
        QPntSurfCoefStokesGetField_surface_normal(&surfQ_cell_coeff[q],&normal);
        QPntSurfCoefStokesGetField_surface_traction(&surfQ_cell_coeff[q],&traction);
        /* Get local coords of quadrature point */
        qp_xi[0] = qp3d[q].xi;
        qp_xi[1] = qp3d[q].eta;
        qp_xi[2] = qp3d[q].zeta;
        /* Evaluate NiQ1 for interpolation on quadrature point */
        EvaluateBasis_Q1_3D(qp_xi,NiQ1);
        /* Interpolate lithostatic pressure at quadrature point */
        litho_pressure_qp = 0.0;
        for (n=0; n<Q1_NODES_PER_EL_3D; n++) {
          litho_pressure_qp += NiQ1[n]*el_lithop[n];
        }
        /* Compute the normal stress: n[i]*sigma[i][j]*n[j] */
        tau_n = 0.0;
        for (i=0; i<NSD; i++) {
          for (j=0; j<NSD; j++) {
            tau_n += normal[i]*sigma[i][j]*normal[j];
          }
        }
        /* Set traction as t = (tau-lp).n (colinear vector of opposite direction and scale tau - lp) */
        traction[0] = (-lambda*tau_n - litho_pressure_qp) * (normal[0]);
        traction[1] = (-lambda*tau_n - litho_pressure_qp) * (normal[1]);
        traction[2] = (-lambda*tau_n - litho_pressure_qp) * (normal[2]);
      }
    }
  }
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyTractionFromLithoPressure(pTatinCtx user, Vec X_stokes)
{
  PDESolveLithoP LP;
  PetscReal      val_P;
  Mat            J = NULL;
  PetscInt       face_list_n;
  HexElementFace face_list[] = {HEX_FACE_Neta, HEX_FACE_Pxi, HEX_FACE_Nxi, HEX_FACE_Pzeta, HEX_FACE_Nzeta};
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  
  ierr = pTatinPhysCompActivate_LithoP(user,PETSC_TRUE);CHKERRQ(ierr);
  ierr = pTatinGetContext_LithoP(user,&LP);CHKERRQ(ierr);
  
  val_P = 0.0;
  ierr = DMDABCListTraverse3d(LP->bclist,LP->da,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  
  ierr = DMSetMatType(LP->da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(LP->da,&J);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  
  ierr = SNESSolve_LithoPressure(LP,J,LP->X,LP->F,user);CHKERRQ(ierr);

  {
    PetscViewer viewer;
    char        fname[256];
    
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/LithoP_step%d.vts",user->outputpath,user->step);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(LP->X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  face_list_n = 5;
  /* Apply the lithostatic pressure on surface quadrature points */
  //ierr = ApplyLithostaticPressure_SurfQuadratureStokes_FullFace(user->stokes_ctx,LP->da,LP->X,face_list,face_list_n);CHKERRQ(ierr);
  ierr = ApplyDeviatoricValuePlusLithostaticPressure_SurfQuadratureStokes_FullFace(user->stokes_ctx,LP->da,LP->X,face_list,face_list_n);CHKERRQ(ierr);

  //ierr = ApplyNormalStress_SurfQuadratureStokes_FullFace(user->stokes_ctx,LP->da,LP->X,X_stokes,face_list,face_list_n);CHKERRQ(ierr);
  //ierr = ModelApplyTractionFromLithoPressureStokesVolumeQuadrature(user,user->stokes_ctx,LP->da,LP->X,X_stokes,face_list,face_list_n);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = PhysCompDestroy_LithoP(&LP);CHKERRQ(ierr);
  user->litho_p_ctx = NULL;
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyTractionFromStokesVolumeQuadrature(pTatinCtx user, Vec X_stokes)
{
  PhysCompStokes     stokes;
  DM                 stokes_pack,dau,dap,cda;
  Vec                Uloc,Ploc,gcoords;
  PetscReal          *LA_Uloc,*LA_Ploc;
  PetscReal          *LA_gcoords;
  QPntVolCoefStokes  *all_gausspoints,*cell_gausspoints;
  SurfaceQuadrature  surfQ_face;
  QPntSurfCoefStokes *surfQ_coeff,*surfQ_cell_coeff;
  QPoint3d           *qp3d;
  Quadrature         volQ;
  PetscInt           VolQ_ngp,SurfQ_nqp,p,q,f,k,c,n,i,j,nfaces,nqp;
  PetscReal          VolQ_weight[NQP],VolQ_xi[NQP][3],VolQ_NiQ2[NQP][NPE],VolQ_GNiQ2[NQP][3][NPE];
  PetscReal          detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];
  PetscInt           *element_list;
  PetscInt           nel_u,nen_u,nen_p;
  const PetscInt     *elnidx_u;
  const PetscInt     *elnidx_p;
  PetscReal          elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal          elu[3*Q2_NODES_PER_EL_3D],elp[P_BASIS_FUNCTIONS];
  PetscReal          el_lithop[Q1_NODES_PER_EL_3D];
  PetscReal          el_eta[MAX_QUAD_PNTS];
  PetscReal          int_sxx,int_syy,int_szz,int_sxy,int_sxz,int_syz;
  PetscReal          el_volume;
  HexElementFace     face_location[] = {HEX_FACE_Nxi, HEX_FACE_Pxi, HEX_FACE_Neta, HEX_FACE_Nzeta, HEX_FACE_Pzeta};
  PetscInt           face_list_n = 5;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* Get volume quadrature structure */
  volQ = user->stokes_ctx->volQ;
  /* Number of Volume quadrature points */
  VolQ_ngp = volQ->npoints;
  /* Get Volume Quadrature Stokes data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);
  
  P3D_prepare_elementQ2(VolQ_ngp,VolQ_weight,VolQ_xi,VolQ_NiQ2,VolQ_GNiQ2);
  /* Get the values of the velocity solution vector at local rank */
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);
  ierr = DMCompositeScatter(stokes_pack,X_stokes,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  /* Get Q2 elements connectivity table for u,p */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel_u,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel_u,&nen_p,&elnidx_p);CHKERRQ(ierr);
  
  for (f=0; f<face_list_n; f++) {
    ierr = PhysCompStokesGetSurfaceQuadrature(stokes,face_location[f],&surfQ_face);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetQuadratureInfo(surfQ_face,&nqp,NULL,&qp3d);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetFaceInfo(surfQ_face,NULL,&nfaces,&element_list);CHKERRQ(ierr);
    
    ierr = SurfaceQuadratureGetAllCellData_Stokes(surfQ_face,&surfQ_coeff);CHKERRQ(ierr);

    for (c=0; c<nfaces; c++) {
      PetscInt  eidx;
      PetscReal sigma[NSD][NSD];
      PetscReal ux[Q2_NODES_PER_EL_3D],uy[Q2_NODES_PER_EL_3D],uz[Q2_NODES_PER_EL_3D];
      
      /* Get the element index */
      eidx = element_list[c];
      /* Get coordinates of the element */
      ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*eidx],LA_gcoords);CHKERRQ(ierr);
      /* Get the velocity in the element */
      ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*eidx],(PetscScalar*)LA_Uloc);CHKERRQ(ierr);
      /* Get the pressure in the element */
      ierr = DMDAGetScalarElementField(elp,nen_p,(PetscInt*)&elnidx_p[nen_p*eidx],(PetscScalar*)LA_Ploc);CHKERRQ(ierr);
      /* Get the volume quadrature data of the element */
      ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,eidx,&cell_gausspoints);CHKERRQ(ierr);
      
      /* get velocity components */
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        ux[k] = elu[3*k  ];
        uy[k] = elu[3*k+1];
        uz[k] = elu[3*k+2];
      }
      /* Evaluate geometry of the Q2 element */
      P3D_evaluate_geometry_elementQ2(VolQ_ngp,elcoords,VolQ_GNiQ2,detJ,dNudx,dNudy,dNudz);
      /* Get the viscosity at volume quadrature points */
      for (p=0; p<VolQ_ngp; p++) {
        el_eta[p] = cell_gausspoints[p].eta;
      }
      
      int_sxx = int_syy = int_szz = 0.0;
      int_sxy = int_sxz = int_syz = 0.0;
      el_volume = 0.0;
      /* Loop over volume quadrature points */
      for (p=0; p<VolQ_ngp; p++) {
        PetscReal fac;
        PetscReal exx,eyy,ezz,exy,exz,eyz;
        PetscReal sxx,syy,szz,sxy,sxz,syz;
        
        fac = VolQ_weight[p] * detJ[p];
        
        /* Compute the strain rate tensor */
        exx=0.0;  eyy=0.0;  ezz=0.0;
        exy=0.0;  exz=0.0;  eyz=0.0;
        for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
          exx += (dNudx[p][k] * ux[k]);
          eyy += (dNudy[p][k] * uy[k]);
          ezz += (dNudz[p][k] * uz[k]);

          exy += dNudy[p][k] * ux[k] + dNudx[p][k] * uy[k];
          exz += dNudz[p][k] * ux[k] + dNudx[p][k] * uz[k];
          eyz += dNudz[p][k] * uy[k] + dNudy[p][k] * uz[k];
        }
        exy = 0.5*exy;
        exz = 0.5*exz;
        eyz = 0.5*eyz;
        /* Compute the Stress tensor components */
        sxx = 2.0*el_eta[p]*exx;
        syy = 2.0*el_eta[p]*eyy;
        szz = 2.0*el_eta[p]*ezz;
        
        sxy = 2.0*el_eta[p]*exy;
        sxz = 2.0*el_eta[p]*exz;
        syz = 2.0*el_eta[p]*eyz;
        /* Integrate the Stress tensor components over the volume */
        int_sxx += sxx * fac;
        int_syy += syy * fac;
        int_szz += szz * fac;
        int_sxy += sxy * fac;
        int_sxz += sxz * fac;
        int_syz += syz * fac;
        /* Compute the volume of the element */
        el_volume += 1.0 * fac;
      }
      /* Construct the stress tensor */
      sigma[0][0] = int_sxx / el_volume;
      sigma[1][1] = int_syy / el_volume;
      sigma[2][2] = int_szz / el_volume;
      
      sigma[0][1] = int_sxy / el_volume;
      sigma[0][2] = int_sxz / el_volume;
      sigma[1][2] = int_syz / el_volume;
      
      sigma[1][0] = sigma[0][1];
      sigma[2][0] = sigma[0][2];
      sigma[2][1] = sigma[1][2];
      
      /* Get the surface quadrature data of the element */
      ierr = SurfaceQuadratureGetCellData_Stokes(surfQ_face,surfQ_coeff,c,&surfQ_cell_coeff);CHKERRQ(ierr);

      /* Loop over surface quadrature points */
      for (q=0; q<nqp; q++) {
        PetscReal pressure,tau_n;
        PetscReal Nip[P_BASIS_FUNCTIONS];
        PetscReal qp_xi[NSD];
        double    *normal,*traction;
        
        /* Get normal and traction arrays at quadrature point (3 components) */
        QPntSurfCoefStokesGetField_surface_normal(&surfQ_cell_coeff[q],&normal);
        QPntSurfCoefStokesGetField_surface_traction(&surfQ_cell_coeff[q],&traction);
        /* Get local coords of quadrature point */
        qp_xi[0] = qp3d[q].xi;
        qp_xi[1] = qp3d[q].eta;
        qp_xi[2] = qp3d[q].zeta;
        /* Evaluate Nip for interpolation on quadrature point */
        ConstructNi_pressure(qp_xi,elcoords,Nip);
        /* Interpolate pressure at quadrature point */
        pressure = 0.0;
        for (n=0; n<P_BASIS_FUNCTIONS; n++) {
          pressure += Nip[n]*elp[n];
        }
        /* Compute the normal stress: n[i]*sigma[i][j]*n[j] */
        tau_n = 0.0;
        for (i=0; i<NSD; i++) {
          for (j=0; j<NSD; j++) {
            tau_n += normal[i]*sigma[i][j]*normal[j];
          }
        }
        /* Set traction as t = (tau-lp).n (colinear vector of opposite direction and scale tau - lp) */
        traction[0] = (-tau_n - pressure) * (normal[0]);
        traction[1] = (-tau_n - pressure) * (normal[1]);
        traction[2] = (-tau_n - pressure) * (normal[2]);
      }
    }
  }
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}