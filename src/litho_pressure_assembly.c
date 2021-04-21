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
#include "litho_pressure_assembly.h"

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
  if (ctx->LP_bclist) { ierr = BCListDestroy(&ctx->LP_bclist);CHKERRQ(ierr); }
  ierr = DMDestroy(&ctx->daLP);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->F);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->X);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);

  *LP = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode DMDACeateQ1FromQ2LithoP_Mesh(DM dmq2, DM *dm)
{
  DM             dmq1;
  PetscInt       stencilq2,sei,sej,sek,lmx,lmy,lmz,MX,MY,MZ,Mp,Np,Pp,i,j,k;
  PetscInt       *siq2,*sjq2,*skq2,*lmxq2,*lmyq2,*lmzq2,*lxq1,*lyq1,*lzq1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDAGetCornersElementQ2(dmq2,&sei,&sej,&sek,&lmx,&lmy,&lmz);CHKERRQ(ierr);
  ierr = DMDAGetSizeElementQ2(dmq2,&MX,&MY,&MZ);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dmq2,0, 0,0,0, &Mp,&Np,&Pp,0,&stencilq2, 0,0,0, 0);CHKERRQ(ierr);
  if (stencilq2 != 2) {
    SETERRQ(PetscObjectComm((PetscObject)dmq2),PETSC_ERR_SUP,"Attempting to create an overlapping DMDA from a DMDA which doesn't have a stencil width of 2... probably the generator isn't a Q2");
  }
  ierr = DMDAGetOwnershipRangesElementQ2(dmq2,0,0,0,&siq2,&sjq2,&skq2,&lmxq2,&lmyq2,&lmzq2);CHKERRQ(ierr);

  ierr = PetscCalloc1(Mp,&lxq1);CHKERRQ(ierr);
  ierr = PetscCalloc1(Np,&lyq1);CHKERRQ(ierr);
  ierr = PetscCalloc1(Pp,&lzq1);CHKERRQ(ierr);

  for (i=0; i<Mp-1; i++) {
    lxq1[i] = siq2[i+1]/2 - siq2[i]/2;
  }
  lxq1[Mp-1] = (2*MX - siq2[Mp-1])/2;
  lxq1[Mp-1]++;

  for (j=0; j<Np-1; j++) {
    lyq1[j] = sjq2[j+1]/2 - sjq2[j]/2;
  }
  lyq1[Np-1] = (2*MY - sjq2[Np-1])/2;
  lyq1[Np-1]++;

  for (k=0; k<Pp-1; k++) {
    lzq1[k] = skq2[k+1]/2 - skq2[k]/2;
  }
  lzq1[Pp-1] = (2*MZ - skq2[Pp-1])/2;
  lzq1[Pp-1]++;

  ierr = DMDACreate3d(PetscObjectComm((PetscObject)dmq2),
                      DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                      DMDA_STENCIL_BOX, /* [NOTE] BOX stencil */
                      MX+1,MY+1,MZ+1,
                      Mp,Np,Pp,
                      1, /* dofs */
                      1, /* [NOTE] stencil width 1 */
                      lxq1,lyq1,lzq1,&dmq1);CHKERRQ(ierr);

  ierr = DMSetUp(dmq1);CHKERRQ(ierr);
  
  ierr = DMDAProjectCoordinatesQ2toOverlappingQ1_3d(dmq2,dmq1);CHKERRQ(ierr);
  
  *dm = dmq1;
  
  ierr = PetscFree(lxq1);CHKERRQ(ierr);
  ierr = PetscFree(lyq1);CHKERRQ(ierr);
  ierr = PetscFree(lzq1);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompCreateMesh_LithoP(PDESolveLithoP LP,DM dav,PetscInt mx,PetscInt my, PetscInt mz,PetscInt mesh_generator_type)
{
  PetscInt       MX,MY,MZ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  LP->LP_mesh_type = mesh_generator_type;

  switch (mesh_generator_type) {

    case 0:
      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_LithoP: Generating standard Q1 DMDA\n");
      LP->mx = mx;
      LP->my = my;
      LP->mz = mz;
      ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, LP->mx+1,LP->my+1,LP->mz+1, PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&dav);CHKERRQ(ierr);
      ierr = DMSetUp(dav);CHKERRQ(ierr);
      LP->daLP = dav;
      ierr = DMDASetElementType(LP->daLP,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
      
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only overlapping Q1 From Q2 mesh supported {1} ");
      break;

    case 1:
      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_LithoP: Generating overlapping Q1 DMDA\n");
      if (!dav) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require valid DM dav");
      }
      ierr = DMDACeateQ1FromQ2LithoP_Mesh(dav,&LP->daLP);CHKERRQ(ierr);
      ierr = DMDASetElementType(LP->daLP,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
      /* Push back the element number info into LP struct */
      ierr = DMDAGetInfo(LP->daLP,0, &MX,&MY,&MZ,0,0,0,0,0, 0,0,0, 0);CHKERRQ(ierr);
      LP->mx = MX-1;
      LP->my = MY-1;
      LP->mz = MZ-1;
      break;

    default:
      LP->LP_mesh_type = 1;

      PetscPrintf(PETSC_COMM_WORLD,"PhysCompCreateMesh_LithoP: Generating overlapping Q1 DMDA\n");
      if (!dav) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require valid DM dav");
      }
      ierr = DMDACeateQ1FromQ2LithoP_Mesh(dav,&LP->daLP);CHKERRQ(ierr);
      ierr = DMDASetElementType(LP->daLP,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
      /* Push back the element number info into LP struct */
      ierr = DMDAGetInfo(LP->daLP,0, &MX,&MY,&MZ,0,0,0,0,0, 0,0,0, 0);CHKERRQ(ierr);
      LP->mx = MX-1;
      LP->my = MY-1;
      LP->mz = MZ-1;
      break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompCreateBoundaryList_LithoP(PDESolveLithoP LP)
{
  DM             daLP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  daLP = LP->daLP;
  if (!daLP) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"daLP must be set"); }
  ierr = DMDABCListCreate(daLP,&LP->LP_bclist);CHKERRQ(ierr);

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
  ierr = DMCreateGlobalVector(litho_p->daLP,&litho_p->F);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(litho_p->daLP,&litho_p->X);CHKERRQ(ierr);

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
  
  user->litho_p_ctx = LP;
  LP->stokes = stokes_ctx;
  
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
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  ierr = pTatinGetContext_LithoP(ptatin,&data);CHKERRQ(ierr);
  da     = data->daLP;
  bclist = data->LP_bclist;
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
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  ierr = pTatinGetContext_LithoP(ptatin,&data);CHKERRQ(ierr);
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
  ierr = BCListResidualDirichlet(data->LP_bclist,X,F);CHKERRQ(ierr);

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
  SNES           snesLP;
  PetscLogDouble time[2];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snesLP);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snesLP,"LP_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snesLP,F,  SNES_FormFunctionLithoPressure,(void*)pctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snesLP,J,J,SNES_FormJacobianLithoPressure,(void*)pctx);CHKERRQ(ierr);
  ierr = SNESSetSolution(snesLP,X);CHKERRQ(ierr);
  ierr = SNESSetType(snesLP,SNESKSPONLY);
  ierr = SNESSetFromOptions(snesLP);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"   [[ COMPUTING LITHOSTATIC PRESSURE ]]\n");

  // insert boundary conditions into solution vector
  ierr = BCListInsert(LP->LP_bclist,X);CHKERRQ(ierr);

  PetscTime(&time[0]);
  ierr = SNESSolve(snesLP,NULL,X);CHKERRQ(ierr);
  PetscTime(&time[1]);
  ierr = pTatinLogBasicSNES(pctx,"LithoPressure",snesLP);CHKERRQ(ierr);
  ierr = pTatinLogBasicCPUtime(pctx,"LithoPressure",time[1]-time[0]);CHKERRQ(ierr);
  ierr = SNESDestroy(&snesLP);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode Test_Assembly(void)
{
  PetscBool      LP_found = PETSC_FALSE;
  pTatinCtx      pctx = NULL;
  PDESolveLithoP LP = NULL;
  Vec            X = NULL, F = NULL;
  Mat            J = NULL;
  PetscReal      grav[] = {0.0,-9.8,0.0};
  PetscScalar    val_P;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatin3dCreateContext(&pctx);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(pctx);CHKERRQ(ierr);
  
  //mesh_generator_type = 1;
  pctx->mx = 2;
  pctx->my = 50;
  pctx->mz = 2;
  // Create a Q2 mesh 
  ierr = pTatin3d_PhysCompStokesCreate(pctx);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(pctx->stokes_ctx->dav,0.0,10.0e-2,-30.0e-2,0.0,0.0,10.0e-2);CHKERRQ(ierr);
  ierr = PhysCompStokesSetGravityVector(pctx->stokes_ctx,grav);CHKERRQ(ierr);
  
  /* 
     Here I manually set the value of rho on stokes volume quadrature points for testing
     the copying of rho from stokes to lithoP in FormFunctionLocal_LithoPressure_dV()
     This will not be necessary in regular models
  */
  {
    QPntVolCoefStokes *all_quadpoints = NULL,*cell_quadpoints = NULL;
    PetscInt nel,nen,e,nqp,q;
    const PetscInt *elnidx;
    
    nqp = pctx->stokes_ctx->volQ->npoints;
    ierr = VolumeQuadratureGetAllCellData_Stokes(pctx->stokes_ctx->volQ,&all_quadpoints);CHKERRQ(ierr);
    ierr = DMDAGetElements_pTatinQ2P1(pctx->stokes_ctx->dav,&nel,&nen,&elnidx);CHKERRQ(ierr);
    for (e=0; e<nel; e++){
      ierr = VolumeQuadratureGetCellData_Stokes(pctx->stokes_ctx->volQ,all_quadpoints,e,&cell_quadpoints);CHKERRQ(ierr);
      /* copy the density */
      for (q=0; q<nqp; q++) {
        cell_quadpoints[q].rho = 27.0; //scale units of rho*g
      }
    }
  }
  
  ierr = pTatinPhysCompCreate_LithoP(pctx);CHKERRQ(ierr);
  ierr = pTatinContextValid_LithoP(pctx,&LP_found);CHKERRQ(ierr);
  if (!LP_found){
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"LP is NULL\n");
  }
  ierr = pTatinGetContext_LithoP(pctx,&LP);CHKERRQ(ierr);
  
  val_P = 0.0;
  ierr = DMDABCListTraverse3d(LP->LP_bclist,LP->daLP,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  /*
  val_P = 1.0;
  ierr = DMDABCListTraverse3d(LP->LP_bclist,LP->daLP,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  val_P = 2.0;
  ierr = DMDABCListTraverse3d(LP->LP_bclist,LP->daLP,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  val_P = 3.0;
  ierr = DMDABCListTraverse3d(LP->LP_bclist,LP->daLP,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  val_P = 4.0;
  ierr = DMDABCListTraverse3d(LP->LP_bclist,LP->daLP,DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  val_P = 5.0;
  ierr = DMDABCListTraverse3d(LP->LP_bclist,LP->daLP,DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  */
  X = LP->X;
  F = LP->F;
  ierr = DMSetMatType(LP->daLP,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(LP->daLP,&J);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  
  ierr = SNESSolve_LithoPressure(LP,J,X,F,pctx);CHKERRQ(ierr);
  
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
  
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = PhysCompDestroy_LithoP(&LP);CHKERRQ(ierr);
  ierr = pTatin3dDestroyContext(&pctx);CHKERRQ(ierr);
  
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
