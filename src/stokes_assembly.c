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
 **    filename:   stokes_assembly.c
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
#include "petscvec.h"
#include "petscdm.h"
#include "petscsnes.h"

#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "element_utils_q2.h"

#include "dmda_bcs.h"
#include "dmda_element_q2p1.h"
#include "quadrature.h"
#include "surfbclist.h"


//#define PTAT3D_LOG_ASM_OP

PetscInt ASS_MAP_wIwDI_uJuDJ(
    PetscInt wi, PetscInt wd, PetscInt w_NPE, PetscInt w_dof,
    PetscInt ui, PetscInt ud, PetscInt u_NPE, PetscInt u_dof )
{
  PetscInt ij;
  PetscInt r,c,nc;

  //nr = w_NPE * w_dof;
  nc = u_NPE * u_dof;

  r = w_dof * wi + wd;
  c = u_dof * ui + ud;

  ij = r*nc + c;

  return ij;
}

void FormStokes3D_transB_isoD_B( const int npe,
                                 double *GNx, double *GNy, double *GNz,
                                 double D[][6], double *Ke )
{
  const PetscInt el_dof = NSD * npe;
  PetscInt       i,j;
  PetscReal      B[6][375]; /* large enough for quartics */
  PetscReal      sum;

  // Use dhdPhys to compute factors in front of strainrate
  for( i = 0; i < 6; i++){
    for( j = 0; j < el_dof; j++){
      B[i][j] = 0.0;
    }
  } // initialize
  for( i=0; i<npe; i++ ){
    B[0][3*i]   = GNx[i];                           // Exx component
    B[1][3*i+1] = GNy[i];                           // Eyy component
    B[2][3*i+2] = GNz[i];                           // Ezz component
    B[3][3*i  ] = GNy[i]; B[3][3*i+1] = GNx[i];               // Exy component
    B[4][3*i  ] = GNz[i];               B[4][3*i+2] = GNx[i]; // Exz component
    B[5][3*i+1] = GNz[i]; B[5][3*i+2] = GNy[i];               // Eyz component
  }

  /* K_ij = trans(B_ik) Dkk B_kj = B_ki.D_kk.B_kj */
  for( i=0; i<el_dof; i++ ) {
    for( j=0; j<el_dof; j++ ) {
      // for -mx 32 -my 32 -mz 32, below does assembly in 13.55 seconds
      //      for( k=0; k<6; k++ ) {
      //        Ke[j*el_dof+i] += B[k][i] * D[k][k] * B[k][j];
      //      }

      // for -mx 32 -my 32 -mz 32, below does assembly in 9.16 seconds
      sum = Ke[j*el_dof+i];

      sum += B[0][i] * D[0][0] * B[0][j];
      sum += B[1][i] * D[1][1] * B[1][j];
      sum += B[2][i] * D[2][2] * B[2][j];

      sum += B[3][i] * D[3][3] * B[3][j];
      sum += B[4][i] * D[4][4] * B[4][j];
      sum += B[5][i] * D[5][5] * B[5][j];

      Ke[j*el_dof+i] = sum;
    }
  }
}


PetscErrorCode _MatAssemble_StokesA_AUU(Mat A,DM dau,BCList u_bclist,Quadrature volQ,SurfBCList surf_bclist)
{
  PetscErrorCode ierr;
  PetscInt       p,ngp;
  DM             cda;
  Vec            gcoords;
  PetscReal      *LA_gcoords;
  PetscInt       nel,nen_u,e,ii,jj,kk;
  PetscInt       vel_el_lidx[3*U_BASIS_FUNCTIONS];
  const PetscInt *elnidx_u;
  PetscReal      elcoords[3*Q2_NODES_PER_EL_3D],el_eta[MAX_QUAD_PNTS];
  ISLocalToGlobalMapping ltog;
  const PetscInt *GINDICES;
  PetscInt       NUM_GINDICES,ge_eqnums[3*Q2_NODES_PER_EL_3D];
  PetscReal      Ae[Q2_NODES_PER_EL_3D * Q2_NODES_PER_EL_3D * U_DOFS * U_DOFS];
  PetscReal      fac,diagD[NSTRESS],B[6][3*Q2_NODES_PER_EL_3D];

  PetscLogDouble t0,t1;
  PetscLogDouble t0c,t1c,tc;
  PetscLogDouble t0q,t1q,tq;
  QPntVolCoefStokes *all_gausspoints,*cell_gausspoints;
  PetscReal WEIGHT[NQP],XI[NQP][3],NI[NQP][NPE],GNI[NQP][3][NPE];
  PetscReal detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];


  PetscFunctionBegin;

  /* quadrature */
  ngp = volQ->npoints;
  P3D_prepare_elementQ2(ngp,WEIGHT,XI,NI,GNI);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(dau, &ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES);CHKERRQ(ierr);
  ierr = BCListApplyDirichletMask(NUM_GINDICES,(PetscInt*)GINDICES,u_bclist);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);

  tc = 0.0;
  tq = 0.0;
  PetscTime(&t0);
  for (e=0;e<nel;e++) {
    /* get local indices */
    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*e]);CHKERRQ(ierr);

    /* get global indices */
    for (ii=0; ii<NPE; ii++) {
      const int NID = elnidx_u[NPE*e + ii];

      ge_eqnums[3*ii  ] = GINDICES[ 3*NID   ];
      ge_eqnums[3*ii+1] = GINDICES[ 3*NID+1 ];
      ge_eqnums[3*ii+2] = GINDICES[ 3*NID+2 ];
    }

    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);

    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /*
       for (p=0; p<ngp; p++) {
       PetscScalar xip[] = { XI[p][0], XI[p][1], XI[p][2] };
       ConstructNi_pressure(xip,elcoords,NIp[p]);
       }
       */
    /* initialise element stiffness matrix */
    PetscMemzero( Ae, sizeof(PetscScalar)* Q2_NODES_PER_EL_3D * Q2_NODES_PER_EL_3D * U_DOFS * U_DOFS );

    PetscTime(&t0c);
    P3D_evaluate_geometry_elementQ2(ngp,elcoords,GNI, detJ,dNudx,dNudy,dNudz);
    PetscTime(&t1c);
    tc += (t1c-t0c);

    /* evaluate the viscosity */
    for (p=0; p<ngp; p++) {
      el_eta[p] = cell_gausspoints[p].eta;
    }

    /*
       PetscTime(&t0q);
       for (p=0; p<ngp; p++) {

       fac = WEIGHT[p] * detJ[p];

       for( ii=0; ii<6; ii++ ) {
       for( jj=0; jj<6; jj++ ) {
       D[ii][jj] = 0.0;
       }
       }
       for( ii=0; ii<3; ii++ ) {  D[ii][ii] = 2.0 * el_eta[p] * fac;  }
       for( ii=3; ii<6; ii++ ) {  D[ii][ii] =       el_eta[p] * fac;  }

       FormStokes3D_transB_isoD_B( Q2_NODES_PER_EL_3D, dNudx[p],dNudy[p],dNudz[p], D, Ae );
       }
       PetscTime(&t1q);
       tq += (t1q-t0q);
       */

    PetscTime(&t0q);
    for (p=0; p<ngp; p++) {

      fac = WEIGHT[p] * detJ[p];

      for (ii = 0; ii < NPE; ii++) {
        PetscScalar d_dx_i = dNudx[p][ii];
        PetscScalar d_dy_i = dNudy[p][ii];
        PetscScalar d_dz_i = dNudz[p][ii];

        B[0][3*ii  ] = d_dx_i; B[0][3*ii+1] = 0.0;     B[0][3*ii+2] = 0.0;
        B[1][3*ii  ] = 0.0;    B[1][3*ii+1] = d_dy_i;  B[1][3*ii+2] = 0.0;
        B[2][3*ii  ] = 0.0;    B[2][3*ii+1] = 0.0;     B[2][3*ii+2] = d_dz_i;

        B[3][3*ii] = d_dy_i;   B[3][3*ii+1] = d_dx_i;  B[3][3*ii+2] = 0.0;   /* e_xy */
        B[4][3*ii] = d_dz_i;   B[4][3*ii+1] = 0.0;     B[4][3*ii+2] = d_dx_i;/* e_xz */
        B[5][3*ii] = 0.0;      B[5][3*ii+1] = d_dz_i;  B[5][3*ii+2] = d_dy_i;/* e_yz */
      }


      diagD[0] = 2.0*fac*el_eta[p];
      diagD[1] = 2.0*fac*el_eta[p];
      diagD[2] = 2.0*fac*el_eta[p];

      diagD[3] =     fac*el_eta[p];
      diagD[4] =     fac*el_eta[p];
      diagD[5] =     fac*el_eta[p];

      /* form Bt tildeD B */
      /*
         Ke_ij = Bt_ik . D_kl . B_lj
         = B_ki . D_kl . B_lj
         = B_ki . D_kk . B_kj
         */
      for (ii = 0; ii < 81; ii++) {
        for (jj = ii; jj < 81; jj++) {
          for (kk = 0; kk < 6; kk++) {
            Ae[ii*81+jj] += B[kk][ii]*diagD[kk]*B[kk][jj];
          }
        }
      }

    }


    /* fill lower triangular part */
    for (ii = 0; ii < 81; ii++) {
      for (jj = ii; jj < 81; jj++) {
        Ae[jj*81+ii] = Ae[ii*81+jj];
      }
    }
    PetscTime(&t1q);
    tq += (t1q-t0q);

    ierr = MatSetValues(A,Q2_NODES_PER_EL_3D * U_DOFS,ge_eqnums,Q2_NODES_PER_EL_3D * U_DOFS,ge_eqnums,Ae,ADD_VALUES);CHKERRQ(ierr);

  }
  ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t1);
#ifdef PTAT3D_LOG_ASM_OP
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A11 <geom>: %1.4e (sec)\n",tc);
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A11 <quad>: %1.4e (sec)\n",tq);
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A11:        %1.4e (sec)[flush]\n",t1-t0);
#endif
  ierr = BCListRemoveDirichletMask(NUM_GINDICES,(PetscInt*)GINDICES,u_bclist);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog, &GINDICES);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES);CHKERRQ(ierr);
  
  ierr = BCListInsertScaling(A,NUM_GINDICES,(PetscInt*)GINDICES,u_bclist);CHKERRQ(ierr);
  
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog, &GINDICES);CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t1);
#ifdef PTAT3D_LOG_ASM_OP
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble Auu:        %1.4e (sec)[final]\n",t1-t0);
#endif
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* assemble surface terms */ // add values -> flush assembly
/* zero rows and columns for dirichlet constraints */ // insert values -> flush assembly
/* assemble volume terms */ // add values -> flush assembly
/* zero rows and or columns and push 1.0 diagonally (internally) for dirichlet constraints */ // add values -> final assembly
PetscErrorCode MatAssemble_StokesA_AUU(Mat A,DM dau,BCList u_bclist,Quadrature volQ,SurfBCList surf_bclist)
{
  PetscErrorCode ierr;
  const PetscInt ij[] = { 0, 0 };
  
  ierr = SurfBCList_AssembleAij(surf_bclist,ij,dau,NULL,A);CHKERRQ(ierr);
  if (surf_bclist->sc_nreg > 0) {
    ierr = MatZeroRows_BCList(A,u_bclist,dau,NULL);CHKERRQ(ierr);
    ierr = MatZeroCols_BCList(A,u_bclist,dau,NULL);CHKERRQ(ierr);
  }
  ierr = _MatAssemble_StokesA_AUU(A,dau,u_bclist,volQ,surf_bclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemble_StokesPC_ScaledMassMatrix(Mat A,DM dau,DM dap,BCList p_bclist,Quadrature volQ)
{
  PetscErrorCode ierr;
  PetscInt       p,ngp;
  DM             cda;
  Vec            gcoords;
  PetscReal      *LA_gcoords;
  PetscInt       nel,e,ii,jj;
  PetscInt       nen_u,nen_p;
  PetscInt       p_el_lidx[P_BASIS_FUNCTIONS];
  const PetscInt *elnidx_u;
  const PetscInt *elnidx_p;
  PetscReal      elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal      el_gp_eta[MAX_QUAD_PNTS],one_el_gp_eta[MAX_QUAD_PNTS];
  ISLocalToGlobalMapping ltog;
  const PetscInt *GINDICES_p;
  PetscInt       NUM_GINDICES_p,ge_eqnums_p[P_BASIS_FUNCTIONS];
  PetscReal      Ae[P_BASIS_FUNCTIONS * P_BASIS_FUNCTIONS];
  PetscReal      fac,el_volume,int_eta;
  PetscInt       IJ;

  PetscLogDouble t0,t1;
  QPntVolCoefStokes *all_gausspoints,*cell_gausspoints;
  PetscReal WEIGHT[NQP],XI[NQP][3],NI[NQP][NPE],GNI[NQP][3][NPE],NIp[NQP][P_BASIS_FUNCTIONS];
  PetscReal detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];


  PetscFunctionBegin;

  /* quadrature */
  ngp = volQ->npoints;
  P3D_prepare_elementQ2(ngp,WEIGHT,XI,NI,GNI);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(dap, &ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES_p);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES_p);CHKERRQ(ierr);
  if (p_bclist) {
    ierr = BCListApplyDirichletMask(NUM_GINDICES_p,(PetscInt*)GINDICES_p,p_bclist);CHKERRQ(ierr);
  }

  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);

  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);

  PetscTime(&t0);
  for (e=0;e<nel;e++) {
    /* get local indices */
    ierr = StokesPressure_GetElementLocalIndices(p_el_lidx,(PetscInt*)&elnidx_p[nen_p*e]);CHKERRQ(ierr);

    /* get global indices */
    for (ii=0; ii<P_BASIS_FUNCTIONS; ii++) {
      const int NID = elnidx_p[nen_p*e + ii];

      ge_eqnums_p[ii] = GINDICES_p[ NID ];
    }

    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);

    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);

    for (p=0; p<ngp; p++) {
      PetscScalar xip[] = { XI[p][0], XI[p][1], XI[p][2] };
      ConstructNi_pressure(xip,elcoords,NIp[p]);
    }

    P3D_evaluate_geometry_elementQ2(ngp,elcoords,GNI, detJ,dNudx,dNudy,dNudz);

    /* initialise element stiffness matrix */
    PetscMemzero( Ae, sizeof(PetscScalar)* P_BASIS_FUNCTIONS * P_BASIS_FUNCTIONS );


    /* evaluate the viscosity, 1/eta, cell volume, average cell viscosity */
    el_volume = 0.0;
    int_eta   = 0.0;
    for (p=0; p<ngp; p++) {
      el_gp_eta[p] = cell_gausspoints[p].eta;
      one_el_gp_eta[p] = 1.0/el_gp_eta[p];

      el_volume += 1.0 * WEIGHT[p] * detJ[p]; /* volume */
      int_eta   += el_gp_eta[p] * WEIGHT[p] * detJ[p]; /* volume */
    }
    //avg_eta = int_eta / el_volume;
    //o_avg_eta = 1.0/ avg_eta;

    // <option 1>> - invert each eta on every gp //
    for (p=0; p<ngp; p++) {
      fac = one_el_gp_eta[p] * WEIGHT[p] * detJ[p];

      for (ii=0; ii<P_BASIS_FUNCTIONS; ii++) {
        for (jj=ii; jj<P_BASIS_FUNCTIONS; jj++) {
          IJ = jj + ii*P_BASIS_FUNCTIONS;

          Ae[IJ] -= fac * ( NIp[p][ii] * NIp[p][jj] );
        }
      }
    }

    // <option 2>> - use inverse of cell averaged eta //
    /*
       for (p=0; p<ngp; p++) {
       fac = o_avg_eta * WEIGHT[p] * detJ[p];

       for (ii=0; ii<P_BASIS_FUNCTIONS; ii++) {
       for (jj=0; jj<P_BASIS_FUNCTIONS; jj++) {
       PetscInt IJ = JJ + II*P_BASIS_FUNCTIONS;

       Ae[IJ] += fac * ( NIp[p][ii] * NIp[p][jj] );
       }
       }
       }
       */
    /* copy symmetric part */
    for (ii=0; ii<P_BASIS_FUNCTIONS; ii++) {
      for (jj=ii; jj<P_BASIS_FUNCTIONS; jj++) {
        PetscInt IJ = jj + ii*P_BASIS_FUNCTIONS;

        Ae[ii + jj*P_BASIS_FUNCTIONS] = Ae[IJ];
      }
    }


    ierr = MatSetValues(A,P_BASIS_FUNCTIONS,ge_eqnums_p,P_BASIS_FUNCTIONS,ge_eqnums_p,Ae,ADD_VALUES);CHKERRQ(ierr);

  }
  ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t1);
#ifdef PTAT3D_LOG_ASM_OP
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A22: %1.4e (sec)[flush]\n",t1-t0);
#endif
  if (p_bclist) {
    ierr = BCListRemoveDirichletMask(NUM_GINDICES_p,(PetscInt*)GINDICES_p,p_bclist);CHKERRQ(ierr);
    ierr = BCListInsertScaling(A,NUM_GINDICES_p,(PetscInt*)GINDICES_p,p_bclist);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog, &GINDICES_p);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t1);
#ifdef PTAT3D_LOG_ASM_OP
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A22: %1.4e (sec)[final]\n",t1-t0);
#endif
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode _MatAssemble_StokesA_A12(Mat A,DM dau,DM dap,BCList u_bclist,BCList p_bclist,Quadrature volQ,SurfBCList surf_bclist)
{
  PetscErrorCode ierr;
  PetscInt       p,ngp;
  DM             cda;
  Vec            gcoords;
  PetscReal      *LA_gcoords;
  PetscInt       nel,e,ii,jj,dr;
  PetscInt       nen_u,nen_p;
  PetscInt       vel_el_lidx[3*U_BASIS_FUNCTIONS];
  PetscInt       p_el_lidx[P_BASIS_FUNCTIONS];
  const PetscInt *elnidx_u;
  const PetscInt *elnidx_p;
  PetscReal      elcoords[3*Q2_NODES_PER_EL_3D];
  ISLocalToGlobalMapping ltog_u,ltog_p;
  const PetscInt *GINDICES_p;
  const PetscInt *GINDICES_u;
  PetscInt       NUM_GINDICES_p,ge_eqnums_p[P_BASIS_FUNCTIONS];
  PetscInt       NUM_GINDICES_u,ge_eqnums_u[3*Q2_NODES_PER_EL_3D];
  PetscReal      Ae[3*Q2_NODES_PER_EL_3D * P_BASIS_FUNCTIONS];
  PetscReal      fac;
  PetscInt       IJ;

  PetscLogDouble t0,t1;
  QPntVolCoefStokes *all_gausspoints,*cell_gausspoints;
  PetscReal WEIGHT[NQP],XI[NQP][3],NI[NQP][NPE],GNI[NQP][3][NPE],NIp[NQP][P_BASIS_FUNCTIONS];
  PetscReal detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];


  PetscFunctionBegin;

  /* quadrature */
  ngp = volQ->npoints;
  P3D_prepare_elementQ2(ngp,WEIGHT,XI,NI,GNI);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(dau, &ltog_u);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog_u, &NUM_GINDICES_u);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog_u, &GINDICES_u);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dap, &ltog_p);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog_p, &NUM_GINDICES_p);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog_p, &GINDICES_p);CHKERRQ(ierr);
  if (u_bclist) {
    ierr = BCListApplyDirichletMask(NUM_GINDICES_u,(PetscInt*)GINDICES_u,u_bclist);CHKERRQ(ierr);
  }
  if (p_bclist) {
    ierr = BCListApplyDirichletMask(NUM_GINDICES_p,(PetscInt*)GINDICES_p,p_bclist);CHKERRQ(ierr);
  }

  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);

  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);

  PetscTime(&t0);
  for (e=0;e<nel;e++) {
    /* get local indices */
    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*e]);CHKERRQ(ierr);
    ierr = StokesPressure_GetElementLocalIndices(p_el_lidx,(PetscInt*)&elnidx_p[nen_p*e]);CHKERRQ(ierr);

    /* get global indices */
    // U
    for (ii=0; ii<NPE; ii++) {
      const int NID = elnidx_u[nen_u*e + ii];
      ge_eqnums_u[3*ii  ] = GINDICES_u[ 3*NID   ];
      ge_eqnums_u[3*ii+1] = GINDICES_u[ 3*NID+1 ];
      ge_eqnums_u[3*ii+2] = GINDICES_u[ 3*NID+2 ];
    }
    // P
    for (ii=0; ii<P_BASIS_FUNCTIONS; ii++) {
      const int NID = elnidx_p[nen_p*e + ii];
      ge_eqnums_p[ii] = GINDICES_p[ NID ];
    }

    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);

    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);

    for (p=0; p<ngp; p++) {
      PetscScalar xip[] = { XI[p][0], XI[p][1], XI[p][2] };
      ConstructNi_pressure(xip,elcoords,NIp[p]);
    }

    P3D_evaluate_geometry_elementQ2(ngp,elcoords,GNI, detJ,dNudx,dNudy,dNudz);

    /* initialise element stiffness matrix */
    PetscMemzero( Ae, sizeof(PetscScalar)* 3*Q2_NODES_PER_EL_3D * P_BASIS_FUNCTIONS );


    for (p=0; p<ngp; p++) {
      fac = - WEIGHT[p] * detJ[p]; /* NOTE MINUS SIGN */

      for( ii=0; ii<Q2_NODES_PER_EL_3D; ii++ ){
        for( jj=0; jj<P_BASIS_FUNCTIONS; jj++ ){

          dr = 0;
          IJ = ASS_MAP_wIwDI_uJuDJ( ii,dr,Q2_NODES_PER_EL_3D,3 , jj,0,P_BASIS_FUNCTIONS,1 );
          Ae[IJ]  +=  dNudx[p][ii] * NIp[p][jj] * fac;

          dr = 1;
          IJ = ASS_MAP_wIwDI_uJuDJ( ii,dr,Q2_NODES_PER_EL_3D,3 , jj,0,P_BASIS_FUNCTIONS,1 );
          Ae[IJ]  +=  dNudy[p][ii] * NIp[p][jj] * fac;

          dr = 2;
          IJ = ASS_MAP_wIwDI_uJuDJ( ii,dr,Q2_NODES_PER_EL_3D,3 , jj,0,P_BASIS_FUNCTIONS,1 );
          Ae[IJ]  +=  dNudz[p][ii] * NIp[p][jj] * fac;
        }
      }
    }

    ierr = MatSetValues(A,3*Q2_NODES_PER_EL_3D,ge_eqnums_u,P_BASIS_FUNCTIONS,ge_eqnums_p,Ae,ADD_VALUES);CHKERRQ(ierr);

  }
  ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t1);
#ifdef PTAT3D_LOG_ASM_OP
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A12: %1.4e (sec)[flush]\n",t1-t0);
#endif
  if (u_bclist) {
    ierr = BCListRemoveDirichletMask(NUM_GINDICES_u,(PetscInt*)GINDICES_u,u_bclist);CHKERRQ(ierr);
  }
  if (p_bclist) {
    ierr = BCListRemoveDirichletMask(NUM_GINDICES_p,(PetscInt*)GINDICES_p,p_bclist);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog_u, &GINDICES_u);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog_p, &GINDICES_p);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t1);
#ifdef PTAT3D_LOG_ASM_OP
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A12: %1.4e (sec)[final]\n",t1-t0);
#endif
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* assemble surface terms */ // add values -> flush assembly
/* zero rows for dirichlet constraints */ // insert values -> flush assembly
/* assemble volume terms */ // add values -> flush assembly
PetscErrorCode MatAssemble_StokesA_A12(Mat A,DM dau,DM dap,BCList u_bclist,BCList p_bclist,Quadrature volQ,SurfBCList surf_bclist)
{
  PetscErrorCode ierr;
  const PetscInt ij[] = { 0, 1 };
  
  ierr = SurfBCList_AssembleAij(surf_bclist,ij,dau,dap,A);CHKERRQ(ierr);
  if (surf_bclist->sc_nreg > 0) {
    ierr = MatZeroRows_BCList(A,u_bclist,dau,dap);CHKERRQ(ierr);
  }
  ierr = _MatAssemble_StokesA_A12(A,dau,dap,u_bclist,p_bclist,volQ,surf_bclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _MatAssemble_StokesA_A21(Mat A,DM dau,DM dap,BCList u_bclist,BCList p_bclist,Quadrature volQ,SurfBCList surf_bclist)
{
  PetscErrorCode ierr;
  PetscInt       p,ngp;
  DM             cda;
  Vec            gcoords;
  PetscReal      *LA_gcoords;
  PetscInt       nel,e,ii,jj,dc;
  PetscInt       nen_u,nen_p;
  PetscInt       vel_el_lidx[3*U_BASIS_FUNCTIONS];
  PetscInt       p_el_lidx[P_BASIS_FUNCTIONS];
  const PetscInt *elnidx_u;
  const PetscInt *elnidx_p;
  PetscReal      elcoords[3*Q2_NODES_PER_EL_3D];
  ISLocalToGlobalMapping ltog_u,ltog_p;
  const PetscInt *GINDICES_p;
  PetscInt       NUM_GINDICES_p,ge_eqnums_p[P_BASIS_FUNCTIONS];
  const PetscInt *GINDICES_u;
  PetscInt       NUM_GINDICES_u,ge_eqnums_u[3*Q2_NODES_PER_EL_3D];
  PetscReal      Ae[3*Q2_NODES_PER_EL_3D * P_BASIS_FUNCTIONS];
  PetscReal      fac;
  PetscInt       IJ;

  PetscLogDouble t0,t1;
  QPntVolCoefStokes *all_gausspoints,*cell_gausspoints;
  PetscReal WEIGHT[NQP],XI[NQP][3],NI[NQP][NPE],GNI[NQP][3][NPE],NIp[NQP][P_BASIS_FUNCTIONS];
  PetscReal detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];


  PetscFunctionBegin;

  /* quadrature */
  ngp = volQ->npoints;
  P3D_prepare_elementQ2(ngp,WEIGHT,XI,NI,GNI);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(dau, &ltog_u);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog_u, &NUM_GINDICES_u);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog_u, &GINDICES_u);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dap, &ltog_p);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog_p, &NUM_GINDICES_p);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog_p, &GINDICES_p);CHKERRQ(ierr);
  if (u_bclist) {
    ierr = BCListApplyDirichletMask(NUM_GINDICES_u,(PetscInt*)GINDICES_u,u_bclist);CHKERRQ(ierr);
  }
  if (p_bclist) {
    ierr = BCListApplyDirichletMask(NUM_GINDICES_p,(PetscInt*)GINDICES_p,p_bclist);CHKERRQ(ierr);
  }

  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);

  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);

  PetscTime(&t0);
  for (e=0;e<nel;e++) {
    /* get local indices */
    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*e]);CHKERRQ(ierr);
    ierr = StokesPressure_GetElementLocalIndices(p_el_lidx,(PetscInt*)&elnidx_p[nen_p*e]);CHKERRQ(ierr);

    /* get global indices */
    // U
    for (ii=0; ii<NPE; ii++) {
      const int NID = elnidx_u[nen_u*e + ii];
      ge_eqnums_u[3*ii  ] = GINDICES_u[ 3*NID   ];
      ge_eqnums_u[3*ii+1] = GINDICES_u[ 3*NID+1 ];
      ge_eqnums_u[3*ii+2] = GINDICES_u[ 3*NID+2 ];
    }
    // P
    for (ii=0; ii<P_BASIS_FUNCTIONS; ii++) {
      const int NID = elnidx_p[nen_p*e + ii];
      ge_eqnums_p[ii] = GINDICES_p[ NID ];
    }

    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);

    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);

    for (p=0; p<ngp; p++) {
      PetscScalar xip[] = { XI[p][0], XI[p][1], XI[p][2] };
      ConstructNi_pressure(xip,elcoords,NIp[p]);
    }

    P3D_evaluate_geometry_elementQ2(ngp,elcoords,GNI, detJ,dNudx,dNudy,dNudz);

    /* initialise element stiffness matrix */
    PetscMemzero( Ae, sizeof(PetscScalar)* 3*Q2_NODES_PER_EL_3D * P_BASIS_FUNCTIONS );


    for (p=0; p<ngp; p++) {
      fac = - WEIGHT[p] * detJ[p]; /* NOTE MINUS SIGN */

      for( ii=0; ii<P_BASIS_FUNCTIONS; ii++ ){
        for( jj=0; jj<Q2_NODES_PER_EL_3D; jj++ ){

          dc = 0;
          IJ = ASS_MAP_wIwDI_uJuDJ( ii,0,P_BASIS_FUNCTIONS,1, jj,dc,Q2_NODES_PER_EL_3D,3 );
          Ae[ IJ ] += fac * ( NIp[p][ii] * dNudx[p][jj] );

          dc = 1;
          IJ = ASS_MAP_wIwDI_uJuDJ( ii,0,P_BASIS_FUNCTIONS,1, jj,dc,Q2_NODES_PER_EL_3D,3 );
          Ae[ IJ ] += fac * ( NIp[p][ii] * dNudy[p][jj] );

          dc = 2;
          IJ = ASS_MAP_wIwDI_uJuDJ( ii,0,P_BASIS_FUNCTIONS,1, jj,dc,Q2_NODES_PER_EL_3D,3 );
          Ae[ IJ ] += fac * ( NIp[p][ii] * dNudz[p][jj] );
        }
      }
    }

    ierr = MatSetValues(A,P_BASIS_FUNCTIONS,ge_eqnums_p,3*Q2_NODES_PER_EL_3D,ge_eqnums_u,Ae,ADD_VALUES);CHKERRQ(ierr);

  }
  ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t1);
#ifdef PTAT3D_LOG_ASM_OP
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A21: %1.4e (sec)[flush]\n",t1-t0);
#endif
  if (u_bclist) {
    ierr = BCListRemoveDirichletMask(NUM_GINDICES_u,(PetscInt*)GINDICES_u,u_bclist);CHKERRQ(ierr);
  }
  if (p_bclist) {
    ierr = BCListRemoveDirichletMask(NUM_GINDICES_p,(PetscInt*)GINDICES_p,p_bclist);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog_u, &GINDICES_u);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog_p, &GINDICES_p);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t1);
#ifdef PTAT3D_LOG_ASM_OP
  PetscPrintf(PETSC_COMM_WORLD,"  Assemble A21: %1.4e (sec)[final]\n",t1-t0);
#endif
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* assemble surface terms */ // add values -> flush assembly
/* zero columns for dirichlet constraints */ // insert values -> flush assembly
/* assemble volume terms */ // add values -> flush assembly
PetscErrorCode MatAssemble_StokesA_A21(Mat A,DM dau,DM dap,BCList u_bclist,BCList p_bclist,Quadrature volQ,SurfBCList surf_bclist)
{
  PetscErrorCode ierr;
  const PetscInt ij[] = { 1, 0 };
  
  ierr = SurfBCList_AssembleAij(surf_bclist,ij,dau,dap,A);CHKERRQ(ierr);
  if (surf_bclist->sc_nreg > 0) {
    ierr = MatZeroCols_BCList(A,u_bclist,dau,dap);CHKERRQ(ierr);
  }
  ierr = _MatAssemble_StokesA_A21(A,dau,dap,u_bclist,p_bclist,volQ,surf_bclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
