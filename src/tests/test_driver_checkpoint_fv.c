#include "petsc/private/dmdaimpl.h"

#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin_init.h"
#include "ptatin_log.h"

#include "material_point_utils.h"
#include "material_point_std_utils.h"
#include "material_point_popcontrol.h"
#include "ptatin_models.h"
#include "ptatin_utils.h"
#include "stokes_form_function.h"
#include "stokes_operators.h"
#include "stokes_operators_mf.h"
#include "stokes_assembly.h"
#include "dmda_element_q2p1.h"
#include "dmda_duplicate.h"
#include "dmda_redundant.h"
#include "dmda_project_coords.h"
#include "dmda_update_coords.h"
#include "dmda_checkpoint.h"
#include "monitors.h"
#include "mp_advection.h"
#include "mesh_update.h"

#include "ptatin3d_energy.h"
#include "energy_assembly.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include <fvda_impl.h>

#include <cjson_utils.h>

#define MAX_MG_LEVELS 20
static const char help[] = "Prototype pTatin3D driver using finite volume transport discretisation and checkpointing\n\n";

typedef enum { OP_TYPE_REDISC_ASM=0, OP_TYPE_REDISC_MF, OP_TYPE_GALERKIN, OP_TYPE_MFGALERKIN, OP_TYPE_SNESFD } OperatorType;

typedef struct {
  PetscInt     nlevels;
  OperatorType *level_type;
  Mat          *operatorA11;
  Mat          *operatorB11;
  DM           *dav_hierarchy;
  Mat          *interpolation_v;
  Mat          *interpolation_eta;
  Quadrature   *volQ;
  BCList       *u_bclist;
  IS           *is_stokes_field;
} AuuMultiLevelCtx;

PetscErrorCode MatAssembleMFGalerkin(DM dav_fine,BCList u_bclist_fine,Quadrature volQ_fine,DM dav_coarse,Mat Ac)
{
  PetscInt    refi,refj,refk,nni,nnj,nnk,n,sei,sej,sek;
  PetscInt    ic,jc,kc,iif,jjf,kkf;
  PetscInt    lmk,lmj,lmi;
  PetscInt    lmk_coarse,lmj_coarse,lmi_coarse;
  DM          daf,dac,cda;
  DMBoundaryType wrap[] = { DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE };
  Mat         Acell,Ac_el,P;
  Vec              gcoords;
  PetscReal        *LA_gcoords;
  PetscInt         q,nqp,ii,jj;
  PetscReal         WEIGHT[NQP],XI[NQP][3];
  QPntVolCoefStokes *quadrature_points,*cell_quadrature_points;
  PetscReal      NI[NQP][NPE],GNI[NQP][3][NPE];
  PetscReal      detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];
  PetscInt       nel,nen_v;
  const PetscInt *elnidx_v;
  PetscInt       nel_cell,nen_v_cell;
  const PetscInt *elnidx_v_cell;
  PetscInt       nel_coarse,nen_v_coarse;
  const PetscInt *elnidx_v_coarse;
  PetscReal      el_coords[3*Q2_NODES_PER_EL_3D],el_eta[MAX_QUAD_PNTS];
  PetscReal      Ae[Q2_NODES_PER_EL_3D * Q2_NODES_PER_EL_3D * U_DOFS * U_DOFS];
  PetscReal      fac,diagD[NSTRESS];
  PetscInt       NUM_GINDICES,ge_eqnums[3*Q2_NODES_PER_EL_3D];
  ISLocalToGlobalMapping ltog;
  const PetscInt *GINDICES;
  const PetscInt *GINDICES_cell;
  const PetscInt *GINDICES_coarse;
  PetscInt       NUM_GINDICES_cell,ge_eqnums_cell[3*Q2_NODES_PER_EL_3D];
  PetscInt       NUM_GINDICES_coarse,ge_eqnums_coarse[3*Q2_NODES_PER_EL_3D];
  PetscScalar    *Ac_entries;
  PetscLogDouble t0,t1,t[6];
  PetscErrorCode ierr;
  
  
  ierr = MatZeroEntries(Ac);CHKERRQ(ierr);
  
  ierr = DMDAGetLocalSizeElementQ2(dav_fine,&lmi,&lmj,&lmk);CHKERRQ(ierr);
  ierr = DMDAGetLocalSizeElementQ2(dav_coarse,&lmi_coarse,&lmj_coarse,&lmk_coarse);CHKERRQ(ierr);
  
  ierr = DMDAGetRefinementFactor(dav_fine,&refi,&refj,&refk);CHKERRQ(ierr);
  
  nni = 2*refi + 1;
  nnj = 2*refj + 1;
  nnk = 2*refk + 1;
  
  PetscPrintf(PETSC_COMM_WORLD,"MatAssembleMFGalerkin:\n");
  PetscPrintf(PETSC_COMM_WORLD,"  Q2 cell problem contains: [%D x %D x %D] elements, [%D x %D x %D] nodes\n",refi,refj,refk,nni,nnj,nnk);
  
  ierr = x_DMDACreate3d(PETSC_COMM_SELF,wrap,DMDA_STENCIL_BOX,nni,nnj,nnk, 1,1,1, 3,2, 0,0,0,&daf);CHKERRQ(ierr);
  ierr = DMSetUp(daf);CHKERRQ(ierr);
  ierr = DMDASetElementType_Q2(daf);CHKERRQ(ierr);
  //ierr = DMDASetUniformCoordinates(daf, 0.0,1.0, 0.0,1.0, 0.0,1.0);CHKERRQ(ierr); /* not compatable with using -da_processors_x etc */
  
  ierr = x_DMDACreate3d(PETSC_COMM_SELF,wrap,DMDA_STENCIL_BOX,3,3,3, 1,1,1, 3,2, 0,0,0,&dac);CHKERRQ(ierr);
  ierr = DMSetUp(dac);CHKERRQ(ierr);
  ierr = DMDASetElementType_Q2(dac);CHKERRQ(ierr);
  //ierr = DMDASetUniformCoordinates(dac, 0.0,1.0, 0.0,1.0, 0.0,1.0);CHKERRQ(ierr);
  
  ierr = DMCreateInterpolation(dac,daf,&P,NULL);CHKERRQ(ierr);
  
  /* check nlocal_elements_i is divisible by ref_i: I don't want to build any off proc elements needed for the cell problem */
  ierr = DMDAGetCornersElementQ2(dav_fine,&sei,&sej,&sek,&lmi,&lmj,&lmk);CHKERRQ(ierr);
  if (sei%refi != 0) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Start element (i) must be divisible by %D",refi); }
  if (sej%refj != 0) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Start element (j) must be divisible by %D",refj); }
  if (sek%refk != 0) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Start element (k) must be divisible by %D",refk); }
  
  if (lmi%refi != 0) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Local element size (i) must be divisible by %D",refi); }
  if (lmj%refj != 0) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Local element size (j) must be divisible by %D",refj); }
  if (lmk%refk != 0) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Local element size (k) must be divisible by %D",refk); }
  
  ierr = DMSetMatType(daf,MATSEQAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(daf,&Acell);CHKERRQ(ierr);
  //ierr = MatPtAPSymbolic(Acell,P,1.0,&Ac_el);CHKERRQ(ierr);
  ierr = MatPtAP(Acell,P,MAT_INITIAL_MATRIX,1.0,&Ac_el);CHKERRQ(ierr);
  
  /* quadrature */
  nqp = volQ_fine->npoints;
  P3D_prepare_elementQ2(nqp,WEIGHT,XI,NI,GNI);
  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ_fine,&quadrature_points);CHKERRQ(ierr);
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(dav_fine,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dav_fine,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  /* indices */
  ierr = DMGetLocalToGlobalMapping(dav_fine, &ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES);CHKERRQ(ierr);
  ierr = BCListApplyDirichletMask(NUM_GINDICES,(PetscInt*)GINDICES,u_bclist_fine);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf, &ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES_cell);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES_cell);CHKERRQ(ierr);
  
  ierr = DMGetLocalToGlobalMapping(dav_coarse, &ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES_coarse);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES_coarse);CHKERRQ(ierr);
  
  /* loop and assemble */
  ierr = DMDAGetElements_pTatinQ2P1(dav_fine,&nel,&nen_v,&elnidx_v);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(daf,&nel_cell,&nen_v_cell,&elnidx_v_cell);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dav_coarse,&nel_coarse,&nen_v_coarse,&elnidx_v_coarse);CHKERRQ(ierr);
  
  /* loop over cells on coarse grid */
  t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = 0.0;
  PetscTime(&t[0]);
  for (kc=0; kc<lmk/refk; kc++) {
    for (jc=0; jc<lmj/refj; jc++) {
      for (ic=0; ic<lmi/refi; ic++) {
        PetscInt cidx_coarse;
        
        cidx_coarse = ic + jc*lmi_coarse + kc*lmi_coarse*lmj_coarse;
        
        
        ierr = MatZeroEntries(Acell);CHKERRQ(ierr);
        ierr = MatZeroEntries(Ac_el);CHKERRQ(ierr);
        
        /* look over fine cells contained in coarse (i call this a cell problem) */
        PetscTime(&t0);
        for (kkf=refk*kc; kkf<refk*kc+refk; kkf++) {
          for (jjf=refj*jc; jjf<refj*jc+refj; jjf++) {
            for (iif=refi*ic; iif<refi*ic+refi; iif++) {
              PetscInt cidx;
              PetscInt cidx_cell;
              
              cidx      = iif + jjf*lmi + kkf*lmi*lmj;
              cidx_cell = (iif-refi*ic) + (jjf-refj*jc)*refi + (kkf-refk*kc)*refi*refj;
              
              /* get global indices */
              for (n=0; n<NPE; n++) {
                PetscInt NID;
                
                /* global indices of FE problem */
                NID = elnidx_v[NPE*cidx + n];
                ge_eqnums[3*n  ] = GINDICES[ 3*NID   ];
                ge_eqnums[3*n+1] = GINDICES[ 3*NID+1 ];
                ge_eqnums[3*n+2] = GINDICES[ 3*NID+2 ];
                
                /* local indices of FE problem relative to cell problem */
                NID = elnidx_v_cell[NPE*cidx_cell + n];
                ge_eqnums_cell[3*n  ] = GINDICES_cell[ 3*NID   ];
                ge_eqnums_cell[3*n+1] = GINDICES_cell[ 3*NID+1 ];
                ge_eqnums_cell[3*n+2] = GINDICES_cell[ 3*NID+2 ];
              }
              
              ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&elnidx_v[nen_v*cidx],LA_gcoords);CHKERRQ(ierr);
              
              ierr = VolumeQuadratureGetCellData_Stokes(volQ_fine,quadrature_points,cidx,&cell_quadrature_points);CHKERRQ(ierr);
              
              /* initialise element stiffness matrix */
              PetscMemzero( Ae, sizeof(PetscScalar)* Q2_NODES_PER_EL_3D * Q2_NODES_PER_EL_3D * U_DOFS * U_DOFS );
              
              P3D_evaluate_geometry_elementQ2(nqp,el_coords,GNI, detJ,dNudx,dNudy,dNudz);
              
              /* evaluate the viscosity */
              for (q=0; q<nqp; q++) {
                el_eta[q] = cell_quadrature_points[q].eta;
              }
              
              /* assemble */
              for (q=0; q<nqp; q++) {
                fac = WEIGHT[q] * detJ[q];
                
                diagD[0] = 2.0*fac*el_eta[q ];
                diagD[1] = 2.0*fac*el_eta[q ];
                diagD[2] = 2.0*fac*el_eta[q ];
                
                diagD[3] =     fac*el_eta[q ];
                diagD[4] =     fac*el_eta[q ];
                diagD[5] =     fac*el_eta[q ];
                
                for (ii = 0; ii<NPE; ii++) {
                  PetscScalar dx_i = dNudx[q ][ii];
                  PetscScalar dy_i = dNudy[q ][ii];
                  PetscScalar dz_i = dNudz[q ][ii];
                  
                  for (jj = ii; jj<NPE; jj++) {
                    PetscScalar dx_j = dNudx[q ][jj];
                    PetscScalar dy_j = dNudy[q ][jj];
                    PetscScalar dz_j = dNudz[q ][jj];
                    
                    Ae[(3*ii+0)*81 + (3*jj+0)] += diagD[0]*dx_i*dx_j + diagD[3]*dy_i*dy_j + diagD[4]*dz_i*dz_j; //
                    Ae[(3*ii+0)*81 + (3*jj+1)] += diagD[3]*dy_i*dx_j; //
                    Ae[(3*ii+0)*81 + (3*jj+2)] += diagD[4]*dz_i*dx_j; //
                    
                    Ae[(3*ii+1)*81 + (3*jj+0)] += diagD[3]*dx_i*dy_j; //
                    Ae[(3*ii+1)*81 + (3*jj+1)] += diagD[1]*dy_i*dy_j + diagD[3]*dx_i*dx_j + diagD[5]*dz_i*dz_j; //
                    Ae[(3*ii+1)*81 + (3*jj+2)] += diagD[5]*dz_i*dy_j; //
                    
                    Ae[(3*ii+2)*81 + (3*jj+0)] += diagD[4]*dx_i*dz_j; //
                    Ae[(3*ii+2)*81 + (3*jj+1)] += diagD[5]*dy_i*dz_j; //
                    Ae[(3*ii+2)*81 + (3*jj+2)] += diagD[2]*dz_i*dz_j + diagD[4]*dx_i*dx_j + diagD[5]*dy_i*dy_j; //
                  }
                }
              }
              /* fill lower triangular part */
              for (ii = 0; ii < 81; ii++) {
                for (jj = ii; jj < 81; jj++) {
                  Ae[jj*81+ii] = Ae[ii*81+jj];
                }
              }
              
              /* mask out any row/cols associated with boundary conditions */
              for (n=0; n<3*NPE; n++) {
                if (ge_eqnums[n] < 0) {
                  
                  ii = n;
                  for (jj=0; jj<81; jj++) {
                    Ae[ii*81+jj] = 0.0;
                  }
                  
                  jj = n;
                  for (ii=0; ii<81; ii++) {
                    Ae[ii*81+jj] = 0.0;
                  }
                }
              }
              ierr = MatSetValues(Acell,Q2_NODES_PER_EL_3D * U_DOFS,ge_eqnums_cell,Q2_NODES_PER_EL_3D * U_DOFS,ge_eqnums_cell,Ae,ADD_VALUES);CHKERRQ(ierr);
              
            }
          }
        }
        ierr = MatAssemblyBegin(Acell,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd (Acell,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        
        /* For each cell problem, define the boundary conditions (kinda ugly hack as this doesnt use bc_list_fine */
        PetscTime(&t1);
        t[2] += (t1-t0);
        
        t0 = t1;
        for (kkf=refk*kc; kkf<refk*kc+refk; kkf++) {
          for (jjf=refj*jc; jjf<refj*jc+refj; jjf++) {
            for (iif=refi*ic; iif<refi*ic+refi; iif++) {
              PetscInt cidx;
              PetscInt cidx_cell;
              
              cidx      = iif + jjf*lmi + kkf*lmi*lmj;
              cidx_cell = (iif-refi*ic) + (jjf-refj*jc)*refi + (kkf-refk*kc)*refi*refj;
              
              for (n=0; n<NPE; n++) {
                PetscInt NID;
                
                NID = elnidx_v[NPE*cidx + n];
                ge_eqnums[3*n  ] = GINDICES[ 3*NID   ];
                ge_eqnums[3*n+1] = GINDICES[ 3*NID+1 ];
                ge_eqnums[3*n+2] = GINDICES[ 3*NID+2 ];
                
                NID = elnidx_v_cell[NPE*cidx_cell + n];
                ge_eqnums_cell[3*n  ] = GINDICES_cell[ 3*NID   ];
                ge_eqnums_cell[3*n+1] = GINDICES_cell[ 3*NID+1 ];
                ge_eqnums_cell[3*n+2] = GINDICES_cell[ 3*NID+2 ];
              }
              
              for (n=0; n<3*NPE; n++) {
                if (ge_eqnums[n] < 0) {
                  ii = ge_eqnums_cell[n];
                  ierr = MatSetValue(Acell,ii,ii,1.0,INSERT_VALUES);CHKERRQ(ierr);
                }
              }
            }
          }
        }
        
        ierr = MatAssemblyBegin(Acell,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd (Acell,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        PetscTime(&t1);
        t[3] += (t1-t0);
        
        t0 = t1;
        //ierr = MatPtAPNumeric(Acell,P,Ac_el);CHKERRQ(ierr);
        ierr = MatPtAP(Acell,P,MAT_REUSE_MATRIX,1.0,&Ac_el);CHKERRQ(ierr);
        PetscTime(&t1);
        t[4] += (t1-t0);
        
        /* assemble coarse grid operator */
        
        /* get global indices */
        t0 = t1;
        for (n=0; n<NPE; n++) {
          PetscInt NID;
          
          NID = elnidx_v_coarse[NPE*cidx_coarse + n];
          ge_eqnums_coarse[3*n  ] = GINDICES_coarse[ 3*NID   ];
          ge_eqnums_coarse[3*n+1] = GINDICES_coarse[ 3*NID+1 ];
          ge_eqnums_coarse[3*n+2] = GINDICES_coarse[ 3*NID+2 ];
        }
        ierr = MatSeqAIJGetArray(Ac_el,&Ac_entries);CHKERRQ(ierr);
        
        ierr = MatSetValues(Ac,Q2_NODES_PER_EL_3D*U_DOFS,ge_eqnums_coarse,Q2_NODES_PER_EL_3D*U_DOFS,ge_eqnums_coarse,Ac_entries,ADD_VALUES);CHKERRQ(ierr);
        
        ierr = MatSeqAIJRestoreArray(Ac_el,&Ac_entries);CHKERRQ(ierr);
        PetscTime(&t1);
        t[5] += (t1-t0);
        
      }
    }
  }
  ierr = MatAssemblyBegin(Ac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd (Ac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscTime(&t[1]);
  
  PetscPrintf(PETSC_COMM_WORLD,"  Time[Assemble cell prb]   %1.4e (sec)\n",t[2]);
  PetscPrintf(PETSC_COMM_WORLD,"  Time[BC insertion]        %1.4e (sec)\n",t[3]);
  PetscPrintf(PETSC_COMM_WORLD,"  Time[Form PtAP]           %1.4e (sec)\n",t[4]);
  PetscPrintf(PETSC_COMM_WORLD,"  Time[Assemble coarse prb] %1.4e (sec)\n",t[5]);
  PetscPrintf(PETSC_COMM_WORLD,"  Time[Total]               %1.4e (sec)\n",t[1]-t[0]);
  
  ierr = BCListRemoveDirichletMask(NUM_GINDICES,(PetscInt*)GINDICES,u_bclist_fine);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog, &GINDICES);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog, &GINDICES_cell);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog, &GINDICES_coarse);CHKERRQ(ierr);
  
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = MatDestroy(&Acell);CHKERRQ(ierr);
  ierr = MatDestroy(&Ac_el);CHKERRQ(ierr);
  ierr = DMDestroy(&daf);CHKERRQ(ierr);
  ierr = DMDestroy(&dac);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SNESGetKSP_(SNES snes,SNES *this_snes,KSP *this_ksp)
{
  PetscBool is_ngmres = PETSC_FALSE;
  PetscErrorCode ierr;

  *this_snes = NULL;
  *this_ksp  = NULL;
  ierr = PetscObjectTypeCompare((PetscObject)snes,SNESNGMRES,&is_ngmres);CHKERRQ(ierr);

  if (is_ngmres) {
    ierr = SNESGetNPC(snes,this_snes);CHKERRQ(ierr);
  } else {
    *this_snes = snes;
  }
  ierr = SNESGetKSP(*this_snes,this_ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESComposeWithMGCtx(SNES snes,AuuMultiLevelCtx *mgctx)
{
  PetscErrorCode ierr;
  PetscContainer container;

  PetscFunctionBegin;
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)snes),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,(void*)mgctx);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)snes,"AuuMultiLevelCtx",(PetscObject)container);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SNESDestroyMGCtx(SNES snes)
{
  PetscErrorCode ierr;
  PetscContainer container;

  PetscFunctionBegin;

  container = NULL;
  ierr = PetscObjectQuery((PetscObject)snes,"AuuMultiLevelCtx",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dStokesKSPConfigureFSGMG(KSP ksp,PetscInt nlevels,Mat operatorA11[],Mat operatorB11[],Mat interpolation_v[],DM dav_hierarchy[])
{
  PetscInt k,nsplits;
  PC       pc,pc_i;
  KSP      *sub_ksp,ksp_coarse,ksp_smoother;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCFieldSplitGetSubKSP(pc,&nsplits,&sub_ksp);CHKERRQ(ierr);

  ierr = KSPSetDM(sub_ksp[0],dav_hierarchy[nlevels-1]);CHKERRQ(ierr);
  ierr = KSPSetDMActive(sub_ksp[0],PETSC_FALSE);CHKERRQ(ierr);

  ierr = KSPGetPC(sub_ksp[0],&pc_i);CHKERRQ(ierr);
  ierr = PCSetType(pc_i,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc_i,nlevels,NULL);CHKERRQ(ierr);
  ierr = PCMGSetType(pc_i,PC_MG_MULTIPLICATIVE);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc_i,PC_MG_GALERKIN_NONE);CHKERRQ(ierr);
  ierr = PCSetDM(pc_i,NULL);CHKERRQ(ierr);

  for( k=1; k<nlevels; k++ ){
    ierr = PCMGSetInterpolation(pc_i,k,interpolation_v[k]);CHKERRQ(ierr);
  }

  /* drop the operators in - i presume this will also need to be performed inside the jacobian each time the operators are modified */
  /* No - it looks like PCSetUp_MG will call set operators on all levels if the SetOperators was called on the finest, which should/is done by the SNES */
  ierr = PCMGGetCoarseSolve(pc_i,&ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_coarse,operatorA11[0],operatorA11[0]);CHKERRQ(ierr);

  ierr = KSPSetDM(ksp_coarse,dav_hierarchy[0]);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp_coarse,PETSC_FALSE);CHKERRQ(ierr);

  for( k=1; k<nlevels; k++ ){
    PetscBool use_low_order_geometry = PETSC_FALSE;

    ierr = PCMGGetSmoother(pc_i,k,&ksp_smoother);CHKERRQ(ierr);

    // use A for smoother, B for residual
    ierr = PetscOptionsGetBool(NULL,NULL,"-use_low_order_geometry",&use_low_order_geometry,NULL);CHKERRQ(ierr);
    if (use_low_order_geometry==PETSC_TRUE) {
      ierr = KSPSetOperators(ksp_smoother,operatorB11[k],operatorB11[k]);CHKERRQ(ierr);
      //ierr = KSPSetOperators(ksp_smoother,operatorA11[k],operatorB11[k]);CHKERRQ(ierr);
    } else {
      // Use A for smoother, lo
      ierr = KSPSetOperators(ksp_smoother,operatorA11[k],operatorA11[k]);CHKERRQ(ierr);
    }
    ierr = KSPSetDM(ksp_smoother,dav_hierarchy[k]);CHKERRQ(ierr);
    ierr = KSPSetDMActive(ksp_smoother,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFree(sub_ksp);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dStokesBuildMeshHierarchy(DM dav,PetscInt nlevels,DM dav_hierarchy[])
{
  PetscErrorCode ierr;
  DM *coarsened_list;
  PetscInt k;

  PetscFunctionBegin;

  /* set up mg */
  dav->ops->coarsenhierarchy = DMCoarsenHierarchy2_DA;

  dav_hierarchy[ nlevels-1 ] = dav;
  ierr = PetscObjectReference((PetscObject)dav);CHKERRQ(ierr);

  /* Coarsen nlevels - 1 times, and add levels into list so that level 0 is the coarsest */
  ierr = PetscMalloc(sizeof(DM)*(nlevels-1),&coarsened_list);CHKERRQ(ierr);
  ierr = DMCoarsenHierarchy(dav,nlevels-1,coarsened_list);CHKERRQ(ierr);
  for (k=0; k<nlevels-1; k++) {
    dav_hierarchy[ nlevels-2-k ] = coarsened_list[k];
  }
  PetscFree(coarsened_list);

  /* Set all dav's to be of type Q2 */
  for (k=0; k<nlevels-1; k++) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)dav_hierarchy[k],"stk_velocity_");CHKERRQ(ierr);
    ierr = DMDASetElementType_Q2(dav_hierarchy[k]);CHKERRQ(ierr);
  }

  /* inject coordinates */
  ierr = DMDARestrictCoordinatesHierarchy(dav_hierarchy,nlevels);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dStokesReportMeshHierarchy(PetscInt nlevels,DM dav_hierarchy[])
{
  PetscErrorCode ierr;
  PetscInt       k,lmx,lmy,lmz;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Report mesh sizes */
  for (k=0; k<nlevels; k++) {
    ierr = DMDAGetSizeElementQ2(dav_hierarchy[k],&lmx,&lmy,&lmz);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"         level [%2D]: global Q2 elements (%D x %D x %D) \n", k,lmx,lmy,lmz );
  }
  
  for (k=0; k<nlevels; k++) {
    PetscInt mp,np,pp,*_mx,*_my,*_mz,ii,jj,kk;

    ierr = DMDAGetOwnershipRangesElementQ2(dav_hierarchy[k],&mp,&np,&pp,NULL,NULL,NULL,&_mx,&_my,&_mz);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"level [%2D]: [total cores %4D]: np-I [%4D]: element range I [ ", k,size,mp );
    for (ii=0; ii<mp; ii++) {
      PetscPrintf(PETSC_COMM_WORLD,"%4D", _mx[ii] );
      if (ii != mp-1) { PetscPrintf(PETSC_COMM_WORLD,", "); }
    }PetscPrintf(PETSC_COMM_WORLD," ]\n");

    PetscPrintf(PETSC_COMM_WORLD,"                                np-J [%4D]: element range J [ ",np);
    for (jj=0; jj<np; jj++) {
      PetscPrintf(PETSC_COMM_WORLD,"%4D", _my[jj] );
      if (jj != np-1) { PetscPrintf(PETSC_COMM_WORLD,", "); }
    }PetscPrintf(PETSC_COMM_WORLD," ]\n");

    PetscPrintf(PETSC_COMM_WORLD,"                                np-K [%4D]: element range K [ ",pp);
    for (kk=0; kk<pp; kk++) {
      PetscPrintf(PETSC_COMM_WORLD,"%4D", _mz[kk] );
      if (kk != pp-1) { PetscPrintf(PETSC_COMM_WORLD,", "); }
    }PetscPrintf(PETSC_COMM_WORLD," ]\n");

    ierr = PetscFree(_mx);CHKERRQ(ierr);
    ierr = PetscFree(_my);CHKERRQ(ierr);
    ierr = PetscFree(_mz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dCreateStokesOperators(PhysCompStokes stokes_ctx,IS is_stokes_field[],
                                             PetscInt nlevels,DM dav_hierarchy[],Mat interpolation_v[],
                                             BCList u_bclist[],Quadrature volQ[],
                                             OperatorType level_type[],
                                             Mat *_A,Mat operatorA11[],Mat *_B,Mat operatorB11[])
{
  Mat            A,B;
  DM             dap;
  PetscInt       k,max;
  PetscBool      flg;
  PetscInt       _level_type[MAX_MG_LEVELS];
  static int     been_here = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dap = stokes_ctx->dap;

  /* A operator */
  ierr = StokesQ2P1CreateMatrix_Operator(stokes_ctx,&A);CHKERRQ(ierr);
  /* memory saving - only need daU IF you want to split A11 into A11uu,A11vv,A11ww */
  {
    MatStokesMF mf;

    ierr = MatShellGetMatStokesMF(A,&mf);CHKERRQ(ierr);
    ierr = DMDestroy(&mf->daU);CHKERRQ(ierr);
    mf->daU = NULL;
    ierr = ISDestroy(&mf->isU);CHKERRQ(ierr);
    ierr = ISDestroy(&mf->isV);CHKERRQ(ierr);
    ierr = ISDestroy(&mf->isW);CHKERRQ(ierr);
  }

  /* B operator */
  {
    Mat         Aup,Apu,Spp,bA[2][2];
    MatStokesMF StkCtx;

    ierr = MatShellGetMatStokesMF(A,&StkCtx);CHKERRQ(ierr);

    /* Schur complement */
    //ierr = DMSetMatType(dap,MATSBAIJ);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dap,&Spp);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(Spp,"S*_");CHKERRQ(ierr);
    ierr = MatSetOption(Spp,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Spp);CHKERRQ(ierr);

    /* A12 */
    ierr = StokesQ2P1CreateMatrix_MFOperator_A12(StkCtx,&Aup);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(Aup,"Bup_");CHKERRQ(ierr);
    ierr = MatSetFromOptions(Aup);CHKERRQ(ierr);

    /* A21 */
    ierr = StokesQ2P1CreateMatrix_MFOperator_A21(StkCtx,&Apu);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(Apu,"Bpu_");CHKERRQ(ierr);
    ierr = MatSetFromOptions(Apu);CHKERRQ(ierr);

    /* nest */
    bA[0][0] = NULL; bA[0][1] = Aup;
    bA[1][0] = Apu;  bA[1][1] = Spp;

    ierr = MatCreateNest(PETSC_COMM_WORLD,2,is_stokes_field,2,is_stokes_field,&bA[0][0],&B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* tidy up - hand back destruction to B */
    ierr = MatDestroy(&Aup);CHKERRQ(ierr);
    ierr = MatDestroy(&Apu);CHKERRQ(ierr);
    ierr = MatDestroy(&Spp);CHKERRQ(ierr);
  }

  /* A11 operator */
  /* defaults */
  _level_type[0] = (PetscInt)OP_TYPE_REDISC_ASM;
  for (k=1; k<nlevels; k++) {
    _level_type[k] = (PetscInt)OP_TYPE_REDISC_MF;
  }

  max = nlevels;
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-A11_operator_type",_level_type,&max,&flg);CHKERRQ(ierr);
  for (k=nlevels-1; k>=0; k--) {
    level_type[k] = (OperatorType)_level_type[k];
  }
  for (k=nlevels-1; k>=0; k--) {

    switch (level_type[k]) {

      case OP_TYPE_REDISC_ASM:
      {
        Mat Auu;
        PetscBool same1 = PETSC_FALSE,same2 = PETSC_FALSE,same3 = PETSC_FALSE;
        Vec X;
        MatNullSpace nullsp;

        /* use -stk_velocity_da_mat_type sbaij or -Buu_da_mat_type sbaij */
        if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: Re-discretisation :: assembled operator \n", k);
        //ierr = DMSetMatType(dav_hierarchy[k],MATSBAIJ);CHKERRQ(ierr);
        ierr = DMCreateMatrix(dav_hierarchy[k],&Auu);CHKERRQ(ierr);
        ierr = MatSetOptionsPrefix(Auu,"Buu_");CHKERRQ(ierr);
        ierr = MatSetFromOptions(Auu);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)Auu,MATSBAIJ,&same1);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)Auu,MATSEQSBAIJ,&same2);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)Auu,MATMPISBAIJ,&same3);CHKERRQ(ierr);
        if (same1||same2||same3) {
          ierr = MatSetOption(Auu,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
        }
        /* should move assembly into jacobian */
        ierr = MatZeroEntries(Auu);CHKERRQ(ierr);
        ierr = MatAssemble_StokesA_AUU(Auu,dav_hierarchy[k],u_bclist[k],volQ[k]);CHKERRQ(ierr);

        operatorA11[k] = Auu;
        operatorB11[k] = Auu;
        ierr = PetscObjectReference((PetscObject)Auu);CHKERRQ(ierr);
        ierr = DMGetCoordinates(dav_hierarchy[k],&X);CHKERRQ(ierr);
        ierr = MatNullSpaceCreateRigidBody(X,&nullsp);CHKERRQ(ierr);
        ierr = MatSetNearNullSpace(Auu,nullsp);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);

      }
        break;

      case OP_TYPE_REDISC_MF:
      {
        Mat Auu;
        MatA11MF mf,A11Ctx;

        if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: Re-discretisation :: matrix free operator \n", k);
        ierr = MatA11MFCreate(&A11Ctx);CHKERRQ(ierr);
        ierr = MatA11MFSetup(A11Ctx,dav_hierarchy[k],volQ[k],u_bclist[k]);CHKERRQ(ierr);

        ierr = StokesQ2P1CreateMatrix_MFOperator_A11(A11Ctx,&Auu);CHKERRQ(ierr);
        /* memory saving - only need daU IF you want to split A11 into A11uu,A11vv,A11ww */
        ierr = MatShellGetMatA11MF(Auu,&mf);CHKERRQ(ierr);
        ierr = DMDestroy(&mf->daU);CHKERRQ(ierr);
        mf->daU = NULL;
        ierr = ISDestroy(&mf->isU);CHKERRQ(ierr);
        ierr = ISDestroy(&mf->isV);CHKERRQ(ierr);
        ierr = ISDestroy(&mf->isW);CHKERRQ(ierr);
        /* --- */
        operatorA11[k] = Auu;

        {
          PetscBool use_low_order_geometry = PETSC_FALSE;

          ierr = PetscOptionsGetBool(NULL,NULL,"-use_low_order_geometry",&use_low_order_geometry,NULL);CHKERRQ(ierr);
          if (use_low_order_geometry==PETSC_TRUE) {
            Mat Buu;

            if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Activiting low order A11 operator \n");
            ierr = StokesQ2P1CreateMatrix_MFOperator_A11LowOrder(A11Ctx,&Buu);CHKERRQ(ierr);
            /* memory saving - only need daU IF you want to split A11 into A11uu,A11vv,A11ww */
            ierr = MatShellGetMatA11MF(Buu,&mf);CHKERRQ(ierr);
            ierr = DMDestroy(&mf->daU);CHKERRQ(ierr);
            mf->daU = NULL;
            ierr = ISDestroy(&mf->isU);CHKERRQ(ierr);
            ierr = ISDestroy(&mf->isV);CHKERRQ(ierr);
            ierr = ISDestroy(&mf->isW);CHKERRQ(ierr);
            /* --- */
            operatorB11[k] = Buu;

          } else {
            operatorB11[k] = Auu;
            ierr = PetscObjectReference((PetscObject)Auu);CHKERRQ(ierr);
          }
        }

        ierr = MatA11MFDestroy(&A11Ctx);CHKERRQ(ierr);
      }
        break;

      case OP_TYPE_GALERKIN:
      {
        Mat Auu;
        Vec X;
        MatNullSpace nullsp;

        if (k==nlevels-1) {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Cannot use galerkin coarse grid on the finest level");
        }

        if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: Galerkin :: assembled operator \n", k);

        /* should move coarse grid assembly into jacobian */
        ierr = MatPtAP(operatorA11[k+1],interpolation_v[k+1],MAT_INITIAL_MATRIX,1.0,&Auu);CHKERRQ(ierr);

        operatorA11[k] = Auu;
        operatorB11[k] = Auu;
        ierr = PetscObjectReference((PetscObject)Auu);CHKERRQ(ierr);
        ierr = DMGetCoordinates(dav_hierarchy[k],&X);CHKERRQ(ierr);
        ierr = MatNullSpaceCreateRigidBody(X,&nullsp);CHKERRQ(ierr);
        ierr = MatSetBlockSize(Auu,3);CHKERRQ(ierr);
        ierr = MatSetNearNullSpace(Auu,nullsp);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
      }
        break;

      case OP_TYPE_MFGALERKIN:
      {
        Mat Auu;
        Vec X;
        MatNullSpace nullsp;
        
        if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: MFGalerkin :: assembled operator \n", k);
        if (k == nlevels-1) {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Cannot use mf-galerkin coarse grid on the finest level");
        }
        if (level_type[k+1] != OP_TYPE_REDISC_MF) {
          SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Cannot use mf-galerkin. Next finest level[%D] must be of type OP_TYPE_REDISC_MF",k+1);
        }
        
        ierr = DMSetMatType(dav_hierarchy[k],MATAIJ);CHKERRQ(ierr);
        ierr = DMCreateMatrix(dav_hierarchy[k],&Auu);CHKERRQ(ierr);
        ierr = MatSetOptionsPrefix(Auu,"Buu_mfg_");CHKERRQ(ierr);
        ierr = MatSetFromOptions(Auu);CHKERRQ(ierr);

        ierr = MatAssembleMFGalerkin(dav_hierarchy[k+1],u_bclist[k+1],volQ[k+1],dav_hierarchy[k],Auu);CHKERRQ(ierr);
        
        operatorA11[k] = Auu;
        operatorB11[k] = Auu;
        ierr = PetscObjectReference((PetscObject)Auu);CHKERRQ(ierr);
        
        ierr = DMGetCoordinates(dav_hierarchy[k],&X);CHKERRQ(ierr);
        ierr = MatNullSpaceCreateRigidBody(X,&nullsp);CHKERRQ(ierr);
        ierr = MatSetBlockSize(Auu,3);CHKERRQ(ierr);
        ierr = MatSetNearNullSpace(Auu,nullsp);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
      }
        break;
        
      case OP_TYPE_SNESFD:
      {
        Mat Auu;
        Vec X;
        MatNullSpace nullsp;
        
        if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: SNESFD :: assembled operator \n", k);
        if (k != nlevels-1) {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Cannot only use snes-fd on the finest level");
        }
        
        ierr = DMSetMatType(dav_hierarchy[k],MATAIJ);CHKERRQ(ierr);
        ierr = DMCreateMatrix(dav_hierarchy[k],&Auu);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(Auu,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Auu,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetOptionsPrefix(Auu,"Buu_fdu_");CHKERRQ(ierr);
        ierr = MatSetFromOptions(Auu);CHKERRQ(ierr);
        
        operatorA11[k] = Auu;
        operatorB11[k] = Auu;
        ierr = PetscObjectReference((PetscObject)Auu);CHKERRQ(ierr);
        
        ierr = DMGetCoordinates(dav_hierarchy[k],&X);CHKERRQ(ierr);
        ierr = MatNullSpaceCreateRigidBody(X,&nullsp);CHKERRQ(ierr);
        ierr = MatSetBlockSize(Auu,3);CHKERRQ(ierr);
        //ierr = MatSetNearNullSpace(Auu,nullsp);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
      }
        break;
        
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must choose a coarse grid constructor");
        break;
    }
  }

  /* Set fine A11 into nest */
  ierr = MatNestSetSubMat(B,0,0,operatorA11[nlevels-1]);CHKERRQ(ierr);

  *_A = A;
  *_B = B;

  been_here = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_StokesMGAuu(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  pTatinCtx         user;
  AuuMultiLevelCtx  *mlctx;
  DM                stokes_pack,dau,dap;
  PhysCompStokes    stokes;
  Vec               Uloc,Ploc;
  PetscScalar       *LA_Uloc,*LA_Ploc;
  PetscBool         is_mffd = PETSC_FALSE;
  PetscBool         is_nest = PETSC_FALSE;
  PetscBool         is_shell = PETSC_FALSE;
  PetscContainer    container;
  PetscInt          k;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  user = (pTatinCtx)ctx;

  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;

  ierr = PetscObjectQuery((PetscObject)snes,"AuuMultiLevelCtx",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"No data with name \"AuuMultiLevelCtx\" was composed with SNES");
  ierr = PetscContainerGetPointer(container,(void*)&mlctx);CHKERRQ(ierr);


  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  ierr = DMCompositeScatter(stokes_pack,X,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);

  /* nonlinearitiers: markers => quad points */
  ierr = pTatin_EvaluateRheologyNonlinearities(user,dau,LA_Uloc,dap,LA_Ploc);CHKERRQ(ierr);
  /* Solve lithostatic pressure and apply on the surface quadrature points for Stokes */
  ierr = ModelApplyTractionFromLithoPressure(user,X);CHKERRQ(ierr);
  
  /* interpolate coefficients */
  {
    int               npoints;
    DataField         PField_std;
    DataField         PField_stokes;
    MPntStd           *mp_std;
    MPntPStokes       *mp_stokes;

    DataBucketGetDataFieldByName(user->materialpoint_db, MPntStd_classname     , &PField_std);
    DataBucketGetDataFieldByName(user->materialpoint_db, MPntPStokes_classname , &PField_stokes);

    DataBucketGetSizes(user->materialpoint_db,&npoints,NULL,NULL);
    mp_std    = PField_std->data; /* should write a function to do this */
    mp_stokes = PField_stokes->data; /* should write a function to do this */

    ierr = SwarmUpdateGaussPropertiesLocalL2Projection_Q1_MPntPStokes_Hierarchy(user->coefficient_projection_type,npoints,mp_std,mp_stokes,mlctx->nlevels,mlctx->interpolation_eta,mlctx->dav_hierarchy,mlctx->volQ);CHKERRQ(ierr);
  }

  /* clean up */
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  
  ierr = DMCompositeRestoreLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);
  
  
  /* operator */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMFFD, &is_mffd);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATNEST, &is_nest);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSHELL,&is_shell);CHKERRQ(ierr);

  if (is_nest) {
    Mat Auu;

    ierr = MatCreateSubMatrix(A,mlctx->is_stokes_field[0],mlctx->is_stokes_field[0],MAT_INITIAL_MATRIX,&Auu);CHKERRQ(ierr);

    is_shell = PETSC_FALSE;
    ierr = PetscObjectTypeCompare((PetscObject)Auu,MATSHELL,&is_shell);CHKERRQ(ierr);
    if (!is_shell) {
      ierr = MatZeroEntries(Auu);CHKERRQ(ierr);
      

      k = mlctx->nlevels-1;
      switch (mlctx->level_type[k]) {
          
        case OP_TYPE_REDISC_ASM:
          ierr = MatAssemble_StokesA_AUU(Auu,dau,user->stokes_ctx->u_bclist,user->stokes_ctx->volQ);CHKERRQ(ierr);
          break;
          
        case OP_TYPE_SNESFD:
        {
          Mat Auu_k;
          Vec Xu,Xp,_X,_F;
          SNES _snes;
          
          Auu_k = Auu;
          
          ierr = DMCreateGlobalVector(mlctx->dav_hierarchy[k],&_F);CHKERRQ(ierr);
          ierr = DMCreateGlobalVector(mlctx->dav_hierarchy[k],&_X);CHKERRQ(ierr);
          ierr = DMCompositeGetAccess(stokes_pack,X,&Xu,&Xp);CHKERRQ(ierr);
          ierr = VecCopy(Xu,_X);CHKERRQ(ierr);
          ierr = DMCompositeRestoreAccess(stokes_pack,X,&Xu,&Xp);CHKERRQ(ierr);
          
          ierr = SNESCreate(PETSC_COMM_WORLD,&_snes);CHKERRQ(ierr);
          ierr = SNESSetDM(_snes,mlctx->dav_hierarchy[k]);CHKERRQ(ierr);
          ierr = SNESSetOptionsPrefix(_snes,"fdu_");CHKERRQ(ierr);
          ierr = SNESSetSolution(_snes,_X);CHKERRQ(ierr);
          ierr = SNESSetFunction(_snes,_F,FormFunction_StokesU,(void*)ctx);CHKERRQ(ierr);
          ierr = SNESSetJacobian(_snes,Auu_k,Auu_k,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
          ierr = SNESSetFromOptions(_snes);CHKERRQ(ierr);
          ierr = SNESSetUp(_snes);CHKERRQ(ierr);
          
          //ierr = SNESSolve(_snes,NULL,_X);CHKERRQ(ierr);
          ierr = SNESComputeFunction(_snes,_X,_F);CHKERRQ(ierr); /* not sure what setup is not done until SNESComputeFunction() or SNESSolve() is called */
          ierr = SNESComputeJacobian(_snes,_X,Auu_k,Auu_k);CHKERRQ(ierr);
          
          ierr = SNESDestroy(&_snes);CHKERRQ(ierr);
          ierr = VecDestroy(&_X);CHKERRQ(ierr);
          ierr = VecDestroy(&_F);CHKERRQ(ierr);
        }
          break;
          
        default:
          break;
      }

    }

    ierr = MatDestroy(&Auu);CHKERRQ(ierr);
  }
  /* If shell, do nothing */
  /* If mffd,  do nothing */

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* preconditioner operator for Jacobian */
  {
    Mat Buu,Bpp;

    ierr = MatCreateSubMatrix(B,mlctx->is_stokes_field[0],mlctx->is_stokes_field[0],MAT_INITIAL_MATRIX,&Buu);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(B,mlctx->is_stokes_field[1],mlctx->is_stokes_field[1],MAT_INITIAL_MATRIX,&Bpp);CHKERRQ(ierr);

    is_shell = PETSC_FALSE;
    ierr = PetscObjectTypeCompare((PetscObject)Buu,MATSHELL,&is_shell);CHKERRQ(ierr);
    if (!is_shell) {
      ierr = MatZeroEntries(Buu);CHKERRQ(ierr);
      
      k = mlctx->nlevels-1;
      switch (mlctx->level_type[k]) {
          
        case OP_TYPE_REDISC_ASM:
          ierr = MatAssemble_StokesA_AUU(Buu,dau,user->stokes_ctx->u_bclist,user->stokes_ctx->volQ);CHKERRQ(ierr);
          break;

        case OP_TYPE_SNESFD:
        {
          Mat Auu_k;
          Vec Xu,Xp,_X,_F;
          SNES _snes;
          
          Auu_k = Buu;
          
          ierr = DMCreateGlobalVector(mlctx->dav_hierarchy[k],&_F);CHKERRQ(ierr);
          ierr = DMCreateGlobalVector(mlctx->dav_hierarchy[k],&_X);CHKERRQ(ierr);
          ierr = DMCompositeGetAccess(stokes_pack,X,&Xu,&Xp);CHKERRQ(ierr);
          ierr = VecCopy(Xu,_X);CHKERRQ(ierr);
          ierr = DMCompositeRestoreAccess(stokes_pack,X,&Xu,&Xp);CHKERRQ(ierr);

          ierr = SNESCreate(PETSC_COMM_WORLD,&_snes);CHKERRQ(ierr);
          ierr = SNESSetDM(_snes,mlctx->dav_hierarchy[k]);CHKERRQ(ierr);
          ierr = SNESSetOptionsPrefix(_snes,"fdu_");CHKERRQ(ierr);
          ierr = SNESSetSolution(_snes,_X);CHKERRQ(ierr);
          ierr = SNESSetFunction(_snes,_F,FormFunction_StokesU,(void*)ctx);CHKERRQ(ierr);
          ierr = SNESSetJacobian(_snes,Auu_k,Auu_k,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
          ierr = SNESSetFromOptions(_snes);CHKERRQ(ierr);
          ierr = SNESSetUp(_snes);CHKERRQ(ierr);
          
          //ierr = SNESSolve(_snes,NULL,_X);CHKERRQ(ierr);
          ierr = SNESComputeFunction(_snes,_X,_F);CHKERRQ(ierr); /* not sure what setup is not done until SNESComputeFunction() or SNESSolve() is called */
          ierr = SNESComputeJacobian(_snes,_X,Auu_k,Auu_k);CHKERRQ(ierr);
          
          ierr = SNESDestroy(&_snes);CHKERRQ(ierr);
          ierr = VecDestroy(&_X);CHKERRQ(ierr);
          ierr = VecDestroy(&_F);CHKERRQ(ierr);
        }
          break;
          
        default:
          break;
      }
      
      
      
    }

    is_shell = PETSC_FALSE;
    ierr = PetscObjectTypeCompare((PetscObject)Bpp,MATSHELL,&is_shell);CHKERRQ(ierr);
    if (!is_shell) {
      ierr = MatZeroEntries(Bpp);CHKERRQ(ierr);
      ierr = MatAssemble_StokesPC_ScaledMassMatrix(Bpp,dau,dap,user->stokes_ctx->p_bclist,user->stokes_ctx->volQ);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&Buu);CHKERRQ(ierr);
    ierr = MatDestroy(&Bpp);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Buu preconditioner for all other levels in the hierarchy */
  {
    PetscBool use_low_order_geometry;
    SNES      this_snes;
    KSP       ksp,*sub_ksp,ksp_smoother;
    PC        pc,pc_i;
    PetscInt  nsplits;

    use_low_order_geometry = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-use_low_order_geometry",&use_low_order_geometry,NULL);CHKERRQ(ierr);

    ierr = SNESGetKSP_(snes,&this_snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCFieldSplitGetSubKSP(pc,&nsplits,&sub_ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(sub_ksp[0],&pc_i);CHKERRQ(ierr);

    for (k=mlctx->nlevels-2; k>=0; k--) {
      /* fetch smoother */
      if (k == 0) {
        ierr = PCMGGetCoarseSolve(pc_i,&ksp_smoother);CHKERRQ(ierr);
      } else {
        ierr = PCMGGetSmoother(pc_i,k,&ksp_smoother);CHKERRQ(ierr);
      }

      switch (mlctx->level_type[k]) {

        case OP_TYPE_REDISC_ASM:
        {
          ierr = MatZeroEntries(mlctx->operatorB11[k]);CHKERRQ(ierr);
          ierr = MatAssemble_StokesA_AUU(mlctx->operatorB11[k],mlctx->dav_hierarchy[k],mlctx->u_bclist[k],mlctx->volQ[k]);CHKERRQ(ierr);

          ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          /* hack for nested coarse solver */
          {
            KSP ksp_nested;
            PC pc_smoother;
            PetscBool is_nested_ksp;

            ierr = KSPGetPC(ksp_smoother,&pc_smoother);CHKERRQ(ierr);
            is_nested_ksp = PETSC_FALSE;
            ierr = PetscObjectTypeCompare((PetscObject)pc_smoother,PCKSP,&is_nested_ksp);CHKERRQ(ierr);
            if (is_nested_ksp) {
              ierr = PCKSPGetKSP(pc_smoother,&ksp_nested);CHKERRQ(ierr);
              ierr = KSPSetOperators(ksp_nested,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
            }
          }

          /* no low order assembly */
          /*
          if (use_low_order_geometry==PETSC_TRUE) {
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          } else {
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorA11[k]);CHKERRQ(ierr);
          }
          */
        }
          break;

        case OP_TYPE_REDISC_MF:
        {
          if (use_low_order_geometry == PETSC_TRUE) {
            //  ierr = KSPSetOperators(ksp_smoother,operatorB11[k],operatorB11[k]);CHKERRQ(ierr);
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          } else {
            //  ierr = KSPSetOperators(ksp_smoother,operatorA11[k],operatorA11[k]);CHKERRQ(ierr);
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorA11[k]);CHKERRQ(ierr);
          }
        }
          break;

        case OP_TYPE_GALERKIN:
        {
          Mat Auu_k;

          /*
          ierr = MatPtAP(mlctx->operatorA11[k+1],mlctx->interpolation_v[k+1],MAT_INITIAL_MATRIX,1.0,&Auu_k);CHKERRQ(ierr);
          ierr = KSPSetOperators(ksp_smoother,Auu_k,Auu_k);CHKERRQ(ierr);
          mlctx->operatorA11[k] = Auu_k;
          mlctx->operatorB11[k] = Auu_k;
          ierr = PetscObjectReference((PetscObject)Auu_k);CHKERRQ(ierr);
          */
          Auu_k = mlctx->operatorA11[k];
          ierr = MatPtAP(mlctx->operatorA11[k+1],mlctx->interpolation_v[k+1],MAT_REUSE_MATRIX,1.0,&Auu_k);CHKERRQ(ierr);
          ierr = KSPSetOperators(ksp_smoother,Auu_k,Auu_k);CHKERRQ(ierr);
          mlctx->operatorB11[k] = Auu_k;
        }
          break;

        case OP_TYPE_MFGALERKIN:
        {
          Mat Auu_k;

          Auu_k = mlctx->operatorA11[k];
          ierr = MatZeroEntries(Auu_k);CHKERRQ(ierr);
          
          ierr = MatAssembleMFGalerkin(mlctx->dav_hierarchy[k+1],mlctx->u_bclist[k+1],mlctx->volQ[k+1],mlctx->dav_hierarchy[k],Auu_k);CHKERRQ(ierr);
          
          ierr = KSPSetOperators(ksp_smoother,Auu_k,Auu_k);CHKERRQ(ierr);
          mlctx->operatorB11[k] = Auu_k;
        }
          break;

        default:
          break;
      }
    }
    PetscFree(sub_ksp);

    /* push operators */
    for (k=mlctx->nlevels-1; k>=0; k--) {
      SNES this_snes;

      ierr = SNESGetKSP_(snes,&this_snes,&ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCFieldSplitGetSubKSP(pc,&nsplits,&sub_ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(sub_ksp[0],&pc_i);CHKERRQ(ierr);

      if (k == 0) {
        ierr = PCMGGetCoarseSolve(pc_i,&ksp_smoother);CHKERRQ(ierr);
      } else {
        ierr = PCMGGetSmoother(pc_i,k,&ksp_smoother);CHKERRQ(ierr);
      }

      switch (mlctx->level_type[k]) {

        case OP_TYPE_REDISC_ASM:
        {
          ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          /* hack for nested coarse solver */
          {
            KSP ksp_nested;
            PC pc_smoother;
            PetscBool is_nested_ksp;

            ierr = KSPGetPC(ksp_smoother,&pc_smoother);CHKERRQ(ierr);
            is_nested_ksp = PETSC_FALSE;
            ierr = PetscObjectTypeCompare((PetscObject)pc_smoother,PCKSP,&is_nested_ksp);CHKERRQ(ierr);
            if (is_nested_ksp) {
              ierr = PCKSPGetKSP(pc_smoother,&ksp_nested);CHKERRQ(ierr);
              ierr = KSPSetOperators(ksp_nested,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
            }
          }
        }
          break;

        case OP_TYPE_REDISC_MF:
        {
          if (use_low_order_geometry == PETSC_TRUE) {
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          } else {
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorA11[k]);CHKERRQ(ierr);
          }
        }
          break;

        case OP_TYPE_GALERKIN:
        {
          ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
        }
          break;
          
        case OP_TYPE_MFGALERKIN:
          ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          break;

        case OP_TYPE_SNESFD:
          ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          break;

          
      }
      PetscFree(sub_ksp);
    }

  }

  {
    PetscBool mg_dump_coarse = PETSC_FALSE;
    char filename[PETSC_MAX_PATH_LEN];
    PetscInt snes_it;
    PetscViewer viewer;

    PetscOptionsGetBool(NULL,NULL,"-ptatin_mg_dump_coarse_operator",&mg_dump_coarse,0);
    SNESGetIterationNumber(snes,&snes_it);

    if (mg_dump_coarse) {
      if (mlctx->level_type[0] != OP_TYPE_REDISC_MF) {

        PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/mg_coarse_operatorA_step%D_snes%D.mat",user->outputpath,user->step,snes_it);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);
        MatView(mlctx->operatorA11[0],viewer);
        PetscViewerDestroy(&viewer);
        if (mlctx->operatorA11[0] != mlctx->operatorB11[0]) {
          PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/mg_coarse_operatorB_step%D_snes%D.mat",user->outputpath,user->step,snes_it);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);
          MatView(mlctx->operatorB11[0],viewer);
          PetscViewerDestroy(&viewer);
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode HMG_SetUp(AuuMultiLevelCtx *mlctx, pTatinCtx user)
{
  DM             dav_hierarchy[MAX_MG_LEVELS];
  Mat            interpolation_v[MAX_MG_LEVELS],interpolation_eta[MAX_MG_LEVELS];
  PetscInt       k,nlevels;
  Quadrature     volQ[MAX_MG_LEVELS];
  BCList         u_bclist[MAX_MG_LEVELS];
  DM             dmstokes,dav;
  IS             *is_stokes_field;
  PhysCompStokes stokes = NULL;
  pTatinModel    model = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatinGetModel(user,&model);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dmstokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&dav,NULL);CHKERRQ(ierr);

  ierr = DMCompositeGetGlobalISs(dmstokes,&is_stokes_field);CHKERRQ(ierr);

  nlevels = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dau_nlevels",&nlevels,NULL);CHKERRQ(ierr);
  if (nlevels >= MAX_MG_LEVELS) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Maximum number of multi-grid levels is set by #define MAX_MG_LEVELS %D",MAX_MG_LEVELS);

  PetscPrintf(PETSC_COMM_WORLD,"Mesh size (%D x %D x %D) : MG levels %D  \n",user->mx,user->my,user->mz,nlevels);
  ierr = pTatin3dStokesBuildMeshHierarchy(dav,nlevels,dav_hierarchy);CHKERRQ(ierr);
  ierr = pTatin3dStokesReportMeshHierarchy(nlevels,dav_hierarchy);CHKERRQ(ierr);
  ierr = pTatinLogNote(user,"  [Velocity multi-grid hierarchy]");CHKERRQ(ierr);
  for (k=nlevels-1; k>=0; k--) {
    char name[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"vel_dmda_Lv%D",k);CHKERRQ(ierr);
    ierr = pTatinLogBasicDMDA(user,name,dav_hierarchy[k]);CHKERRQ(ierr);
  }

  /* Coarse grid setup: Define interpolation operators for velocity space */
  interpolation_v[0] = NULL;
  for (k=0; k<nlevels-1; k++) {
    ierr = DMCreateInterpolation(dav_hierarchy[k],dav_hierarchy[k+1],&interpolation_v[k+1],NULL);CHKERRQ(ierr);
  }

  /* Coarse grid setup: Define interpolation operators for scalar space */
  interpolation_eta[0] = NULL;
  for (k=1; k<nlevels; k++) {
    ierr = MatMAIJRedimension(interpolation_v[k],1,&interpolation_eta[k]);CHKERRQ(ierr);
  }
  /*
  PetscPrintf(PETSC_COMM_WORLD,"Generated velocity mesh hierarchy --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);
  */

  /* Coarse grid setup: Define material properties on gauss points */
  for (k=0; k<nlevels-1; k++) {
    PetscInt ncells,lmx,lmy,lmz;
    PetscInt np_per_dim;

    np_per_dim = 3;
    ierr = DMDAGetLocalSizeElementQ2(dav_hierarchy[k],&lmx,&lmy,&lmz);CHKERRQ(ierr);
    ncells = lmx * lmy * lmz;
    ierr = VolumeQuadratureCreate_GaussLegendreStokes(3,np_per_dim,ncells,&volQ[k]);CHKERRQ(ierr);
  }
  volQ[nlevels-1] = stokes->volQ;
  /*
  PetscPrintf(PETSC_COMM_WORLD,"Generated quadrature point hierarchy --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);
  */

  /* Coarse grid setup: Define boundary conditions */
  for (k=0; k<nlevels-1; k++) {
    ierr = DMDABCListCreate(dav_hierarchy[k],&u_bclist[k]);CHKERRQ(ierr);
  }
  u_bclist[nlevels-1] = stokes->u_bclist;

  /* Coarse grid setup: Configure boundary conditions */
  ierr = pTatinModel_ApplyBoundaryConditionMG(nlevels,u_bclist,dav_hierarchy,model,user);CHKERRQ(ierr);

  /* set all pointers into mg context */
  mlctx->is_stokes_field     = is_stokes_field;

  ierr = PetscMalloc1(nlevels,&mlctx->dav_hierarchy);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlevels,&mlctx->interpolation_v);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlevels,&mlctx->interpolation_eta);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlevels,&mlctx->volQ);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlevels,&mlctx->u_bclist);CHKERRQ(ierr);

  ierr = PetscMalloc1(nlevels,&mlctx->level_type);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlevels,&mlctx->operatorA11);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlevels,&mlctx->operatorB11);CHKERRQ(ierr);

  for (k=0; k<nlevels; k++) {
    mlctx->dav_hierarchy[k]       = NULL;
    mlctx->interpolation_v[k]     = NULL;
    mlctx->interpolation_eta[k]   = NULL;
    mlctx->volQ[k]                = NULL;
    mlctx->u_bclist[k]            = NULL;

    mlctx->level_type[k]          = OP_TYPE_MFGALERKIN;
    mlctx->operatorA11[k]         = NULL;
    mlctx->operatorB11[k]         = NULL;
  }

  mlctx->nlevels                  = nlevels;
  for (k=0; k<nlevels; k++) {
    mlctx->dav_hierarchy[k]       = dav_hierarchy[k];
    mlctx->interpolation_v[k]     = interpolation_v[k];
    mlctx->interpolation_eta[k]   = interpolation_eta[k];
    mlctx->volQ[k]                = volQ[k];
    mlctx->u_bclist[k]            = u_bclist[k];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode HMG_Destroy(AuuMultiLevelCtx *mlctx)
{
  PetscInt       k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (k=0; k<mlctx->nlevels-1; k++) {
    ierr = BCListDestroy(&mlctx->u_bclist[k]);CHKERRQ(ierr);
    ierr = QuadratureDestroy(&mlctx->volQ[k]);CHKERRQ(ierr);
  }
  for (k=0; k<mlctx->nlevels; k++) {
    ierr = MatDestroy(&mlctx->operatorA11[k]);CHKERRQ(ierr);
    ierr = MatDestroy(&mlctx->operatorB11[k]);CHKERRQ(ierr);
    ierr = MatDestroy(&mlctx->interpolation_v[k]);CHKERRQ(ierr);
    ierr = MatDestroy(&mlctx->interpolation_eta[k]);CHKERRQ(ierr);
    ierr = DMDestroy(&mlctx->dav_hierarchy[k]);CHKERRQ(ierr);
  }

  ierr = ISDestroy(&mlctx->is_stokes_field[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&mlctx->is_stokes_field[1]);CHKERRQ(ierr);
  ierr = PetscFree(mlctx->is_stokes_field);CHKERRQ(ierr);

  ierr = PetscFree(mlctx->level_type);CHKERRQ(ierr);
  ierr = PetscFree(mlctx->operatorB11);CHKERRQ(ierr);
  ierr = PetscFree(mlctx->operatorA11);CHKERRQ(ierr);
  ierr = PetscFree(mlctx->dav_hierarchy);CHKERRQ(ierr);
  ierr = PetscFree(mlctx->interpolation_v);CHKERRQ(ierr);
  ierr = PetscFree(mlctx->interpolation_eta);CHKERRQ(ierr);
  ierr = PetscFree(mlctx->volQ);CHKERRQ(ierr);
  ierr = PetscFree(mlctx->u_bclist);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode HMGOperator_SetUp(AuuMultiLevelCtx *mlctx,pTatinCtx user,Mat *A,Mat *B)
{
  OperatorType   level_type[MAX_MG_LEVELS];
  Mat            operatorA11[MAX_MG_LEVELS],operatorB11[MAX_MG_LEVELS];
  PhysCompStokes stokes = NULL;
  PetscInt       k,nlevels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);

  /* configure stokes opertors */
  ierr = pTatin3dCreateStokesOperators(stokes,mlctx->is_stokes_field,
                                       mlctx->nlevels,
                                       mlctx->dav_hierarchy,
                                       mlctx->interpolation_v,
                                       mlctx->u_bclist,
                                       mlctx->volQ,
                                       level_type,
                                       A,operatorA11,B,operatorB11);CHKERRQ(ierr);

  nlevels = mlctx->nlevels;
  for (k=0; k<nlevels; k++) {
    mlctx->level_type[k]  = level_type[k];
    mlctx->operatorA11[k] = operatorA11[k];
    mlctx->operatorB11[k] = operatorB11[k];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode HMGOperator_Destroy(AuuMultiLevelCtx *mlctx)
{
  PetscInt       k,nlevels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nlevels = mlctx->nlevels;
  for (k=0; k<nlevels; k++) {
    ierr = MatDestroy(&mlctx->operatorA11[k]);CHKERRQ(ierr);
    ierr = MatDestroy(&mlctx->operatorB11[k]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinNonlinearStokesSolveCreate(pTatinCtx user,Mat A,Mat B,Vec F,AuuMultiLevelCtx *mgctx,SNES *s)
{
  SNES           snes;
  KSP            ksp;
  PC             pc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,FormFunction_Stokes,user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,A,B,FormJacobian_StokesMGAuu,user);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESComposeWithMGCtx(snes,mgctx);CHKERRQ(ierr);
  ierr = pTatin_Stokes_ActivateMonitors(user,snes);CHKERRQ(ierr);

  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc,"u",mgctx->is_stokes_field[0]);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc,"p",mgctx->is_stokes_field[1]);CHKERRQ(ierr);

  /* configure uu split for galerkin multi-grid */
  ierr = pTatin3dStokesKSPConfigureFSGMG(ksp,mgctx->nlevels,mgctx->operatorA11,mgctx->operatorB11,mgctx->interpolation_v,mgctx->dav_hierarchy);CHKERRQ(ierr);

  *s = snes;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinNonlinearStokesSolve(pTatinCtx user,SNES snes,Vec X,const char stagename[])
{
  PetscLogDouble time[2];
  PhysCompStokes stokes = NULL;
  DM             dmstokes;
  char           title[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dmstokes);CHKERRQ(ierr);

  /* insert boundary conditions into solution vector */
  {
    Vec Xu,Xp;
    
    ierr = DMCompositeGetAccess(dmstokes,X,&Xu,&Xp);CHKERRQ(ierr);
    ierr = BCListInsert(stokes->u_bclist,Xu);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(dmstokes,X,&Xu,&Xp);CHKERRQ(ierr);
  }

  if (stagename) {
    PetscPrintf(PETSC_COMM_WORLD,"   --------- Stokes[%s] ---------\n",stagename);
  }
  PetscTime(&time[0]);
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  PetscTime(&time[1]);
  if (stagename) {
    ierr = PetscSNPrintf(title,PETSC_MAX_PATH_LEN-1,"Stokes[%s]",stagename);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(title,PETSC_MAX_PATH_LEN-1,"Stokes");CHKERRQ(ierr);
  }
  ierr = pTatinLogBasicSNES(user,   title,snes);CHKERRQ(ierr);
  ierr = pTatinLogBasicCPUtime(user,title,time[1]-time[0]);CHKERRQ(ierr);
  ierr = pTatinLogBasicStokesSolution(user,dmstokes,X);CHKERRQ(ierr);
  ierr = pTatinLogBasicStokesSolutionResiduals(user,snes,dmstokes,X);CHKERRQ(ierr);
  //ierr = pTatinLogPetscLog(user,    title);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ProjectStokesVariablesOnQuadraturePoints(pTatinCtx user)
{
  int               npoints;
  DataField         PField_std;
  DataField         PField_stokes;
  MPntStd           *mp_std;
  MPntPStokes       *mp_stokes;
  PhysCompStokes    stokes;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  
  /* Marker -> quadrature point projection */
  DataBucketGetDataFieldByName(user->materialpoint_db, MPntStd_classname     , &PField_std);
  DataBucketGetDataFieldByName(user->materialpoint_db, MPntPStokes_classname , &PField_stokes);

  DataBucketGetSizes(user->materialpoint_db,&npoints,NULL,NULL);
  DataFieldGetEntries(PField_std,(void**)&mp_std);
  DataFieldGetEntries(PField_stokes,(void**)&mp_stokes);
  
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);

  switch (user->coefficient_projection_type) {

    case -1:      /* Perform null projection use the values currently defined on the quadrature points */
      break;

    case 0:     /* Perform P0 projection over Q2 element directly onto quadrature points */
      //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P0 [arithmetic avg] marker->quadrature projection not supported");
            ierr = MPntPStokesProj_P0(CoefAvgARITHMETIC,npoints,mp_std,mp_stokes,stokes->dav,stokes->volQ);CHKERRQ(ierr);
      break;
    case 10:
      //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P0 [harmonic avg] marker->quadrature projection not supported");
            ierr = MPntPStokesProj_P0(CoefAvgHARMONIC,npoints,mp_std,mp_stokes,stokes->dav,stokes->volQ);CHKERRQ(ierr);
      break;
    case 20:
      //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P0 [geometric avg] marker->quadrature projection not supported");
            ierr = MPntPStokesProj_P0(CoefAvgGEOMETRIC,npoints,mp_std,mp_stokes,stokes->dav,stokes->volQ);CHKERRQ(ierr);
      break;
    case 30:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P0 [dominant phase] marker->quadrature projection not supported");
      break;

    case 1:     /* Perform Q1 projection over Q2 element and interpolate back to quadrature points */
      ierr = SwarmUpdateGaussPropertiesLocalL2Projection_Q1_MPntPStokes(npoints,mp_std,mp_stokes,stokes->dav,stokes->volQ);CHKERRQ(ierr);
      break;

    case 2:       /* Perform Q2 projection and interpolate back to quadrature points */
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Q2 marker->quadrature projection not supported");
      break;

    case 3:       /* Perform P1 projection and interpolate back to quadrature points */
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P1 marker->quadrature projection not supported");
      break;
        case 4:
            ierr = SwarmUpdateGaussPropertiesOne2OneMap_MPntPStokes(npoints,mp_std,mp_stokes,stokes->volQ);CHKERRQ(ierr);
            break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Viscosity projection type is not defined");
      break;
  }
  
  DataFieldRestoreEntries(PField_stokes,(void**)&mp_stokes);
  DataFieldRestoreEntries(PField_std,(void**)&mp_std);
  
  ierr = AverageVolumeQuadraturePointsToSurfaceQuadraturePointsStokes(user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CheckpointWrite_EnergyFV(PhysCompEnergyFV energyfv,PetscBool write_dmda,const char path[],const char prefix[])
{
  PetscMPIInt    commsize,commrank;
  FVDA           fv;
  char           jprefix_fv[PETSC_MAX_PATH_LEN];
  char           jprefix_geom[PETSC_MAX_PATH_LEN];
  char           jfilename[PETSC_MAX_PATH_LEN];
  char           vfilename[3][PETSC_MAX_PATH_LEN],daprefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  fv = energyfv->fv;
  ierr = MPI_Comm_size(fv->comm,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(fv->comm,&commrank);CHKERRQ(ierr);
  
  if (path && prefix) SETERRQ(fv->comm,PETSC_ERR_SUP,"Only support for path/file or prefix_file");
  
  daprefix[0] = '\0';
  if (write_dmda) {
    if (path) {
      if (prefix) { PetscSNPrintf(jprefix_fv,PETSC_MAX_PATH_LEN-1,"%s/%s_fvda_fvspace",path,prefix); }
      else { PetscSNPrintf(jprefix_fv,PETSC_MAX_PATH_LEN-1,"%s/fvda_fvspace",path); }
    } else {
      if (prefix) { PetscSNPrintf(jprefix_fv,PETSC_MAX_PATH_LEN-1,"%s_fvda_fvspace",prefix); }
      else { PetscSNPrintf(jprefix_fv,PETSC_MAX_PATH_LEN-1,"fvda_fvspace"); }
    }
    ierr = DMDACheckpointWrite(fv->dm_fv,jprefix_fv);CHKERRQ(ierr);
    
    if (path) {
      if (prefix) { PetscSNPrintf(jprefix_geom,PETSC_MAX_PATH_LEN-1,"%s/%s_fvda_geom",path,prefix); }
      else { PetscSNPrintf(jprefix_geom,PETSC_MAX_PATH_LEN-1,"%s/fvda_geom",path); }
    } else {
      if (prefix) { PetscSNPrintf(jprefix_geom,PETSC_MAX_PATH_LEN-1,"%s_fvda_geom",prefix); }
      else { PetscSNPrintf(jprefix_geom,PETSC_MAX_PATH_LEN-1,"fvda_geom"); }
    }
    ierr = DMDACheckpointWrite(fv->dm_geometry,jprefix_geom);CHKERRQ(ierr);
    
    char cfilename[PETSC_MAX_PATH_LEN];
    PetscSNPrintf(cfilename,PETSC_MAX_PATH_LEN-1,"%s_coords",jprefix_geom);
    ierr =  PetscVecWriteJSON(fv->vertex_coor_geometry,0,cfilename);CHKERRQ(ierr);
  }
  
  if (path) {
    PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s/physcomp_energy_fvda.json",path);
    PetscSNPrintf(vfilename[0],PETSC_MAX_PATH_LEN-1,"%s/physcomp_energy_fvda_Told.pbvec",path);
    PetscSNPrintf(vfilename[1],PETSC_MAX_PATH_LEN-1,"%s/physcomp_energy_fvda_Xold.pbvec",path);
  } else {
    PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s_physcomp_energy_fvda.json",prefix);
    PetscSNPrintf(vfilename[0],PETSC_MAX_PATH_LEN-1,"%s_physcomp_energy_fvda_Told.pbvec",prefix);
    PetscSNPrintf(vfilename[1],PETSC_MAX_PATH_LEN-1,"%s_physcomp_energy_fvda_Xold.pbvec",prefix);
  }
  
  /* Write vectors required to restart */
  if (energyfv->Told) {
    ierr = DMDAWriteVectorToFile(energyfv->Told,vfilename[0],PETSC_FALSE);CHKERRQ(ierr);
  }
  if (energyfv->Xold) {
    ierr = DMDAWriteVectorToFile(energyfv->Xold,vfilename[1],PETSC_FALSE);CHKERRQ(ierr);
  }
  
  if (commrank == 0) {
    cJSON *jso_file = NULL,*jso_energy = NULL,*content;

    /* create json meta data file */
    jso_file = cJSON_CreateObject();

    jso_energy = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_file,"PhysCompEnergyFV",jso_energy);

    content = cJSON_CreateInt((int)fv->Mi[0]);    cJSON_AddItemToObject(jso_energy,"mx",content);
    content = cJSON_CreateInt((int)fv->Mi[1]);    cJSON_AddItemToObject(jso_energy,"my",content);
    content = cJSON_CreateInt((int)fv->Mi[2]);    cJSON_AddItemToObject(jso_energy,"mz",content);
    
    content = cJSON_CreateInt((int)energyfv->nsubdivision[0]); cJSON_AddItemToObject(jso_energy,"nsubdivision_x",content);
    content = cJSON_CreateInt((int)energyfv->nsubdivision[1]); cJSON_AddItemToObject(jso_energy,"nsubdivision_y",content);
    content = cJSON_CreateInt((int)energyfv->nsubdivision[2]); cJSON_AddItemToObject(jso_energy,"nsubdivision_z",content);
    
    content = cJSON_CreateNumber((double)energyfv->time);   cJSON_AddItemToObject(jso_energy,"time",content);
    content = cJSON_CreateNumber((double)energyfv->dt);     cJSON_AddItemToObject(jso_energy,"timeStepSize",content);

    if (write_dmda) {
      cJSON *jso_dmda;
      char subdmfilename[PETSC_MAX_PATH_LEN];

      jso_dmda = cJSON_CreateObject();
      cJSON_AddItemToObject(jso_energy,"sub_dmda",jso_dmda);

      PetscSNPrintf(subdmfilename,PETSC_MAX_PATH_LEN-1,"%s_dmda.json",daprefix);
      content = cJSON_CreateString(subdmfilename);       cJSON_AddItemToObject(jso_dmda,"fileName",content);
      content = cJSON_CreateString("json-meta");         cJSON_AddItemToObject(jso_dmda,"dataFormat",content);
    }

    {
      cJSON *jso_petscvec;

      jso_petscvec = cJSON_CreateObject();
      cJSON_AddItemToObject(jso_energy,"Told",jso_petscvec);

      content = cJSON_CreateString(vfilename[0]);   cJSON_AddItemToObject(jso_petscvec,"fileName",content);
      content = cJSON_CreateString("petsc-binary"); cJSON_AddItemToObject(jso_petscvec,"dataFormat",content);
    }
    {
      cJSON *jso_petscvec;

      jso_petscvec = cJSON_CreateObject();
      cJSON_AddItemToObject(jso_energy,"Xold",jso_petscvec);

      content = cJSON_CreateString(vfilename[1]);   cJSON_AddItemToObject(jso_petscvec,"fileName",content);
      content = cJSON_CreateString("petsc-binary"); cJSON_AddItemToObject(jso_petscvec,"dataFormat",content);
    }

    /* write json meta data file */
    {
      FILE *fp;
      char *jbuff = cJSON_Print(jso_file);

      fp = fopen(jfilename,"w");
      if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open file %s",jfilename);
      fprintf(fp,"%s\n",jbuff);
      fclose(fp);
      free(jbuff);
    }

    cJSON_Delete(jso_file);
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompLoad_EnergyFV(pTatinCtx user,DM dav,const char jfilename[])
{
  PhysCompEnergyFV energyfv;
  PetscErrorCode ierr;
  PetscMPIInt rank;
  char pathtovec[PETSC_MAX_PATH_LEN];
  cJSON *jfile = NULL,*jphys = NULL;
  //PetscInt mx,my,mz;
  PetscInt nsub[] = {0,0,0};
  PetscReal time,dt;
  MPI_Comm comm;
  PetscBool found;


  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dav,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (rank == 0) {
    cJSON_FileView(jfilename,&jfile);
    if (!jfile) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open JSON file \"%s\"",jfilename);
    jphys = cJSON_GetObjectItem(jfile,"PhysCompEnergyFV");
  }
  
  /* query JSON file for input parameters */
  ierr = cJSONGetPetscInt(comm,jphys,"nsubdivision_x",&nsub[0],&found);CHKERRQ(ierr);
  if (!found) SETERRQ(comm,PETSC_ERR_USER,"Failed to locate key \"nsubdivision_x\"");

  ierr = cJSONGetPetscInt(comm,jphys,"nsubdivision_y",&nsub[1],&found);CHKERRQ(ierr);
  if (!found) SETERRQ(comm,PETSC_ERR_USER,"Failed to locate key \"nsubdivision_y\"");

  ierr = cJSONGetPetscInt(comm,jphys,"nsubdivision_z",&nsub[2],&found);CHKERRQ(ierr);
  if (!found) SETERRQ(comm,PETSC_ERR_USER,"Failed to locate key \"nsubdivision_z\"");

  ierr = cJSONGetPetscReal(comm,jphys,"time",&time,&found);CHKERRQ(ierr);
  if (!found) SETERRQ(comm,PETSC_ERR_USER,"Failed to locate key \"time\"");

  ierr = cJSONGetPetscReal(comm,jphys,"timeStepSize",&dt,&found);CHKERRQ(ierr);
  if (!found) SETERRQ(comm,PETSC_ERR_USER,"Failed to locate key \"timeStepSize\"");

  /*
   This function creates the DMDA for temperature.
   I elect to perform a self creation, rather than creating via DMDACheckpointLoad() as
   this particular DMDA possess extra content associated with finite elements in the
   form of an attached struct.
  */
  ierr = PhysCompEnergyFVCreate(PETSC_COMM_WORLD,&energyfv);CHKERRQ(ierr);
  ierr = PhysCompEnergyFVSetParams(energyfv,time,dt,nsub);CHKERRQ(ierr);
  ierr = PhysCompEnergyFVSetUp(energyfv,user);CHKERRQ(ierr);
  ierr = PhysCompEnergyFVUpdateGeometry(energyfv,user->stokes_ctx);CHKERRQ(ierr);
  /* query file for state vector filenames - load vectors into energy struct */
  {
    cJSON *jso_petscvec;

    /* T^{k} */
    if (jphys) {
      jso_petscvec = NULL;
      jso_petscvec = cJSON_GetObjectItem(jphys,"Told");
      if (!jso_petscvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Failed to locate key \"Told\"");
    }

    ierr = cJSONGetPetscString(comm,jso_petscvec,"fileName",pathtovec,&found);CHKERRQ(ierr);
    if (found) { ierr = VecLoadFromFile(energyfv->Told,pathtovec);CHKERRQ(ierr);
    } else       SETERRQ(comm,PETSC_ERR_USER,"Failed to locate key \"fileName\"");

    /* X^{k} */
    if (jphys) {
      jso_petscvec = NULL;
      jso_petscvec = cJSON_GetObjectItem(jphys,"Xold");
      if (!jso_petscvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Failed to locate key \"Xold\"");
    }

    ierr = cJSONGetPetscString(comm,jso_petscvec,"fileName",pathtovec,&found);CHKERRQ(ierr);
    if (found) { ierr = VecLoadFromFile(energyfv->Xold,pathtovec);CHKERRQ(ierr);
    } else       SETERRQ(comm,PETSC_ERR_USER,"Failed to locate key \"fileName\"");
  }

  if (rank == 0) {
    cJSON_Delete(jfile);
  }
  
  user->energyfv_ctx = energyfv;

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinPhysCompActivate_EnergyFV_FromFile(pTatinCtx ctx)
{
  PetscErrorCode   ierr;
  PhysCompStokes   stokes;
  MPI_Comm         comm;
  PetscMPIInt      commrank;
  char             jfilename[PETSC_MAX_PATH_LEN],field_string[PETSC_MAX_PATH_LEN];
  cJSON            *jfile = NULL,*jptat = NULL,*jobj;
  PetscBool        found;

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&commrank);CHKERRQ(ierr);
  PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s/ptatin3dctx.json",ctx->restart_dir);
  if (commrank == 0) {
    cJSON_FileView(jfilename,&jfile);
    if (!jfile) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open JSON file \"%s\"",jfilename);
    jptat = cJSON_GetObjectItem(jfile,"pTatinCtx");
  }

  jobj = NULL;
  if (commrank == 0) { jobj = cJSON_GetObjectItem(jptat,"energy"); if (!jobj) SETERRQ_JSONKEY(PETSC_COMM_SELF,"energy"); }
  ierr = cJSONGetPetscString(comm,jobj,"fileName",field_string,&found);CHKERRQ(ierr);
  if (!found) SETERRQ_JSONKEY(comm,"fileName");

  stokes = ctx->stokes_ctx;
  ierr = PhysCompLoad_EnergyFV(ctx,stokes->dav,field_string);CHKERRQ(ierr);
  
  if (commrank == 0) {
    cJSON_Delete(jfile);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinCtxCheckpointWriteFV(pTatinCtx ctx,const char path[],const char prefix[],
                                          DM dms,DM dme,
                                          PetscInt nfields,const char *dmnames[],DM dmlist[],
                                          Vec Xs,Vec Xe,const char *fieldnames[],Vec veclist[])
{
  PetscErrorCode   ierr;
  MPI_Comm         comm;
  PetscMPIInt      commsize,commrank;
  char             jfilename[PETSC_MAX_PATH_LEN];
  char             vfilename[3][PETSC_MAX_PATH_LEN],checkpoint_prefix[3][PETSC_MAX_PATH_LEN];
  PetscBool        energy_activated;
  PhysCompStokes   stokes = NULL;
  PhysCompEnergyFV energy = NULL;
  DM               dmv,dmp;
  DataBucket       materialpoint_db = NULL,material_constants_db = NULL;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ctx->pack,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&commrank);CHKERRQ(ierr);

  if (path && prefix) SETERRQ(comm,PETSC_ERR_SUP,"Intended support for either ${path}/FILENAME or ${prefix}_FILENAME");
  if (prefix) SETERRQ(comm,PETSC_ERR_SUP,"Current implementation only supports ${path}/FILENAME format");
  if (nfields > 0) SETERRQ(comm,PETSC_ERR_SUP,"Only support for stokes + energy");

  if (path) {
    PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s/ptatin3dctx.json",path);

    PetscSNPrintf(vfilename[0],PETSC_MAX_PATH_LEN-1,"%s/ptatinstate_stokes_Xv.pbvec",path);
    PetscSNPrintf(vfilename[1],PETSC_MAX_PATH_LEN-1,"%s/ptatinstate_stokes_Xp.pbvec",path);
    PetscSNPrintf(vfilename[2],PETSC_MAX_PATH_LEN-1,"%s/ptatinstate_energy_Xt.pbvec",path);

    PetscSNPrintf(checkpoint_prefix[0],PETSC_MAX_PATH_LEN-1,"%s/stokes_v",path);
    PetscSNPrintf(checkpoint_prefix[1],PETSC_MAX_PATH_LEN-1,"%s/materialpoint",path);
    PetscSNPrintf(checkpoint_prefix[2],PETSC_MAX_PATH_LEN-1,"%s/materialconstants",path);
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Current implementation only supports ${path}/FILENAME format");
  }

  /* stokes */
  ierr = pTatinGetStokesContext(ctx,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&dmv,&dmp);CHKERRQ(ierr);

  ierr = DMDACheckpointWrite(dmv,checkpoint_prefix[0]);CHKERRQ(ierr);
  {
    Vec velocity,pressure;

    ierr = DMCompositeGetAccess(dms,Xs,&velocity,&pressure);CHKERRQ(ierr);

    ierr = DMDAWriteVectorToFile(velocity,vfilename[0],PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAWriteVectorToFile(pressure,vfilename[1],PETSC_FALSE);CHKERRQ(ierr);

    ierr = DMCompositeRestoreAccess(dms,Xs,&velocity,&pressure);CHKERRQ(ierr);
  }

  /* energy */
  ierr = pTatinContextValid_EnergyFV(ctx,&energy_activated);CHKERRQ(ierr);
  if (energy_activated) {
    ierr = pTatinGetContext_EnergyFV(ctx,&energy);CHKERRQ(ierr);
    ierr = CheckpointWrite_EnergyFV(energy,PETSC_FALSE,path,NULL);CHKERRQ(ierr);
    ierr = DMDAWriteVectorToFile(Xe,vfilename[2],PETSC_FALSE);CHKERRQ(ierr);
  }

  /* material points */
  ierr = pTatinGetMaterialPoints(ctx,&materialpoint_db,NULL);CHKERRQ(ierr);
  DataBucketView(PETSC_COMM_WORLD,materialpoint_db,checkpoint_prefix[1],DATABUCKET_VIEW_NATIVE);

  /* material constants */
  /* material_constants_db is a redundant object, e.g. it is identical on all ranks */
  /* Hence, we let only 1 rank write out the data file during checkpoint.write() */
  ierr = pTatinGetMaterialConstants(ctx,&material_constants_db);CHKERRQ(ierr);
  if (commrank == 0) {
    DataBucketView(PETSC_COMM_SELF,material_constants_db,checkpoint_prefix[2],DATABUCKET_VIEW_NATIVE);
  }

  if (commrank == 0) {
    cJSON *jso_file = NULL,*jso_ptat = NULL,*jso_state,*jso_object, *content;
    char relpathtofile[PETSC_MAX_PATH_LEN];

    /* create json meta data file */
    jso_file = cJSON_CreateObject();

    jso_ptat = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_file,"pTatinCtx",jso_ptat);

    content = cJSON_CreateInt((int)commsize);  cJSON_AddItemToObject(jso_ptat,"commSize",content);
    content = cJSON_CreateInt((int)ctx->mx);  cJSON_AddItemToObject(jso_ptat,"mx",content);
    content = cJSON_CreateInt((int)ctx->my);  cJSON_AddItemToObject(jso_ptat,"my",content);
    content = cJSON_CreateInt((int)ctx->mz);  cJSON_AddItemToObject(jso_ptat,"mz",content);

    content = cJSON_CreateBool((int)ctx->restart_from_file);        cJSON_AddItemToObject(jso_ptat,"restartFromFile",content);
    content = cJSON_CreateString(ctx->restart_dir);                 cJSON_AddItemToObject(jso_ptat,"restartPath",content);
    content = cJSON_CreateInt((int)ctx->checkpoint_every);          cJSON_AddItemToObject(jso_ptat,"checkpointEvery",content);
    content = cJSON_CreateInt((int)ctx->checkpoint_every_nsteps);   cJSON_AddItemToObject(jso_ptat,"checkpointEveryNSteps",content);
    content = cJSON_CreateDouble(ctx->checkpoint_every_ncpumins);   cJSON_AddItemToObject(jso_ptat,"checkpointEveryNCPUMins",content);

    content = cJSON_CreateBool((int)ctx->use_mf_stokes);  cJSON_AddItemToObject(jso_ptat,"useMFStokes",content);

    content = cJSON_CreateString(ctx->formatted_timestamp);            cJSON_AddItemToObject(jso_ptat,"formattedTimestamp",content);
    content = cJSON_CreateString(ctx->outputpath);                     cJSON_AddItemToObject(jso_ptat,"outputPath",content);
    content = cJSON_CreateInt((int)ctx->coefficient_projection_type);  cJSON_AddItemToObject(jso_ptat,"coefficientProjectionType",content);

    content = cJSON_CreateBool((int)ctx->solverstatistics);  cJSON_AddItemToObject(jso_ptat,"solverStatistics",content);
    content = cJSON_CreateInt((int)ctx->continuation_m);     cJSON_AddItemToObject(jso_ptat,"continuation_m",content);
    content = cJSON_CreateInt((int)ctx->continuation_M);     cJSON_AddItemToObject(jso_ptat,"continuation_M",content);

    content = cJSON_CreateInt((int)ctx->step);     cJSON_AddItemToObject(jso_ptat,"timeStep",content);
    content = cJSON_CreateInt((int)ctx->nsteps);   cJSON_AddItemToObject(jso_ptat,"timeStepMax",content);

    content = cJSON_CreateDouble(ctx->dt);          cJSON_AddItemToObject(jso_ptat,"timeStepSize",content);
    content = cJSON_CreateDouble(ctx->dt_max);      cJSON_AddItemToObject(jso_ptat,"timeStepSizeMax",content);
    content = cJSON_CreateDouble(ctx->dt_min);      cJSON_AddItemToObject(jso_ptat,"timeStepSizeMin",content);
    content = cJSON_CreateDouble(ctx->dt_adv);      cJSON_AddItemToObject(jso_ptat,"timeStepSizeAdv",content);
    content = cJSON_CreateDouble(ctx->constant_dt); cJSON_AddItemToObject(jso_ptat,"constantTimeStepSize",content);
    content = cJSON_CreateBool((int)ctx->use_constant_dt);  cJSON_AddItemToObject(jso_ptat,"useConstantTimeStepSize",content);

    content = cJSON_CreateDouble(ctx->time);                cJSON_AddItemToObject(jso_ptat,"time",content);
    content = cJSON_CreateDouble(ctx->time_max);            cJSON_AddItemToObject(jso_ptat,"timeMax",content);

    content = cJSON_CreateInt((int)ctx->output_frequency);  cJSON_AddItemToObject(jso_ptat,"outputFrequency",content);

    /* references to data files */
    /* PhysCompStokes */
    jso_object = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_ptat,"stokes->gravity_vector",jso_object);
    content = cJSON_CreateString("double");          cJSON_AddItemToObject(jso_object,"ctype",content);
    content = cJSON_CreateInt((int)3);               cJSON_AddItemToObject(jso_object,"length",content);
    /* todo - change this to base64 encoded */
    content = cJSON_CreateString("ascii");           cJSON_AddItemToObject(jso_object,"dataFormat",content);
    content = cJSON_CreateDoubleArray(stokes->gravity_vector,3);   cJSON_AddItemToObject(jso_object,"data",content);
    content = cJSON_CreateDoubleArray(stokes->gravity_vector,3);   cJSON_AddItemToObject(jso_object,"data[ascii]",content);

    /* DMDA for velocity */
    jso_object = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_ptat,"stokes->dmv",jso_object);
    content = cJSON_CreateString("DMDA");                cJSON_AddItemToObject(jso_object,"ctype",content);
    content = cJSON_CreateString("json-meta");           cJSON_AddItemToObject(jso_object,"dataFormat",content);
    //content = cJSON_CreateString(checkpoint_prefix[0]);  cJSON_AddItemToObject(jso_object,"prefix",content);
    PetscSNPrintf(relpathtofile,PETSC_MAX_PATH_LEN-1,"%s_dmda.json",checkpoint_prefix[0]);
    content = cJSON_CreateString(relpathtofile);         cJSON_AddItemToObject(jso_object,"fileName",content);

    /* PhysCompEnergy */
    if (energy_activated) {
      jso_object = cJSON_CreateObject();
      cJSON_AddItemToObject(jso_ptat,"energy",jso_object);
      content = cJSON_CreateString("PhysCompEnergyFV");      cJSON_AddItemToObject(jso_object,"ctype",content);
      content = cJSON_CreateString("json-meta");           cJSON_AddItemToObject(jso_object,"dataFormat",content);
      //content = cJSON_CreateString(path);  cJSON_AddItemToObject(jso_object,"prefix",content);
      PetscSNPrintf(relpathtofile,PETSC_MAX_PATH_LEN-1,"%s/physcomp_energy_fvda.json",path);
      content = cJSON_CreateString(relpathtofile);         cJSON_AddItemToObject(jso_object,"fileName",content);
    }

    /* Material points */
    jso_object = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_ptat,"materialpoint_db",jso_object);
    content = cJSON_CreateString("DataBucket");          cJSON_AddItemToObject(jso_object,"ctype",content);
    content = cJSON_CreateString("native");              cJSON_AddItemToObject(jso_object,"dataFormat",content);
    //content = cJSON_CreateString(checkpoint_prefix[1]);  cJSON_AddItemToObject(jso_object,"prefix",content);
    PetscSNPrintf(relpathtofile,PETSC_MAX_PATH_LEN-1,"%s_db.json",checkpoint_prefix[1]);
    content = cJSON_CreateString(relpathtofile);         cJSON_AddItemToObject(jso_object,"fileName",content);

    /* Material constants */
    jso_object = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_ptat,"material_constants",jso_object);
    content = cJSON_CreateString("DataBucket");          cJSON_AddItemToObject(jso_object,"ctype",content);
    content = cJSON_CreateString("native");              cJSON_AddItemToObject(jso_object,"dataFormat",content);
    //content = cJSON_CreateString(checkpoint_prefix[2]);  cJSON_AddItemToObject(jso_object,"prefix",content);
    PetscSNPrintf(relpathtofile,PETSC_MAX_PATH_LEN-1,"%s_db.json",checkpoint_prefix[2]);
    content = cJSON_CreateString(relpathtofile);         cJSON_AddItemToObject(jso_object,"fileName",content);

    /* State vectors */
    jso_state = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_ptat,"stokes.Xs->Xv",jso_state);
    content = cJSON_CreateString("Vec");                      cJSON_AddItemToObject(jso_state,"ctype",content);
    content = cJSON_CreateString("petsc-binary");             cJSON_AddItemToObject(jso_state,"dataFormat",content);
    content = cJSON_CreateString(vfilename[0]);               cJSON_AddItemToObject(jso_state,"fileName",content);

    jso_state = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_ptat,"stokes.Xs->Xp",jso_state);
    content = cJSON_CreateString("Vec");                      cJSON_AddItemToObject(jso_state,"ctype",content);
    content = cJSON_CreateString("petsc-binary");             cJSON_AddItemToObject(jso_state,"dataFormat",content);
    content = cJSON_CreateString(vfilename[1]);               cJSON_AddItemToObject(jso_state,"fileName",content);

    if (energy_activated) {
      jso_state = cJSON_CreateObject();
      cJSON_AddItemToObject(jso_ptat,"energy.Xt",jso_state);
      content = cJSON_CreateString("Vec");                    cJSON_AddItemToObject(jso_state,"ctype",content);
      content = cJSON_CreateString("petsc-binary");           cJSON_AddItemToObject(jso_state,"dataFormat",content);
      content = cJSON_CreateString(vfilename[2]);             cJSON_AddItemToObject(jso_state,"fileName",content);
    }

    /*
    jso_state = cJSON_CreateArray();
    cJSON_AddItemToObject(jso_ptat,"userstate",jso_state);

    jso_object = cJSON_CreateObject();
    content = cJSON_CreateString();            cJSON_AddItemToObject(jso_object,"points",content);
    cJSON_AddItemToArray(jso_state,jso_object);
    */

    /* write json meta data file */
    {
      FILE *fp;
      char *jbuff = cJSON_Print(jso_file);

      fp = fopen(jfilename,"w");
      if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open file %s",jfilename);
      fprintf(fp,"%s\n",jbuff);
      fclose(fp);
      free(jbuff);
    }

    cJSON_Delete(jso_file);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dLoadState_FromFile_FV(pTatinCtx ctx,DM dmstokes,DM dmenergy,Vec Xs,Vec Xt)
{
  PetscErrorCode ierr;
  MPI_Comm comm;
  PetscMPIInt commrank;
  char jfilename[PETSC_MAX_PATH_LEN];
  cJSON *jfile = NULL,*jptat = NULL,*jobj;
  PetscBool found,energy_activated;


  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&commrank);CHKERRQ(ierr);
  PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s/ptatin3dctx.json",ctx->restart_dir);
  if (commrank == 0) {
    cJSON_FileView(jfilename,&jfile);
    if (!jfile) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open JSON file \"%s\"",jfilename);
    jptat = cJSON_GetObjectItem(jfile,"pTatinCtx");
  }

  {
    char field_string_v[PETSC_MAX_PATH_LEN];
    char field_string_p[PETSC_MAX_PATH_LEN];
    Vec velocity,pressure;

    jobj = NULL;
    if (commrank == 0) { jobj = cJSON_GetObjectItem(jptat,"stokes.Xs->Xv"); if (!jobj) SETERRQ_JSONKEY(PETSC_COMM_SELF,"stokes.Xs->Xv"); }
    ierr = cJSONGetPetscString(comm,jobj,"fileName",field_string_v,&found);CHKERRQ(ierr);
    if (!found) SETERRQ_JSONKEY(comm,"fileName");

    jobj = NULL;
    if (commrank == 0) { jobj = cJSON_GetObjectItem(jptat,"stokes.Xs->Xp"); if (!jobj) SETERRQ_JSONKEY(PETSC_COMM_SELF,"stokes.Xs->Xp");}
    ierr = cJSONGetPetscString(comm,jobj,"fileName",field_string_p,&found);CHKERRQ(ierr);
    if (!found) SETERRQ_JSONKEY(comm,"fileName");

    ierr = DMCompositeGetAccess(dmstokes,Xs,&velocity,&pressure);CHKERRQ(ierr);

    ierr = VecLoadFromFile(velocity,field_string_v);CHKERRQ(ierr);
    ierr = VecLoadFromFile(pressure,field_string_p);CHKERRQ(ierr);

    ierr = DMCompositeRestoreAccess(dmstokes,Xs,&velocity,&pressure);CHKERRQ(ierr);
  }

  ierr = pTatinContextValid_EnergyFV(ctx,&energy_activated);CHKERRQ(ierr);
  if (energy_activated) {
    char field_string_t[PETSC_MAX_PATH_LEN];

    jobj = NULL;
    if (commrank == 0) { jobj = cJSON_GetObjectItem(jptat,"energy.Xt"); if (!jobj) SETERRQ_JSONKEY(PETSC_COMM_SELF,"energy.Xt");}
    ierr = cJSONGetPetscString(comm,jobj,"fileName",field_string_t,&found);CHKERRQ(ierr);
    if (!found) SETERRQ_JSONKEY(comm,"fileName");

    ierr = VecLoadFromFile(Xt,field_string_t);CHKERRQ(ierr);
  }

  if (commrank == 0) {
    cJSON_Delete(jfile);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode LoadStateFromModelDefinitionFV(pTatinCtx *pctx,Vec *v1,Vec *v2,PetscBool write_checkpoint)
{
  pTatinCtx         user;
  pTatinModel       model = NULL;
  PhysCompStokes    stokes = NULL;
  PhysCompEnergyFV  energyfv = NULL;
  DM                dmstokes,dmv,dmp,dmenergy = NULL;
  Vec               X_s,X_e = NULL;
  PetscBool         activate_energy = PETSC_FALSE;
  DataBucket        materialpoint_db,material_constants_db;
  PetscLogDouble    time[2];
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscTime(&time[0]);
  ierr = pTatin3dLoadContext_FromFile(&user);CHKERRQ(ierr);
  PetscTime(&time[1]);
  ierr = pTatin3dSetFromOptions(user);CHKERRQ(ierr);
  ierr = pTatinLogNote(user,"  [ptatin_driver.Load]");CHKERRQ(ierr);
  ierr = pTatinLogBasicCPUtime(user,"Checkpoint.read()",time[1]-time[0]);CHKERRQ(ierr);

  /* driver specific options parsed here */

  /* Register all models */
  ierr = pTatinModelLoad(user);CHKERRQ(ierr);
  ierr = pTatinGetModel(user,&model);CHKERRQ(ierr);

  ierr = pTatinModel_Initialize(model,user);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-activate_energyfv",&activate_energy,NULL);CHKERRQ(ierr);

  /* Create Stokes context */
  ierr = pTatin3d_PhysCompStokesLoad_FromFile(user);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dmstokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&dmv,&dmp);CHKERRQ(ierr);

  { /* IF I DON'T DO THIS, THE IS's OBTAINED FROM DMCompositeGetGlobalISs() are wrong !! */
    Vec X;

    ierr = DMGetGlobalVector(dmstokes,&X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmstokes,&X);CHKERRQ(ierr);
  }
  /* Pack all physics together */
  ierr = PetscObjectReference((PetscObject)dmstokes);CHKERRQ(ierr);
  user->pack = dmstokes;

  /* Create material points */
  // material point load //ierr = pTatin3dCreateMaterialPoints(user,dmv);CHKERRQ(ierr);
  ierr = pTatin3dLoadMaterialPoints_FromFile(user,dmv);CHKERRQ(ierr);
  ierr = pTatinGetMaterialPoints(user,&materialpoint_db,NULL);CHKERRQ(ierr);
  
  /* Create energy context */
  /* NOTE - Calling pTatinPhysCompActivate_Energy() after pTatin3dCreateMaterialPoints() is not essential when restarting */
  /* The reason for this is pTatinPhysCompActivate_Energy_FromFile() does not register new fields, rather DataBucketLoad() does */
  if (activate_energy) {
    ierr = pTatinPhysCompActivate_EnergyFV_FromFile(user);CHKERRQ(ierr);
    ierr = pTatinGetContext_EnergyFV(user,&energyfv);CHKERRQ(ierr);
    ierr = FVDAGetDM(energyfv->fv,&dmenergy);CHKERRQ(ierr);
  }
  
  /* work vector for solution */
  ierr = DMCreateGlobalVector(dmstokes,&X_s);CHKERRQ(ierr);
  if (activate_energy) {
    //X_e = energyfv->T;
    //ierr = pTatinCtxAttachModelData(user,"PhysCompEnergy_T",(void*)X_e);CHKERRQ(ierr);
    ierr = pTatinCtxAttachModelData(user,"PhysCompEnergy_T",(void*)energyfv->T);CHKERRQ(ierr);
  }

  /* initial condition - call user method, then clobber */
  ierr = pTatinModel_ApplyInitialSolution(model,user,X_s);CHKERRQ(ierr);
  if (activate_energy) {
    ierr = pTatin3dLoadState_FromFile_FV(user,dmstokes,dmenergy,X_s,energyfv->T);CHKERRQ(ierr);
  } else{
    ierr = pTatin3dLoadState_FromFile_FV(user,dmstokes,dmenergy,X_s,NULL);CHKERRQ(ierr);
  }
  ierr = ProjectStokesVariablesOnQuadraturePoints(user);CHKERRQ(ierr);
  /* boundary conditions */
  ierr = pTatinModel_ApplyBoundaryCondition(model,user);CHKERRQ(ierr);

  ierr = pTatinGetMaterialConstants(user,&material_constants_db);CHKERRQ(ierr);

  {
    char output_path[PETSC_MAX_PATH_LEN];
    char output_path_ic[PETSC_MAX_PATH_LEN];
    
    ierr = PetscSNPrintf(output_path,PETSC_MAX_PATH_LEN-1,"%s",user->outputpath);CHKERRQ(ierr);
    
    ierr = PetscSNPrintf(output_path_ic,PETSC_MAX_PATH_LEN-1,"%s/fromfile",output_path);CHKERRQ(ierr);
    ierr = pTatinCreateDirectory(output_path_ic);CHKERRQ(ierr);
    
    ierr = PetscSNPrintf(user->outputpath,PETSC_MAX_PATH_LEN-1,"%s",output_path_ic);CHKERRQ(ierr);
    ierr = pTatinModel_Output(model,user,X_s,"icbc");CHKERRQ(ierr);
    
    ierr = PetscSNPrintf(user->outputpath,PETSC_MAX_PATH_LEN-1,"%s",output_path);CHKERRQ(ierr);
  }

  if (write_checkpoint) {
    char checkpoints_path[PETSC_MAX_PATH_LEN];
    char checkpoint_path[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(checkpoints_path,PETSC_MAX_PATH_LEN-1,"%s/checkpoints",user->outputpath);CHKERRQ(ierr);
    ierr = pTatinCreateDirectory(checkpoints_path);CHKERRQ(ierr);

    ierr = PetscSNPrintf(checkpoint_path,PETSC_MAX_PATH_LEN-1,"%s/initial_condition",checkpoints_path);CHKERRQ(ierr);
    ierr = pTatinCreateDirectory(checkpoint_path);CHKERRQ(ierr);
    /*
    ierr = pTatinCtxCheckpointWriteFV(user,checkpoint_path,NULL,
                                    dmstokes,dmenergy,
                                    0,NULL,NULL,
                                    X_s,X_e,NULL,NULL);CHKERRQ(ierr);
    */
    if (activate_energy) {
      ierr = pTatinCtxCheckpointWriteFV(user,checkpoint_path,NULL,
                                    dmstokes,dmenergy,
                                    0,NULL,NULL,
                                    X_s,energyfv->T,NULL,NULL);CHKERRQ(ierr);
    } else {
      ierr = pTatinCtxCheckpointWriteFV(user,checkpoint_path,NULL,
                                    dmstokes,dmenergy,
                                    0,NULL,NULL,
                                    X_s,NULL,NULL,NULL);CHKERRQ(ierr);
    }
  }

  if (v2) {
    if (activate_energy) {
      *v2 = energyfv->T;
    } else {
      *v2 = NULL;
    }
  }
  if (v1) { *v1 = X_s; }
  else    { ierr = VecDestroy(&X_s);CHKERRQ(ierr); }

  *pctx = user;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dCheckpointManagerFV(pTatinCtx ctx,Vec Xs)
{
  PetscErrorCode     ierr;
  PetscInt           checkpoint_every;
  PetscInt           checkpoint_every_nsteps,step;
  double             checkpoint_every_ncpumins, max_current_cpu_time, current_cpu_time;
  static double      last_cpu_time = 0.0;
  /*PetscBool        skip_existence_test = PETSC_TRUE;*/
  PhysCompStokes     stokes = NULL;
  PetscBool          energy_activated;
  PhysCompEnergyFV   energy = NULL;
  Vec                Xe = NULL;
  DM                 dmv,dmp,dmstokes = NULL,dmenergy = NULL;
  PetscBool          exists,write_step_checkpoint;
  char               checkpoints_basedir[PETSC_MAX_PATH_LEN];
  char               test_dir[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = pTatinGetStokesContext(ctx,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dmstokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&dmv,&dmp);CHKERRQ(ierr);
  ierr = pTatinContextValid_EnergyFV(ctx,&energy_activated);CHKERRQ(ierr);
  if (energy_activated) {
    ierr = pTatinGetContext_EnergyFV(ctx,&energy);CHKERRQ(ierr);
    ierr = FVDAGetDM(energy->fv,&dmenergy);CHKERRQ(ierr);
    //ierr = pTatinPhysCompGetData_Energy(ctx,&Xe,NULL);CHKERRQ(ierr);
    Xe = energy->T;
    ierr = pTatinCtxGetModelData(ctx,"PhysCompEnergy_T",(void**)Xe);CHKERRQ(ierr);
  }

  ierr = PetscSNPrintf(checkpoints_basedir,PETSC_MAX_PATH_LEN-1,"%s/checkpoints",ctx->outputpath);CHKERRQ(ierr);
  ierr = pTatinTestDirectory(checkpoints_basedir,'w',&exists);CHKERRQ(ierr);
  if (!exists) {
    ierr = pTatinCreateDirectory(checkpoints_basedir);CHKERRQ(ierr);
  }

  step                      = ctx->step;
  checkpoint_every          = ctx->checkpoint_every;
  checkpoint_every_nsteps   = ctx->checkpoint_every_nsteps;
  checkpoint_every_ncpumins = ctx->checkpoint_every_ncpumins;

  /* -------------------------------------- */
  /* check one - this file has a fixed name */
  if (step%checkpoint_every == 0) {

    ierr = PetscSNPrintf(test_dir,PETSC_MAX_PATH_LEN-1,"%s/checkpoints/default",ctx->outputpath);CHKERRQ(ierr);
    ierr = pTatinTestDirectory(test_dir,'w',&exists);CHKERRQ(ierr);
    if (!exists) {
      ierr = pTatinCreateDirectory(test_dir);CHKERRQ(ierr);
    }
    PetscPrintf(PETSC_COMM_WORLD,"CheckpointManager[every]: Writing to dir %s\n",test_dir);
    // call checkpoint routine //
    //ierr = pTatin3dCheckpoint(ctx,X,NULL);CHKERRQ(ierr);
    ierr = pTatinCtxCheckpointWriteFV(ctx,test_dir,NULL,dmstokes,dmenergy,0,NULL,NULL,Xs,Xe,NULL,NULL);CHKERRQ(ierr);
  }

  write_step_checkpoint = PETSC_FALSE;

  /* -------------------------------------------------------------------- */
  /* check three - look at cpu time and decide if we need to write or not */
  PetscTime(&current_cpu_time);
  ierr = MPI_Allreduce(&current_cpu_time,&max_current_cpu_time,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
  max_current_cpu_time = max_current_cpu_time/60.0; /* convert sec to mins */

  if (max_current_cpu_time > last_cpu_time + checkpoint_every_ncpumins) {

    PetscPrintf(PETSC_COMM_WORLD,"CheckpointManager[checkpoint_every_ncpumins]: Activated\n");
    write_step_checkpoint = PETSC_TRUE;

    last_cpu_time = max_current_cpu_time;
  }

  /* ----------------------------------------------------------------- */
  /* check two - these files have a file name related to the time step */
  if (step%checkpoint_every_nsteps == 0) {

    PetscPrintf(PETSC_COMM_WORLD,"CheckpointManager[checkpoint_every_nsteps]: Activated\n");
    write_step_checkpoint = PETSC_TRUE;
  }

  /* if either the cpu based or step based checks returned true, write a checkpoint file */
  if (write_step_checkpoint) {
    PetscLogDouble time[2];
    char           restartfile[PETSC_MAX_PATH_LEN];
    char           restartstring[PETSC_MAX_PATH_LEN];
    PetscMPIInt    rank;

    ierr = PetscSNPrintf(test_dir,PETSC_MAX_PATH_LEN-1,"%s/step%d",checkpoints_basedir,step);CHKERRQ(ierr);

    ierr = pTatinTestDirectory(test_dir,'w',&exists);CHKERRQ(ierr);
    if (!exists) {
      ierr = pTatinCreateDirectory(test_dir);CHKERRQ(ierr);
    }
    PetscPrintf(PETSC_COMM_WORLD,"CheckpointManager: Writing to dir %s\n",test_dir);
    PetscTime(&time[0]);
    ierr = pTatinCtxCheckpointWriteFV(ctx,test_dir,NULL,dmstokes,dmenergy,0,NULL,NULL,Xs,Xe,NULL,NULL);CHKERRQ(ierr);
    PetscTime(&time[1]);
    ierr = pTatinLogBasicCPUtime(ctx,"Checkpoint.write()",time[1]-time[0]);CHKERRQ(ierr);

    /* write out a default string for restarting the job */
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    ierr = PetscSNPrintf(restartfile,PETSC_MAX_PATH_LEN-1,"%s/restart.default",ctx->outputpath);CHKERRQ(ierr);
    ierr = PetscSNPrintf(restartstring,PETSC_MAX_PATH_LEN-1,"-restart_directory %s/checkpoints/step%d",ctx->outputpath,step);CHKERRQ(ierr);
    if (rank == 0) {
      FILE *fp;
      fp = fopen(restartfile,"w");
      fprintf(fp,"%s",restartstring);
      fclose(fp);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode GenerateICStateFromModelDefinition_FV(pTatinCtx *pctx)
{
  pTatinCtx         user;
  pTatinModel       model = NULL;
  PhysCompStokes    stokes = NULL;
  PhysCompEnergyFV  energyfv = NULL;
  DM                multipys_pack,dav,dap,dmfv = NULL;
  Vec               X_s,X_e = NULL;
  PetscBool         active_energy = PETSC_FALSE;
  DataBucket        materialpoint_db;
  PetscReal         surface_displacement_max = 1.0e32;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = pTatin3dCreateContext(&user);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(user);CHKERRQ(ierr);
  
  /* driver specific options parsed here */
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt_max_surface_displacement",&surface_displacement_max,NULL);CHKERRQ(ierr);
  
  /* Register all models */
  ierr = pTatinModelLoad(user);CHKERRQ(ierr);
  ierr = pTatinGetModel(user,&model);CHKERRQ(ierr);

  ierr = pTatinModel_Initialize(model,user);CHKERRQ(ierr);

 /* Generate physics modules */
  ierr = pTatin3d_PhysCompStokesCreate(user);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&multipys_pack);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&dav,&dap);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Generated vel/pressure mesh --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* Pack all physics together */
  /* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
  ierr = PetscObjectReference((PetscObject)stokes->stokes_pack);CHKERRQ(ierr);
  user->pack = stokes->stokes_pack;

  /* IF I DON'T DO THIS, THE IS's OBTAINED FROM DMCompositeGetGlobalISs() are wrong !! */
  {
    Vec X;

    ierr = DMGetGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
  }

  ierr = pTatin3dCreateMaterialPoints(user,dav);CHKERRQ(ierr);
  ierr = pTatinGetMaterialPoints(user,&materialpoint_db,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Generated material points --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* mesh geometry */
  ierr = pTatinModel_ApplyInitialMeshGeometry(model,user);CHKERRQ(ierr);

  ierr = pTatinLogBasicDMDA(user,"Velocity",dav);CHKERRQ(ierr);
  ierr = pTatinLogBasicDMDA(user,"Pressure",dap);CHKERRQ(ierr);

  /* generate energy solver */
  PetscBool load_energy = PETSC_FALSE;
  PetscOptionsGetBool(NULL,NULL,"-activate_energyfv",&load_energy,NULL);
  ierr = pTatinPhysCompActivate_EnergyFV(user,load_energy);CHKERRQ(ierr);
  ierr = pTatinContextValid_EnergyFV(user,&active_energy);CHKERRQ(ierr);
  
  if (active_energy) {
    ierr = pTatinGetContext_EnergyFV(user,&energyfv);CHKERRQ(ierr);
    ierr = FVDAGetDM(energyfv->fv,&dmfv);CHKERRQ(ierr);
    ierr = pTatinLogBasicDMDA(user,"EnergyFV",dmfv);CHKERRQ(ierr);
    X_e  = energyfv->T;
    ierr = pTatinCtxAttachModelData(user,"PhysCompEnergy_T",(void*)X_e);CHKERRQ(ierr);
    pTatinGetRangeCurrentMemoryUsage(NULL);
  }
  
  /* interpolate material point coordinates (needed if mesh was modified) */
  ierr = MaterialPointCoordinateSetUp(user,dav);CHKERRQ(ierr);

  /* material geometry */
  ierr = pTatinModel_ApplyInitialMaterialGeometry(model,user);CHKERRQ(ierr);
  if (active_energy) {
    PetscPrintf(PETSC_COMM_WORLD,"********* <FV SUPPORT NOTE> IS THIS REQUIRED?? pTatinPhysCompEnergy_MPProjectionQ1 ****************\n");
    
    ierr = EnergyFVEvaluateCoefficients(user,0.0,energyfv,NULL,NULL);CHKERRQ(ierr);
    
    ierr = pTatinPhysCompEnergyFV_MPProjection(energyfv,user);CHKERRQ(ierr);
    
    ierr = FVDACellPropertyProjectToFace_HarmonicMean(energyfv->fv,"k","k");CHKERRQ(ierr);
  }
  DataBucketView(PetscObjectComm((PetscObject)multipys_pack), materialpoint_db,"MaterialPoints StokesCoefficients",DATABUCKET_VIEW_STDOUT);

  /* work vector for solution and residual */
  ierr = DMCreateGlobalVector(multipys_pack,&X_s);CHKERRQ(ierr);

  /* initial condition */
  ierr = pTatinModel_ApplyInitialSolution(model,user,X_s);CHKERRQ(ierr);

  /* initial viscosity  */
  ierr = pTatinModel_ApplyInitialStokesVariableMarkers(model,user,X_s);CHKERRQ(ierr);

  /* boundary conditions */
  ierr = pTatinModel_ApplyBoundaryCondition(model,user);CHKERRQ(ierr);
  
  /* insert boundary conditions into solution vector */
  {
    Vec velocity,pressure;

    ierr = DMCompositeGetAccess(multipys_pack,X_s,&velocity,&pressure);CHKERRQ(ierr);
    ierr = BCListInsert(stokes->u_bclist,velocity);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(multipys_pack,X_s,&velocity,&pressure);CHKERRQ(ierr);
  }

  /* Configure for the initial condition */
  user->step = 0;
  user->time = 0.0;
  user->dt = 1.0e-10;

  {
    AuuMultiLevelCtx mgctx;
    Mat              A,B;
    Vec              F;
    SNES             snes;
    SNESLineSearch   linesearch;
    RheologyType     init_rheology_type;

    ierr = VecDuplicate(X_s,&F);CHKERRQ(ierr);
    ierr = HMG_SetUp(&mgctx,user);CHKERRQ(ierr);
    
    /* linear stage */
    ierr = HMGOperator_SetUp(&mgctx,user,&A,&B);CHKERRQ(ierr);
    ierr = pTatinNonlinearStokesSolveCreate(user,A,B,F,&mgctx,&snes);CHKERRQ(ierr);
    
    /* Solve lithostatic pressure and apply on the surface quadrature points for Stokes */
    /* Check if required to do it again before Picard solve */
    ierr = ModelApplyTractionFromLithoPressure(user,X_s);CHKERRQ(ierr);
    
    /* configure as a linear solve */
    init_rheology_type = user->rheology_constants.rheology_type;
    user->rheology_constants.rheology_type = RHEOLOGY_VISCOUS;
    ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1,PETSC_DEFAULT);CHKERRQ(ierr);

    ierr = SNESSetType(snes,SNESNEWTONLS);CHKERRQ(ierr);
    ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);CHKERRQ(ierr);

    ierr = pTatinNonlinearStokesSolve(user,snes,X_s,"Linear Stage");CHKERRQ(ierr);

    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = SNESDestroyMGCtx(snes);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = HMGOperator_Destroy(&mgctx);CHKERRQ(ierr);

    /* picard stage */
    ierr = HMGOperator_SetUp(&mgctx,user,&A,&B);CHKERRQ(ierr);
    ierr = pTatinNonlinearStokesSolveCreate(user,A,B,F,&mgctx,&snes);CHKERRQ(ierr);

    user->rheology_constants.rheology_type = init_rheology_type;
    ierr = pTatinNonlinearStokesSolve(user,snes,X_s,"Picard Stage");CHKERRQ(ierr);

    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = SNESDestroyMGCtx(snes);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = HMGOperator_Destroy(&mgctx);CHKERRQ(ierr);

    ierr = HMG_Destroy(&mgctx);CHKERRQ(ierr);
    ierr = VecDestroy(&F);CHKERRQ(ierr);
  }
  
  /* compute timestep */
  user->dt = 1.0e32;
  {
    Vec       velocity,pressure;
    PetscReal timestep;

    ierr = DMCompositeGetAccess(multipys_pack,X_s,&velocity,&pressure);CHKERRQ(ierr);
    ierr = SwarmUpdatePosition_ComputeCourantStep(dav,velocity,&timestep);CHKERRQ(ierr);
    ierr = pTatin_SetTimestep(user,"StkCourant",timestep);CHKERRQ(ierr);

    ierr = UpdateMeshGeometry_ComputeSurfaceCourantTimestep(dav,velocity,surface_displacement_max,&timestep);CHKERRQ(ierr);
    ierr = pTatin_SetTimestep(user,"StkSurfaceCourant",timestep);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(multipys_pack,X_s,&velocity,&pressure);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"  timestep[stokes] dt_courant = %1.4e \n", user->dt );
  }
  
  /* first time step, enforce to be super small */
  user->dt = user->dt * 1.0e-10;
  
  /* initialise the energy solver */
  if (active_energy) {
    ierr = pTatinPhysCompEnergyFV_Initialise(energyfv,energyfv->T);CHKERRQ(ierr);
    
    energyfv->dt = user->dt;
    PetscPrintf(PETSC_COMM_WORLD,"  timestep[adv-diff] dt_courant = <UNAVAILABLE BUT NOT REQUIRED>\n");
  }
  
  {
    char prefix[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step%D",user->step);CHKERRQ(ierr);
    ierr = pTatinModel_Output(model,user,X_s,prefix);CHKERRQ(ierr);
  }
  
  /* last thing we do */
  {
    char           checkpoints_path[PETSC_MAX_PATH_LEN];
    char           checkpoint_path[PETSC_MAX_PATH_LEN];
    char           filename[PETSC_MAX_PATH_LEN];
    PetscLogDouble time[2];

    ierr = PetscSNPrintf(checkpoints_path,PETSC_MAX_PATH_LEN-1,"%s/checkpoints",user->outputpath);CHKERRQ(ierr);
    ierr = pTatinCreateDirectory(checkpoints_path);CHKERRQ(ierr);

    ierr = PetscSNPrintf(checkpoint_path,PETSC_MAX_PATH_LEN-1,"%s/initial_condition",checkpoints_path);CHKERRQ(ierr);
    ierr = pTatinCreateDirectory(checkpoint_path);CHKERRQ(ierr);

    PetscTime(&time[0]);
    ierr = pTatinCtxCheckpointWriteFV(user,checkpoint_path,NULL,
                                      multipys_pack,dmfv,
                                      0,NULL,NULL,
                                      X_s,X_e,NULL,NULL);CHKERRQ(ierr);
    PetscTime(&time[1]);
    ierr = pTatinLogBasicCPUtime(user,"Checkpoint.write()",time[1]-time[0]);CHKERRQ(ierr);

    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/initial_condition/ptatin.options",checkpoints_path);CHKERRQ(ierr);
    ierr = pTatinWriteOptionsFile(filename);CHKERRQ(ierr);
  }

  /* write out a default string for restarting the job */
  {
    char        restartfile[PETSC_MAX_PATH_LEN];
    char        restartstring[PETSC_MAX_PATH_LEN];
    PetscMPIInt rank;

    ierr = PetscSNPrintf(restartfile,PETSC_MAX_PATH_LEN-1,"%s/restart.default",user->outputpath);CHKERRQ(ierr);
    ierr = PetscSNPrintf(restartstring,PETSC_MAX_PATH_LEN-1,"-restart_directory %s/checkpoints/initial_condition",user->outputpath);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    if (rank == 0) {
      FILE *fp;

      fp = fopen(restartfile,"w");
      fprintf(fp,"%s",restartstring);
      fclose(fp);
    }
  }
  
  ierr = VecDestroy(&X_s);CHKERRQ(ierr);

  *pctx = user;
  PetscFunctionReturn(0);
}

PetscErrorCode Run_NonLinearFV(pTatinCtx user,Vec v1,Vec v2)
{
  PetscInt           step,step0;
  Vec                X = NULL,velocity,pressure;
  Vec                F_s = NULL;
  PetscBool          active_energy,monitor_stages,activate_quasi_newton_coord_update;
  PhysCompStokes     stokes = NULL;
  PhysCompEnergyFV   energyfv = NULL;
  DM                 dmstokes,dav,dap,dmfv = NULL;
  PetscErrorCode     ierr;
  pTatinModel        model = NULL;
  PetscReal          surface_displacement_max = 1.0e32;
  PetscLogDouble     time[4];
  AuuMultiLevelCtx   mgctx;
  PetscReal          timestep,dt_factor = 1.0;
  PetscMPIInt        rank;
  
  PetscFunctionBegin;
  ierr = pTatinLogNote(user,"  [ptatin_driver.Execute]");CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor_stages",&monitor_stages,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_quasi_newton_coordinate_update",&activate_quasi_newton_coord_update,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt_max_surface_displacement",&surface_displacement_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt_factor",&dt_factor,NULL);CHKERRQ(ierr);

  ierr = pTatinGetModel(user,&model);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dmstokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&dav,&dap);CHKERRQ(ierr);
  
  /* Pack all physics together */
  /* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
  ierr = PetscObjectReference((PetscObject)stokes->stokes_pack);CHKERRQ(ierr);
  user->pack = stokes->stokes_pack;
  
  if (v1) {
    X = v1;
  } else {
    ierr = DMCreateGlobalVector(dmstokes,&X);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(X,&F_s);CHKERRQ(ierr);
  
  ierr = pTatinContextValid_EnergyFV(user,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = pTatinGetContext_EnergyFV(user,&energyfv);CHKERRQ(ierr);
    ierr = FVDAGetDM(energyfv->fv,&dmfv);CHKERRQ(ierr);
    ierr = pTatinLogBasicDMDA(user,"EnergyFV",dmfv);CHKERRQ(ierr);
    if (v2) {
      energyfv->T = v2;
      ierr = pTatinCtxAttachModelData(user,"PhysCompEnergy_T",(void*)energyfv->T);CHKERRQ(ierr);
    }
  }
  
  ierr = HMG_SetUp(&mgctx,user);CHKERRQ(ierr);
  step0 = user->step + 1;
  
  /* Timestep 0 forces a supersmall dt (computed dt * 1e-10)
     However, the JSON file does not support to write a number that small
     This result in a dt = 0.0 and produces a division by 0.0 in the EnergyFV solver
     To prevent this I introduce the following test */
  if (user->dt <= 1.0e-17) {
    user->dt = 1.0e-17;
  }
  
  for (step=step0; step<=user->nsteps; step++) {
    char stepname[PETSC_MAX_PATH_LEN];
    Vec  fv_coor_k,q2_coor_k;
    
    /* Update time */
    user->step = step;
    user->time += user->dt;
    if (active_energy) {
      energyfv->time = user->time;
      energyfv->dt   = user->dt;
    }
    
    PetscPrintf(PETSC_COMM_WORLD,"<<----------------------------------------------------------------------------------------------->>\n");
    PetscPrintf(PETSC_COMM_WORLD,"   [[ EXECUTING TIME STEP : %D ]]\n", step );
    PetscPrintf(PETSC_COMM_WORLD,"     dt    : %1.4e \n", user->dt );
    PetscPrintf(PETSC_COMM_WORLD,"     time  : %1.4e \n", user->time );

    ierr = pTatinLogBasic(user);CHKERRQ(ierr);
    
    /* update mesh coordinates then restore */
    {
      DM  cdm;
      Vec tmp,q2_coor;

      ierr = DMGetCoordinateDM(dav,&cdm);CHKERRQ(ierr);
      ierr = DMGetCoordinates(dav,&q2_coor);CHKERRQ(ierr);
      
      ierr = VecDuplicate(q2_coor,&tmp);CHKERRQ(ierr);
      ierr = VecCopy(q2_coor,tmp);CHKERRQ(ierr);
      
      ierr = DMCreateGlobalVector(cdm,&q2_coor_k);CHKERRQ(ierr);
      ierr = pTatinModel_UpdateMeshGeometry(model,user,X);CHKERRQ(ierr);
      
      ierr = DMGetCoordinates(dav,&q2_coor);CHKERRQ(ierr);
      ierr = VecCopy(q2_coor,q2_coor_k);CHKERRQ(ierr);
      ierr = VecCopy(tmp,q2_coor);CHKERRQ(ierr);
      ierr = DMDAUpdateGhostedCoordinates(dav);CHKERRQ(ierr);
      
      ierr = VecDestroy(&tmp);CHKERRQ(ierr);
    }
    
    /* solve energy equation with FV + ALE */
    // [FV EXTENSION] (1) Evaluate thermal properties on material points, project onto FV cell / cell faces
    if (active_energy) {
      ierr = EnergyFVEvaluateCoefficients(user,0.0,energyfv,NULL,NULL);CHKERRQ(ierr);
      
      ierr = pTatinPhysCompEnergyFV_MPProjection(energyfv,user);CHKERRQ(ierr);
      
      ierr = FVDACellPropertyProjectToFace_HarmonicMean(energyfv->fv,"k","k");CHKERRQ(ierr);
      
      ierr = EvalRHS_HeatProd(energyfv->fv,energyfv->G);CHKERRQ(ierr);
      /* Scale by dt, note the minus sign */
      ierr = VecScale(energyfv->G,-energyfv->dt);CHKERRQ(ierr);
    }
    
    // [FV EXTENSION] (1) Interpolate Q2 velocity onto vertices of FV geometry mesh, (2) interpolate the nodal velocity onto the cell faces
    if (active_energy) {
      PetscTime(&time[0]);
      ierr = DMCompositeGetAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
      
      // (1)
      ierr = PhysCompEnergyFVInterpolateMacroQ2ToSubQ1(dav,velocity,energyfv,energyfv->dmv,energyfv->velocity);CHKERRQ(ierr);
      
      ierr = DMCompositeRestoreAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
      
      // (2)
      ierr = PhysCompEnergyFVInterpolateNormalVectorToFace(energyfv,energyfv->velocity,"v.n");CHKERRQ(ierr);
      
      ierr = PhysCompEnergyFVInterpolateVectorToFace(energyfv,energyfv->velocity,"v");CHKERRQ(ierr);
      PetscTime(&time[1]);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV-InterpV",time[1]-time[0]);CHKERRQ(ierr);
    }
    
    // [FV EXTENSION] Compute ALE velocity
    if (active_energy) {
      DM  dm_fv_geometry;
      Vec fv_vertex_coor_geometry;
      
      PetscTime(&time[0]);
      ierr = FVDAGetGeometryDM(energyfv->fv,&dm_fv_geometry);CHKERRQ(ierr);
      ierr = FVDAGetGeometryCoordinates(energyfv->fv,&fv_vertex_coor_geometry);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(dm_fv_geometry,&fv_coor_k);CHKERRQ(ierr);
      
      ierr = PhysCompEnergyFVInterpolateMacroQ2ToSubQ1(dav,q2_coor_k,energyfv,dm_fv_geometry,fv_coor_k);CHKERRQ(ierr);
      {
        Vec x_target;
        
        ierr = FVDAAccessData_ALE(energyfv->fv,NULL,NULL,&x_target);CHKERRQ(ierr);
        ierr = VecCopy(fv_coor_k,x_target);CHKERRQ(ierr);
      }
      ierr = pTatinPhysCompEnergyFV_ComputeALEVelocity(dm_fv_geometry,fv_vertex_coor_geometry,fv_coor_k,user->dt,energyfv->velocity);CHKERRQ(ierr); /* note we re-use storage for velocity here */
      
      ierr = PhysCompEnergyFVInterpolateVectorToFace(energyfv,energyfv->velocity,"xDot");CHKERRQ(ierr);
      //ierr = PhysCompEnergyFVInterpolateNormalVectorToFace(energyfv,energyfv->velocity,"xDot.n");CHKERRQ(ierr);
      PetscTime(&time[1]);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV-ALE-ComputeV",time[1]-time[0]);CHKERRQ(ierr);
    }
    
    // [FV EXTENSION] Make v.n define on each face compatible (e.g. ensure it satisfies \int v.n dS = \int div(v) dV = 0
    if (active_energy) {
      KSP ksp_pp;
      Vec source,fv_vertex_coor_geometry;
      
      ierr = FVDAGetGeometryCoordinates(energyfv->fv,&fv_vertex_coor_geometry);CHKERRQ(ierr);
      ierr = VecDuplicate(energyfv->T,&source);CHKERRQ(ierr);
      
      ierr = FVDAPPCompatibleVelocityCreate(energyfv->fv,&ksp_pp);CHKERRQ(ierr);
      
      PetscTime(&time[0]);
      ierr = VecZeroEntries(source);CHKERRQ(ierr); /* div(u) = 0 */
      ierr = FVDAPostProcessCompatibleVelocity(energyfv->fv,"v","v.n",source,ksp_pp);CHKERRQ(ierr);
      PetscTime(&time[1]);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV-PostProc-v",time[1]-time[0]);CHKERRQ(ierr);

      PetscTime(&time[0]);
      ierr = VecZeroEntries(source);CHKERRQ(ierr);
      ierr = pTatinPhysCompEnergyFV_ComputeALESource(energyfv->fv,fv_vertex_coor_geometry,fv_coor_k,energyfv->dt,source,PETSC_TRUE);CHKERRQ(ierr);
      ierr = FVDAPostProcessCompatibleVelocity(energyfv->fv,"xDot","xDot.n",source,ksp_pp);CHKERRQ(ierr);
      PetscTime(&time[1]);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV-PostProc-xDot",time[1]-time[0]);CHKERRQ(ierr);

      ierr = KSPDestroy(&ksp_pp);CHKERRQ(ierr);
      ierr = VecDestroy(&source);CHKERRQ(ierr);
    }

    // [FV EXTENSION] Combine (v - v_mesh)
    {
      FVDA      fv;
      PetscInt  f,nfaces;
      PetscReal *vdotn,*xDotdotn;
      
      fv = energyfv->fv;
      
      ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyByNameArray(fv,"v.n",&vdotn);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyByNameArray(fv,"xDot.n",&xDotdotn);CHKERRQ(ierr);
      
      for (f=0; f<nfaces; f++) {
        vdotn[f] = vdotn[f] - xDotdotn[f];
      }
    }
    
    /* Update boundary conditions */
    /* Fine level setup */
    ierr = pTatinModel_ApplyBoundaryCondition(model,user);CHKERRQ(ierr);
    
    /* (i) solve energy equation */
    if (active_energy) {
      PetscReal *dt;
      Vec       E,Ek,Ekk;
      
      E    = energyfv->T;
      Ek   = energyfv->Told;
      //ierr = FVDAAccessData_TimeDep(energyfv->fv,&dt,&Ekk);CHKERRQ(ierr);
      ierr = FVDAAccessData_ALE(energyfv->fv,&dt,&Ekk,NULL);CHKERRQ(ierr);
      *dt = energyfv->dt;
      
      /* Push current state into old state */
      ierr = VecCopy(E,Ek);CHKERRQ(ierr);
      ierr = VecCopy(E,Ekk);CHKERRQ(ierr);
      
      /* Three choices */
      {
        //const PetscReal range[] = {0.0,1.0e32};
        
        //ierr = EnergyFV_RK1(energyfv->snes,range,energyfv->time,energyfv->dt,energyfv->Told,energyfv->T);CHKERRQ(ierr);
        
        // high-order no limiting (negative temperature)
        //ierr = EnergyFV_RK2SSP(energyfv->snes,NULL,energyfv->time,energyfv->dt,energyfv->Told,energyfv->T);CHKERRQ(ierr);
        
        //ierr = EnergyFV_RK2SSP(energyfv->snes,range,energyfv->time,energyfv->dt,energyfv->Told,energyfv->T);CHKERRQ(ierr);
        
        PetscTime(&time[0]);
        ierr = SNESSolve(energyfv->snes,energyfv->G,energyfv->T);CHKERRQ(ierr);
        PetscTime(&time[1]);
        
        ierr = pTatinLogBasicSNES(user,"EnergyFV",energyfv->snes);CHKERRQ(ierr);
        ierr = pTatinLogBasicCPUtime(user,"EnergyFV-Solve",time[1]-time[0]);CHKERRQ(ierr);
      } 
    }
    
    /* update marker time dependent terms */
    /* e.g. e_plastic^1 = e_plastic^0 + dt * [ strain_rate_inv(u^0) ] */
    /*
     NOTE: for a consistent forward difference time integration we evaluate u^0 at x^0
     - thus this update is performed BEFORE we advect the markers
     */
    ierr = pTatin_UpdateCoefficientTemporalDependence_Stokes(user,X);CHKERRQ(ierr);
    
    /* update marker positions */
    ierr = DMCompositeGetAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
    ierr = MaterialPointStd_UpdateGlobalCoordinates(user->materialpoint_db,dav,velocity,user->dt);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
    
    /* Q2: update mesh coordinates */
    {
      Vec q2_coor;
      
      ierr = DMGetCoordinates(dav,&q2_coor);CHKERRQ(ierr);
      
      ierr = VecCopy(q2_coor_k,q2_coor);CHKERRQ(ierr);
      ierr = DMDAUpdateGhostedCoordinates(dav);CHKERRQ(ierr);
    }
    
    /* FV: update mesh coordinates */
    if (active_energy) {
      Vec fv_vertex_coor_geometry;
      
      ierr = FVDAGetGeometryCoordinates(energyfv->fv,&fv_vertex_coor_geometry);CHKERRQ(ierr);
      ierr = VecCopy(fv_coor_k,fv_vertex_coor_geometry);CHKERRQ(ierr);
      
      ierr = PhysCompEnergyFVUpdateGeometry(energyfv,stokes);CHKERRQ(ierr);
    }
    
    /* update mesh coordinate hierarchy */
    ierr = DMDARestrictCoordinatesHierarchy(mgctx.dav_hierarchy,mgctx.nlevels);CHKERRQ(ierr);

    /* 3 Update local coordinates and communicate */
    ierr = MaterialPointStd_UpdateCoordinates(user->materialpoint_db,dav,user->materialpoint_ex);CHKERRQ(ierr);

    /* add / remove points if cells are over populated or depleted of points */
    //ierr = MaterialPointPopulationControl_v1(user);CHKERRQ(ierr);

    /* 3a - Add material */
    //ierr = pTatinModel_ApplyMaterialBoundaryCondition(model,user);CHKERRQ(ierr);

    ierr = pTatinModel_AdaptMaterialPointResolution(model,user);CHKERRQ(ierr);
    
    /* update markers = >> gauss points */
    {
      int               npoints;
      DataField         PField_std;
      DataField         PField_stokes;
      MPntStd           *mp_std;
      MPntPStokes       *mp_stokes;

      DataBucketGetDataFieldByName(user->materialpoint_db, MPntStd_classname     , &PField_std);
      DataBucketGetDataFieldByName(user->materialpoint_db, MPntPStokes_classname , &PField_stokes);

      DataBucketGetSizes(user->materialpoint_db,&npoints,NULL,NULL);
      DataFieldGetEntries(PField_std,(void**)&mp_std);
      DataFieldGetEntries(PField_stokes,(void**)&mp_stokes);
      
      ierr = SwarmUpdateGaussPropertiesLocalL2Projection_Q1_MPntPStokes_Hierarchy(user->coefficient_projection_type,npoints,mp_std,mp_stokes,mgctx.nlevels,mgctx.interpolation_eta,mgctx.dav_hierarchy,mgctx.volQ);CHKERRQ(ierr);

#if 0
      /* START OF DEBUG PART*/
      PetscInt   np_points=0;
      DataField  PField_energy;
      
      DataFieldGetAccess(PField_std);
      DataFieldVerifyAccess(PField_std,sizeof(MPntStd));
      
      DataFieldGetAccess(PField_stokes);
      DataFieldVerifyAccess(PField_stokes,sizeof(MPntPStokes));
      
      DataBucketGetDataFieldByName(user->materialpoint_db,MPntPEnergy_classname,&PField_energy);
      DataFieldGetAccess(PField_energy);
      DataFieldVerifyAccess(PField_energy,sizeof(MPntPEnergy));
      
      for (np_points=0;np_points < npoints; np_points++) {
        MPntStd           *material_point;
        MPntPStokes       *material_point_stokes;
        MPntPEnergy       *material_point_energy;
        
        DataFieldAccessPoint(PField_std,np_points,(void**)&material_point);
        DataFieldAccessPoint(PField_stokes,np_points,(void**)&material_point_stokes);
        DataFieldAccessPoint(PField_energy,np_points,(void**)&material_point_energy);
        
        PetscPrintf(PETSC_COMM_WORLD,"density on mp_stokes[%d] = %1.4e; mp_energy diffusivity = %1.4e\n",np_points,material_point_stokes->rho,material_point_energy->diffusivity);
        
        
      }
      DataFieldRestoreAccess(PField_std);
      DataFieldRestoreAccess(PField_stokes);
      DataFieldRestoreAccess(PField_energy);
      /* END OF DEBUG PART */
#endif
      DataFieldRestoreEntries(PField_std,(void**)&mp_std);
      DataFieldRestoreEntries(PField_stokes,(void**)&mp_stokes);
    }
    
    /* Solve lithostatic pressure and apply on the surface quadrature points for Stokes */
    ierr = ModelApplyTractionFromLithoPressure(user,X);CHKERRQ(ierr);
    
    /* Coarse grid setup: Configure boundary conditions */
    ierr = pTatinModel_ApplyBoundaryConditionMG(mgctx.nlevels,mgctx.u_bclist,mgctx.dav_hierarchy,model,user);CHKERRQ(ierr);
    
    /* (ii) solve stokes */
    {
      SNES snes;
      Mat  A,B;

      ierr = HMGOperator_SetUp(&mgctx,user,&A,&B);CHKERRQ(ierr);
      ierr = pTatinNonlinearStokesSolveCreate(user,A,B,F_s,&mgctx,&snes);CHKERRQ(ierr);

      PetscPrintf(PETSC_COMM_WORLD,"   [[ COMPUTING FLOW FIELD FOR STEP : %D ]]\n",step);
      ierr = pTatinNonlinearStokesSolve(user,snes,X,NULL);CHKERRQ(ierr);

      ierr = MatDestroy(&A);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
      ierr = SNESDestroyMGCtx(snes);CHKERRQ(ierr);
      ierr = SNESDestroy(&snes);CHKERRQ(ierr);
      ierr = HMGOperator_Destroy(&mgctx);CHKERRQ(ierr);
    }
    
    /* compute timestep */
    user->dt = 1.0e32;
    ierr = DMCompositeGetAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
    ierr = SwarmUpdatePosition_ComputeCourantStep(dav,velocity,&timestep);CHKERRQ(ierr);
    timestep = timestep * dt_factor;
    ierr = pTatin_SetTimestep(user,"StkCourant",timestep);CHKERRQ(ierr);

    ierr = UpdateMeshGeometry_ComputeSurfaceCourantTimestep(dav,velocity,surface_displacement_max,&timestep);CHKERRQ(ierr);
    timestep = timestep * dt_factor;
    ierr = pTatin_SetTimestep(user,"StkSurfaceCourant",timestep);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"  timestep_stokes[%D] dt_courant = %1.4e \n", step,user->dt );
    if (active_energy) {
      {
        ierr = DMCompositeGetAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr); 
        // (1)
        ierr = PhysCompEnergyFVInterpolateMacroQ2ToSubQ1(dav,velocity,energyfv,energyfv->dmv,energyfv->velocity);CHKERRQ(ierr);
        
        ierr = pTatinPhysCompEnergyFV_ComputeAdvectiveTimestep(energyfv,energyfv->velocity,&timestep);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"  PhysCompEnergyFV_ComputeAdvectiveTimestep[%D] dt_courant = %1.4e \n", step,timestep );
        
        ierr = DMCompositeRestoreAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
      }
      //ierr = pTatin_SetTimestep(user,"AdvDiffCourant",timestep);CHKERRQ(ierr);
      //PetscPrintf(PETSC_COMM_WORLD,"  timestep_advdiff[%D] dt_courant = %1.4e \n", step,user->dt );
    }
    
    /* output */
    if (step%user->output_frequency == 0) {
      PetscSNPrintf(stepname,PETSC_MAX_PATH_LEN-1,"step%1.6D",step);
      ierr = pTatinModel_Output(model,user,X,stepname);CHKERRQ(ierr);
    }
    
    ierr = pTatin3dCheckpointManagerFV(user,X);CHKERRQ(ierr);
    
    /* tidy up */
    if (active_energy) {
      ierr = VecDestroy(&fv_coor_k);CHKERRQ(ierr);
      ierr = VecDestroy(&q2_coor_k);CHKERRQ(ierr);
    }
    
    //if (user->time > user->time_max) break;
  }
  
  ierr = VecDestroy(&F_s);CHKERRQ(ierr);
  ierr = HMG_Destroy(&mgctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  PetscBool      init = PETSC_FALSE,load = PETSC_FALSE, run = PETSC_FALSE;
  pTatinCtx      pctx = NULL;

  ierr = pTatinInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = pTatinModelRegisterAll();CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-init",&init,NULL);CHKERRQ(ierr);
  if (init) {
    ierr = GenerateICStateFromModelDefinition_FV(&pctx);CHKERRQ(ierr);
    if (pctx) { ierr = pTatin3dDestroyContext(&pctx); }
    pctx = NULL;
  }

  ierr = PetscOptionsGetBool(NULL,NULL,"-load",&load,NULL);CHKERRQ(ierr);
  if (load) {
    ierr = LoadStateFromModelDefinitionFV(&pctx,NULL,NULL,PETSC_TRUE);CHKERRQ(ierr);
    /* do something */
    if (pctx) { ierr = pTatin3dDestroyContext(&pctx); }
    pctx = NULL;
  }

  ierr = PetscOptionsGetBool(NULL,NULL,"-run",&run,NULL);CHKERRQ(ierr);
  if (run || (!init && !load)) {
    Vec       Xup,Xt;
    PetscBool restart_string_found = PETSC_FALSE,flg = PETSC_FALSE;
    char      outputpath[PETSC_MAX_PATH_LEN];

    /* look for a default restart file */
    ierr = PetscOptionsGetString(NULL,NULL,"-output_path",outputpath,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
    if (flg) {
      char fname[PETSC_MAX_PATH_LEN];

      ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/restart.default",outputpath);CHKERRQ(ierr);
      ierr = pTatinTestFile(fname,'r',&restart_string_found);CHKERRQ(ierr);
      if (restart_string_found) {
        PetscPrintf(PETSC_COMM_WORLD,"[pTatin] Detected default restart option file helper: %s\n",fname);
        //ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL,fname,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscOptionsInsert(NULL,&argc,&argv,fname);CHKERRQ(ierr);
      }
    }

    ierr = LoadStateFromModelDefinitionFV(&pctx,&Xup,&Xt,PETSC_FALSE);CHKERRQ(ierr);

    ierr = Run_NonLinearFV(pctx,Xup,Xt);CHKERRQ(ierr);

    ierr = VecDestroy(&Xup);CHKERRQ(ierr);
    ierr = VecDestroy(&Xt);CHKERRQ(ierr);
    if (pctx) { ierr = pTatin3dDestroyContext(&pctx); }
    pctx = NULL;
  }

  if (pctx) { ierr = pTatin3dDestroyContext(&pctx); }

  ierr = pTatinModelDeRegisterAll();CHKERRQ(ierr);
  ierr = pTatinFinalize();CHKERRQ(ierr);
  return(0);
}
