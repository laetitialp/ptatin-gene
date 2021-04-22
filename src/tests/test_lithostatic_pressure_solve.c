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
 **    filename:   test_lithostatic_pressure_solve.c
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
#include "litho_pressure_PDESolve.h"

PetscErrorCode test_lithostatic_pressure_solve(void)
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
  ierr = DMDABCListTraverse3d(LP->bclist,LP->da,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  /*
  val_P = 1.0;
  ierr = DMDABCListTraverse3d(LP->bclist,LP->da,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  val_P = 2.0;
  ierr = DMDABCListTraverse3d(LP->bclist,LP->da,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  val_P = 3.0;
  ierr = DMDABCListTraverse3d(LP->bclist,LP->da,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  val_P = 4.0;
  ierr = DMDABCListTraverse3d(LP->bclist,LP->da,DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  val_P = 5.0;
  ierr = DMDABCListTraverse3d(LP->bclist,LP->da,DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  */
  X = LP->X;
  F = LP->F;
  ierr = DMSetMatType(LP->da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(LP->da,&J);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  
  ierr = SNESSolve_LithoPressure(LP,J,X,F,pctx);CHKERRQ(ierr);
  
  /* Apply the computed lithostatic pressure to the stokes surface quadrature points on the face HEX_FACE_Neta (bottom) */
  ierr = ApplyLithostaticPressure_SurfQuadratureStokes_FullFace(pctx->stokes_ctx,LP->da,X,HEX_FACE_Neta);CHKERRQ(ierr);
  
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
  ierr = test_lithostatic_pressure_solve();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
