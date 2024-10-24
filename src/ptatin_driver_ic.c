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
 **    filename:   ptatin_driver_ic.c
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

static const char help[] = "Stokes solver using Q2-Pm1 mixed finite elements.\n"
"3D prototype of the (p)ragmatic version of Tatin. (pTatin3d_v0.0)\n\n";


#include "ptatin3d.h"
#include "ptatin3d_energy.h"
#include "private/ptatin_impl.h"
#include "ptatin_init.h"

#include "material_point_utils.h"
#include "material_point_std_utils.h"
#include "ptatin_log.h"
#include "ptatin_models.h"
#include "ptatin_utils.h"
#include "stokes_form_function.h"
#include "stokes_operators.h"
#include "sub_comm.h"
#include "dmda_redundant.h"
#include "stokes_output.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include <fvda_impl.h>


PetscErrorCode pTatin3d_material_points_check_ic(int argc,char **argv)
{
  DM               multipys_pack,dav,dmfv=NULL;
  PetscErrorCode   ierr;
  pTatinCtx        user;
  Vec              X,F,T,X_e;
  PetscBool        active_energy,active_energyfv;
  PhysCompEnergy   energy;
  PhysCompEnergyFV energyfv;

  PetscFunctionBegin;

  ierr = pTatin3dCreateContext(&user);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(user);CHKERRQ(ierr);

  /* Register all models */
  ierr = pTatinModelRegisterAll();CHKERRQ(ierr);
  /* Load model, call an initialization routines */
  ierr = pTatinModelLoad(user);CHKERRQ(ierr);

  ierr = pTatinModel_Initialize(user->model,user);CHKERRQ(ierr);

  /* Generate physics modules */
  ierr = pTatin3d_PhysCompStokesCreate(user);CHKERRQ(ierr);

  /* Pack all physics together */
  /* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
  ierr = PetscObjectReference((PetscObject)user->stokes_ctx->stokes_pack);CHKERRQ(ierr);
  user->pack = user->stokes_ctx->stokes_pack;

  /* fetch some local variables */
  multipys_pack = user->pack;
  dav           = user->stokes_ctx->dav;

  ierr = DMCreateGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = pTatinPhysCompAttachData_Stokes(user,X);CHKERRQ(ierr);
  
  ierr = pTatin3dCreateMaterialPoints(user,dav);CHKERRQ(ierr);

  /* mesh geometry */
  ierr = pTatinModel_ApplyInitialMeshGeometry(user->model,user);CHKERRQ(ierr);
  ierr = PhysCompStokesUpdateSurfaceQuadratureGeometry(user->stokes_ctx);CHKERRQ(ierr);
  ierr = SurfaceQuadratureViewParaview_Stokes(user->stokes_ctx,user->outputpath,"def");CHKERRQ(ierr);

  /* generate energy solver */
  /* NOTE - Generating the thermal solver here will ensure that the initial geometry on the mechanical model is copied */
  /* NOTE - Calling pTatinPhysCompActivate_Energy() after pTatin3dCreateMaterialPoints() is essential */
  {
    PetscBool load_energy = PETSC_FALSE;

    PetscOptionsGetBool(NULL,NULL,"-activate_energy",&load_energy,NULL);
    ierr = pTatinPhysCompActivate_Energy(user,load_energy);CHKERRQ(ierr);
    ierr = pTatinContextValid_Energy(user,&active_energy);CHKERRQ(ierr);
  }
  if (active_energy) {
    ierr = pTatinGetContext_Energy(user,&energy);CHKERRQ(ierr);

    ierr = pTatinLogBasicDMDA(user,"Energy",energy->daT);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(energy->daT,&T);CHKERRQ(ierr);
    ierr = pTatinPhysCompAttachData_Energy(user,T,NULL);CHKERRQ(ierr);
  }
  /* generate energyfv solver */
  {
    PetscBool load_energyfv = PETSC_FALSE;
    PetscOptionsGetBool(NULL,NULL,"-activate_energyfv",&load_energyfv,NULL);
    ierr = pTatinPhysCompActivate_EnergyFV(user,load_energyfv);CHKERRQ(ierr);
    ierr = pTatinContextValid_EnergyFV(user,&active_energyfv);CHKERRQ(ierr);
  }
  if (active_energyfv) {
    ierr = pTatinGetContext_EnergyFV(user,&energyfv);CHKERRQ(ierr);
    ierr = FVDAGetDM(energyfv->fv,&dmfv);CHKERRQ(ierr);
    ierr = pTatinLogBasicDMDA(user,"EnergyFV",dmfv);CHKERRQ(ierr);
    X_e  = energyfv->T;
    ierr = pTatinCtxAttachModelData(user,"PhysCompEnergy_T",(void*)X_e);CHKERRQ(ierr);
  }
  /* interpolate point coordinates (needed if mesh was modified) */
  /* interpolate material point coordinates (needed if mesh was modified) */
  ierr = MaterialPointCoordinateSetUp(user,dav);CHKERRQ(ierr);

  /* material geometry */
  ierr = pTatinModel_ApplyInitialMaterialGeometry(user->model,user);CHKERRQ(ierr);
  ierr = PhysCompStokesUpdateSurfaceQuadratureGeometry(user->stokes_ctx);CHKERRQ(ierr);

  /* boundary conditions */
  ierr = pTatinModel_ApplyBoundaryCondition(user->model,user);CHKERRQ(ierr);

  /* initial condition */
  ierr = pTatinModel_ApplyInitialSolution(user->model,user,X);CHKERRQ(ierr);
  {
    Vec Xu,Xp;
    ierr = DMCompositeGetAccess(multipys_pack,X,&Xu,&Xp);CHKERRQ(ierr);
    ierr = BCListInsert(user->stokes_ctx->u_bclist,Xu);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(multipys_pack,X,&Xu,&Xp);CHKERRQ(ierr);
  }

  if (active_energyfv) {
    /* Evaluate coefficients (diffusivity and heat source) on markers for initial conditions report */
    ierr = EnergyFVEvaluateCoefficients(user,0.0,energyfv,NULL,X);CHKERRQ(ierr);
  }

  /* test form function */
  {
    SNES snes;


    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,F,FormFunction_Stokes,user);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  }

  /* test data bucket viewer */
  DataBucketView(PetscObjectComm((PetscObject)multipys_pack), user->materialpoint_db,"materialpoint_stokes",DATABUCKET_VIEW_STDOUT);
  DataBucketView(PETSC_COMM_SELF, user->material_constants,"material_constants",DATABUCKET_VIEW_STDOUT);


  /* write out the initial condition */
  ierr = pTatinModel_Output(user->model,user,X,"icbc");CHKERRQ(ierr);

  /* test generic viewer */
  {
    const int nf = 1;
    const MaterialPointField mp_prop_list[] = { MPField_Std };
    ierr = SwarmViewGeneric_ParaView(user->materialpoint_db,nf,mp_prop_list,user->outputpath,"test_MPStd");CHKERRQ(ierr);
  }
  {
    const int nf = 1;
    const MaterialPointField mp_prop_list[] = { MPField_Stokes };
    ierr = SwarmViewGeneric_ParaView(user->materialpoint_db,nf,mp_prop_list,user->outputpath,"test_MPStokes");CHKERRQ(ierr);
  }

  {
    const int nf = 2;
    const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes };
    ierr = SwarmViewGeneric_ParaView(user->materialpoint_db,nf,mp_prop_list,user->outputpath,"test_MPStd_MPStokes");CHKERRQ(ierr);
  }


  if (active_energy) {
    ierr = VecDestroy(&T);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = pTatin3dDestroyContext(&user);

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = pTatinInitialize(&argc,&argv,0,help);CHKERRQ(ierr);

  ierr = pTatin3d_material_points_check_ic(argc,argv);CHKERRQ(ierr);

  ierr = pTatinFinalize();CHKERRQ(ierr);
  return 0;
}
