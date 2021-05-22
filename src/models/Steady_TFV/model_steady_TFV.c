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
 **    filename:   model_Steady_TFV_ops.c
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

#include "ptatin3d.h"
#include "private/ptatin_impl.h"

#include "dmda_bcs.h"
#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPEnergy_def.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include "dmda_iterator.h"
#include "mesh_deformation.h"
#include "mesh_update.h"
#include "dmda_remesh.h"
#include "dmda_element_q2p1.h"
#include "material_point_std_utils.h"
#include "material_point_popcontrol.h"
#include "output_material_points.h"
#include "xdmf_writer.h"
#include "output_material_points_p0.h"
#include "private/quadrature_impl.h"
#include "quadrature.h"
#include "QPntSurfCoefStokes_def.h"
#include <material_constants_energy.h>
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include "ptatin3d_energy.h"
#include "geometry_object.h"
#include "output_paraview.h"

#include "steady_TFV_ctx.h"

PetscErrorCode ModelInitialize_Steady_TFV(pTatinCtx c,void *ctx)
{
  ModelSteady_TFVCtx      *data = (ModelSteady_TFVCtx*)ctx;
  RheologyConstants       *rheology;
  int                     source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  EnergyConductivityConst *data_k;
  EnergySourceConst       *data_Q;
  DataField               PField_k,PField_Q,PField;
  DataBucket              materialconstants;
  EnergyMaterialConstants *matconstants_e;
  PetscReal               const_eta0[2],const_rho0[2],heat_source[2];
  PetscBool flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  rheology                 = &c->rheology_constants;
  rheology->rheology_type  = RHEOLOGY_VP_STD;
  rheology->nphases_active = 2;
  
  /* box geometry */
  data->Lx = 1.0;
  data->Ly = 1.0;
  data->Lz = 1.0;
  
  /* parse from command line */
  data->Tbottom = 10.0;
  data->Ttop = 0.0;
  
  const_eta0[0] = 1.0;
  const_eta0[1] = 1.0;

  const_rho0[0] = 0.0;
  const_rho0[1] = 1.0;
  
  heat_source[0] = 1.0;
  heat_source[1] = 0.0;

  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_eta0",&const_eta0[0],&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_eta1",&const_eta0[1],&flg);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_rho0",&const_rho0[0],&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_rho1",&const_rho0[1],&flg);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_Ttop",&data->Ttop,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_Tbottom",&data->Tbottom,&flg);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_heatsource0",&heat_source[0],&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_heatsource1",&heat_source[1],&flg);CHKERRQ(ierr);

  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);

  DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
  DataFieldGetEntries(PField_k,(void**)&data_k);
  
  DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
  DataFieldGetEntries(PField_Q,(void**)&data_Q);

  source_type[0] = ENERGYSOURCE_CONSTANT;

  MaterialConstantsSetValues_MaterialType(materialconstants,0,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
  MaterialConstantsSetValues_ViscosityConst(materialconstants,0,const_eta0[0]);
  MaterialConstantsSetValues_EnergyMaterialConstants(0,matconstants_e,0.0,0.0,const_rho0[0],1.0,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
  MaterialConstantsSetValues_DensityConst(materialconstants,0,const_rho0[0]);
  EnergyConductivityConstSetField_k0(&data_k[0],1.0e-6);
  EnergySourceConstSetField_HeatSource(&data_Q[0],heat_source[0]);
  
  MaterialConstantsSetValues_MaterialType(materialconstants,1,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
  MaterialConstantsSetValues_ViscosityConst(materialconstants,1,const_eta0[1]);
  MaterialConstantsSetValues_EnergyMaterialConstants(1,matconstants_e,0.0,0.0,const_rho0[1],1.0,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
  MaterialConstantsSetValues_DensityConst(materialconstants,1,const_rho0[1]);
  EnergyConductivityConstSetField_k0(&data_k[1],1.0e-6);
  EnergySourceConstSetField_HeatSource(&data_Q[1],heat_source[1]);

  /* set initial values for model parameters */
  /* material properties */
  data->nmaterials = rheology->nphases_active;
  data->eta0 = const_eta0[0];
  data->eta1 = const_eta0[1];
  data->rho0 = const_rho0[0];
  data->rho1 = const_rho0[1];

  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_Lx",&data->Lx,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_Ly",&data->Ly,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_Lz",&data->Lz,&flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryCondition_Steady_TFV(pTatinCtx user,void *ctx)
{
  ModelSteady_TFVCtx *data = (ModelSteady_TFVCtx*)ctx;
  PetscScalar zero = 0.0;
  PetscBool   active_energy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_JMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_JMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_JMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_KMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_KMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  ierr = pTatinContextValid_EnergyFV(user,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PetscInt         l;
    PhysCompEnergyFV energy;
    PetscReal        val_T;

    ierr = pTatinGetContext_EnergyFV(user,&energy);CHKERRQ(ierr);

    val_T = data->Tbottom;
    ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_S,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);
    
    val_T = data->Ttop;
    ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_N,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);

    ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_E,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
    ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_W,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
    ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_F,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
    ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_B,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_Steady_TFV(PetscInt nl,BCList bclist[],DM dav[],pTatinCtx user,void *ctx)
{
  ModelSteady_TFVCtx *data = (ModelSteady_TFVCtx*)ctx;
  PetscScalar zero = 0.0;
  PetscInt n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  for (n=0; n<nl; n++) {

    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_IMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_IMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_IMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_IMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_JMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_JMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_JMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_KMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_KMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist[n],dav[n],DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyMaterialBoundaryCondition_Steady_TFV(pTatinCtx c,void *ctx)
{
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  PetscPrintf(PETSC_COMM_WORLD,"  NOT IMPLEMENTED \n", PETSC_FUNCTION_NAME);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometry_Steady_TFV(pTatinCtx c,void *ctx)
{
  ModelSteady_TFVCtx *data = (ModelSteady_TFVCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = DMDASetUniformCoordinates(c->stokes_ctx->dav,0.0,data->Lx, 0.0,data->Ly, 0.0,data->Lz);CHKERRQ(ierr);

  //
  //PetscPrintf(PETSC_COMM_WORLD,"RUNNING DEFORMED MESH EXAMPLE \n");
  //ierr = MeshDeformation_GaussianBump_YMAX(c->stokes_ctx->dav,-0.3,-5.6);CHKERRQ(ierr);
  //ierr = DMDASetGraduatedCoordinates1D(c->stokes_ctx->dav,2,1,2.0);CHKERRQ(ierr);
  // [test c] remesh interp
  //ierr = DMDASetCoordinatesCentralSqueeze1D(c->stokes_ctx->dav,0,4.0,0.0,0.4,0.8,1.0);CHKERRQ(ierr);
  //ierr = DMDASetCoordinatesColumnRefinement(c->stokes_ctx->dav,1,4.0,0.66,1.0);CHKERRQ(ierr);

  /* diffusion example */
  /*
     ierr = UpdateMeshGeometry_ApplyDiffusionJMAX(c->stokes_ctx->dav,1.0e-2,0.44,
     PETSC_TRUE,PETSC_TRUE,PETSC_FALSE,PETSC_FALSE, PETSC_FALSE);CHKERRQ(ierr);
  */
  // [test d] remesh vertically and preserve topography
  /*
     {
     PetscInt npoints,dir;
     PetscReal xref[10],xnat[10];

     npoints = 6;
     xref[0] = 0.0;
     xref[1] = 0.2;
     xref[2] = 0.4;
     xref[3] = 0.6;
     xref[4] = 0.8;
     xref[5] = 1.0;

     xnat[0] = 0.0;
     xnat[1] = 0.67;
     xnat[2] = 0.92;
     xnat[3] = 0.97;
     xnat[4] = 0.985;
     xnat[5] = 1.0;

     dir = 1;
     ierr = DMDACoordinateRefinementTransferFunction(c->stokes_ctx->dav,dir,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);
     ierr = DMDABilinearizeQ2Elements(c->stokes_ctx->dav);CHKERRQ(ierr);
     }
  */

  PetscFunctionReturn(0);
}

PetscErrorCode Steady_TFV_ApplyInitialMaterialGeometry_Layer(pTatinCtx c,ModelSteady_TFVCtx *data)
{
  int                    p,n_mp_points;
  DataBucket             db;
  PetscReal              Layer1;
  DataField              PField_std,PField_stokes;
  PetscErrorCode         ierr;
  int                    phase;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_Layer1",&Layer1,NULL);CHKERRQ(ierr);
  
  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetSizes(db,&n_mp_points,0,0);

  for (p=0; p<n_mp_points; p++) {
    MPntStd     *material_point;
    MPntPStokes *mpprop_stokes;
    double      *position;

    DataFieldAccessPoint(PField_std,p,   (void**)&material_point);

    /* Access using the getter function provided for you (recommeneded for beginner user) */
    MPntStdGetField_global_coord(material_point,&position);

    if (position[1] > Layer1){
      phase = 0;
    } else {
      phase = 1;
    }

    /* user the setters provided for you */
    MPntStdSetField_phase_index(material_point,phase);

  }

  DataFieldRestoreAccess(PField_std);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_Steady_TFV(pTatinCtx c,void *ctx)
{
  ModelSteady_TFVCtx *data = (ModelSteady_TFVCtx*)ctx;
  PetscErrorCode        ierr;

  ierr = Steady_TFV_ApplyInitialMaterialGeometry_Layer(c,data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyUpdateMeshGeometry_Steady_TFV(pTatinCtx c,Vec X,void *ctx)
{
  ModelSteady_TFVCtx *data = (ModelSteady_TFVCtx*)ctx;
  Vec velocity,pressure;
  DM stokes_pack,dav,dap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  stokes_pack = c->stokes_ctx->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  //ierr = UpdateMeshGeometry_FullLagrangian(dav,velocity,c->dt);CHKERRQ(ierr);
  ierr = UpdateMeshGeometry_VerticalLagrangianSurfaceRemesh(dav,velocity,c->dt);CHKERRQ(ierr);

  // [test c] remesh interp
  //ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,c->dt);CHKERRQ(ierr);
  //ierr = DMDASetCoordinatesColumnRefinement(dav,1,4.0,0.75,1.0);CHKERRQ(ierr);

  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_Steady_TFV(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  PetscErrorCode ierr;
  DataBucket materialpoint_db;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  // testing
  //ierr = pTatin3d_ModelOutputLite_Velocity_Stokes(c,X,prefix);CHKERRQ(ierr);
  //ierr = pTatinOutputLiteMeshVelocitySlicedPVTS(c->stokes_ctx->stokes_pack,c->outputpath,prefix);CHKERRQ(ierr);
  //ierr = ptatin3d_StokesOutput_VelocityXDMF(c,X,prefix);CHKERRQ(ierr);
  // testing
  //ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  //ierr = pTatin3d_ModelOutput_MPntStd(c,prefix);CHKERRQ(ierr);
  ierr = pTatinGetMaterialPoints(c,&materialpoint_db,NULL);CHKERRQ(ierr);
  {
    const int nf = 3;
    const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_Energy };
    char mp_file_prefix[256];

    sprintf(mp_file_prefix,"%s_mpoints",prefix);
    ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  }
  // tests for alternate (output/load)ing of "single file marker" formats
  //ierr = SwarmDataWriteToPetscVec(c->materialpoint_db,prefix);CHKERRQ(ierr);
  //ierr = SwarmDataLoadFromPetscVec(c->materialpoint_db,prefix);CHKERRQ(ierr);
  /*
     {
     MaterialPointVariable vars[] = { MPV_viscosity, MPV_density, MPV_region };
  //ierr = pTatin3dModelOutput_MarkerCellFieldsP0_ParaView(c,sizeof(vars)/sizeof(MaterialPointVariable),vars,PETSC_TRUE,prefix);CHKERRQ(ierr);
  ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(c,PETSC_TRUE,sizeof(vars)/sizeof(MaterialPointVariable),vars,prefix);CHKERRQ(ierr);
  }
  */
  PetscFunctionReturn(0);
}

PetscErrorCode ModelInitialCondition_Steady_TFV(pTatinCtx c,Vec X,void *ctx)
{
  DM stokes_pack,dau,dap;
  Vec velocity,pressure;
  //DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
  //DMDAVecTraverse3d_InterpCtx IntpCtx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  stokes_pack = c->stokes_ctx->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);
  /* apply -5 < vx 5 across the domain x \in [0,1] */
  //ierr = DMDAVecTraverse3d_InterpCtxSetUp_X(&IntpCtx,10.0,-5.0,0.0);CHKERRQ(ierr);
  //ierr = DMDAVecTraverse3d(dau,velocity,0,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);

  ierr = VecZeroEntries(pressure);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /*
     ierr = PetscOptionsGetReal(NULL,NULL,"-model_Steady_TFV_rho0",&rho0,0);CHKERRQ(ierr);

     HPctx.surface_pressure = 0.0;
     HPctx.ref_height = data->Ly;
     HPctx.ref_N = c->stokes_ctx->my-1;
     HPctx.grav = 1.0;
     HPctx.rho = rho0;

     ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
     ierr = DMDAVecTraverseIJK(dap,pressure,0,DMDAVecTraverseIJK_HydroStaticPressure_v2,(void*)&HPctx);
     ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

     ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,"testHP");CHKERRQ(ierr);
     */

  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_Steady_TFV(pTatinCtx c,void *ctx)
{
  ModelSteady_TFVCtx *data = (ModelSteady_TFVCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  /* Free contents of structure */

  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_Steady_TFV(void)
{
  ModelSteady_TFVCtx *data;
  pTatinModel m;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelSteady_TFVCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelSteady_TFVCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"Steady_TFV");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelInitialCondition_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_Steady_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_Steady_TFV);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
