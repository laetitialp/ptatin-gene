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
**    filename:   model_ops_subduction_oblique.c
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

/* Developed by Anthony Jourdon [jourdon.anthon@gmail.com] */

#include "petsc.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin_utils.h"
#include "dmda_bcs.h"
#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPStokesPl_def.h"
#include "MPntPEnergy_def.h"
#include "output_material_points.h"
#include "output_material_points_p0.h"
#include "energy_output.h"
#include "output_paraview.h"
#include "material_point_std_utils.h"
#include "material_point_utils.h"
#include "material_point_popcontrol.h"
#include "stokes_form_function.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include "dmda_iterator.h"
#include "mesh_update.h"
#include "dmda_update_coords.h"
#include "dmda_remesh.h"
#include "ptatin3d_stokes.h"
#include "ptatin3d_energy.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include "quadrature.h"
#include "QPntSurfCoefStokes_def.h"
#include "geometry_object.h"
#include <material_constants_energy.h>
#include "model_utils.h"
#include "debug_ctx.h"

#include "element_utils_q1.h"
#include "element_type_Q2.h"
#include "dmda_element_q2p1.h"

static const char MODEL_NAME_DB[] = "model_debug_";

PetscErrorCode Debug_SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(DM da,PetscInt Nxp[],PetscReal perturb, PetscReal epsilon, PetscInt face_idx,DataBucket db);
PetscErrorCode Model_SetParameters_Debug(RheologyConstants *rheology, DataBucket materialconstants, ModelDebugCtx *data);

PetscErrorCode ModelInitialize_Debug(pTatinCtx c,void *ctx)
{
  ModelDebugCtx             *data;
  RheologyConstants         *rheology;
  DataBucket                materialconstants;
  PetscBool                 flg,found;
  PetscErrorCode            ierr;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelDebugCtx*)ctx;
  
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);
  /* force energy equation to be introduced */
  
  data->n_phases = 3;
  rheology->nphases_active = data->n_phases;
  rheology->apply_viscosity_cutoff_global = PETSC_TRUE;
  rheology->eta_upper_cutoff_global = 1.e+25;
  rheology->eta_lower_cutoff_global = 1.e+19;
  
  /* cutoff */
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_DB,"-apply_viscosity_cutoff_global",&rheology->apply_viscosity_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-eta_lower_cutoff_global",&rheology->eta_lower_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-eta_upper_cutoff_global",&rheology->eta_upper_cutoff_global,NULL);CHKERRQ(ierr);
  
  /* box geometry, [m] */
  data->Lx = 1000.0e3; 
  data->Ly = 0.0e3;
  data->Lz = 600.0e3;
  data->Ox = 0.0e3;
  data->Oy = -250.0e3;
  data->Oz = 0.0e3;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Lx",&data->Lx,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Ly",&data->Ly,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Lz",&data->Lz,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Ox",&data->Ox,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Oy",&data->Oy,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Oz",&data->Oz,&flg);CHKERRQ(ierr);
  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"********** Box Geometry **********\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[Ox,Lx] = [%+1.4e [m], %+1.4e [m]]\n", data->Ox ,data->Lx );
  PetscPrintf(PETSC_COMM_WORLD,"[Oy,Ly] = [%+1.4e [m], %+1.4e [m]]\n", data->Oy ,data->Ly );
  PetscPrintf(PETSC_COMM_WORLD,"[Oz,Lz] = [%+1.4e [m], %+1.4e [m]]\n", data->Oz ,data->Lz );

  /* Layering */
  data->layer1 = -40.0e3;
  data->layer2 = -100.0e3;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-layer1",&data->layer1,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-layer2",&data->layer2,&flg);CHKERRQ(ierr);

  /* Velocity */
  data->normV = 1.0;
  /* Angle of the velocity vector with the face on which it is applied */
  data->angle_v = 30.0;
  
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-normV",  &data->normV,NULL);CHKERRQ(ierr);  
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-angle_v",&data->angle_v,NULL);CHKERRQ(ierr);
  
  data->viscous = PETSC_FALSE;
  data->vp_std  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_DB,"-rheology_viscous",&data->viscous,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_DB,"-rheology_vp_std", &data->vp_std,NULL);CHKERRQ(ierr);

  data->output_markers = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_DB,"-output_markers", &data->output_markers,NULL);CHKERRQ(ierr);  

  if(!data->viscous && !data->vp_std) {
    PetscPrintf(PETSC_COMM_WORLD,"**** NO RHEOLOGY SELECTED: USING DEFAULT RHEOLOGY_VISCOUS ****\n");CHKERRQ(ierr);
    data->viscous = PETSC_TRUE;
  }

  /* Material constants */
  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);
  
  /* Call rheology assignement function */
  ierr = Model_SetParameters_Debug(rheology,materialconstants,data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode Model_SetRheology_VP_STD(RheologyConstants *rheology, DataBucket materialconstants, ModelDebugCtx *data)
{
  EnergyConductivityConst   *data_k;
  EnergySourceConst         *data_Q;
  EnergyMaterialConstants   *matconstants_e;
  DataField                 PField,PField_k,PField_Q;
  PetscInt                  region_idx;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);
  
  rheology->rheology_type = RHEOLOGY_VP_STD;

  data->Ttop = 0.0;
  data->Tbottom = 1600.0;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Ttop",   &data->Ttop,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Tbottom",&data->Tbottom,NULL);CHKERRQ(ierr);
  
  /* Energy material constants */
  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  /* Conductivity */
  DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
  DataFieldGetEntries(PField_k,(void**)&data_k);
  
  /* Heat source */
  DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
  DataFieldGetEntries(PField_Q,(void**)&data_Q);
  
  // HERE ASSIGN RHEOLOGY PARAMETERS FOR VP_STD
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"RHEOLOGY VP_STD NOT IMPLEMENTED FOR THAT MODEL\n");CHKERRQ(ierr);

  /* Report all material parameters values */
  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsPrintAll(materialconstants,region_idx);
    MaterialConstantsEnergyPrintAll(materialconstants,region_idx);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode Model_SetRheology_VISCOUS(RheologyConstants *rheology, DataBucket materialconstants, ModelDebugCtx *data)
{
  PetscInt       region_idx;
  PetscReal      density,viscosity;
  char           *option_name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  rheology->rheology_type = RHEOLOGY_VP_STD;

  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
    /* Set region viscosity */
    viscosity = 1.0e+23;
    if (asprintf (&option_name, "-eta0_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB, option_name,&viscosity,NULL);CHKERRQ(ierr); 
    free (option_name);
    ierr = MaterialConstantsSetValues_ViscosityConst(materialconstants,region_idx,viscosity);CHKERRQ(ierr);
    
    /* Set region density */
    density = 2700.0;
    if (asprintf (&option_name, "-rho0_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB, option_name,&density,NULL);CHKERRQ(ierr);
    free (option_name);
    ierr = MaterialConstantsSetValues_DensityConst(materialconstants,region_idx,density);CHKERRQ(ierr);
  }

  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsPrintAll(materialconstants,region_idx);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode Model_SetParameters_Debug(RheologyConstants *rheology, DataBucket materialconstants, ModelDebugCtx *data)
{
  PetscInt       region_idx;
  PetscReal      cm_per_yer2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 );
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* scaling */
  data->length_bar       = 100.0 * 1.0e3;
  data->viscosity_bar    = 1e22;
  data->velocity_bar     = 1.0e-10;
  data->time_bar         = data->length_bar / data->velocity_bar;
  data->pressure_bar     = data->viscosity_bar/data->time_bar;
  data->density_bar      = data->pressure_bar * (data->time_bar*data->time_bar)/(data->length_bar*data->length_bar); // kg.m^-3
  data->acceleration_bar = data->length_bar / (data->time_bar*data->time_bar);

  PetscPrintf(PETSC_COMM_WORLD,"[subduction_oblique]:  during the solve scaling will be done using \n");
  PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n", data->length_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n", data->velocity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n", data->time_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n", data->viscosity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->density_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n", data->pressure_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  a*    : %1.4e [m.s^-2]\n", data->acceleration_bar );
  /* Scale viscosity cutoff */
  rheology->eta_lower_cutoff_global = rheology->eta_lower_cutoff_global / data->viscosity_bar;
  rheology->eta_upper_cutoff_global = rheology->eta_upper_cutoff_global / data->viscosity_bar;
  /* Scale length */
  data->Lx = data->Lx / data->length_bar;
  data->Ly = data->Ly / data->length_bar;
  data->Lz = data->Lz / data->length_bar;
  data->Ox = data->Ox / data->length_bar;
  data->Oy = data->Oy / data->length_bar;
  data->Oz = data->Oz / data->length_bar;
  data->layer1 = data->layer1 / data->length_bar;
  data->layer2 = data->layer2 / data->length_bar;
  /* Scale velocity */
  data->normV = data->normV*cm_per_yer2m_per_sec/data->velocity_bar;
  data->angle_v = data->angle_v*M_PI/180.0;

  if(data->vp_std) {
    ierr = Model_SetRheology_VP_STD(rheology,materialconstants,data);CHKERRQ(ierr);
    /* scale material properties */
    for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
      MaterialConstantsScaleAll(materialconstants,region_idx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);
      MaterialConstantsEnergyScaleAll(materialconstants,region_idx,data->length_bar,data->time_bar,data->pressure_bar);
    }
  } else if (data->viscous) {
    ierr = Model_SetRheology_VISCOUS(rheology,materialconstants,data);CHKERRQ(ierr);
    for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
      MaterialConstantsScaleAll(materialconstants,region_idx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometry_Debug(pTatinCtx c,void *ctx)
{
  ModelDebugCtx    *data = (ModelDebugCtx*)ctx;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);CHKERRQ(ierr);
  if (data->is_2D) {
    ierr = pTatin3d_DefineVelocityMeshGeometryQuasi2D(c);CHKERRQ(ierr);
  }
  
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  
  PetscReal gvec[] = { 0.0, -9.8, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(c->stokes_ctx,1.0/data->acceleration_bar);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_VISCOUS(pTatinCtx c,void *ctx)
{
  ModelDebugCtx *data = (ModelDebugCtx*)ctx;
  DataBucket                db;
  DataField                 PField_std;
  PetscInt                  p;
  int                       n_mp_points;
  
  PetscFunctionBegin;
  
  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    int           region_idx;
    double        *position;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);

    /* Access using the getter function */
    MPntStdGetField_global_coord(material_point,&position);

    region_idx = 0;
    if (position[1] < data->layer1) { region_idx = 1; }
    if (position[1] < data->layer2) { region_idx = 2; }
        
    MPntStdSetField_phase_index(material_point,region_idx);
  }
  DataFieldRestoreAccess(PField_std);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_Debug(pTatinCtx c,void *ctx)
{
  ModelDebugCtx *data = (ModelDebugCtx*)ctx;
  PetscErrorCode            ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  if (data->viscous) {
    ierr = ModelApplyInitialMaterialGeometry_VISCOUS(c,data);CHKERRQ(ierr);
  } else if (data->vp_std) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"RHEOLOGY VP_STD NOT IMPLEMENTED FOR THAT MODEL\n");
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"A RHEOLOGY MUST BE SET, use -viscous or -vp_std\n");
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialSolution_Debug(pTatinCtx c,Vec X,void *ctx)
{
  ModelDebugCtx                                *data;
  DM                                           stokes_pack,dau,dap;
  Vec                                          velocity,pressure;
  PetscReal                                    vx,vz,vxl,vxr,vzl,vzr;
  DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
  DMDAVecTraverse3d_InterpCtx                  IntpCtx;
  PetscReal                                    MeshMin[3],MeshMax[3],domain_height;
  PetscBool                                    active_energy;
  PetscErrorCode                               ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelDebugCtx*)ctx;
  
  stokes_pack = c->stokes_ctx->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  
  /* Velocity IC */
  vz = data->normV*PetscCosReal(data->angle_v);
  vx = PetscSqrtReal(data->normV*data->normV - vz*vz);

  /* Left face */  
  vxl = -vx;
  vzl = -vz;
  /* Right face */
  vxr = vx;
  vzr = vz;
  
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);

  ierr = DMDAVecTraverse3d_InterpCtxSetUp_X(&IntpCtx,(vxr-vxl)/(data->Lx-data->Ox),vxl,0.0);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d(dau,velocity,0,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d_InterpCtxSetUp_Y(&IntpCtx,0.0,0.0,0.0);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d(dau,velocity,1,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d_InterpCtxSetUp_Z(&IntpCtx,(vzl-vzr)/(data->Lz-data->Oz),vzl,0.0);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d(dau,velocity,2,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);
  
  /* Pressure IC */
  ierr = VecZeroEntries(pressure);CHKERRQ(ierr);

  ierr = DMGetBoundingBox(dau,MeshMin,MeshMax);CHKERRQ(ierr);
  domain_height = MeshMax[1] - MeshMin[1];

  HPctx.surface_pressure = 0.0;
  HPctx.ref_height = domain_height;
  HPctx.ref_N      = c->stokes_ctx->my-1;
  HPctx.grav       = 9.8 / data->acceleration_bar;
  HPctx.rho        = 3300.0 / data->density_bar;

  ierr = DMDAVecTraverseIJK(dap,pressure,0,DMDAVecTraverseIJK_HydroStaticPressure_v2,     (void*)&HPctx);CHKERRQ(ierr);
  ierr = DMDAVecTraverseIJK(dap,pressure,2,DMDAVecTraverseIJK_HydroStaticPressure_dpdy_v2,(void*)&HPctx);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /* Temperature IC */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergyFV energy;
    PetscBool        flg = PETSC_FALSE,subduction_temperature_ic_from_file = PETSC_FALSE;
    char             fname[PETSC_MAX_PATH_LEN],temperature_file[PETSC_MAX_PATH_LEN];
    PetscViewer      viewer;
    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    
    ierr = PetscOptionsGetBool(NULL,MODEL_NAME_DB,"-temperature_ic_from_file",&subduction_temperature_ic_from_file,NULL);CHKERRQ(ierr);
    if (subduction_temperature_ic_from_file) {
      /* Check if a file is provided */
      ierr = PetscOptionsGetString(NULL,MODEL_NAME_DB,"-temperature_file",temperature_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,temperature_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      } else {
        PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/temperature_steady.pbvec",c->outputpath);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      }
      ierr = VecLoad(energy->T,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Providing a temperature file for initial state is required\n");
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialStokesVariableMarkers_Debug(pTatinCtx user,Vec X,void *ctx)
{
  DM                         stokes_pack,dau,dap;
  PhysCompStokes             stokes;
  Vec                        Uloc,Ploc;
  PetscScalar                *LA_Uloc,*LA_Ploc;
  DataField                  PField;
  MaterialConst_MaterialType *truc;
  PetscInt                   regionidx;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  DataBucketGetDataFieldByName(user->material_constants,MaterialConst_MaterialType_classname,&PField);

  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  ierr = DMCompositeScatter(stokes_pack,X,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);

  ierr = pTatin_EvaluateRheologyNonlinearities(user,dau,LA_Uloc,dap,LA_Ploc);CHKERRQ(ierr);

  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscBool BCListEvaluator_Lithosphere_DEBUG(PetscScalar position[],PetscScalar *value,void *ctx)
{
  PetscBool            impose_dirichlet;
  BC_Lithosphere_DEBUG data_ctx = (BC_Lithosphere_DEBUG)ctx;
  
  PetscFunctionBegin;
  
  if (position[1] >= data_ctx->y_lab) {
    *value = data_ctx->v;
    impose_dirichlet = PETSC_TRUE;
  } else {
    impose_dirichlet = PETSC_FALSE;
  }
  
  return impose_dirichlet;
}

PetscErrorCode Debug_VelocityBC_Oblique(BCList bclist,DM dav,pTatinCtx c,ModelDebugCtx *data)
{
  BC_Lithosphere_DEBUG bcdata;
  PetscReal      vx,vz,vxl,vxr,vzl,vzr;
  PetscReal      zero = 0.0;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  ierr = PetscMalloc(sizeof(struct _p_BC_Lithosphere_DEBUG),&bcdata);CHKERRQ(ierr);

  /* Computing of the 2 velocity component required to get a vector of norm normV and angle angle_v */
  vz = data->normV*PetscCosReal(data->angle_v);
  vx = PetscSqrtReal(data->normV*data->normV - vz*vz);

  /* Left face */  
  vxl = -vx;
  vzl = -vz;
  /* Right face */
  vxr = vx;
  vzr = vz;

  bcdata->y_lab = data->layer2;
  bcdata->v = vxl;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_Lithosphere_DEBUG,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = vzl;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_Lithosphere_DEBUG,(void*)bcdata);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  bcdata->y_lab = data->layer2;
  bcdata->v = vxr;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_Lithosphere_DEBUG,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = vzr;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_Lithosphere_DEBUG,(void*)bcdata);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  if (data->is_2D) {
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  }
  
  /* Free slip bottom */
  if (!data->open_base) {
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  }

  ierr = PetscFree(bcdata);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTimeDependantEnergy_BCs(pTatinCtx c,ModelDebugCtx *data)
{
  PhysCompEnergyFV energy;
  PetscReal        val_T;
  PetscInt         l;
  PetscErrorCode   ierr;
  
  PetscFunctionBegin;

  ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);

  val_T = data->Tbottom;
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_S,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);
  
  val_T = data->Ttop;
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_N,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);

  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_E,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_W,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_F,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_B,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);

  /* Iterate through all boundary faces, if there is inflow redefine the bc value and bc flux method */
  const DACellFace flist[] = { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_B, DACELL_FACE_F };
  for (l=0; l<sizeof(flist)/sizeof(DACellFace); l++) {
    ierr = FVSetDirichletFromInflow(energy->fv,energy->T,flist[l]);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryCondition_Debug(pTatinCtx c,void *ctx)
{
  ModelDebugCtx *data;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscBool        active_energy;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  data = (ModelDebugCtx*)ctx;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Define velocity boundary conditions */
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = Debug_VelocityBC_Oblique(stokes->u_bclist,dav,c,data);CHKERRQ(ierr);
  /* Insert here the litho pressure functions */
  /* The faces with litho pressure BCs are:
     -  IMIN, IMAX below the lithosphere
     -  KMIN, KMAX on all face 
  */
  
  /* Define boundary conditions for any other physics */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = SetTimeDependantEnergy_BCs(c,data);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_Debug(PetscInt nl,BCList bclist[],DM dav[],pTatinCtx c,void *ctx)
{
  ModelDebugCtx *data;
  PetscInt         n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data = (ModelDebugCtx*)ctx;
  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = Debug_VelocityBC_Oblique(bclist[n],dav[n],c,data);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyMaterialBoundaryCondition_Debug(pTatinCtx c,void *ctx)
{
  ModelDebugCtx *data;
  PhysCompStokes  stokes;
  DM              stokes_pack,dav,dap;
  PetscInt        Nxp[2];
  PetscInt        *face_list;
  PetscReal       perturb, epsilon;
  DataBucket      material_point_db,material_point_face_db;
  PetscInt        f, n_face_list;
  int             p,n_mp_points;
  MPAccess        mpX;
  PetscErrorCode  ierr;

  const int nf = 3;
  const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl};//, MPField_Energy };
  char mp_file_prefix[256],prefix[256],stepprefix[256];

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelDebugCtx*)ctx;
  
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(c,&material_point_db,NULL);CHKERRQ(ierr);

  if (data->output_markers)
  {
    sprintf(prefix,"step%1.6d",(int)c->step);
    sprintf(mp_file_prefix,"%s_mpoints_bfr_injection",prefix);
    ierr = SwarmViewGeneric_ParaView(material_point_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  }

  /* create face storage for markers */
  DataBucketDuplicateFields(material_point_db,&material_point_face_db);
  
  if (data->is_2D){
    n_face_list = 2;
    ierr = PetscMalloc1(n_face_list,&face_list);CHKERRQ(ierr);
    face_list[0] = 0;
    face_list[1] = 1;
  } else {
    n_face_list = 4;
    ierr = PetscMalloc1(n_face_list,&face_list);CHKERRQ(ierr);
    face_list[0] = 0;
    face_list[1] = 1;
    face_list[2] = 4;
    face_list[3] = 5;
  }
  
  for (f=0; f<n_face_list; f++) {

    /* traverse */
    /* [0,1/east,west] ; [2,3/north,south] ; [4,5/front,back] */
    Nxp[0]  = 4;
    Nxp[1]  = 4;
    perturb = 0.1;

    /* reset size */
    DataBucketSetSizes(material_point_face_db,0,-1);

    /* assign coords */
    epsilon = 1.0e-6;
    //ierr = SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d(dav,Nxp,perturb, face_list[f], material_point_face_db);CHKERRQ(ierr);
    ierr = Debug_SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(dav,Nxp,perturb,epsilon,face_list[f],material_point_face_db);CHKERRQ(ierr);

    if (data->output_markers)
    {
      sprintf(stepprefix,"step%1.6d",(int)c->step);
      sprintf(prefix,"%s_face_%d_",stepprefix,(int)f);
      sprintf(mp_file_prefix,"%s_mpoints_bfr_region",prefix);
      ierr = SwarmViewGeneric_ParaView(material_point_face_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
    }

    /* assign values */
    DataBucketGetSizes(material_point_face_db,&n_mp_points,0,0);
    ierr = MaterialPointGetAccess(material_point_face_db,&mpX);CHKERRQ(ierr);
    for (p=0; p<n_mp_points; p++) {
      ierr = MaterialPointSet_phase_index(mpX,p,MATERIAL_POINT_PHASE_UNASSIGNED);CHKERRQ(ierr);
    }
    ierr = MaterialPointRestoreAccess(material_point_face_db,&mpX);CHKERRQ(ierr);

    if (data->output_markers)
    {
      sprintf(stepprefix,"step%1.6d",(int)c->step);
      sprintf(prefix,"%s_face_%d_",stepprefix,(int)f);
      sprintf(mp_file_prefix,"%s_mpoints_bfr_assignement",prefix);
      ierr = SwarmViewGeneric_ParaView(material_point_face_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
    }

    /* insert into volume bucket */
    DataBucketInsertValues(material_point_db,material_point_face_db);
  }
  if (data->output_markers)
  { 
    sprintf(prefix,"step%1.6d",(int)c->step);
    sprintf(mp_file_prefix,"%s_mpoints_aftr_db_insrt",prefix);
    ierr = SwarmViewGeneric_ParaView(material_point_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  }
  /* Copy ALL values from nearest markers to newly inserted markers except (xi,xip,pid) */
  //ierr = MaterialPointRegionAssignment_v1(material_point_db,dav);CHKERRQ(ierr);
  ierr = MaterialPointRegionAssignment_KDTree(material_point_db,PETSC_TRUE);CHKERRQ(ierr);
  if (data->output_markers)
  { 
    sprintf(prefix,"step%1.6d",(int)c->step);
    sprintf(mp_file_prefix,"%s_mpoints_aftr_assignement",prefix);
    ierr = SwarmViewGeneric_ParaView(material_point_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  }
  /* delete */
  DataBucketDestroy(&material_point_face_db);
  
  ierr = PetscFree(face_list);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyUpdateMeshGeometry_Debug_Noair(pTatinCtx c,Vec X,ModelDebugCtx *data)
{
  PhysCompStokes  stokes;
  DM              stokes_pack,dav,dap;
  Vec             velocity,pressure;
  PetscReal       dt,Kero;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* fully lagrangian update */
  ierr = pTatinGetTimestep(c,&dt);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);

  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /* SURFACE REMESHING */
  Kero = 1.0e-6;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_DB,"-Kero",&Kero,NULL);CHKERRQ(ierr);
  Kero = Kero / (data->length_bar*data->length_bar/data->time_bar);
  
  ierr = UpdateMeshGeometry_ApplyDiffusionJMAX(dav,Kero,dt,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,dt);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyUpdateMeshGeometry_Debug(pTatinCtx c,Vec X,void *ctx)
{
  ModelDebugCtx  *data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  data = (ModelDebugCtx*)ctx;
  
  ierr = ModelApplyUpdateMeshGeometry_Debug_Noair(c,X,data);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_Debug(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  ModelDebugCtx             *data;
  PetscBool                 active_energy;
  DataBucket                materialpoint_db;
  PetscErrorCode            ierr;
  static PetscBool          been_here = PETSC_FALSE;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelDebugCtx*)ctx;
  
  /* Output Velocity and pressure */
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  
  /* Output markers cell fields (for production runs) */
  {
    const MaterialPointVariable mp_prop_list[] = { MPV_region, MPV_viscosity, MPV_density, MPV_plastic_strain }; //MPV_viscous_strain,

    ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(c,PETSC_FALSE,sizeof(mp_prop_list)/sizeof(MaterialPointVariable),mp_prop_list,prefix);CHKERRQ(ierr);
  }
  
  /* Output raw markers (for testing and debugging) */
  if (data->output_markers)
  {
    ierr = pTatinGetMaterialPoints(c,&materialpoint_db,NULL);CHKERRQ(ierr);

    const int nf = 3;
    const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl};//, MPField_Energy };
    char mp_file_prefix[256];

    sprintf(mp_file_prefix,"%s_mpoints",prefix);
    ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  }
  
  /* Output temperature (FV) */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergyFV energy;
    char             root[PETSC_MAX_PATH_LEN],pvoutputdir[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    
    // PVD
    {
      char pvdfilename[PETSC_MAX_PATH_LEN],vtkfilename[PETSC_MAX_PATH_LEN];
      char stepprefix[PETSC_MAX_PATH_LEN];
      
      PetscSNPrintf(pvdfilename,PETSC_MAX_PATH_LEN-1,"%s/timeseries_T_fv.pvd",c->outputpath);
      if (prefix) { PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "%s_T_fv.pvts",prefix);
      } else {      PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "T_fv.pvts");           }
      
      PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%D",c->step);
      //ierr = ParaviewPVDOpenAppend(PETSC_FALSE,c->step,pvdfilename,c->time, vtkfilename, stepprefix);CHKERRQ(ierr);
      if (!been_here) { /* new file */
        ierr = ParaviewPVDOpen(pvdfilename);CHKERRQ(ierr);
        ierr = ParaviewPVDAppend(pvdfilename,c->time,vtkfilename,stepprefix);CHKERRQ(ierr);
      } else {
        ierr = ParaviewPVDAppend(pvdfilename,c->time,vtkfilename,stepprefix);CHKERRQ(ierr);
      }
    }
    ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",c->outputpath);CHKERRQ(ierr);
    ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",root,c->step);CHKERRQ(ierr);
    
    /* PetscVec */
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_energy",prefix);
    ierr = FVDAView_JSON(energy->fv,pvoutputdir,fname);CHKERRQ(ierr); /* write meta data abour fv mesh, its DMDA and the coords */
    ierr = FVDAView_Heavy(energy->fv,pvoutputdir,fname);CHKERRQ(ierr);  /* write cell fields */
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s_energy_T",pvoutputdir,prefix);
    ierr = PetscVecWriteJSON(energy->T,0,fname);CHKERRQ(ierr); /* write cell temperature */
    
    if (data->output_markers) {
      PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s-Tfv",pvoutputdir,prefix);
      ierr = FVDAView_CellData(energy->fv,energy->T,PETSC_TRUE,fname);CHKERRQ(ierr);
    }
  }
  
  been_here = PETSC_TRUE;
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_Debug(pTatinCtx c,void *ctx)
{
  ModelDebugCtx *data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelDebugCtx*)ctx;

  /* Free contents of structure */

  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_Debug(void)
{
  ModelDebugCtx *data;
  pTatinModel      m;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelDebugCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelDebugCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"debug");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void))ModelApplyInitialStokesVariableMarkers_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_Debug);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode Debug_SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(DM da,PetscInt Nxp[],PetscReal perturb, PetscReal epsilon, PetscInt face_idx,DataBucket db)
{
  DataField      PField;
  PetscInt       e,ei,ej,ek,eij2d;
  Vec            gcoords;
  PetscScalar    *LA_coords;
  PetscScalar    el_coords[Q2_NODES_PER_EL_3D*NSD];
  int            ncells,ncells_face,np_per_cell,points_face,points_face_local=0;
  PetscInt       nel,nen,lmx,lmy,lmz,MX,MY,MZ;
  const PetscInt *elnidx;
  PetscInt       p,k,pi,pj;
  PetscReal      dxi,deta;
  int            np_current,np_new;
  PetscInt       si,sj,sk,M,N,P,lnx,lny,lnz;
  PetscBool      contains_east,contains_west,contains_north,contains_south,contains_front,contains_back;
  PetscErrorCode ierr;


  PetscFunctionBegin;

  ierr = DMDAGetElements_pTatinQ2P1(da,&nel,&nen,&elnidx);CHKERRQ(ierr);
  ncells = nel;
  ierr = DMDAGetLocalSizeElementQ2(da,&lmx,&lmy,&lmz);CHKERRQ(ierr);

  switch (face_idx) {
    case 0:// east-west
      ncells_face = lmy * lmz; // east
      break;
    case 1:
      ncells_face = lmy * lmz; // west
      break;

    case 2:// north-south
      ncells_face = lmx * lmz; // north
      break;
    case 3:
      ncells_face = lmx * lmz; // south
      break;

    case 4: // front-back
      ncells_face = lmx * lmy; // front
      break;
    case 5:
      ncells_face = lmx * lmy; // back
      break;

    default:
      SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Unknown face index");
      break;
  }

  np_per_cell = Nxp[0] * Nxp[1];
  points_face = ncells_face * np_per_cell;

  if (perturb < 0.0) {
    SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Cannot use a negative perturbation");
  }
  if (perturb > 1.0) {
    SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Cannot use a perturbation greater than 1.0");
  }

  ierr = DMDAGetSizeElementQ2(da,&MX,&MY,&MZ);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&lnx,&lny,&lnz);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &M,&N,&P, 0,0,0, 0,0, 0,0,0, 0);CHKERRQ(ierr);

  contains_east  = PETSC_FALSE; if (si+lnx == M) { contains_east  = PETSC_TRUE; }
  contains_west  = PETSC_FALSE; if (si == 0)     { contains_west  = PETSC_TRUE; }
  contains_north = PETSC_FALSE; if (sj+lny == N) { contains_north = PETSC_TRUE; }
  contains_south = PETSC_FALSE; if (sj == 0)     { contains_south = PETSC_TRUE; }
  contains_front = PETSC_FALSE; if (sk+lnz == P) { contains_front = PETSC_TRUE; }
  contains_back  = PETSC_FALSE; if (sk == 0)     { contains_back  = PETSC_TRUE; }

  // re-size //
  switch (face_idx) {
    case 0:
      if (contains_east) points_face_local = points_face;
      break;
    case 1:
      if (contains_west) points_face_local = points_face;
      break;

    case 2:
      if (contains_north) points_face_local = points_face;
      break;
    case 3:
      if (contains_south) points_face_local = points_face;
      break;

    case 4:
      if (contains_front) points_face_local = points_face;
      break;
    case 5:
      if (contains_back) points_face_local = points_face;
      break;
  }
  
  DataBucketGetSizes(db,&np_current,NULL,NULL);
  np_new = np_current + points_face_local;
  
  DataBucketSetSizes(db,np_new,-1);

  /* setup for coords */
  ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_coords);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField);
  DataFieldGetAccess(PField);
  DataFieldVerifyAccess( PField,sizeof(MPntStd));

  dxi  = 2.0/(PetscReal)Nxp[0];
  deta = 2.0/(PetscReal)Nxp[1];

  p = np_current;
  for (e = 0; e < ncells; e++) {
    /* get coords for the element */
    ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&elnidx[nen*e],LA_coords);CHKERRQ(ierr);

    ek = e / (lmx*lmy);
    eij2d = e - ek * (lmx*lmy);
    ej = eij2d / lmx;
    ei = eij2d - ej * lmx;

    switch (face_idx) {
      case 0:// east-west
        if (!contains_east) { continue; }
        if (ei != lmx-1) { continue; }
        break;
      case 1:
        if (!contains_west) { continue; }
        if (ei != 0) { continue; }
        break;

      case 2:// north-south
        if (!contains_north) { continue; }
        if (ej != lmy-1) { continue; }
        break;
      case 3:
        if (!contains_south) { continue; }
        if (ej != 0) { continue; }
        break;

      case 4: // front-back
        if (!contains_front) { continue; }
        if (ek != lmz-1) { continue; }
        break;
      case 5:
        if (!contains_back) { continue; }
        if (ek != 0) { continue; }
        break;
    }
    
    for (pj=0; pj<Nxp[1]; pj++) {
      for (pi=0; pi<Nxp[0]; pi++) {
        MPntStd *marker;
        double xip2d[2],xip_shift2d[2],xip_rand2d[2];
        double xip[NSD],xp_rand[NSD],Ni[Q2_NODES_PER_EL_3D];

        /* define coordinates in 2d layout */
        xip2d[0] = -1.0 + dxi    * (pi + 0.5);
        xip2d[1] = -1.0 + deta   * (pj + 0.5);

        /* random between -0.5 <= shift <= 0.5 */
        xip_shift2d[0] = 1.0*(rand()/(RAND_MAX+1.0)) - 0.5;
        xip_shift2d[1] = 1.0*(rand()/(RAND_MAX+1.0)) - 0.5;

        xip_rand2d[0] = xip2d[0] + perturb * dxi    * xip_shift2d[0];
        xip_rand2d[1] = xip2d[1] + perturb * deta   * xip_shift2d[1];

        if (fabs(xip_rand2d[0]) > 1.0) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"fabs(x-point coord) greater than 1.0");
        }
        if (fabs(xip_rand2d[1]) > 1.0) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"fabs(y-point coord) greater than 1.0");
        }

        /* set to 3d dependnent on face */
        // case 0:// east-west
        // case 2:// north-south
        // case 4: // front-back
        switch (face_idx) {
          case 0:// east-west
            xip[0] = 1.0 - epsilon;
            xip[1] = xip_rand2d[0];
            xip[2] = xip_rand2d[1];
            break;
          case 1:
            xip[0] = -1.0 + epsilon;
            xip[1] = xip_rand2d[0];
            xip[2] = xip_rand2d[1];
            break;

          case 2:// north-south
            xip[0] = xip_rand2d[0];
            xip[1] = 1.0 - epsilon;
            xip[2] = xip_rand2d[1];
            break;
          case 3:
            xip[0] = xip_rand2d[0];
            xip[1] = -1.0 + epsilon;
            xip[2] = xip_rand2d[1];
            break;

          case 4: // front-back
            xip[0] = xip_rand2d[0];
            xip[1] = xip_rand2d[1];
            xip[2] = 1.0 - epsilon;
            break;
          case 5:
            xip[0] = xip_rand2d[0];
            xip[1] = xip_rand2d[1];
            xip[2] = -1.0 + epsilon;
            break;
        }

        pTatin_ConstructNi_Q2_3D(xip,Ni);

        xp_rand[0] = xp_rand[1] = xp_rand[2] = 0.0;
        for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
          xp_rand[0] += Ni[k] * el_coords[NSD*k+0];
          xp_rand[1] += Ni[k] * el_coords[NSD*k+1];
          xp_rand[2] += Ni[k] * el_coords[NSD*k+2];
        }

        DataFieldAccessPoint(PField,p,(void**)&marker);

        marker->coor[0] = xp_rand[0];
        marker->coor[1] = xp_rand[1];
        marker->coor[2] = xp_rand[2];

        marker->xi[0] = xip[0];
        marker->xi[1] = xip[1];
        marker->xi[2] = xip[2];

        marker->wil    = e;
        marker->pid    = 0;
        p++;
      }
    }

  }
  DataFieldRestoreAccess(PField);
  ierr = VecRestoreArray(gcoords,&LA_coords);CHKERRQ(ierr);

  ierr = SwarmMPntStd_AssignUniquePointIdentifiers(PetscObjectComm((PetscObject)da),db,np_current,np_new);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}