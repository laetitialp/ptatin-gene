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
 **    filename:   model_rift3D_T_ops.c
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

/*
   Developed by Laetitia Le Pourhiet [laetitia.le_pourhiet@upmc.fr]
 */


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
#include "stokes_form_function.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include "dmda_iterator.h"
#include "mesh_update.h"
#include "output_material_points.h"
#include "material_point_std_utils.h"
#include "material_point_utils.h"
#include "material_point_popcontrol.h"
#include "energy_output.h"
#include "ptatin3d_stokes.h"
#include "ptatin3d_energy.h"
#include "geometry_object.h"
#include "output_paraview.h"
#include <material_constants_energy.h>
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include "material_point_popcontrol.h"

#include "rift3D_T_ctx.h"

#define REMOVE_FACE_INJECTION

PetscErrorCode GeometryObjectSetFromOptions_Box(GeometryObject go);
PetscErrorCode GeometryObjectSetFromOptions_InfLayer(GeometryObject go);
PetscErrorCode GeometryObjectSetFromOptions_EllipticCylinder(GeometryObject go);

static PetscErrorCode ModelApplyUpdateMeshGeometry_Rift3D_T_semi_eulerian(pTatinCtx c,Vec X,void *ctx);
static PetscErrorCode ModelAdaptMaterialPointResolution_Rift3D_T_semi_eulerian(pTatinCtx c,void *ctx);

static PetscErrorCode ModelApplyInitialMaterialGeometry_Notchtest(pTatinCtx c,void *ctx);

static PetscErrorCode ModelSetMaterialConstants_DislDiff(DataBucket materialconstants, EnergyMaterialConstants *matconstants_e, RheologyConstants *rheology)
{
  /* Function designed to test DislDiff viscosity, Plastic and Energy are missing */
  PetscInt       i;
  PetscReal      preexpA_disl,Ascale_disl,entalpy_disl,Vmol_disl,nexp_disl,preexpA_diff,Ascale_diff,entalpy_diff,Vmol_diff,pexp_diff,gsize,Tref;
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  preexpA_disl = 1.1e5;
  Ascale_disl  = 1.0e6;
  entalpy_disl = 530.0e3;
  Vmol_disl    = 1.0e-5;
  nexp_disl    = 3.5;
  preexpA_diff = 1.5e9;
  Ascale_diff  = 1.0e6;
  entalpy_diff = 375.0e3;
  Vmol_diff    = 1.8e-5;
  pexp_diff    = 3.0;
  gsize        = 1.0e-2;
  Tref         = 273.0;

  for (i=0; i<rheology->nphases_active; i++) {
    MaterialConstantsSetValues_MaterialType(materialconstants,i,VISCOUS_ARRHENIUS_DISLDIFF,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
    MaterialConstantsSetValues_DensityConst(materialconstants,i,2700);
    EnergyMaterialConstantsSetFieldAll_SourceMethod(&matconstants_e[i],ENERGYSOURCE_NONE);
    MaterialConstantsSetValues_ViscosityArrh_DislDiff(materialconstants,i,preexpA_disl,Ascale_disl,entalpy_disl,Vmol_disl,nexp_disl,preexpA_diff,Ascale_diff,entalpy_diff,Vmol_diff,pexp_diff,gsize,Tref);
    MaterialConstantsPrintValues_ViscosityArrh_DislDiff(materialconstants,i);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialConstants_FK(DataBucket materialconstants, EnergyMaterialConstants *matconstants_e) 
{
  PetscReal      Cp,rho_ref;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  rho_ref = 1.0;
  Cp  = 1.0;
  // UPPER CRUST WITH STRIPES OF 4
  MaterialConstantsSetValues_MaterialType(materialconstants,0,VISCOUS_FRANKK,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);
  ierr = MaterialConstantsSetValues_EnergyMaterialConstants(0,matconstants_e,0.0,0.0,rho_ref,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,NULL);CHKERRQ(ierr);
  EnergyMaterialConstantsSetFieldAll_SourceMethod(&matconstants_e[0],ENERGYSOURCE_NONE);
  EnergyMaterialConstantsSetFieldByIndex_SourceMethod(&matconstants_e[0],0,ENERGYSOURCE_CONSTANT);

  MaterialConstantsSetValues_ViscosityFK(materialconstants,0,1.0e27,0.025);
  MaterialConstantsSetValues_DensityBoussinesq(materialconstants,0,2700,2.e-5,0.0);
  MaterialConstantsSetValues_DensityConst(materialconstants,0,2700);
  MaterialConstantsSetValues_PlasticDP(materialconstants,0,0.6,0.1,2.e7,2.e7,1.e7,2.e8);
  MaterialConstantsSetValues_PlasticMises(materialconstants,0,1.e8,1.e8);
  MaterialConstantsSetValues_SoftLin(materialconstants,0,0.0,0.3);

  MaterialConstantsSetValues_MaterialType(materialconstants,1,VISCOUS_FRANKK,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);

  MaterialConstantsSetValues_ViscosityFK(materialconstants,1,1.0e27,0.03);
  MaterialConstantsSetValues_DensityBoussinesq(materialconstants,1,2800,2.e-5,3.e-12);
  MaterialConstantsSetValues_DensityConst(materialconstants,1,2800);
  MaterialConstantsSetValues_PlasticDP(materialconstants,1,0.6,0.1,2.e7,2.e7,1.e7,2.e8);
  MaterialConstantsSetValues_PlasticMises(materialconstants,1,1.e8,1.e8);
  MaterialConstantsSetValues_SoftLin(materialconstants,1,0.0,0.3);

  ierr = MaterialConstantsSetValues_EnergyMaterialConstants(1,matconstants_e,0.0,0.0,rho_ref,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,NULL);CHKERRQ(ierr);
  EnergyMaterialConstantsSetFieldAll_SourceMethod(&matconstants_e[1],ENERGYSOURCE_NONE);
  EnergyMaterialConstantsSetFieldByIndex_SourceMethod(&matconstants_e[1],0,ENERGYSOURCE_CONSTANT);
  EnergyMaterialConstantsSetFieldByIndex_SourceMethod(&matconstants_e[1],0,ENERGYSOURCE_SHEAR_HEATING);

  MaterialConstantsSetValues_MaterialType(materialconstants,2,VISCOUS_FRANKK,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);
  MaterialConstantsSetValues_ViscosityFK(materialconstants,2,1.0e30,0.018);
  MaterialConstantsSetValues_DensityBoussinesq(materialconstants,2,3300,2.e-5,3.e-12);
  MaterialConstantsSetValues_DensityConst(materialconstants,2,3300);
  MaterialConstantsSetValues_PlasticDP(materialconstants,2,0.6,0.1,2.e7,2.e7,2.e7,3.e8);
  MaterialConstantsSetValues_PlasticMises(materialconstants,2,3.e8,3.e8);
  MaterialConstantsSetValues_SoftLin(materialconstants,2,0.0,0.3);

  ierr = MaterialConstantsSetValues_EnergyMaterialConstants(2,matconstants_e,0.0,0.0,rho_ref,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,NULL);CHKERRQ(ierr);
  EnergyMaterialConstantsSetFieldAll_SourceMethod(&matconstants_e[2],ENERGYSOURCE_NONE);
  EnergyMaterialConstantsSetFieldByIndex_SourceMethod(&matconstants_e[2],0,ENERGYSOURCE_CONSTANT);

  MaterialConstantsSetValues_MaterialType(materialconstants,3,VISCOUS_FRANKK,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);
  MaterialConstantsSetValues_ViscosityFK(materialconstants,3,1.0e30,0.018);
  MaterialConstantsSetValues_DensityBoussinesq(materialconstants,3,3300,2.e-5,3.e-12);
  MaterialConstantsSetValues_DensityConst(materialconstants,3,3300);
  MaterialConstantsSetValues_PlasticDP(materialconstants,3,0.6,0.1,2.e7,2.e7,2.e7,3.e8);
  MaterialConstantsSetValues_PlasticMises(materialconstants,3,3.e8,3.e8);
  MaterialConstantsSetValues_SoftLin(materialconstants,3,0.0,0.3);

  ierr = MaterialConstantsSetValues_EnergyMaterialConstants(3,matconstants_e,0.0,0.0,rho_ref,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,NULL);CHKERRQ(ierr);
  EnergyMaterialConstantsSetFieldAll_SourceMethod(&matconstants_e[3],ENERGYSOURCE_NONE);
  EnergyMaterialConstantsSetFieldByIndex_SourceMethod(&matconstants_e[3],0,ENERGYSOURCE_CONSTANT);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialConstants_Rift3D_TFV(DataBucket materialconstants, RheologyConstants *rheology) 
{
  EnergyMaterialConstants *matconstants_e;
  DataField               PField;
  PetscInt                regionidx,viscous_type;
  PetscErrorCode          ierr;

  PetscFunctionBegin;

  
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  viscous_type = 0;
  ierr = PetscOptionsGetInt(NULL,NULL,"-viscous_type",&viscous_type,NULL);CHKERRQ(ierr);

  switch (viscous_type)
  {
    case 0:
      ierr = ModelSetMaterialConstants_FK(materialconstants,matconstants_e);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelSetMaterialConstants_DislDiff(materialconstants,matconstants_e,rheology);CHKERRQ(ierr);
      break;

    default:
      ierr = ModelSetMaterialConstants_FK(materialconstants,matconstants_e);CHKERRQ(ierr);
      break;
  }

  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
    EnergyConductivityConst *data_k;
    EnergySourceConst       *data_Q;
    DataField               PField_k,PField_Q;

    DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
    DataFieldGetEntries(PField_k,(void**)&data_k);
    EnergyConductivityConstSetField_k0(&data_k[regionidx],1.0e-6);

    DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
    DataFieldGetEntries(PField_Q,(void**)&data_Q);
    EnergySourceConstSetField_HeatSource(&data_Q[regionidx],0.0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelInitialize_Rift3D_T(pTatinCtx c,void *ctx)
{
  ModelRift3D_TCtx *data = (ModelRift3D_TCtx*)ctx;
  RheologyConstants       *rheology;
  DataBucket              materialconstants;
  PetscBool               nondim;
  PetscScalar             vx,vy,vz,Sx,Sy,Sz;
  PetscInt                regionidx;
  PetscReal               cm_per_yer2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 ) ;
  PetscErrorCode          ierr;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  PetscPrintf(PETSC_COMM_WORLD,"Rift model expects the following dimensions for input\n");
  PetscPrintf(PETSC_COMM_WORLD," Box geometry: [m] \n");
  PetscPrintf(PETSC_COMM_WORLD," Viscosity:    [Pa.s] \n");
  PetscPrintf(PETSC_COMM_WORLD," Velocity:     [m/sec] \n");
  PetscPrintf(PETSC_COMM_WORLD," Density:      [kg/m^3] \n");

  PetscPrintf(PETSC_COMM_WORLD,"if you wish to use non dimensional input you must add -model_rift3D_T_dimensional \n");
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);

  rheology->rheology_type = RHEOLOGY_VP_STD;
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);
  /* I REALLY DONT LIKE THE FOLLOWING ONE, SHOULD BE  in model data */
  rheology->nphases_active = 4;
  rheology->apply_viscosity_cutoff_global = PETSC_TRUE;
  rheology->eta_upper_cutoff_global = 1.e+25;
  rheology->eta_lower_cutoff_global = 1.e+19;
  data->runmises = PETSC_FALSE;
  /* set the deffault values of the material constant for this particular model */
  /*scaling */
  data->length_bar    = 100.0 * 1.0e3;
  data->viscosity_bar = 1e22;
  data->velocity_bar  = 1.0e-10;
  data->dimensional   = PETSC_TRUE;

  /* box geometry, m */
  data->Lx =  12.0e5;
  data->Ly =  0.0e5;
  data->Lz =  6.0e5;
  //data->Ox =  -6.0e5;
  data->Ox =  0.0e5;
  data->Oy =  -1.5e5;
  data->Oz =  0.0e5;
  /* velocity cm/y */
  vx = 1.0*cm_per_yer2m_per_sec;
  vz = 0.25*cm_per_yer2m_per_sec;
  /* rho0 for initial pressure*/
  data->rho0 = 3140.0;
  /*Temperature */
  data->Tbottom = 1400.0;
  data->Ttop    = 0.0;
  data->thermal_age0 = 300;
  data->thermal_age_anom = 300;
  data->wx_anom  = 1;
  data->wz_anom  = 0.5;
  data->cx_anom  = 3.0;
  data->cz_anom  = 0.0;

  /* Material constant */
  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
  /* Material parameters */
  ierr = ModelSetMaterialConstants_Rift3D_TFV(materialconstants,rheology);

  /* Read the options */
  /*cutoff */
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_apply_viscosity_cutoff_global",&rheology->apply_viscosity_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_eta_lower_cutoff_global",&rheology->eta_lower_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_eta_upper_cutoff_global",&rheology->eta_upper_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_runwithmises",&data->runmises,NULL);CHKERRQ(ierr);
  /*scaling */
  nondim = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_nondimensional",&nondim,NULL);CHKERRQ(ierr);
  if (nondim){
    data->dimensional = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_vis_bar",&data->viscosity_bar,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_vel_bar",&data->velocity_bar,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_length_bar",&data->length_bar,NULL);CHKERRQ(ierr);
  }

  /* box geometry, m */
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_Lx",&data->Lx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_Ly",&data->Ly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_Lz",&data->Lz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_Ox",&data->Ox,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_Oy",&data->Oy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_Oz",&data->Oz,NULL);CHKERRQ(ierr);

  /* velocity cm/y */
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_vx",&vx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_vz",&vz,NULL);CHKERRQ(ierr);

  /* rho0 for initial pressure */
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_rho0",&data->rho0,NULL);CHKERRQ(ierr);

  /* temperature initial condition */
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_Tbot",&data->Tbottom,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_Ttop",&data->Ttop,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_age0",&data->thermal_age0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_ageAnom",&data->thermal_age_anom,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_wx",&data->wx_anom,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_wz",&data->wz_anom,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_cx",&data->cx_anom,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_cz",&data->cz_anom,NULL);CHKERRQ(ierr);

  /* Material constant */
  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
    PetscPrintf(PETSC_COMM_WORLD,"reading options");
    ierr= MaterialConstantsSetFromOptions(materialconstants,"model_rift3D_T",regionidx,PETSC_FALSE);CHKERRQ(ierr);
  }

  /*Compute velocity at bottom*/
  Sx = (data->Ly - data->Oy)*(data->Lz - data->Oz);
  Sz = (data->Ly - data->Oy)*(data->Lx - data->Ox);
  Sy = (data->Lx - data->Ox)*(data->Lz - data->Oz);
  vy = (2*vx*Sx-vz*Sz)/Sy;

  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"  input: -model_rift3D_T_Ox %+1.4e [SI] -model_rift3D_T_Lx : %+1.4e [SI]\n", data->Ox ,data->Lx );
  PetscPrintf(PETSC_COMM_WORLD,"  input: -model_rift3D_T_Oy %+1.4e [SI] -model_rift3D_T_Ly : %+1.4e [SI]\n", data->Oy ,data->Ly );
  PetscPrintf(PETSC_COMM_WORLD,"  input: -model_rift3D_T_Oz %+1.4e [SI] -model_rift3D_T_Lz : %+1.4e [SI]\n", data->Oz ,data->Lz );
  PetscPrintf(PETSC_COMM_WORLD,"  -model_rift3D_T_vx [m/s]:  %+1.4e  -model_rift3D_T_vz [m/s]:  %+1.4e : computed vy [m/s]:  %+1.4e \n", vx,vz,vy);
  PetscPrintf(PETSC_COMM_WORLD,"-model_rift3D_T_rho0 [kg/m^3] :%+1.4e \n", data->rho0 );
  PetscPrintf(PETSC_COMM_WORLD,"-model_rift3D_T_Tbot:%+1.4e \t -model_rift3D_T_Ttop:%+1.4e \t -model_rift3D_T_age0:%+1.4e \n",data->Tbottom,data->Ttop,  data->thermal_age0);
  PetscPrintf(PETSC_COMM_WORLD,"ageAnom:%+1.4e \t wx:%+1.4e \t wz:%+1.4e cx:%+1.4e \t cz:%+1.4e \n",data->thermal_age_anom,data->wx_anom,data->wz_anom,data->cx_anom,data->cz_anom);

  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
    MaterialConstantsPrintAll(materialconstants,regionidx);
    MaterialConstantsEnergyPrintAll(materialconstants,regionidx);
  }

  if (data->dimensional) {
    /*Compute additional scaling parameters*/
    data->time_bar      = data->length_bar / data->velocity_bar;
    data->pressure_bar  = data->viscosity_bar/data->time_bar;
    data->density_bar   = data->pressure_bar / data->length_bar;

    PetscPrintf(PETSC_COMM_WORLD,"[rift3D_T]:  during the solve scaling will be done using \n");
    PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n", data->length_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n", data->velocity_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n", data->time_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n", data->viscosity_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->density_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n", data->pressure_bar );
    //scale viscosity cutoff
    rheology->eta_lower_cutoff_global = rheology->eta_lower_cutoff_global / data->viscosity_bar;
    rheology->eta_upper_cutoff_global = rheology->eta_upper_cutoff_global / data->viscosity_bar;
    PetscPrintf(PETSC_COMM_WORLD,"[[ Viscosity cutoff ]]: (min/max) = %1.2e/%1.2e\n",rheology->eta_lower_cutoff_global,rheology->eta_upper_cutoff_global);
    //scale length
    data->Lx = data->Lx / data->length_bar;
    data->Ly = data->Ly / data->length_bar;
    data->Lz = data->Lz / data->length_bar;
    data->Ox = data->Ox / data->length_bar;
    data->Oy = data->Oy / data->length_bar;
    data->Oz = data->Oz / data->length_bar;

    //scale velocity
    data->vx = vx/data->velocity_bar;
    data->vy = vy/data->velocity_bar;
    data->vz = vz/data->velocity_bar;
    //scale rho0
    data->rho0 = data->rho0/data->density_bar;

    // scale material properties
    for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
      MaterialConstantsScaleAll(materialconstants,regionidx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);
      MaterialConstantsEnergyScaleAll(materialconstants,regionidx,data->length_bar,data->time_bar,
          data->pressure_bar);
    }

    /*Reports scaled values*/

    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_rift3D_T_Ox   :  %+1.4e    -model_rift3D_T_Lx   :  %+1.4e  \n", data->Ox ,data->Lx );
    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_rift3D_T_Oy   :  %+1.4e    -model_rift3D_T_Ly   :  %+1.4e \n", data->Oy, data->Ly );
    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_rift3D_T_Oz   :  %+1.4e    -model_rift3D_T_Lz   :  %+1.4e\n", data->Oz , data->Lz );

    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_rift3D_T_Vx:%+1.4e    -model_rift3D_T_vy:%+1.4e    -model_rift3D_T_vz:  %+1.4e \n", data->vx ,data->vy, data->vz);
    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_rift3D_T_rho0:%+1.4e \n", data->rho0 );
    PetscPrintf(PETSC_COMM_WORLD,"scaled value for material parameters\n");
    for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
      MaterialConstantsPrintAll(materialconstants,regionidx);
      MaterialConstantsEnergyPrintAll(materialconstants,regionidx);
    }
  }

  data->use_semi_eulerian_mesh = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_use_semi_eulerian",&data->use_semi_eulerian_mesh,NULL);CHKERRQ(ierr);
  if (data->use_semi_eulerian_mesh) {
    pTatinModel model;

    PetscPrintf(PETSC_COMM_WORLD,"rift3D_T: activating semi Eulerian mesh advection\n");
    ierr = pTatinGetModel(c,&model);CHKERRQ(ierr);
    ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_Rift3D_T_semi_eulerian);CHKERRQ(ierr);
    ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void))ModelAdaptMaterialPointResolution_Rift3D_T_semi_eulerian);CHKERRQ(ierr);
  }

  data->output_markers = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_output_markers",&data->output_markers,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
   Returns the parameters and function need to define initial thermal field.
   The function returned can be used to define either the initial condition for T or the boundary condition for T.
 */
static PetscErrorCode ModelRift3D_T_GetDescription_InitialThermalField(ModelRift3D_TCtx *data,PetscReal coeffs[],PetscBool (**func)(PetscScalar*,PetscScalar*,void*) )
{
  PetscFunctionBegin;
  /* assign params */
  coeffs[0] = data->cx_anom ;
  coeffs[1] = data->cz_anom ;
  coeffs[2] = data->thermal_age0;
  coeffs[3] = data->thermal_age_anom;
  coeffs[4] = data->length_bar;
  coeffs[5] = data->Tbottom;
  coeffs[6] = data->wx_anom;
  coeffs[7] = data->wz_anom;

  /* assign function to use */
  *func = DMDAVecTraverse3d_ERFC3DFunctionXYZ;

  PetscFunctionReturn(0);
}

//PetscBool DMDAVecTraverse3d_ERFC3DFunctionXYZ(PetscScalar pos[],PetscScalar *val,void *ctx);

PetscErrorCode FVDABCMethod_ERFC3DFunctionXYZ(FVDA fv,
                                             DACellFace face,
                                             PetscInt nfaces,
                                             const PetscReal coor[],
                                             const PetscReal normal[],
                                             const PetscInt cell[],
                                             PetscReal time,
                                             FVFluxType flux[],
                                             PetscReal bcvalue[],
                                             void *ctx)
{
  PetscInt    f;
  PetscScalar value;
  
  for (f=0; f<nfaces; f++) {
    PetscBool set = DMDAVecTraverse3d_ERFC3DFunctionXYZ((PetscScalar*)&coor[3*f],&value,ctx);
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    bcvalue[f] = value;
  }
  PetscFunctionReturn(0);
}


static PetscErrorCode ModelRift3D_T_GetDescription_InitialThermalField_FV(ModelRift3D_TCtx *data,PetscReal coeffs[],
                               PetscErrorCode (**func)(FVDA,
                                                             DACellFace,
                                                             PetscInt,
                                                             const PetscReal*,
                                                             const PetscReal*,
                                                             const PetscInt*,
                                                             PetscReal,FVFluxType*,PetscReal*,void*))
{
  PetscFunctionBegin;
  /* assign params */
  coeffs[0] = data->cx_anom ;
  coeffs[1] = data->cz_anom ;
  coeffs[2] = data->thermal_age0;
  coeffs[3] = data->thermal_age_anom;
  coeffs[4] = data->length_bar;
  coeffs[5] = data->Tbottom;
  coeffs[6] = data->wx_anom;
  coeffs[7] = data->wz_anom;
  
  /* assign function to use */
  *func = FVDABCMethod_ERFC3DFunctionXYZ;
  
  PetscFunctionReturn(0);
}

/*

   1/ Define boundary conditions in one place for this model.

   2/ Calling pattern should always be
   PetscErrorCode ModelRift3D_T_DefineBCList(BCList bclist,DM dav,pTatinCtx user,ModelRift3D_TCtx data)
   where ModelRift3D_TCtx data is a different type for each model.

   3/ Re-use this function in
   ModelApplyBoundaryCondition_Rift3D_T();
   ModelApplyBoundaryConditionMG_Rift3D_T();

*/
static PetscErrorCode ModelRift3D_T_DefineBCList(BCList bclist,DM dav,pTatinCtx user,ModelRift3D_TCtx *data)
{
  PetscScalar    vy;
  PetscBool      scissor;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  scissor = PETSC_FALSE;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_scissors",&scissor,NULL);CHKERRQ(ierr);

  vy  =  data->vy;

  /* infilling free slip base */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&vy);CHKERRQ(ierr);
  /* free surface top*/

  if(scissor == PETSC_FALSE){
    PetscScalar    vxl,vxr,vzf,vzb;
    PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
    vxl = -data->vx;
    vxr =  data->vx;
    vzf = -data->vz;
    vzb =  0.0;//data->vz;

    /*extension along face of normal x */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&(vxl));CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&(vxr));CHKERRQ(ierr);

    /*compression along face of normal z */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&(vzb));CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&(vzf));CHKERRQ(ierr);
  }else{
    PetscReal    coeffs[5];
    PetscBool    rigid=PETSC_FALSE;
    PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
    // set center of rotation x0
    coeffs[0]= 6.0;
    PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
    // set center of rotation z0
    coeffs[1]= 4.0;
    // set direction of interpolation
    coeffs[2]= 0;
    // set lenght in direction of interpolation
    coeffs[3]= data->Lx-data->Ox;
    // set x a;gular velocity based on vx at the right back corner
    coeffs[4]=data->vx/(data->Oz-4.0);

    ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_scissors_rigid",&rigid,NULL);CHKERRQ(ierr);
    if (rigid){
      ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,DMDAVecTraverse3d_ROTXZ_Z,(void*)coeffs);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,DMDAVecTraverse3d_ROTXZ_Z,(void*)coeffs);CHKERRQ(ierr);

    }

    /*extension along face of normal x */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,DMDAVecTraverse3d_ROTXZ_X,(void*)coeffs);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,DMDAVecTraverse3d_ROTXZ_X,(void*)coeffs);CHKERRQ(ierr);

    /*compression along face of normal z */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,DMDAVecTraverse3d_ROTXZ_Z,(void*)coeffs);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,DMDAVecTraverse3d_ROTXZ_Z,(void*)coeffs);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* This function serves as an exemple to show how to use a BCEvaluator with its own struct statically allocated */
static PetscErrorCode ModelApplyOutflowInflowOnVerticalFaces(BCList bclist, DM dav, ModelRift3D_TCtx *data)
{
  BCSplitFaceCtx bc_data;
  PetscReal      zero=0.0;
  PetscReal      y_lab,y_lith,y_asth,v_lith,v_asth;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  y_lab  = -40.0e3;
  y_lith = (y_lab + 5.0e3) / data->length_bar;
  y_asth = (y_lab - 5.0e3) / data->length_bar;
  
  bc_data.dim = 1;
  bc_data.x0  = y_asth;
  bc_data.x1  = y_lith;
  
  /* left face */
  v_lith = -data->vx;
  v_asth = -v_lith * (data->Ly - y_lith)/(y_asth - data->Oy);
  bc_data.v0  = v_asth;
  bc_data.v1  = v_lith;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_SplitLinear,(void*)&bc_data);CHKERRQ(ierr);
  /* right face */
  v_lith = data->vx;
  v_asth = -v_lith * (data->Ly - y_lith)/(y_asth - data->Oy);
  bc_data.v0  = v_asth;
  bc_data.v1  = v_lith;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_SplitLinear,(void*)&bc_data);CHKERRQ(ierr);

  /* free-slip along face of normal z */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  /* free-slip along face of normal y */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_Rift3D_T(pTatinCtx user,void *ctx)
{
  ModelRift3D_TCtx *data = (ModelRift3D_TCtx*)ctx;
  PetscBool        active_energy;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = ModelRift3D_T_DefineBCList(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,user,data);CHKERRQ(ierr);

  /* set boundary conditions for temperature */
  ierr = pTatinContextValid_Energy(user,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PetscReal      val_T;
    PhysCompEnergy energy;
    BCList         bclist;
    DM             daT;
    PetscBool      (*iterator_initial_thermal_field)(PetscScalar*,PetscScalar*,void*);
    PetscReal      coeffs[8];

    ierr   = pTatinGetContext_Energy(user,&energy);CHKERRQ(ierr);
    daT    = energy->daT;
    bclist = energy->T_bclist;

    ierr = ModelRift3D_T_GetDescription_InitialThermalField(data,coeffs,&iterator_initial_thermal_field);CHKERRQ(ierr);

    if (data->use_semi_eulerian_mesh) {
      /* use the erfc function */
      ierr = DMDABCListTraverse3d(bclist,daT,DMDABCList_JMIN_LOC,0,iterator_initial_thermal_field,(void*)coeffs);CHKERRQ(ierr);

      val_T = data->Ttop;
      ierr = DMDABCListTraverse3d(bclist,daT,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&val_T);CHKERRQ(ierr);
    } else {
      val_T = data->Tbottom;
      ierr = DMDABCListTraverse3d(bclist,daT,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&val_T);CHKERRQ(ierr);

      val_T = data->Ttop;
      ierr = DMDABCListTraverse3d(bclist,daT,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&val_T);CHKERRQ(ierr);
    }
  }

  ierr = pTatinContextValid_EnergyFV(user,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergyFV energy;
    PetscReal        val_T;
    PetscReal        coeffs[8];
    PetscErrorCode   (*iterator_initial_thermal_field)(FVDA,
                                                       DACellFace,
                                                       PetscInt,
                                                       const PetscReal*,
                                                       const PetscReal*,
                                                       const PetscInt*,
                                                       PetscReal,FVFluxType*,PetscReal*,void*);

    ierr = ModelRift3D_T_GetDescription_InitialThermalField_FV(data,coeffs,&iterator_initial_thermal_field);CHKERRQ(ierr);

    ierr = pTatinGetContext_EnergyFV(user,&energy);CHKERRQ(ierr);
    if (data->use_semi_eulerian_mesh) {

      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_S,PETSC_FALSE,0.0,iterator_initial_thermal_field,(void*)coeffs);CHKERRQ(ierr);
      
      val_T = data->Ttop;
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_N,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);

      
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_E,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_W,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_F,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_B,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);

    } else {
      val_T = data->Tbottom;
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_S,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);
      
      val_T = data->Ttop;
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_N,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);
      
      
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_E,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_W,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_F,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_B,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
    }
    
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryConditionMG_Rift3D_T(PetscInt nl,BCList bclist[],SurfBCList surf_bclist[],DM dav[],pTatinCtx user,void *ctx)
{
  ModelRift3D_TCtx *data = (ModelRift3D_TCtx*)ctx;
  PetscInt         n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  for (n=0; n<nl; n++) {
    ierr = ModelRift3D_T_DefineBCList(bclist[n],dav[n],user,data);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelAdaptMaterialPointResolution_Rift3D_T(pTatinCtx c,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]] - MARKER INJECTION IGNORED\n", PETSC_FUNCTION_NAME);

  ierr = MaterialPointPopulationControl_v1(c);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelAdaptMaterialPointResolution_Rift3D_T_semi_eulerian(pTatinCtx c,void *ctx)
{
  PhysCompStokes  stokes;
  DM              stokes_pack,dav,dap;
  PetscInt        Nxp[2];
  PetscReal       perturb;
  DataBucket      material_point_db,material_point_face_db;
  PetscInt        f, n_face_list=2, face_list[] = { 3, 4 }; // ymin, zmax //
  //  PetscInt        f, n_face_list=1, face_list[] = { 3 }; /* base */
  int             p,n_mp_points;
  MPAccess        mpX;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

#ifndef REMOVE_FACE_INJECTION
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]] FACE_INJECTION IS BEING IGNORED - POTENTIAL BUG DETECTED \n", PETSC_FUNCTION_NAME);
#endif

#ifdef REMOVE_FACE_INJECTION

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(c,&material_point_db,NULL);CHKERRQ(ierr);

  /* create face storage for markers */
  DataBucketDuplicateFields(material_point_db,&material_point_face_db);

  for (f=0; f<n_face_list; f++) {

    /* traverse */
    /* [0,1/east,west] ; [2,3/north,south] ; [4,5/front,back] */
    Nxp[0]  = 1;
    Nxp[1]  = 1;
    perturb = 0.1;

    /* reset size */
    DataBucketSetSizes(material_point_face_db,0,-1);

    /* assign coords */
    ierr = SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d(dav,Nxp,perturb, face_list[f], material_point_face_db);CHKERRQ(ierr);

    /* assign values */
    DataBucketGetSizes(material_point_face_db,&n_mp_points,0,0);
    ierr = MaterialPointGetAccess(material_point_face_db,&mpX);CHKERRQ(ierr);
    for (p=0; p<n_mp_points; p++) {
      ierr = MaterialPointSet_phase_index(mpX,p,MATERIAL_POINT_PHASE_UNASSIGNED);CHKERRQ(ierr);
    }
    ierr = MaterialPointRestoreAccess(material_point_face_db,&mpX);CHKERRQ(ierr);

    /* insert into volume bucket */
    DataBucketInsertValues(material_point_db,material_point_face_db);
  }

  /* Copy ALL values from nearest markers to newly inserted markers expect (xi,xip,pid) */
  ierr = MaterialPointRegionAssignment_v1(material_point_db,dav);CHKERRQ(ierr);

  /* reset any variables */
  DataBucketGetSizes(material_point_face_db,&n_mp_points,0,0);
  ierr = MaterialPointGetAccess(material_point_face_db,&mpX);CHKERRQ(ierr);
  for (p=0; p<n_mp_points; p++) {
    ierr = MaterialPointSet_plastic_strain(mpX,p,0.0);CHKERRQ(ierr);
    ierr = MaterialPointSet_yield_indicator(mpX,p,0);CHKERRQ(ierr);
  }
  ierr = MaterialPointRestoreAccess(material_point_face_db,&mpX);CHKERRQ(ierr);

  /* delete */
  DataBucketDestroy(&material_point_face_db);

#endif

  ierr = MaterialPointPopulationControl_v1(c);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMeshGeometry_Rift3D_T(pTatinCtx c,void *ctx)
{
  ModelRift3D_TCtx *data = (ModelRift3D_TCtx*)ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = DMDASetUniformCoordinates(c->stokes_ctx->dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);
  CHKERRQ(ierr);

  /* note - Don't access the energy mesh here, its not yet created */
  /* note - The initial velocity mesh geometry will be copied into the energy mesh */

  PetscReal gvec[] = { 0.0, -10.0, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_Rift3D_T(pTatinCtx c,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = ModelApplyInitialMaterialGeometry_Notchtest(c,ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyUpdateMeshGeometry_Rift3D_T(pTatinCtx c,Vec X,void *ctx)
{
  PetscReal       step;
  PhysCompStokes  stokes;
  DM              stokes_pack,dav,dap;
  Vec             velocity,pressure;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  /* fully lagrangian update */
  ierr = pTatinGetTimestep(c,&step);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);

  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  ierr = UpdateMeshGeometry_FullLagrangian(dav,velocity,step);CHKERRQ(ierr);

  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  //ierr = DMDAGetInfo(dav,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  //ierr = DMDARemeshSetUniformCoordinatesBetweenJLayers3d(dav,0,N);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyUpdateMeshGeometry_Rift3D_T_semi_eulerian(pTatinCtx c,Vec X,void *ctx)
{
  PetscReal       step;
  PhysCompStokes  stokes;
  DM              stokes_pack,dav,dap;
  Vec             velocity,pressure;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  /* fully lagrangian update */
  ierr = pTatinGetTimestep(c,&step);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);

  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  ierr = UpdateMeshGeometry_VerticalLagrangianSurfaceRemesh(dav,velocity,step);CHKERRQ(ierr);

  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutput_Rift3D_T(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  ModelRift3D_TCtx *data = (ModelRift3D_TCtx*)ctx;
  PetscBool        active_energy;
  DataBucket       materialpoint_db;
  PetscErrorCode   ierr;
  static PetscBool been_here = PETSC_FALSE;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  //ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  // just plot the velocity field (coords and vel stored in file as floats)
  ierr = pTatin3d_ModelOutputLite_Velocity_Stokes(c,X,prefix);CHKERRQ(ierr);
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);

  if (data->output_markers)
  {
    ierr = pTatinGetMaterialPoints(c,&materialpoint_db,NULL);CHKERRQ(ierr);
    //  Write out just the stokes variable?
    //  const int nf = 1;
    //  const MaterialPointField mp_prop_list[] = { MPField_Stokes };
    //
    //  Write out just std, stokes and plastic variables
    const int nf = 4;
    const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl, MPField_Energy };
    char mp_file_prefix[256];

    sprintf(mp_file_prefix,"%s_mpoints",prefix);
    ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  }

  {
    const int                   nf = 4;
    const MaterialPointVariable mp_prop_list[] = { MPV_region, MPV_viscosity, MPV_density, MPV_plastic_strain };

    ierr = pTatin3d_ModelOutput_MarkerCellFields(c,nf,mp_prop_list,prefix);CHKERRQ(ierr);
  }

  /* standard viewer */
  ierr = pTatinContextValid_Energy(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergy energy;
    Vec            temperature;

    ierr = pTatinGetContext_Energy(c,&energy);CHKERRQ(ierr);
    ierr = pTatinPhysCompGetData_Energy(c,&temperature,NULL);CHKERRQ(ierr);

    ierr = pTatin3d_ModelOutput_Temperature_Energy(c,temperature,prefix);CHKERRQ(ierr);
  }

  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
#if 1
  if (active_energy) {
    PhysCompEnergyFV energy;
    char             root[PETSC_MAX_PATH_LEN],pvoutputdir[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
    
    ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",c->outputpath);CHKERRQ(ierr);
    ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",root,c->step);CHKERRQ(ierr);
    //ierr = pTatinTestDirectory(pvoutputdir,'w',&found);CHKERRQ(ierr);
    //if (!found) { ierr = pTatinCreateDirectory(pvoutputdir);CHKERRQ(ierr); }

    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s-Tfv",pvoutputdir,prefix);
    ierr = FVDAView_CellData(energy->fv,energy->T,PETSC_TRUE,fname);CHKERRQ(ierr);
    // PVD
    {
      char pvdfilename[PETSC_MAX_PATH_LEN],vtkfilename[PETSC_MAX_PATH_LEN];
      char stepprefix[PETSC_MAX_PATH_LEN];
      
      PetscSNPrintf(pvdfilename,PETSC_MAX_PATH_LEN-1,"%s/timeseries_Tfv.pvd",root);
      if (prefix) { PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "%s-Tfv.pvtu",prefix);
      } else {      PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "Tfv.pvtu");           }
      
      PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%D",c->step);
      //ierr = ParaviewPVDOpenAppend(PETSC_FALSE,c->step,pvdfilename,c->time, vtkfilename, stepprefix);CHKERRQ(ierr);
      if (!been_here) { /* new file */
        ierr = ParaviewPVDOpen(pvdfilename);CHKERRQ(ierr);
        ierr = ParaviewPVDAppend(pvdfilename,c->time,vtkfilename,stepprefix);CHKERRQ(ierr);
      } else {
        ierr = ParaviewPVDAppend(pvdfilename,c->time,vtkfilename,stepprefix);CHKERRQ(ierr);
      }
    }
    
    /*
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s-Tfv_face",c->outputpath,prefix);
    ierr = FVDAView_FaceData_local(energy->fv,fname);CHKERRQ(ierr);
    */
  }
#endif

#if 0
  if (active_energy) {
    PhysCompEnergyFV energy;
    char             root[PETSC_MAX_PATH_LEN],pvoutputdir[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
    
    ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",c->outputpath);CHKERRQ(ierr);
    ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",root,c->step);CHKERRQ(ierr);
    
    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    if (prefix) { PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_T_fv",prefix);
    } else {      PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"T_fv",prefix);    }

    ierr = FVDAOutputParaView(energy->fv,energy->T,PETSC_TRUE,pvoutputdir,fname);CHKERRQ(ierr);
    
    // PVD
    {
      char pvdfilename[PETSC_MAX_PATH_LEN],vtkfilename[PETSC_MAX_PATH_LEN];
      char stepprefix[PETSC_MAX_PATH_LEN];
      
      PetscSNPrintf(pvdfilename,PETSC_MAX_PATH_LEN-1,"%s/timeseries_T_fv.pvd",root);
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
  }
#endif
  
#if 0
  if (active_energy) {
    PhysCompEnergyFV energy;
    char             root[PETSC_MAX_PATH_LEN],pvoutputdir[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
    
    ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",c->outputpath);CHKERRQ(ierr);
    ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",root,c->step);CHKERRQ(ierr);
    
    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_energy",prefix);
    ierr = FVDAView_JSON(energy->fv,pvoutputdir,fname);CHKERRQ(ierr); /* write meta data abour fv mesh, its DMDA and the coords */
    ierr = FVDAView_Heavy(energy->fv,pvoutputdir,fname);CHKERRQ(ierr);  /* write cell fields */
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s_energy_T",pvoutputdir,prefix);
    ierr = PetscVecWriteJSON(energy->T,0,fname);CHKERRQ(ierr); /* write cell temperature */
  }
#endif

  been_here = PETSC_TRUE;
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelDestroy_Rift3D_T(pTatinCtx c,void *ctx)
{
  ModelRift3D_TCtx *data = (ModelRift3D_TCtx*)ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  /* Free contents of structure */

  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialStokesVariableMarkers_Rift3D_T(pTatinCtx user,Vec X,void *ctx)
{
  DM                         stokes_pack,dau,dap;
  PhysCompStokes             stokes;
  Vec                        Uloc,Ploc;
  PetscScalar                *LA_Uloc,*LA_Ploc;
  ModelRift3D_TCtx           *data = (ModelRift3D_TCtx*)ctx;
  DataField                  PField;
  MaterialConst_MaterialType *truc;
  PetscInt                   regionidx;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  if (!data->runmises) {
    DataBucketGetDataFieldByName(user->material_constants,MaterialConst_MaterialType_classname,&PField);
    DataFieldGetAccess(PField);
    for (regionidx=0; regionidx<user->rheology_constants.nphases_active;regionidx++) {
      DataFieldAccessPoint(PField,regionidx,(void**)&truc);
      MaterialConst_MaterialTypeSetField_plastic_type(truc,PLASTIC_MISES);
    }
    DataFieldRestoreAccess(PField);
  }

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

  ierr = DMCompositeRestoreLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  if (!data->runmises) {
    for (regionidx=0; regionidx<user->rheology_constants.nphases_active;regionidx++) {
      DataBucketGetDataFieldByName(user->material_constants,MaterialConst_MaterialType_classname,&PField);
      DataFieldGetAccess(PField);
      for (regionidx=0; regionidx<user->rheology_constants.nphases_active;regionidx++) {
        DataFieldAccessPoint(PField,regionidx,(void**)&truc);
        MaterialConst_MaterialTypeSetField_plastic_type(truc,PLASTIC_DP);
      }
      DataFieldRestoreAccess(PField);
    }
  }
  PetscFunctionReturn(0);
}

PetscBool iterator_plume_thermal_field(PetscScalar coor[],PetscScalar *val,void *ctx)
{
  PetscBool         impose = PETSC_FALSE;
  const PetscScalar origin[] = {9.0, -1.0, 0.0};
  PetscScalar       r;
  
  r = (coor[0] - origin[0])*(coor[0] - origin[0]) + (coor[1] - origin[1])*(coor[1] - origin[1]);
  r = PetscSqrtReal(r);
  if (r < 0.25) {
    impose = PETSC_TRUE;
    *val = 1500.0;
  }
  return impose;
}

PetscBool iterator_plume_thermal_field_2(PetscScalar coor[],PetscScalar *val,void *ctx)
{
  PetscBool         impose = PETSC_FALSE, set;
  const PetscScalar origin[] = {7.0, -0.7, 0.0};
  PetscScalar       r;
  
  set = DMDAVecTraverse3d_ERFC3DFunctionXYZ(coor,val,ctx);
  r = (coor[0] - origin[0])*(coor[0] - origin[0]) + (coor[1] - origin[1])*(coor[1] - origin[1]);
  r = PetscSqrtReal(r);
  if (r < 0.25) {
    impose = PETSC_TRUE;
    *val  += 450.0;
  }
  return impose;
}

static PetscErrorCode ModelApplyInitialCondition_Rift3D_T(pTatinCtx c,Vec X,void *ctx)
{
  ModelRift3D_TCtx                             *data = (ModelRift3D_TCtx*)ctx;
  DM                                           stokes_pack,dau,dap;
  Vec                                          velocity,pressure;
  PetscReal                                    vxl,vxr,vzb,vzf,vy;
  DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
  DMDAVecTraverse3d_InterpCtx                  IntpCtx;
  PetscReal                                    MeshMin[3],MeshMax[3],domain_height;
  PetscBool                                    active_energy, scissor;
  PetscErrorCode                               ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  stokes_pack = c->stokes_ctx->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  scissor = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_scissors",&scissor,NULL);CHKERRQ(ierr);

  if(scissor == PETSC_FALSE){

    vxl = -data->vx;
    vxr =  data->vx;

    vzf = -data->vz;
    vzb =  0.0;//data->vz;

    ierr = VecZeroEntries(velocity);CHKERRQ(ierr);

    ierr = DMDAVecTraverse3d_InterpCtxSetUp_X(&IntpCtx,(vxr-vxl)/(data->Lx-data->Ox),vxl,0.0);CHKERRQ(ierr);
    ierr = DMDAVecTraverse3d(dau,velocity,0,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);
    ierr = DMDAVecTraverse3d_InterpCtxSetUp_Z(&IntpCtx,(vzf-vzb)/(data->Lz-data->Oz),vzb,0.0);CHKERRQ(ierr);
    ierr = DMDAVecTraverse3d(dau,velocity,2,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);

  }else{
    PetscReal    coeffs[5];

    // set center of rotation x0
    coeffs[0]= 6.0;
    // set center of rotation z0
    coeffs[1]= 4.0;
    // set direction of interpolation
    coeffs[2]= 0;
    // set lenght in direction of interpolation
    coeffs[3]= data->Lx-data->Ox;
    // set x a;gular velocity based on vx at the right back corner
    coeffs[4]=data->vx/(data->Oz-4.0);

    ierr = DMDAVecTraverse3d(dau,velocity,0,DMDAVecTraverse3d_ROTXZ_X,(void*)coeffs);CHKERRQ(ierr);

    ierr = DMDAVecTraverse3d(dau,velocity,2,DMDAVecTraverse3d_ROTXZ_Z,(void*)coeffs);CHKERRQ(ierr);
  }
  vy= data->vy;
  ierr = DMDAVecTraverse3d_InterpCtxSetUp_Y(&IntpCtx,-vy/(data->Ly-data->Oy),0.0,0.0);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d(dau,velocity,1,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);

  ierr = VecZeroEntries(pressure);CHKERRQ(ierr);

  ierr = DMGetBoundingBox(dau,MeshMin,MeshMax);CHKERRQ(ierr);
  domain_height = MeshMax[1] - MeshMin[1];

  HPctx.surface_pressure = 0.0;
  HPctx.ref_height = domain_height;
  HPctx.ref_N      = c->stokes_ctx->my-1;
  HPctx.grav       = 10.0;
  HPctx.rho        = data->rho0;

  ierr = DMDAVecTraverseIJK(dap,pressure,0,DMDAVecTraverseIJK_HydroStaticPressure_v2,     (void*)&HPctx);CHKERRQ(ierr); /* P = P0 + a.x + b.y + c.z, modify P0 (idx=0) */
  ierr = DMDAVecTraverseIJK(dap,pressure,2,DMDAVecTraverseIJK_HydroStaticPressure_dpdy_v2,(void*)&HPctx);CHKERRQ(ierr); /* P = P0 + a.x + b.y + c.z, modify b  (idx=2) */
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /*ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,"testHP");CHKERRQ(ierr);*/

  /* initial condition for temperature */
  ierr = pTatinContextValid_Energy(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergy energy;
    Vec            temperature;
    DM             daT;
    PetscBool      (*iterator_initial_thermal_field)(PetscScalar*,PetscScalar*,void*);
    PetscReal      coeffs[8];

    ierr = pTatinGetContext_Energy(c,&energy);CHKERRQ(ierr);
    ierr = pTatinPhysCompGetData_Energy(c,&temperature,NULL);CHKERRQ(ierr);
    daT  = energy->daT;

    ierr = ModelRift3D_T_GetDescription_InitialThermalField(data,coeffs,&iterator_initial_thermal_field);CHKERRQ(ierr);
    ierr = DMDAVecTraverse3d(daT,temperature,0,iterator_initial_thermal_field,(void*)coeffs);CHKERRQ(ierr);
  }

  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergyFV energy;
    PetscBool        (*iterator_initial_thermal_field)(PetscScalar*,PetscScalar*,void*);
    PetscReal        coeffs[8];
    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    ierr = ModelRift3D_T_GetDescription_InitialThermalField(data,coeffs,&iterator_initial_thermal_field);CHKERRQ(ierr);

    ierr = FVDAVecTraverse(energy->fv,energy->T,0.0,0,iterator_initial_thermal_field,(void*)coeffs);CHKERRQ(ierr);
    
    // <DAM FV TESTING: Insert a cylinderical hot region centered at x = 9, y = -1.0>
    //ierr = FVDAVecTraverse(energy->fv,energy->T,0.0,0,iterator_plume_thermal_field,NULL);CHKERRQ(ierr);
    //ierr = FVDAVecTraverse(energy->fv,energy->T,0.0,0,iterator_plume_thermal_field_2,(void*)coeffs);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_Rift3D_TFV(void)
{
  ModelRift3D_TCtx *data;
  pTatinModel      m;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelRift3D_TCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelRift3D_TCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"rift3D_TFV");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_Rift3D_T);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialCondition_Rift3D_T);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,   (void (*)(void))ModelApplyInitialStokesVariableMarkers_Rift3D_T);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_Rift3D_T);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_Rift3D_T);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_Rift3D_T);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_Rift3D_T);CHKERRQ(ierr);

  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_Rift3D_T);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void))ModelAdaptMaterialPointResolution_Rift3D_T);CHKERRQ(ierr);

  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_Rift3D_T);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_Rift3D_T);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_Notchtest(pTatinCtx c,void *ctx)
{
  ModelRift3D_TCtx *data = (ModelRift3D_TCtx*)ctx;
  int              p,n_mp_points;
  PetscScalar      y_lab,y_moho,y_midcrust,notch_l,notch_w2,xc,notchspace;
  DataBucket       db;
  DataField        PField_std,PField_pls;
  int              phase;
  PetscBool        norandomiseplastic,double_notch;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));

  /* m */
  y_lab      = -120.0e3;
  y_moho     = -40.0e3;
  y_midcrust = -20.0e3;
  notch_w2   = 50.e3;
  notch_l    = 100.e3;
  xc         = (data->Lx + data->Ox)/2.0* data->length_bar;
  notchspace = 200.e3;
  //xc         = 0.0;
  DataBucketGetSizes(db,&n_mp_points,0,0);

  ptatin_RandomNumberSetSeedRank(PETSC_COMM_WORLD);

  double_notch = PETSC_FALSE;
  norandomiseplastic = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_norandom",&norandomiseplastic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_rift3D_T_DoubleNotch",&double_notch,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_rift3D_T_notchspace",&notchspace,NULL);CHKERRQ(ierr);

  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    double        *position,ycoord,xcoord,zcoord;
    float         pls;
    char          yield;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access using the getter function provided for you (recommeneded for beginner user) */
    MPntStdGetField_global_coord(material_point,&position);

    /* convert to scaled units */
    xcoord = position[0] * data->length_bar;
    ycoord = position[1] * data->length_bar;
    zcoord = position[2] * data->length_bar;

    if (ycoord < y_lab) {
      phase = 3;
    } else if (ycoord < y_moho) {
      phase = 2;
    } else if (ycoord < y_midcrust) {
      phase = 1;
    } else {
      phase = 0;
    }

    if (!double_notch){
      if (norandomiseplastic) {
        pls   = 0.0;
        if ( (fabs(xcoord - xc) < notch_w2) && (zcoord < notch_l) && (ycoord > y_lab) ) {
          pls = 0.05;
        }
      } else {
        pls = ptatin_RandomNumberGetDouble(0.0,0.03);
        if ( (fabs(xcoord - xc) < notch_w2) && (zcoord < notch_l) && (ycoord > y_lab) ) {
          pls = ptatin_RandomNumberGetDouble(0.0,0.3);
        }
      }
    }else{
      PetscScalar xc1,xc2,Lz;
      xc1 = (data->Lx + data->Ox)/2.0* data->length_bar - notchspace/2.0;
      xc2 = (data->Lx + data->Ox)/2.0* data->length_bar + notchspace/2.0;
      Lz  = data->Lz*data->length_bar;

      if (norandomiseplastic) {
        pls   = 0.0;
        if ( (fabs(xcoord - xc1) < notch_w2) && (zcoord < notch_l) && (ycoord > y_lab) )  {
          pls = 0.05;
        }
        if ( (fabs(xcoord - xc2) < notch_w2) && (zcoord > Lz-notch_l) && (ycoord > y_lab) )  {
          pls = 0.05;
        }
      } else {

        pls = ptatin_RandomNumberGetDouble(0.0,0.03);
        if ( (fabs(xcoord - xc1) < notch_w2) && (zcoord < notch_l) && (ycoord > y_lab) )  {
          pls = ptatin_RandomNumberGetDouble(0.0,0.3);
        }
        if ( (fabs(xcoord - xc2) < notch_w2) && (zcoord > Lz-notch_l) && (ycoord > y_lab) ) {

          pls = ptatin_RandomNumberGetDouble(0.0,0.3);
        }
      }
    }

    yield = 0;
    /* user the setters provided for you */
    MPntStdSetField_phase_index(material_point,phase);
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);
  }

  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);

  PetscFunctionReturn(0);
}
