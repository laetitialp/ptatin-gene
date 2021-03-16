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
 **    filename:   model_kinematic_TFV_ops.c
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
#include "output_material_points_p0.h"
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
#include "dmda_remesh.h"
#include "model_utils.h"
#include "kinematic_TFV_ctx.h"
#include "cartgrid.h"

#define REMOVE_FACE_INJECTION

const char MODEL_NAME_KINE[] = "model_kinematic_TFV_";

PetscErrorCode GeometryObjectSetFromOptions_Box(GeometryObject go);
PetscErrorCode GeometryObjectSetFromOptions_InfLayer(GeometryObject go);
PetscErrorCode GeometryObjectSetFromOptions_EllipticCylinder(GeometryObject go);

static PetscErrorCode ModelApplyUpdateMeshGeometry_kinematic_TFV_semi_eulerian(pTatinCtx c,Vec X,void *ctx);
static PetscErrorCode ModelApplyMaterialBoundaryCondition_kinematic_TFV_semi_eulerian(pTatinCtx c,void *ctx);

static PetscErrorCode ModelApplyInitialMaterialGeometry_FromMap(pTatinCtx c,void *ctx);
static PetscErrorCode ModelApplyInitialMaterialGeometry_FromIndex(pTatinCtx c,void *ctx);
static PetscErrorCode ModelApplyInitialMaterialGeometry_Notchtest(pTatinCtx c,void *ctx);
static PetscErrorCode ComputeSphericalVelocities(PetscScalar position[],PetscReal phi_0,PetscReal lat_pole,PetscReal lon_pole,PetscReal w, PetscScalar *v_e, PetscScalar *v_n);

static PetscErrorCode ModelInitialize_kinematic_TFV(pTatinCtx c,void *ctx)
{
  Modelkinematic_TFVCtx   *data = (Modelkinematic_TFVCtx*)ctx;
  RheologyConstants       *rheology;
  DataBucket              materialconstants;
  PetscBool               nondim;
  PetscScalar             vx,vy,vz,Sx,Sy,Sz;
  PetscInt                regionidx;
  PetscReal               cm_per_yer2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 ) ;
  PetscReal               rho_ref,Cp;
  PetscInt                region_idx;
  PetscReal               phi_rad,phi_inf_rad;
  DataField               PField;
  PetscReal               *preexpA,*Ascale,*entalpy,*Vmol,*nexp,*Tref;
  PetscReal               *phi,*phi_inf,*Co,*Co_inf,*Tens_cutoff,*Hst_cutoff,*eps_min,*eps_max;
  PetscReal               *beta,*alpha,*rho;
  char                    *option_name;
  EnergyMaterialConstants *matconstants_e;
  PetscErrorCode          ierr;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  PetscPrintf(PETSC_COMM_WORLD,"Rift model expects the following dimensions for input\n");
  PetscPrintf(PETSC_COMM_WORLD," Box geometry: [m] \n");
  PetscPrintf(PETSC_COMM_WORLD," Viscosity:    [Pa.s] \n");
  PetscPrintf(PETSC_COMM_WORLD," Velocity:     [m/sec] \n");
  PetscPrintf(PETSC_COMM_WORLD," Density:      [kg/m^3] \n");

  PetscPrintf(PETSC_COMM_WORLD,"if you wish to use non dimensional input you must add -model_kinematic_TFV_dimensional \n");
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);

  rheology->rheology_type = RHEOLOGY_VP_STD;
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);

  data->from_map = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_KINE,"-PhaseFromMap",&data->from_map,NULL);CHKERRQ(ierr);
  if (data->from_map) {
    data->n_phase_map = 3;
    data->nlayers = 4;
    ierr = PetscOptionsGetInt(NULL,MODEL_NAME_KINE,"-n_phase_map",&data->n_phase_map,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,MODEL_NAME_KINE,"-nlayers",&data->nlayers,NULL);CHKERRQ(ierr);
    data->n_phases = 8;
    //data->n_phases = data->n_phase_map*data->nlayers;
  } else {
     data->n_phase_map = 0;
     data->nlayers = 0;
     data->n_phases = 4;
  }    
  rheology->nphases_active = data->n_phases;
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
  data->Lx =  20.0e5;
  data->Ly =  0.0e5;
  data->Lz =  10.0e5;
  //data->Ox =  -6.0e5;
  data->Ox =  0.0e5;
  data->Oy =  -2.5e5;
  data->Oz =  0.0e5;
  /* velocity cm/y */
  vx = 1.0*cm_per_yer2m_per_sec;
  vz = 0.25*cm_per_yer2m_per_sec;
  /* rho0 for initial pressure*/
  data->rho0 = 3140.0;
  /*Temperature */
  data->Tbottom = 1364.0;
  data->Ttop    = 0.0;
  data->Tlitho  = 1300.0;
  data->h_prod  = 1.2e-6;
  data->y_prod  = -40.0e3;
  data->ylab    = -122.0e3;
  data->qm      = 20.0e-3;
  data->k       = 3.3;
  /* Material constant */
  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);

  /* Allocate memory for arrays */
  ierr = PetscMalloc1(rheology->nphases_active,&preexpA);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&Ascale);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&entalpy);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&Vmol);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&nexp);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&Tref);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&phi);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&phi_inf);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&Co);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&Co_inf);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&Tens_cutoff);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&Hst_cutoff);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&eps_min);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&eps_max);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&rho);CHKERRQ(ierr);

  /* Set default values for parameters */
  for (region_idx=0;region_idx<rheology->nphases_active;region_idx++) {
    preexpA[region_idx] = 6.3e-6;
    Ascale[region_idx] = 1.0e6;
    entalpy[region_idx] = 156.0e3;
    Vmol[region_idx] = 0.0;
    nexp[region_idx] = 2.4;
    Tref[region_idx] = 273.0;
    phi[region_idx] = 30.0;
    phi_inf[region_idx] = 5.0;
    Co[region_idx] = 2.0e7;
    Co_inf[region_idx] = 2.0e7;
    Tens_cutoff[region_idx] = 1.0e7;
    Hst_cutoff[region_idx] = 2.0e8;
    eps_min[region_idx] = 0.0;
    eps_max[region_idx] = 0.5;
    beta[region_idx] = 0.0;
    alpha[region_idx] = 2.0e-5;
    rho[region_idx] = 2700.0;
  }  


  rho_ref     = 1.0;
  Cp          = 1.0;

  // With this method EVERY phases will have the same constitutive laws (but can have different parameters values)
  for (region_idx=0;region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_ARRHENIUS_2,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);

    /* VISCOUS PARAMETERS */
    if (asprintf (&option_name, "-preexpA_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&preexpA[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    
    if (asprintf (&option_name, "-Ascale_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&Ascale[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
 
    if (asprintf (&option_name, "-entalpy_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&entalpy[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
 
    if (asprintf (&option_name, "-Vmol_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&Vmol[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-nexp_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&nexp[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
 
    if (asprintf (&option_name, "-Tref_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&Tref[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    MaterialConstantsSetValues_ViscosityArrh(materialconstants,region_idx,preexpA[region_idx],Ascale[region_idx],entalpy[region_idx],Vmol[region_idx],nexp[region_idx],Tref[region_idx]);  

    /* PLASTIC PARAMETERS */

    if (asprintf (&option_name, "-phi_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&phi[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-phi_inf_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&phi_inf[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-Co_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&Co[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-Co_inf_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&Co_inf[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-Tens_cutoff_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&Tens_cutoff[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-Hst_cutoff_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&Hst_cutoff[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-eps_min_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&eps_min[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-eps_max_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&eps_max[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    phi_rad     = M_PI * phi[region_idx]/180.0;
    phi_inf_rad = M_PI * phi_inf[region_idx]/180.0;

    MaterialConstantsSetValues_PlasticDP(materialconstants,region_idx,phi_rad,phi_inf_rad,Co[region_idx],Co_inf[region_idx],Tens_cutoff[region_idx],Hst_cutoff[region_idx]);
    MaterialConstantsSetValues_PlasticMises(materialconstants,region_idx,Tens_cutoff[region_idx],Hst_cutoff[region_idx]);
    MaterialConstantsSetValues_SoftLin(materialconstants,region_idx,eps_min[region_idx],eps_max[region_idx]);

    /* ENERGY PARAMETERS */
    if (asprintf (&option_name, "-alpha_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&alpha[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-beta_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&beta[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    if (asprintf (&option_name, "-rho_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE, option_name,&rho[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

 ierr = MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha[region_idx],beta[region_idx],rho_ref,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,NULL);CHKERRQ(ierr);
    EnergyMaterialConstantsSetFieldAll_SourceMethod(&matconstants_e[region_idx],ENERGYSOURCE_NONE);
    EnergyMaterialConstantsSetFieldByIndex_SourceMethod(&matconstants_e[region_idx],0,ENERGYSOURCE_CONSTANT);

    MaterialConstantsSetValues_DensityBoussinesq(materialconstants,region_idx,rho[region_idx],alpha[region_idx],beta[region_idx]);
    MaterialConstantsSetValues_DensityConst(materialconstants,region_idx,rho[region_idx]);

  }

  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {

    EnergyConductivityConst *data_k;
    EnergySourceConst *data_Q;
    DataField               PField_k,PField_Q;

    DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
    DataFieldGetEntries(PField_k,(void**)&data_k);
    EnergyConductivityConstSetField_k0(&data_k[regionidx],1.0e-6);

    DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
    DataFieldGetEntries(PField_Q,(void**)&data_Q);
    EnergySourceConstSetField_HeatSource(&data_Q[regionidx],0.0);

  }
  /* Read the options */
  /*cutoff */
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_kinematic_TFV_apply_viscosity_cutoff_global",&rheology->apply_viscosity_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_eta_lower_cutoff_global",&rheology->eta_lower_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_eta_upper_cutoff_global",&rheology->eta_upper_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_kinematic_TFV_runwithmises",&data->runmises,NULL);CHKERRQ(ierr);
  /*scaling */
  nondim = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_kinematic_TFV_nondimensional",&nondim,NULL);CHKERRQ(ierr);
  if (nondim){
    data->dimensional = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_vis_bar",&data->viscosity_bar,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_vel_bar",&data->velocity_bar,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_length_bar",&data->length_bar,NULL);CHKERRQ(ierr);
  }

  /* box geometry, m */
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_Lx",&data->Lx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_Ly",&data->Ly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_Lz",&data->Lz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_Ox",&data->Ox,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_Oy",&data->Oy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_Oz",&data->Oz,NULL);CHKERRQ(ierr);

  /* velocity cm/y */
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_vx",&vx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_vz",&vz,NULL);CHKERRQ(ierr);

  /* rho0 for initial pressure */
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_rho0",&data->rho0,NULL);CHKERRQ(ierr);

  /* temperature initial condition */
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_Tbot",&data->Tbottom,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_Ttop",&data->Ttop,NULL);CHKERRQ(ierr);

  /* Material constant */
  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
    PetscPrintf(PETSC_COMM_WORLD,"reading options");
    ierr= MaterialConstantsSetFromOptions(materialconstants,"model_kinematic_TFV",regionidx,PETSC_FALSE);CHKERRQ(ierr);
  }

  /*Compute velocity at bottom*/
  Sx = (data->Ly - data->Oy)*(data->Lz - data->Oz);
  Sz = (data->Ly - data->Oy)*(data->Lx - data->Ox);
  Sy = (data->Lx - data->Ox)*(data->Lz - data->Oz);
  vy = (2*vx*Sx-vz*Sz)/Sy;

  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"  input: -model_kinematic_TFV_Ox %+1.4e [SI] -model_kinematic_TFV_Lx : %+1.4e [SI]\n", data->Ox ,data->Lx );
  PetscPrintf(PETSC_COMM_WORLD,"  input: -model_kinematic_TFV_Oy %+1.4e [SI] -model_kinematic_TFV_Ly : %+1.4e [SI]\n", data->Oy ,data->Ly );
  PetscPrintf(PETSC_COMM_WORLD,"  input: -model_kinematic_TFV_Oz %+1.4e [SI] -model_kinematic_TFV_Lz : %+1.4e [SI]\n", data->Oz ,data->Lz );
  PetscPrintf(PETSC_COMM_WORLD,"  -model_kinematic_TFV_vx [m/s]:  %+1.4e  -model_kinematic_TFV_vz [m/s]:  %+1.4e : computed vy [m/s]:  %+1.4e \n", vx,vz,vy);
  PetscPrintf(PETSC_COMM_WORLD,"-model_kinematic_TFV_rho0 [kg/m^3] :%+1.4e \n", data->rho0 );
  PetscPrintf(PETSC_COMM_WORLD,"-model_kinematic_TFV_Tbot:%+1.4e \t -model_kinematic_TFV_Ttop:%+1.4e \n",data->Tbottom,data->Ttop);

  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
    MaterialConstantsPrintAll(materialconstants,regionidx);
    MaterialConstantsEnergyPrintAll(materialconstants,regionidx);
  }

  if (data->dimensional) {
    /*Compute additional scaling parameters*/
    data->time_bar      = data->length_bar / data->velocity_bar;
    data->pressure_bar  = data->viscosity_bar/data->time_bar;
    data->density_bar   = data->pressure_bar / data->length_bar;

    PetscPrintf(PETSC_COMM_WORLD,"[kinematic_TFV]:  during the solve scaling will be done using \n");
    PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n", data->length_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n", data->velocity_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n", data->time_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n", data->viscosity_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->density_bar );
    PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n", data->pressure_bar );
    //scale viscosity cutoff
    rheology->eta_lower_cutoff_global = rheology->eta_lower_cutoff_global / data->viscosity_bar;
    rheology->eta_upper_cutoff_global = rheology->eta_upper_cutoff_global / data->viscosity_bar;
    //scale length
    data->Lx = data->Lx / data->length_bar;
    data->Ly = data->Ly / data->length_bar;
    data->Lz = data->Lz / data->length_bar;
    data->Ox = data->Ox / data->length_bar;
    data->Oy = data->Oy / data->length_bar;
    data->Oz = data->Oz / data->length_bar;

    //scale velocity
    data->vx = vx/data->velocity_bar;
    //data->vy = vy/data->velocity_bar;
    data->vz = vz/data->velocity_bar;
    //scale rho0
    data->rho0 = data->rho0/data->density_bar;
    //scale thermal params
    data->h_prod = data->h_prod/(data->pressure_bar / data->time_bar);
    data->k      = data->k/(data->pressure_bar*data->length_bar*data->length_bar/data->time_bar);
    data->qm     = data->qm/(data->pressure_bar*data->velocity_bar);
    data->ylab   = data->ylab/data->length_bar;
    data->y_prod = data->y_prod/data->length_bar;

    // scale material properties
    for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
      MaterialConstantsScaleAll(materialconstants,regionidx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);
      MaterialConstantsEnergyScaleAll(materialconstants,regionidx,data->length_bar,data->time_bar,
          data->pressure_bar);
    }

    /*Reports scaled values*/

    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_kinematic_TFV_Ox   :  %+1.4e    -model_kinematic_TFV_Lx   :  %+1.4e  \n", data->Ox ,data->Lx );
    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_kinematic_TFV_Oy   :  %+1.4e    -model_kinematic_TFV_Ly   :  %+1.4e \n", data->Oy, data->Ly );
    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_kinematic_TFV_Oz   :  %+1.4e    -model_kinematic_TFV_Lz   :  %+1.4e\n", data->Oz , data->Lz );

    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_kinematic_TFV_Vx:%+1.4e         -model_kinematic_TFV_vz:  %+1.4e \n", data->vx , data->vz);
    PetscPrintf(PETSC_COMM_WORLD,"scaled value   -model_kinematic_TFV_rho0:%+1.4e \n", data->rho0 );
    PetscPrintf(PETSC_COMM_WORLD,"scaled value for material parameters\n");
    for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
      MaterialConstantsPrintAll(materialconstants,regionidx);
      MaterialConstantsEnergyPrintAll(materialconstants,regionidx);
    }
  }

  data->use_semi_eulerian_mesh = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_kinematic_TFV_use_semi_eulerian",&data->use_semi_eulerian_mesh,NULL);CHKERRQ(ierr);
  if (data->use_semi_eulerian_mesh) {
    pTatinModel model;

    PetscPrintf(PETSC_COMM_WORLD,"kinematic_TFV: activating semi Eulerian mesh advection\n");
    ierr = pTatinGetModel(c,&model);CHKERRQ(ierr);
    ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_kinematic_TFV_semi_eulerian);CHKERRQ(ierr);
    ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_kinematic_TFV_semi_eulerian);CHKERRQ(ierr);
  }

  data->output_markers = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_kinematic_TFV_output_markers",&data->output_markers,NULL);CHKERRQ(ierr);

  PetscFree(preexpA);
  PetscFree(Ascale);
  PetscFree(entalpy);
  PetscFree(Vmol);
  PetscFree(nexp);
  PetscFree(Tref);
  PetscFree(phi);
  PetscFree(phi_inf);
  PetscFree(Co);
  PetscFree(Co_inf);
  PetscFree(Tens_cutoff);
  PetscFree(Hst_cutoff);
  PetscFree(eps_min);
  PetscFree(eps_max);
  PetscFree(beta);
  PetscFree(alpha);
  PetscFree(rho);

  PetscFunctionReturn(0);
}

/*
   Returns the parameters and function need to define initial thermal field.
   The function returned can be used to define either the initial condition for T or the boundary condition for T.
 */

PetscBool DMDATraverse3d_ContinentalGeothermSteady(PetscScalar position[],PetscScalar *value, void *ctx)
{
  PetscBool   impose_dirichlet=PETSC_TRUE;
  PetscScalar y;
  PetscReal   *coeffs;
  PetscReal   Tbot,Ttop,Tlitho,h_prod,y_prod,ylab,k,ymin,qm;

  y = position[1];

  coeffs = (PetscReal*)ctx;
  Tbot     = coeffs[0];
  Ttop     = coeffs[1];
  Tlitho   = coeffs[2];
  h_prod   = coeffs[3];
  y_prod   = coeffs[4];
  ylab     = coeffs[5];
  k        = coeffs[6];
  ymin     = coeffs[7];
  qm       = coeffs[8];


  *value = Ttop + qm*(-y)/k + h_prod*pow(y_prod,2)/k * (1-exp(-y/y_prod));

  if (*value >= Tlitho){
    *value = -((Tbot-Tlitho)/(ymin-ylab)) * (ylab-y) + Tlitho;
  }

  return impose_dirichlet;
}

static PetscErrorCode Modelkinematic_TFV_GetDescription_InitialThermalField(Modelkinematic_TFVCtx *data,PetscReal coeffs[],PetscBool (**func)(PetscScalar*,PetscScalar*,void*) )
{
  PetscFunctionBegin;
  /* assign params */
  coeffs[0] = data->Tbottom;
  coeffs[1] = data->Ttop;
  coeffs[2] = data->Tlitho;
  coeffs[3] = data->h_prod;
  coeffs[4] = data->y_prod;
  coeffs[5] = data->ylab;
  coeffs[6] = data->k;
  coeffs[7] = data->Oy;
  coeffs[8] = data->qm;

  /* assign function to use */
  *func = DMDATraverse3d_ContinentalGeothermSteady;

  PetscFunctionReturn(0);
}

//PetscBool DMDAVecTraverse3d_ERFC3DFunctionXYZ(PetscScalar pos[],PetscScalar *val,void *ctx);
PetscErrorCode FVDABCMethod_ContinentalGeothermSteady(FVDA fv,
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
    PetscBool set = DMDATraverse3d_ContinentalGeothermSteady((PetscScalar*)&coor[3*f],&value,ctx);
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    bcvalue[f] = value;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode Modelkinematic_TFV_GetDescription_InitialThermalField_FV(Modelkinematic_TFVCtx *data,PetscReal coeffs[],
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
  coeffs[0] = data->Tbottom;
  coeffs[1] = data->Ttop;
  coeffs[2] = data->Tlitho;
  coeffs[3] = data->h_prod;
  coeffs[4] = data->y_prod;
  coeffs[5] = data->ylab;
  coeffs[6] = data->k;
  coeffs[7] = data->Oy;
  coeffs[8] = data->qm;
  
  /* assign function to use */
  *func = FVDABCMethod_ContinentalGeothermSteady;
  
  PetscFunctionReturn(0);
}
/*

   1/ Define boundary conditions in one place for this model.

   2/ Calling pattern should always be
   PetscErrorCode Modelkinematic_TFV_DefineBCList(BCList bclist,DM dav,pTatinCtx user,Modelkinematic_TFVCtx data)
   where Modelkinematic_TFVCtx data is a different type for each model.

   3/ Re-use this function in
   ModelApplyBoundaryCondition_kinematic_TFV();
   ModelApplyBoundaryConditionMG_kinematic_TFV();

*/

PetscErrorCode ModelComputeBottomFlow_kinematic_TFV(pTatinCtx c,Vec X, Modelkinematic_TFVCtx *data)
{
  PhysCompStokes stokes;
  DM             dms;
  Vec            velocity,pressure;
  PetscReal      int_u_dot_n[HEX_EDGES];
  PetscErrorCode ierr;

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dms);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(dms,X,&velocity,&pressure);CHKERRQ(ierr);  
  
  ierr = StokesComputeVdotN(stokes,velocity,int_u_dot_n);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"imin: %+1.4e\n",int_u_dot_n[ WEST_FACE  -1]);
  PetscPrintf(PETSC_COMM_WORLD,"imax: %+1.4e\n",int_u_dot_n[ EAST_FACE  -1]);
  PetscPrintf(PETSC_COMM_WORLD,"jmin: %+1.4e\n",int_u_dot_n[ SOUTH_FACE -1]);
  PetscPrintf(PETSC_COMM_WORLD,"jmax: [free surface] %+1.4e\n",int_u_dot_n[ NORTH_FACE -1]);
  PetscPrintf(PETSC_COMM_WORLD,"kmin: %+1.4e\n",int_u_dot_n[ BACK_FACE  -1]);
  PetscPrintf(PETSC_COMM_WORLD,"kmax: %+1.4e\n",int_u_dot_n[ FRONT_FACE -1]);

  ierr = DMCompositeRestoreAccess(dms,X,&velocity,&pressure);CHKERRQ(ierr);

  if (c->step == 0) {
    data->vy = 0.0;
  } else {
    data->vy = (int_u_dot_n[WEST_FACE-1]+int_u_dot_n[EAST_FACE-1]+int_u_dot_n[BACK_FACE-1]+int_u_dot_n[FRONT_FACE-1])/(data->Lx*data->Lz);
    PetscPrintf(PETSC_COMM_WORLD,"v.n = %+1.4e\n",data->vy);    
  }
  PetscFunctionReturn(0);
}

PetscBool BCListEvaluator_RotationPole(PetscScalar position[],PetscScalar *value, void *ctx)
{
  BC_RotationPole data_ctx = (BC_RotationPole)ctx;
  PetscScalar     rx,rz,vr;
  PetscBool       impose_dirichlet = PETSC_TRUE;

  PetscFunctionBegin;

  if(data_ctx->component == 0) {
    /* x component of the tangent vector is -z component of the radius vector scaled by the angular velocity */
    rz = position[2] - data_ctx->zp;
    vr = -rz * data_ctx->v0;
  } else if (data_ctx->component == 2) {
    /* z component of the tangent vector is x component of the radius vector scaled by the angular velocity */
    rx = position[0] - data_ctx->xp;
    vr = rx * data_ctx->v0;
  }

  *value = vr;

  return impose_dirichlet;
}

PetscBool BCListEvaluator_RotationPoleFreeslip(PetscScalar position[],PetscScalar *value, void *ctx)
{
  BC_RotationPoleFreeslip data_ctx = (BC_RotationPoleFreeslip)ctx;
  PetscScalar     rx,rz,vr;
  PetscBool       impose_dirichlet = PETSC_TRUE;

  PetscFunctionBegin;

  if(data_ctx->component == 0) {
    /* x component of the tangent vector is -z component of the radius vector scaled by the angular velocity */
    rz = position[2] - data_ctx->zp;
    vr = -rz * data_ctx->v0;
  } else if (data_ctx->component == 2) {
    /* z component of the tangent vector is x component of the radius vector scaled by the angular velocity */
    rx = position[0] - data_ctx->xp;
    vr = rx * data_ctx->v0;
  }

  switch(data_ctx->normal) { /* Determine (user choice) on which face we are */

    /* Face of normal x */
    case 0:
    { 
      if (position[2] < data_ctx->x0) {
        *value = vr;
      } else if (position[2] > data_ctx->x1) {
        *value = -vr;
      } else { /* if we are between x0 and x1 apply freeslip */
        if (data_ctx->component == 0) { 
          *value = 0.0; /* Set to 0.0 the x velocity component */
        } else {
          impose_dirichlet = PETSC_FALSE; /* Do not apply value to the z velocity component */ 
        }
      }
    }
      break;

    /* Face of normal z */
    case 1:
    { 
      if (position[0] < data_ctx->x0) {
        *value = vr;
      } else if (position [0] > data_ctx->x1) {
        *value = -vr;
      } else { 
        if (data_ctx->component == 2) {
          *value = 0.0;
        } else {
          impose_dirichlet = PETSC_FALSE;
        }
      }
    }
      break;
  }
  return impose_dirichlet;
}

PetscBool BCListEvaluator_RotationPoleFreeslipLinear(PetscScalar position[],PetscScalar *value, void *ctx)
{
  BC_RotationPoleFreeslipLinear data_ctx = (BC_RotationPoleFreeslipLinear)ctx;
  PetscScalar     rx,rz,vr;
  PetscBool       impose_dirichlet = PETSC_TRUE;

  PetscFunctionBegin;

  if(data_ctx->component == 0) {
    /* x component of the tangent vector is -z component of the radius vector scaled by the angular velocity */
    rz = position[2] - data_ctx->zp;
    vr = -rz * data_ctx->v0;
  } else if (data_ctx->component == 2) {
    /* z component of the tangent vector is x component of the radius vector scaled by the angular velocity */
    rx = position[0] - data_ctx->xp;
    vr = rx * data_ctx->v0;
  }

  switch(data_ctx->normal) { /* Determine (user choice) on which face we are */

    /* Face of normal x */
    case 0:
    { 
      if (position[2] <= data_ctx->x0) {
        *value = vr;
      } else if (position[2] > data_ctx->x0 && position[2] < data_ctx->x1 && data_ctx->component == 0) {
        // Interpolation between vr and 0
        *value = vr + (position[2]-data_ctx->x0)*(0.0-vr)/(data_ctx->x1-data_ctx->x0);
      } else if (position[2] > data_ctx->x2 && position[2] < data_ctx->x3 && data_ctx->component == 0) {
        // Interpolation between 0 and -vr
        *value = 0.0 + (position[2]-data_ctx->x2)*(-vr-0.0)/(data_ctx->x3-data_ctx->x2);
      } else if (position[2] >= data_ctx->x3) {
        *value = -vr;
      } else { /* if we are between x0 and x1 apply freeslip */
        if (data_ctx->component == 0) { 
          *value = 0.0; /* Set to 0.0 the x velocity component */
        } else {
          impose_dirichlet = PETSC_FALSE; /* Do not apply value to the z velocity component */ 
        }
      }
    }
      break;

    /* Face of normal z */
    case 1:
    { 
      if (data_ctx->component == 2) {
        if (position[0] <= data_ctx->x0) {
          *value = vr;
        } else if (position[0] >= data_ctx->x3) {
          *value = -vr;
        } else if (position[0] > data_ctx->x0 && position[0] < data_ctx->x1) {
          // Interpolation between vr and 0
          *value = vr + (position[0]-data_ctx->x0)*(0.0-vr)/(data_ctx->x1-data_ctx->x0);
        } else if (position[0] > data_ctx->x2 && position[0] < data_ctx->x3) {
          // Interpolation between 0 and -vr
          *value = 0.0 + (position[0]-data_ctx->x2)*(-vr-0.0)/(data_ctx->x3-data_ctx->x2);
        } else {
          *value = 0.0;
        }
      } else {
        if (position[0] <= data_ctx->x0) {
          *value = vr;
        } else if (position[0] >= data_ctx->x3) {
          *value = -vr;
        } else {
          impose_dirichlet = PETSC_FALSE;
        }
      }
    }    
      break;
  }  

  return impose_dirichlet;
}

PetscBool BCListEvaluator_RotationPoleFreeslipSpherical(PetscScalar position[],PetscScalar *value, void *ctx) 
{
  BC_RotationPoleFreeslipSpherical data_ctx = (BC_RotationPoleFreeslipSpherical)ctx;
  PetscScalar                   v_e,v_n;  
  PetscBool                     impose_dirichlet = PETSC_TRUE;
  PetscErrorCode                ierr;

  PetscFunctionBegin;

  ierr = ComputeSphericalVelocities(position,data_ctx->phi_0,data_ctx->lat_pole,data_ctx->lon_pole,data_ctx->v0,&v_e,&v_n);CHKERRQ(ierr);

  switch(data_ctx->normal) { /* Determine (user choice) on which face we are */

    /* Face of normal x */
    case 0:
    { 
      if (data_ctx->component == 0) {
        if (position[2] <= data_ctx->x0) {
          *value = v_n;
        } else if (position[2] > data_ctx->x0 && position[2] < data_ctx->x1) {
          // Interpolation between v_n and 0
          *value = v_n + (position[2]-data_ctx->x0)*(0.0-v_n)/(data_ctx->x1-data_ctx->x0);
        } else if (position[2] > data_ctx->x2 && position[2] < data_ctx->x3) {
          // Interpolation between 0 and -vr
          *value = 0.0 + (position[2]-data_ctx->x2)*(-v_n-0.0)/(data_ctx->x3-data_ctx->x2);
        } else if (position[2] >= data_ctx->x3) {
          *value = -v_n;
        } else { /* if we are between x0 and x1 apply freeslip */
          *value = 0.0; /* Set to 0.0 the x velocity component */
        }
      } else {
        if (position[2] <= data_ctx->x0) {
          *value = v_e;
        } else if (position[2] >= data_ctx->x3) {
          *value = -v_e;
        } else {
          impose_dirichlet = PETSC_FALSE;
        }
      }
    }
      break;

    /* Face of normal z */
    case 1:
    { 
      if (data_ctx->component == 2) {
        if (position[0] <= data_ctx->x0) {
          *value = v_e;
        } else if (position[0] >= data_ctx->x3) {
          *value = -v_e;
        } else if (position[0] > data_ctx->x0 && position[0] < data_ctx->x1) {
          // Interpolation between vr and 0
          *value = v_e + (position[0]-data_ctx->x0)*(0.0-v_e)/(data_ctx->x1-data_ctx->x0);
        } else if (position[0] > data_ctx->x2 && position[0] < data_ctx->x3) {
          // Interpolation between 0 and -vr
          *value = 0.0 + (position[0]-data_ctx->x2)*(-v_e-0.0)/(data_ctx->x3-data_ctx->x2);
        } else {
          *value = 0.0;
        }
      } else {
        if (position[0] <= data_ctx->x0) {
          *value = v_n;
        } else if (position[0] >= data_ctx->x3) {
          *value = -v_n;
        } else {
          impose_dirichlet = PETSC_FALSE;
        }
      }
    }    
      break;
  }  
  return impose_dirichlet;
}

PetscBool BCListEvaluator_RotationPoleSpherical(PetscScalar position[],PetscScalar *value, void *ctx)
{
  BC_RotationPoleSpherical data_ctx = (BC_RotationPoleSpherical)ctx;
  PetscScalar              v_e,v_n;
  PetscBool                impose_dirichlet = PETSC_TRUE;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = ComputeSphericalVelocities(position,data_ctx->phi_0,data_ctx->lat_pole,data_ctx->lon_pole,data_ctx->v0,&v_e,&v_n);CHKERRQ(ierr);

  if(data_ctx->component == 0) {
    *value = v_n;
  } else if (data_ctx->component == 2) {
    *value = v_e;
  }
  return impose_dirichlet;
}

static PetscErrorCode ComputeSphericalVelocities(PetscScalar position[],PetscReal phi_0,PetscReal lat_pole,PetscReal lon_pole,PetscReal w, PetscScalar *v_e, PetscScalar *v_n)
{
  PetscScalar R_E,wx,wy,wz;
  PetscScalar lat_point,lon_point;
  PetscScalar x_point,y_point,z_point;
  PetscScalar vx,vy,vz;

  PetscFunctionBegin;

  // Earth Radius; Except if you're not an Earth, do not change
  R_E = 63.78137;

  // Inverse Mercator ==> planar to spherical
  lon_point = phi_0 + (position[0]/R_E); 
  lat_point = 2*atan(exp(position[2]/R_E))-M_PI/2;
 
  // Rotation vector, Cartesian coords on a sphere 
  wx = w*cos(lat_pole)*cos(lon_pole);
  wy = w*cos(lat_pole)*sin(lon_pole);
  wz = w*sin(lat_pole);

  //Cartesian Point coords on a sphere
  x_point = R_E*cos(lat_point)*cos(lon_point);
  y_point = R_E*cos(lat_point)*sin(lon_point);
  z_point = R_E*sin(lat_point);

  // Linear velocities cross_product(w,x)
  vx = z_point*wy - y_point*wz;
  vy = x_point*wz - z_point*wx;
  vz = y_point*wx - x_point*wy;

  *v_n = - (sin(lat_point)*cos(lon_point)*vx) - (sin(lat_point)*sin(lon_point)*vy) + (cos(lat_point)*vz);
  *v_e = - (sin(lon_point)*vx) + (cos(lon_point)*vy);
//  v_u = (cos(lat_point)*cos(lon_point)*vx) + (cos(lat_point)*sin(lon_point)*vy) + (sin(lat_point)*vz);

  PetscFunctionReturn(0);
}

static PetscErrorCode Modelkinematic_TFV_DefineBCList_RotationSplitFreeslip(BCList bclist,DM dav,pTatinCtx user,Modelkinematic_TFVCtx *data)
{
  BC_RotationPole         bcdata;
  //BC_RotationPoleFreeslip bcdata_free;
  BC_RotationPoleFreeslipLinear bcdata_free;
  PetscReal               xp[2] = {0.0,0.0};
  PetscReal               x_freeslip[2] = {0.0,0.0};
  PetscInt                nn;
  PetscBool               found = PETSC_FALSE;
  PetscScalar             vy,v_ang;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  /* Allocate memory for the variables in the struct of BC Functions */
  ierr = PetscMalloc(sizeof(struct _p_BC_RotationPole),        &bcdata     );CHKERRQ(ierr);
  //ierr = PetscMalloc(sizeof(struct _p_BC_RotationPoleFreeslip),&bcdata_free);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(struct _p_BC_RotationPoleFreeslipLinear),&bcdata_free);CHKERRQ(ierr);

  /* Angular velocity in °/Myr */
  v_ang = 0.15;
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_v_ang",&v_ang,NULL);CHKERRQ(ierr);
  /* Scaling to rad/s and pTatin units [time] */
  v_ang = (M_PI * v_ang/180.0) / (3.14e13/data->time_bar);
  
  /* Rotation Pole coordinates in m */
  xp[0] = -500.0e3;
  xp[1] = 3000.0e3;

  nn = 2;
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-model_kinematic_TFV_PoleCoords",xp,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"The option -model_kinematic_TFV_PoleCoords needs 2 coordinates, %d values were passed",nn);
    }
  }

  /* Scaling to pTatin units [length] */
  xp[0] = xp[0]/data->length_bar;
  xp[1] = xp[1]/data->length_bar;

  /* Give to BCFunction the values */
  bcdata->xp = xp[0];
  bcdata->zp = xp[1]; 
  /* Faces of normal x */
 
    // ---------- //
   // FACE X = 0 //
  // ---------- //  
  bcdata->v0        = 0.5*v_ang;
  /* x component of the velocity vector */
  bcdata->component = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_RotationPole,(void*)bcdata);CHKERRQ(ierr);
  /* z component of the velocity vector */
  bcdata->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_RotationPole,(void*)bcdata);CHKERRQ(ierr);

    // --------------- //
   // FACE X = MAX(X) //
  // --------------- //  
  bcdata->v0        = -0.5*v_ang;
  /* x component of the velocity vector */
  bcdata->component = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_RotationPole,(void*)bcdata);CHKERRQ(ierr);
  /* z component of the velocity vector */
  bcdata->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_RotationPole,(void*)bcdata);CHKERRQ(ierr);

  /* Faces of normal z */

  bcdata_free->normal = 1;
  bcdata_free->xp = xp[0];
  bcdata_free->zp = xp[1];
  bcdata_free->v0 = 0.5*v_ang;
  /* x coordinates between which the BCs are freeslip (in m) */
  x_freeslip[0] = 500.0e3;
  x_freeslip[1] = 1500.0e3;
  nn = 2;
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-model_kinematic_TFV_FreeslipCoords",x_freeslip,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"The option -model_kinematic_TFV_FreeslipCoords needs 2 values, %d values were passed",nn);
    }
  }

 /* Scaling to pTatin units [length] */
  x_freeslip[0] = x_freeslip[0]/data->length_bar; 
  x_freeslip[1] = x_freeslip[1]/data->length_bar;
  bcdata_free->x0 = 400.0e3/data->length_bar;   //x_freeslip[0];
  bcdata_free->x1 = (500.0e3+100.0e3)/data->length_bar;   //x_freeslip[0]+(250.0e3/data->length_bar);
  bcdata_free->x2 = (1100.0e3-100.0e3)/data->length_bar;   //x_freeslip[1]-(250.0e3/data->length_bar);
  bcdata_free->x3 = 1100.0e3/data->length_bar;   //x_freeslip[1];
    // ---------- //
   // FACE Z = 0 //
  // ---------- //  
  bcdata_free->component = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_RotationPoleFreeslipLinear,(void*)bcdata_free);CHKERRQ(ierr);
  bcdata_free->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_RotationPoleFreeslipLinear,(void*)bcdata_free);CHKERRQ(ierr);

    // --------------- //
   // FACE Z = MAX(Z) //
  // --------------- //  
  bcdata_free->component = 0;  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_RotationPoleFreeslipLinear,(void*)bcdata_free);CHKERRQ(ierr);
  bcdata_free->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_RotationPoleFreeslipLinear,(void*)bcdata_free);CHKERRQ(ierr);

  /* vy computed in the function ModelComputeBottomFlow_kinematic_TFV */
  vy  = data->vy;

  /* infilling free slip base */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&vy);CHKERRQ(ierr);
  /* free surface top*/

  PetscFree(bcdata);
  PetscFree(bcdata_free);

  PetscFunctionReturn(0);
}

static PetscErrorCode Modelkinematic_TFV_DefineBCList_RotationSpherical(BCList bclist,DM dav,pTatinCtx user,Modelkinematic_TFVCtx *data)
{
  BC_RotationPoleSpherical         bcdata;
  BC_RotationPoleFreeslipSpherical bcdata_free;
  PetscReal                        xp[2] = {0.0,0.0};
  PetscReal                        x_freeslip[4] = {0.0,0.0,0.0,0.0};
  PetscInt                         nn;
  PetscBool                        found = PETSC_FALSE;
  PetscScalar                      vy,v_ang;
  PetscErrorCode                   ierr;

  PetscFunctionBegin;
  /* Allocate memory for the variables in the struct of BC Functions */
  ierr = PetscMalloc(sizeof(struct _p_BC_RotationPoleSpherical),        &bcdata     );CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(struct _p_BC_RotationPoleFreeslipSpherical),&bcdata_free);CHKERRQ(ierr);

  /* Angular velocity in °/Myr */
  v_ang = 0.15;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE,"-v_ang",&v_ang,NULL);CHKERRQ(ierr);
  /* Scaling to rad/s and pTatin units [time] */
  v_ang = (M_PI * v_ang/180.0) / (3.14e13/data->time_bar);

  xp[0] = 75.6308; 
  xp[1] = 2.95007;

  nn = 2;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_KINE,"-PoleLatLon",xp,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"The option -model_kinematic_TFV_PoleLatLon needs 2 coordinates, %d values were passed",nn);
    }
  }
  
  x_freeslip[0] = 300.0e3;
  x_freeslip[1] = 400.0e3;
  x_freeslip[2] = 900.0e3;
  x_freeslip[3] = 1000.0e3;

  nn = 4;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_KINE,"-FreeslipCoords",x_freeslip,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"The option -model_kinematic_TFV_FreeslipCoords needs 4 coordinates, %d values were passed",nn);
    }
  }

  bcdata->phi_0 = 0.0;
  bcdata_free->phi_0 = bcdata->phi_0; 
  bcdata->lat_pole = xp[0]*M_PI/180.0;
  bcdata->lon_pole = xp[1]*M_PI/180.0;
  bcdata_free->lat_pole = bcdata->lat_pole;
  bcdata_free->lon_pole = bcdata->lon_pole;
  // Face Zmin
  bcdata->v0 = 0.5 * v_ang;
  bcdata->component = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_RotationPoleSpherical,(void*)bcdata);CHKERRQ(ierr);
  bcdata->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_RotationPoleSpherical,(void*)bcdata);CHKERRQ(ierr);
 
  // Face Zmax
  bcdata->v0 = -0.5 * v_ang;
  bcdata->component = 0; 
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_RotationPoleSpherical,(void*)bcdata);CHKERRQ(ierr);
  bcdata->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_RotationPoleSpherical,(void*)bcdata);CHKERRQ(ierr);

  bcdata_free->x0 = x_freeslip[0]/data->length_bar;   
  bcdata_free->x1 = x_freeslip[1]/data->length_bar;   
  bcdata_free->x2 = x_freeslip[2]/data->length_bar;  
  bcdata_free->x3 = x_freeslip[3]/data->length_bar; 

  // Face Xmin
  bcdata_free->v0 = 0.5 * v_ang;
  bcdata_free->normal = 0;
  bcdata_free->component = 0; 
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_RotationPoleFreeslipSpherical,(void*)bcdata_free);CHKERRQ(ierr);
  bcdata_free->component = 2; 
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_RotationPoleFreeslipSpherical,(void*)bcdata_free);CHKERRQ(ierr);

  // Face Xmax
  bcdata_free->component = 0; 
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_RotationPoleFreeslipSpherical,(void*)bcdata_free);CHKERRQ(ierr);
  bcdata_free->component = 2; 
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_RotationPoleFreeslipSpherical,(void*)bcdata_free);CHKERRQ(ierr);
  
  /* vy computed in the function ModelComputeBottomFlow_kinematic_TFV */
  vy  = data->vy;

  /* infilling free slip base */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&vy);CHKERRQ(ierr);
  /* free surface top*/

  PetscFree(bcdata);
  PetscFree(bcdata_free);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_kinematic_TFV(pTatinCtx user,void *ctx)
{
  Modelkinematic_TFVCtx *data = (Modelkinematic_TFVCtx*)ctx;
  PetscBool        active_energy;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  BCList           u_bclist;
  Vec              X = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&stokes_pack);CHKERRQ(ierr);
  ierr = PhysCompStokesGetBCList(stokes,&u_bclist,NULL);CHKERRQ(ierr);
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  
  ierr = pTatinPhysCompGetData_Stokes(user,&X);CHKERRQ(ierr); 
 
  ierr = ModelComputeBottomFlow_kinematic_TFV(user,X,data);CHKERRQ(ierr);
  //ierr = Modelkinematic_TFV_DefineBCList_RotationSplitFreeslip(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,user,data);CHKERRQ(ierr);
  ierr = Modelkinematic_TFV_DefineBCList_RotationSpherical(user->stokes_ctx->u_bclist,user->stokes_ctx->dav,user,data);CHKERRQ(ierr);
  /* set boundary conditions for temperature */
  ierr = pTatinContextValid_Energy(user,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PetscReal      val_T;
    PhysCompEnergy energy;
    BCList         bclist;
    DM             daT;
    PetscBool      (*iterator_initial_thermal_field)(PetscScalar*,PetscScalar*,void*);
    PetscReal      coeffs[9];

    ierr   = pTatinGetContext_Energy(user,&energy);CHKERRQ(ierr);
    daT    = energy->daT;
    bclist = energy->T_bclist;

    ierr = Modelkinematic_TFV_GetDescription_InitialThermalField(data,coeffs,&iterator_initial_thermal_field);CHKERRQ(ierr);

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
    PetscInt         l;
    PhysCompEnergyFV energy;
    PetscReal        val_T;
    PetscReal        coeffs[9];
    PetscErrorCode   (*iterator_initial_thermal_field)(FVDA,
                                                       DACellFace,
                                                       PetscInt,
                                                       const PetscReal*,
                                                       const PetscReal*,
                                                       const PetscInt*,
                                                       PetscReal,FVFluxType*,PetscReal*,void*);

    ierr = Modelkinematic_TFV_GetDescription_InitialThermalField_FV(data,coeffs,&iterator_initial_thermal_field);CHKERRQ(ierr);

    ierr = pTatinGetContext_EnergyFV(user,&energy);CHKERRQ(ierr);
    if (data->use_semi_eulerian_mesh) {

      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_S,PETSC_FALSE,0.0,iterator_initial_thermal_field,(void*)coeffs);CHKERRQ(ierr);
      
      val_T = data->Ttop;
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_N,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);

      
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_E,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_W,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_F,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_B,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);

      const DACellFace facelist[] = {DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_B, DACELL_FACE_F};
      for(l=0;l<sizeof(facelist)/sizeof(DACellFace);l++) {
        ierr = FVSetDirichletFromInflow(energy->fv,energy->T,facelist[l]);
      }

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

static PetscErrorCode ModelApplyBoundaryConditionMG_kinematic_TFV(PetscInt nl,BCList bclist[],DM dav[],pTatinCtx user,void *ctx)
{
  Modelkinematic_TFVCtx *data = (Modelkinematic_TFVCtx*)ctx;
  Vec              X = NULL;
  PetscInt         n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinPhysCompGetData_Stokes(user,&X);CHKERRQ(ierr);  
  ierr = ModelComputeBottomFlow_kinematic_TFV(user,X,data);

  for (n=0; n<nl; n++) {
    //ierr = Modelkinematic_TFV_DefineBCList_RotationSplitFreeslip(bclist[n],dav[n],user,data);CHKERRQ(ierr);
    ierr = Modelkinematic_TFV_DefineBCList_RotationSpherical(bclist[n],dav[n],user,data);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyMaterialBoundaryCondition_kinematic_TFV(pTatinCtx c,void *ctx)
{
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]] - Not implemented \n", PETSC_FUNCTION_NAME);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyMaterialBoundaryCondition_kinematic_TFV_semi_eulerian(pTatinCtx c,void *ctx)
{
  PhysCompStokes  stokes;
  DM              stokes_pack,dav,dap;
  PetscInt        Nxp[2];
  PetscReal       perturb;
  DataBucket      material_point_db,material_point_face_db;
  PetscInt        f, n_face_list=5, face_list[] = { 0, 1, 3, 4, 5 }; // ymin, zmax //
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

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMeshGeometry_kinematic_TFV(pTatinCtx c,void *ctx)
{
  Modelkinematic_TFVCtx *data = (Modelkinematic_TFVCtx*)ctx;
  PetscInt              npoints,dir;
  PetscReal             *xref,*xnat;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = DMDASetUniformCoordinates(c->stokes_ctx->dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);
  CHKERRQ(ierr);

  dir = 1; // 0 = x; 1 = y; 2 = z;
  npoints = 3;

  ierr = PetscMalloc1(npoints,&xref);CHKERRQ(ierr); 
  ierr = PetscMalloc1(npoints,&xnat);CHKERRQ(ierr); 

  xref[0] = 0.0;
  xref[1] = 0.3;
  xref[2] = 1.0;

  xnat[0] = 0.0;
  xnat[1] = 0.6;
  xnat[2] = 1.0;

  ierr = DMDACoordinateRefinementTransferFunction(c->stokes_ctx->dav,dir,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(c->stokes_ctx->dav);CHKERRQ(ierr);

  /* note - Don't access the energy mesh here, its not yet created */
  /* note - The initial velocity mesh geometry will be copied into the energy mesh */

  PetscReal gvec[] = { 0.0, -10.0, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);

  ierr = PhysCompStokesUpdateSurfaceQuadrature(c->stokes_ctx);CHKERRQ(ierr);

  PetscFree(xnat);
  PetscFree(xref);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_kinematic_TFV(pTatinCtx c,void *ctx)
{
  Modelkinematic_TFVCtx *data = (Modelkinematic_TFVCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  if (data->from_map) {
    ierr = ModelApplyInitialMaterialGeometry_FromMap(c,ctx);CHKERRQ(ierr);
    ierr = ModelApplyInitialMaterialGeometry_FromIndex(c,ctx);CHKERRQ(ierr);
  } else {
    ierr = ModelApplyInitialMaterialGeometry_Notchtest(c,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyUpdateMeshGeometry_kinematic_TFV(pTatinCtx c,Vec X,void *ctx)
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
  ierr = PhysCompStokesUpdateSurfaceQuadrature(c->stokes_ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyUpdateMeshGeometry_kinematic_TFV_semi_eulerian(pTatinCtx c,Vec X,void *ctx)
{
  PetscReal       step;
  PetscInt        dir,npoints;
  PetscReal       *xnat,*xref;
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

  //ierr = UpdateMeshGeometry_VerticalLagrangianSurfaceRemesh(dav,velocity,step);CHKERRQ(ierr);
  ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,step);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  dir = 1;
  npoints = 3;
  ierr = PetscMalloc1(npoints,&xref);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints,&xnat);CHKERRQ(ierr);

  xref[0] = 0.0;
  xref[1] = 0.3;
  xref[2] = 1.0;

  xnat[0] = 0.0;
  xnat[1] = 0.6;
  xnat[2] = 1.0;

  ierr = DMDACoordinateRefinementTransferFunction(dav,dir,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);

  PetscFree(xref);
  PetscFree(xnat);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutput_kinematic_TFV(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  Modelkinematic_TFVCtx *data = (Modelkinematic_TFVCtx*)ctx;
  PetscBool        active_energy;
  DataBucket       materialpoint_db;
  PetscErrorCode   ierr;
  static PetscBool been_here = PETSC_FALSE;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  //ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  // just plot the velocity field (coords and vel stored in file as floats)
  //ierr = pTatin3d_ModelOutputLite_Velocity_Stokes(c,X,prefix);CHKERRQ(ierr);
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

  //  ierr = pTatin3d_ModelOutput_MarkerCellFields(c,nf,mp_prop_list,prefix);CHKERRQ(ierr);
    ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(c,PETSC_FALSE,sizeof(mp_prop_list)/sizeof(MaterialPointVariable),mp_prop_list,prefix);CHKERRQ(ierr);
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
#if 0
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

#if 1
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
  
#if 1
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

static PetscErrorCode ModelDestroy_kinematic_TFV(pTatinCtx c,void *ctx)
{
  Modelkinematic_TFVCtx *data = (Modelkinematic_TFVCtx*)ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  /* Free contents of structure */

  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialStokesVariableMarkers_kinematic_TFV(pTatinCtx user,Vec X,void *ctx)
{
  DM                         stokes_pack,dau,dap;
  PhysCompStokes             stokes;
  Vec                        Uloc,Ploc;
  PetscScalar                *LA_Uloc,*LA_Ploc;
  Modelkinematic_TFVCtx           *data = (Modelkinematic_TFVCtx*)ctx;
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

static PetscBool iterator_plume_thermal_field(PetscScalar coor[],PetscScalar *val,void *ctx)
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

static PetscBool iterator_plume_thermal_field_2(PetscScalar coor[],PetscScalar *val,void *ctx)
{
  PetscBool         impose = PETSC_FALSE, set;
  const PetscScalar origin[] = {7.0, -1.0, 4.0};
  PetscScalar       r,dl;
  
  set = DMDAVecTraverse3d_ERFC3DFunctionXYZ(coor,val,ctx);
  { PetscInt d;
    r = 0.0;
    for (d=0; d<3; d++) { dl = coor[d] - origin[d]; r += dl*dl; }
  }
  r = PetscSqrtReal(r);
  if (r < 0.25) {
    impose = PETSC_TRUE;
    /**val  += 450.0;*/ /* shift background gradient by constant */
    *val = 1600.0; /* impose constant value */
  }
  return impose;
}

static PetscErrorCode ModelApplyInitialCondition_kinematic_TFV(pTatinCtx c,Vec X,void *ctx)
{
  Modelkinematic_TFVCtx                        *data = (Modelkinematic_TFVCtx*)ctx;
  DM                                           stokes_pack,dau,dap;
  Vec                                          velocity,pressure;
  DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
  DMDAVecTraverse3d_InterpCtx                  IntpCtx;
  PetscReal                                    MeshMin[3],MeshMax[3],domain_height;
  PetscBool                                    active_energy;
  BC_RotationPole                              bcdata;
  PetscReal                                    xp[2] = {0.0,0.0};
  PetscInt                                     nn;
  PetscBool                                    found = PETSC_FALSE;
  PetscScalar                                  vy,v_ang;
  PetscErrorCode                               ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  stokes_pack = c->stokes_ctx->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /* Allocate memory for the variables in the struct of BC Functions */
  ierr = PetscMalloc(sizeof(struct _p_BC_RotationPole),        &bcdata     );CHKERRQ(ierr);

  /* Angular velocity in °/Myr */
  v_ang = 0.3;
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_v_ang",&v_ang,NULL);CHKERRQ(ierr);
  /* Scaling to rad/s and pTatin units [time] */
  v_ang = (M_PI * v_ang/180.0) / (3.14e13/data->time_bar);
  
  /* Rotation Pole coordinates in m */
  xp[0] = -500.0e3;
  xp[1] = 3000.0e3;

  nn = 2;
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-model_kinematic_TFV_PoleCoords",xp,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"The option -model_kinematic_TFV_PoleCoords needs 2 coordinates, %d values were passed",nn);
    }
  }

  /* Scaling to pTatin units [length] */
  xp[0] = xp[0]/data->length_bar;
  xp[1] = xp[1]/data->length_bar;

  /* Give to BCFunction the values */
  bcdata->xp = xp[0];
  bcdata->zp = xp[1]; 
  bcdata->v0        = 0.5*v_ang;
  /* x component of the velocity vector */
  bcdata->component = 0;
  ierr = DMDAVecTraverse3d(dau,velocity,0,BCListEvaluator_RotationPole,(void*)bcdata);CHKERRQ(ierr);
  /* z component of the velocity vector */
  bcdata->component = 2;
  ierr = DMDAVecTraverse3d(dau,velocity,2,BCListEvaluator_RotationPole,(void*)bcdata);CHKERRQ(ierr);
  /* vy computed in the function ModelComputeBottomFlow_kinematic_TFV */
  vy  = data->vy;
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
    PetscReal      coeffs[9];

    ierr = pTatinGetContext_Energy(c,&energy);CHKERRQ(ierr);
    ierr = pTatinPhysCompGetData_Energy(c,&temperature,NULL);CHKERRQ(ierr);
    daT  = energy->daT;

    ierr = Modelkinematic_TFV_GetDescription_InitialThermalField(data,coeffs,&iterator_initial_thermal_field);CHKERRQ(ierr);
    ierr = DMDAVecTraverse3d(daT,temperature,0,iterator_initial_thermal_field,(void*)coeffs);CHKERRQ(ierr);
  }

  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergyFV energy;
    PetscBool        (*iterator_initial_thermal_field)(PetscScalar*,PetscScalar*,void*);
    PetscReal        coeffs[9];
    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    ierr = Modelkinematic_TFV_GetDescription_InitialThermalField(data,coeffs,&iterator_initial_thermal_field);CHKERRQ(ierr);

    ierr = FVDAVecTraverse(energy->fv,energy->T,0.0,0,iterator_initial_thermal_field,(void*)coeffs);CHKERRQ(ierr);
    
    // <FV TESTING: Insert a hot spherical region centered at x = 7, y = -1.0, z = 4.0 >
    //ierr = FVDAVecTraverse(energy->fv,energy->T,0.0,0,iterator_plume_thermal_field,NULL);CHKERRQ(ierr);
//    ierr = FVDAVecTraverse(energy->fv,energy->T,0.0,0,iterator_plume_thermal_field_2,(void*)coeffs);CHKERRQ(ierr); /* <plume tests> */
  }
 
  PetscFree(bcdata);
 
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_kinematic_TFV(void)
{
  Modelkinematic_TFVCtx *data;
  pTatinModel      m;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(Modelkinematic_TFVCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(Modelkinematic_TFVCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"kinematic_TFV");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_kinematic_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialCondition_kinematic_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,   (void (*)(void))ModelApplyInitialStokesVariableMarkers_kinematic_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_kinematic_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_kinematic_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_kinematic_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_kinematic_TFV);CHKERRQ(ierr);

  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_kinematic_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_kinematic_TFV);CHKERRQ(ierr);

  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_kinematic_TFV);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_kinematic_TFV);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_FromMap(pTatinCtx c,void *ctx)
{
  PetscInt       dir_0,dir_1,direction;
  int            p,n_mp_points;
  int            phase_init, phase, phase_index;
  char           map_file[PETSC_MAX_PATH_LEN], *name;
  CartGrid       phasemap;
  DataBucket     db;
  DataField      PField_std;
  PetscBool      flg,phasefound;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf (PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = PetscOptionsGetString(NULL,NULL,"-map_file",map_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (flg == PETSC_FALSE) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Expected user to provide a map file \n");
  }
  ierr = PetscOptionsGetInt(NULL,NULL,"-extrude_dir",&direction,&flg);CHKERRQ(ierr);
  if (flg == PETSC_FALSE) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Expected user to provide an extrusion direction \n");
  }

  switch (direction){
    case 0:{
             dir_0 = 2;
             dir_1 = 1;
           }
           break;
    case 1:{
             dir_0 = 0;
             dir_1 = 2;
           }
           break;
    case 2:{
             dir_0 = 0;
             dir_1 = 1;
           }
           break;
    default :
           SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"-extrude_dir %d not valid, it must be 0(x), 1(y) or 2(z)",direction);
  }

  if (asprintf(&name,"./%s.pmap",map_file) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = CartGridCreate(&phasemap);CHKERRQ(ierr);
  ierr = CartGridSetFilename(phasemap,map_file);CHKERRQ(ierr);
  ierr = CartGridSetUp(phasemap);CHKERRQ(ierr);
  free(name);

  //if (asprintf(&name,"./%s_phase_map.gp",map_file) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  //ierr = CartGridViewPV(phasemap,name);CHKERRQ(ierr);
  //free(name);


  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof (MPntStd));

  DataBucketGetSizes(db,&n_mp_points,0,0);

  for (p=0; p<n_mp_points; p++)
  {
    MPntStd *material_point;
    double position2D[2],*pos;
    DataFieldAccessPoint(PField_std, p, (void **) &material_point);
    MPntStdGetField_global_coord(material_point,&pos);

    position2D[0] = pos[dir_0];
    position2D[1] = pos[dir_1];

    //MPntStdGetField_phase_index(material_point, &phase_init);

    ierr = CartGridGetValue(phasemap, position2D, (int*)&phase_index, &phasefound);CHKERRQ(ierr);

    if (phasefound) {     /* point located in the phase map */
      phase = phase_index;
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"marker outside the domain\n your phasemap is smaller than the domain \n please check your parameters and retry");
    }
    /* user the setters provided for you */
    MPntStdSetField_phase_index(material_point, phase);

  }
  ierr = CartGridDestroy(&phasemap);CHKERRQ(ierr);
  DataFieldRestoreAccess(PField_std);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_FromIndex(pTatinCtx c, void *ctx)
{
  Modelkinematic_TFVCtx *data = (Modelkinematic_TFVCtx*)ctx;
  PetscInt         p,n_mp_points,ni,n;
  PetscReal        y_lab,y_moho,y_midcrust,ylayers;
  DataBucket       db;
  DataField        PField_std,PField_pls;
  PetscInt         phase,phase_init;
  PetscInt         nn;
  PetscBool        found;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
/*
  ierr = PetscMalloc1(data->n_phase_map,&y_midcrust);CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phase_map,&y_moho);CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phase_map,&y_lab);CHKERRQ(ierr);
*/
  /* Set the default depth values of each layer */
/*
  for (n=0;n<data->n_phase_map;n++) {
    y_midcrust[n] = -20.0e3;
    y_moho[n]     = -40.0e3;
    y_lab[n]      = -120.0e3;
  }

  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_KINE,"-y_midcrust",y_midcrust,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != data->n_phase_map) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"The option -model_kinematic_TFV_y_midcrust needs %d entries but %d values were passed",data->n_phase_map,nn);
    }
  }

  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_KINE,"-y_moho",y_moho,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != data->n_phase_map) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"The option -model_kinematic_TFV_y_moho needs %d entries but %d values were passed",data->n_phase_map,nn);
    }
  }

  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_KINE,"-y_lab",y_lab,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != data->n_phase_map) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"The option -model_kinematic_TFV_y_lab needs %d entries but %d values were passed",data->n_phase_map,nn);
    }
  }
*/
  /* Scale values */
/*
  for (n=0;n<data->n_phase_map;n++) {
    y_midcrust[n] = y_midcrust[n]/data->length_bar;
    y_moho[n]     = y_moho[n]/data->length_bar;
    y_lab[n]      = y_lab[n]/data->length_bar;
  }
*/
//  ierr = PetscMalloc1(data->n_phase_map*(data->nlayers-1),&ylayers);CHKERRQ(ierr);
  /* Fill an array that contains all the depth values for each layer in the model */
/*
  for (n=0;n<data->n_phase_map;n++) {
    ylayers[n*(data->nlayers-1) + 0] = y_lab[n];
    ylayers[n*(data->nlayers-1) + 1] = y_moho[n];
    ylayers[n*(data->nlayers-1) + 2] = y_midcrust[n];
  }
*/

  y_lab = -120.0e3;
  y_moho = -40.0e3;
  y_midcrust = -20.0e3;

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE,"-y_lab",&y_lab,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE,"-y_moho",&y_moho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_KINE,"-y_midcrust",&y_midcrust,NULL);CHKERRQ(ierr);
  
  y_lab = y_lab/data->length_bar;
  y_moho = y_moho/data->length_bar;
  y_midcrust = y_midcrust/data->length_bar;

  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));

  DataBucketGetSizes(db,&n_mp_points,0,0);

  for (p=0; p<n_mp_points; p++)
  {
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    double        *pos;
    char          yield;
    double        pls;

    DataFieldAccessPoint(PField_std, p, (void **) &material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    MPntStdGetField_global_coord(material_point,&pos);
    /* Get the phase index set by the FromMap function */
    MPntStdGetField_phase_index(material_point, &phase_init);

    /* Depending on the phase index determined by the map re-index phase below
       (to not have the same phase on the entire vertical direction) */
/*
    for (n=0;n<data->n_phase_map;n++) {
      if (phase_init == n) {
        for (ni=1;ni<data->nlayers;ni++) {
          if (pos[1] < ylayers[n*(data->nlayers-1) + (ni-1)]) {
            phase = n + ni*data->n_phase_map;
          }
        }
      }
    }
*/
    pls = ptatin_RandomNumberGetDouble(0.0,0.03);
    if (pos[1] < y_lab) {
      phase = 7;
    } else if (pos[1] < y_moho) {
      phase = 6;
    } else if (pos[1] < y_midcrust) {
      for (n=0;n<data->n_phase_map;n++) {
        if (n == phase_init) {
          phase = n + 3;
        }
      }
    } else {
      phase = phase_init;
    }

    yield = 0;
    MPntStdSetField_phase_index(material_point, phase);
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);
  }

  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);
/*  PetscFree(y_midcrust);
  PetscFree(y_moho);
  PetscFree(y_lab);
  PetscFree(ylayers);
*/
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_Notchtest(pTatinCtx c,void *ctx)
{
  Modelkinematic_TFVCtx *data = (Modelkinematic_TFVCtx*)ctx;
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
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_kinematic_TFV_norandom",&norandomiseplastic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-model_kinematic_TFV_DoubleNotch",&double_notch,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-model_kinematic_TFV_notchspace",&notchspace,NULL);CHKERRQ(ierr);

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
