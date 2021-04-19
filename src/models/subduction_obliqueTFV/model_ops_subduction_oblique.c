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
#include "subduction_oblique_ctx.h"

static const char MODEL_NAME_SO[] = "model_subduction_oblique_"

PetscErrorCode ModelInitialize_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data;
  RheologyConstants         *rheology;
  PetscInt                  nn,region_idx;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscBool                 flg;
  PetscErrorCode            ierr;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelSubductionObliqueCtx*)ctx;
  
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);
  rheology->rheology_type = RHEOLOGY_VP_STD;
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);
  
  data->n_phases = 8;
  rheology->nphases_active = data->n_phases;
  rheology->apply_viscosity_cutoff_global = PETSC_TRUE;
  rheology->eta_upper_cutoff_global = 1.e+25;
  rheology->eta_lower_cutoff_global = 1.e+19;
  
  /* set the deffault values of the material constant for this particular model */
  /*scaling */
  data->length_bar    = 100.0 * 1.0e3;
  data->viscosity_bar = 1e22;
  data->velocity_bar  = 1.0e-10;
  /*cutoff */
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-apply_viscosity_cutoff_global",&rheology->apply_viscosity_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-eta_lower_cutoff_global",&rheology->eta_lower_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-eta_upper_cutoff_global",&rheology->eta_upper_cutoff_global,NULL);CHKERRQ(ierr);
  
  /* box geometry, m */
  data->Lx = 1000.0e3; 
  data->Ly = 0.0e3;
  data->Lz = 600.0e3;
  data->Ox = 0.0e3;
  data->Oy = -680.0e3;
  data->Oz = 0.0e3;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Lx",&data->Lx,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Ly",&data->Ly,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Lz",&data->Lz,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Ox",&data->Ox,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Oy",&data->Oy,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Oz",&data->Oz,&flg);CHKERRQ(ierr);
  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"********** Box Geometry **********\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[Ox,Lx] = [%+1.4e [m], %+1.4e [m]]\n", data->Ox ,data->Lx );
  PetscPrintf(PETSC_COMM_WORLD,"[Oy,Ly] = [%+1.4e [m], %+1.4e [m]]\n", data->Oy ,data->Ly );
  PetscPrintf(PETSC_COMM_WORLD,"[Oz,Lz] = [%+1.4e [m], %+1.4e [m]]\n", data->Oz ,data->Lz );
  
  data->y_continent[0] = -25.0e3;
  data->y_continent[1] = -35.0e3;
  data->y_continent[2] = -120.0e3;
  data->y_ocean[0] = -5.0e3;
  data->y_ocean[1] = -10.0e3;
  data->y_ocean[2] = -80.0e3;
  
  data->y0 = 0.0; // depth of the first rock layer
  data->alpha_subd = 15.0; // angle of the trench
  data->theta_subd = 30.0; // angle of the subduction 
  data->wz = 20.0e3; // weak zone width
  
  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_SO,"-y_continent",data->y_continent,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -y_continent. Found %d",nn);
    }
  }
  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_SO,"-y_ocean",data->y_ocean,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -y_ocean. Found %d",nn);
    }
  }
  
  PetscPrintf(PETSC_COMM_WORLD,"*************** Layering ***************\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"CONTINENT: Upper Crust Depth %+1.4e [m] \n", data->y_continent[0] );
  PetscPrintf(PETSC_COMM_WORLD,"           Lower Crust Depth %+1.4e [m] \n", data->y_continent[1] );
  PetscPrintf(PETSC_COMM_WORLD,"           LAB         Depth %+1.4e [m] \n", data->y_continent[2] );
  PetscPrintf(PETSC_COMM_WORLD,"OCEAN:     Upper Crust Depth %+1.4e [m] \n", data->y_ocean[0] );
  PetscPrintf(PETSC_COMM_WORLD,"           Lower Crust Depth %+1.4e [m] \n", data->y_ocean[1] );
  PetscPrintf(PETSC_COMM_WORLD,"           LAB         Depth %+1.4e [m] \n", data->y_ocean[2] );
  
  data->oblique_IC = PETSC_FALSE;
  data->oblique_BC = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-oblique_IC",&data->oblique_IC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-oblique_BC",&data->oblique_BC,NULL);CHKERRQ(ierr);
  
  /* Material constant */
  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  /* Allocate memory for arrays of material parameters */
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
  source_type[0] = ENERGYSOURCE_CONSTANT;
  rho_ref        = 1.0;
  Cp             = 1.0;
  for (region_idx=0;region_idx<rheology->nphases_active;region_idx++) {
    preexpA[region_idx]     = 6.3e-6;
    Ascale[region_idx]      = 1.0e6;
    entalpy[region_idx]     = 156.0e3;
    Vmol[region_idx]        = 0.0;
    nexp[region_idx]        = 2.4;
    Tref[region_idx]        = 273.0;
    phi[region_idx]         = 30.0;
    phi_inf[region_idx]     = 5.0;
    Co[region_idx]          = 2.0e7;
    Co_inf[region_idx]      = 2.0e7;
    Tens_cutoff[region_idx] = 1.0e7;
    Hst_cutoff[region_idx]  = 2.0e8;
    eps_min[region_idx]     = 0.0;
    eps_max[region_idx]     = 0.5;
    beta[region_idx]        = 0.0;
    alpha[region_idx]       = 2.0e-5;
    rho[region_idx]         = 2700.0;
  }
  for (region_idx=0;region_idx<rheology->nphases_active;region_idx++) {
    /* Set material constitutive laws */
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_ARRHENIUS_2,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);

    /* VISCOUS PARAMETERS */
    if (asprintf (&option_name, "-preexpA_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&preexpA[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Ascale_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&Ascale[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-entalpy_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&entalpy[region_idx],NULL);CHKERRQ(ierr);
    free (option_name); 
    if (asprintf (&option_name, "-Vmol_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&Vmol[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-nexp_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&nexp[region_idx],NULL);CHKERRQ(ierr);
    free (option_name); 
    if (asprintf (&option_name, "-Tref_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&Tref[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    /* Set viscous params for region_idx */
    MaterialConstantsSetValues_ViscosityArrh(materialconstants,region_idx,preexpA[region_idx],Ascale[region_idx],entalpy[region_idx],Vmol[region_idx],nexp[region_idx],Tref[region_idx]);  

    /* PLASTIC PARAMETERS */
    if (asprintf (&option_name, "-phi_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&phi[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-phi_inf_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&phi_inf[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Co_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&Co[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Co_inf_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&Co_inf[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Tens_cutoff_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&Tens_cutoff[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Hst_cutoff_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&Hst_cutoff[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-eps_min_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&eps_min[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-eps_max_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&eps_max[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    phi_rad     = M_PI * phi[region_idx]/180.0;
    phi_inf_rad = M_PI * phi_inf[region_idx]/180.0;
    /* Set plastic params for region_idx */
    MaterialConstantsSetValues_PlasticDP(materialconstants,region_idx,phi_rad,phi_inf_rad,Co[region_idx],Co_inf[region_idx],Tens_cutoff[region_idx],Hst_cutoff[region_idx]);
    MaterialConstantsSetValues_SoftLin(materialconstants,region_idx,eps_min[region_idx],eps_max[region_idx]);

    /* ENERGY PARAMETERS */
    if (asprintf (&option_name, "-alpha_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&alpha[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-beta_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&beta[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-rho_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&rho[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    /* Set energy params for region_idx */
    MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha[region_idx],beta[region_idx],rho_ref,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
    MaterialConstantsSetValues_DensityBoussinesq(materialconstants,region_idx,rho[region_idx],alpha[region_idx],beta[region_idx]);
  }
  
  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
    EnergyConductivityConst *data_k;
    EnergySourceConst       *data_Q;
    DataField               PField_k,PField_Q;

    DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
    DataFieldGetEntries(PField_k,(void**)&data_k);
    EnergyConductivityConstSetField_k0(&data_k[regionidx],1.0e-6);
    // TODO Move heat source before loops and set it one by one
    DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
    DataFieldGetEntries(PField_Q,(void**)&data_Q);
    EnergySourceConstSetField_HeatSource(&data_Q[regionidx],0.0);
  }
  /* Report all material parameters values */
  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
    MaterialConstantsPrintAll(materialconstants,regionidx);
    MaterialConstantsEnergyPrintAll(materialconstants,regionidx);
  }
  /* Compute additional scaling parameters */
  data->time_bar      = data->length_bar / data->velocity_bar;
  data->pressure_bar  = data->viscosity_bar/data->time_bar;
  data->density_bar   = data->pressure_bar / data->length_bar;

  PetscPrintf(PETSC_COMM_WORLD,"[subduction_oblique]:  during the solve scaling will be done using \n");
  PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n", data->length_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n", data->velocity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n", data->time_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n", data->viscosity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->density_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n", data->pressure_bar );
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
  
  data->y_continent[0] = data->y_continent[0] / data->length_bar;
  data->y_continent[1] = data->y_continent[1] / data->length_bar;
  data->y_continent[2] = data->y_continent[2] / data->length_bar;
  data->y_ocean[0]     = data->y_ocean[0]     / data->length_bar;
  data->y_ocean[1]     = data->y_ocean[1]     / data->length_bar;
  data->y_ocean[2]     = data->y_ocean[2]     / data->length_bar;
  data->y0             = data->y0             / data->length_bar;
  data->wz             = data->wz             / data->length_bar;
  data->alpha_subd     = data->alpha_subd * M_PI/180.0;
  data->theta_subd     = data->theta_subd * M_PI/180.0;

  /* Scale velocity */
  data->vx = vx/data->velocity_bar;

  //scale thermal params
  /*
  data->h_prod = data->h_prod/(data->pressure_bar / data->time_bar);
  data->k      = data->k/(data->pressure_bar*data->length_bar*data->length_bar/data->time_bar);
  data->qm     = data->qm/(data->pressure_bar*data->velocity_bar);
  data->ylab   = data->ylab/data->length_bar;
  data->y_prod = data->y_prod/data->length_bar;
  */
  // scale material properties
  for (regionidx=0; regionidx<rheology->nphases_active;regionidx++) {
    MaterialConstantsScaleAll(materialconstants,regionidx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);
    MaterialConstantsEnergyScaleAll(materialconstants,regionidx,data->length_bar,data->time_bar,data->pressure_bar);
  }
    
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

PetscErrorCode ModelApplyInitialMeshGeometry_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data = (ModelSubductionObliqueCtx*)ctx;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscInt         dir,npoints;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);CHKERRQ(ierr);
  
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

  ierr = DMDACoordinateRefinementTransferFunction(dav,dir,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  
  PetscReal gvec[] = { 0.0, -9.8, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesUpdateSurfaceQuadrature(c->stokes_ctx);CHKERRQ(ierr);
  
  PetscFree(xref);
  PetscFree(xnat);
  
  PetscFunctionReturn(0);
}

/* GEOMETRY: Obliquity through initial conditions */
PetscErrorCode ModelApplyInitialMaterialGeometry_ObliqueIC(pTatinCtx c,void *ctx)
{
  DataBucket       db;
  DataField        PField_std,PField_pls;
  PetscInt         p;
  PetscReal xc,zc;
  
  PetscFunctionBegin;
  
  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));
  
  xc = (data->Lx + data->Ox)/2.0;
  zc = (data->Lz + data->Oz)/2.0;
  
  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    int           region_idx;
    double        *position;
    float         pls;
    char          yield;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access using the getter function */
    MPntStdGetField_global_coord(material_point,&position);
    
    x_trench =  (position[2] - zc      ) / tan(data->alpha_subd) + xc;
    x_subd   = -(position[1] - data->y0) / tan(data->theta_subd) + x_trench;
    
    if (position[0] <= x_subd && position[1] >= data->y_ocean[0]) {
      region_idx = 0;
    } else if (position[0] <= x_subd && position[1] >= data->y_ocean[1]) {
      region_idx = 1;
    } else if (position[0] <= (x_subd-data->wz) && position[1] >= data->y_ocean[2]) {
      region_idx = 4;
    } else if (position[0] >= (x_subd-data->wz) && 
               position[0] <= (x_subd+data->wz) &&
               position[1] < data->y_ocean[1]   && 
               position[1] >= data->y_continent[2]) {
      region_idx = 7;
    } else if (position[0] > x_subd && position[1] >= data->y_continent[0]) {
      region_idx = 2;
    } else if (position[0] > x_subd && position[1] >= data->y_continent[1]) {
      region_idx = 3;
    } else if (position[0] > x_subd && position[1] >= data->y_continent[2]) {
      region_idx = 5;
    } else {
      region_idx = 6;
    }
    
    MPntStdSetField_phase_index(material_point,region_idx);
  }
  
  PetscFunctionReturn(0);
}

/* GEOMETRY: Obliquity through boundary conditions */
PetscErrorCode ModelApplyInitialMaterialGeometry_ObliqueBC(pTatinCtx c,void *ctx)
{
  PetscFunctionBegin;
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data = (ModelSubductionObliqueCtx*)ctx;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  if (data->oblique_IC) {
    ierr = ModelApplyInitialMaterialGeometry_ObliqueIC(c,ctx);CHKERRQ(ierr);
  } else if (data->oblique_BC) {
    ierr = ModelApplyInitialMaterialGeometry_ObliqueBC(c,ctx);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialSolution_SubductionOblique(pTatinCtx c,Vec X,void *ctx)
{
  /* ModelSubductionObliqueCtx *data; */

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* data = (ModelSubductionObliqueCtx*)ctx; */

  PetscFunctionReturn(0);
}

PetscErrorCode SubductionOblique_VelocityBC(BCList bclist,DM dav,pTatinCtx c,ModelSubductionObliqueCtx *data)
{
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryCondition_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  data = (ModelSubductionObliqueCtx*)ctx;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  PetscPrintf(PETSC_COMM_WORLD,"param1 = %lf \n", data->param1 );

  /* Define velocity boundary conditions */
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = SubductionOblique_VelocityBC(stokes->u_bclist,dav,c,data);CHKERRQ(ierr);

  /* Define boundary conditions for any other physics */

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_SubductionOblique(PetscInt nl,BCList bclist[],DM dav[],pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data;
  PetscInt         n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data = (ModelSubductionObliqueCtx*)ctx;
  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = SubductionOblique_VelocityBC(bclist[n],dav[n],c,data);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyMaterialBoundaryCondition_SubductionOblique(pTatinCtx c,void *ctx)
{
  /* ModelSubductionObliqueCtx *data; */

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* data = (ModelSubductionObliqueCtx*)ctx; */

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyUpdateMeshGeometry_SubductionOblique(pTatinCtx c,Vec X,void *ctx)
{
  /* ModelSubductionObliqueCtx *data; */

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* data = (ModelSubductionObliqueCtx*)ctx; */

  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_SubductionOblique(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  /* ModelSubductionObliqueCtx *data; */

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* data = (ModelSubductionObliqueCtx*)ctx; */

  /* ---- Velocity-Pressure Mesh Output ---- */
  /* [1] Standard viewer: v,p written out as binary in double */
  /*
  ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  */
  /* [2] Light weight viewer: Only v is written out. v and coords are expressed as floats */
  /*
  ierr = pTatin3d_ModelOutputLite_Velocity_Stokes(c,X,prefix);CHKERRQ(ierr);
  */
  /* [3] Write out v,p into PETSc Vec. These can be used to restart pTatin */
  /*
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  */

  /* ---- Material Point Output ---- */
  /* [1] Basic viewer: Only reports coords, regionid and other internal data */
  /*
  ierr = pTatin3d_ModelOutput_MPntStd(c,prefix);CHKERRQ(ierr);
  */

  /* [2] Customized viewer: User defines specific fields they want to view - NOTE not .pvd file will be created */
  /*
  {
  DataBucket                materialpoint_db;
  const int                 nf = 4;
  const MaterialPointField  mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl, MPField_Energy };
  char                      mp_file_prefix[256];

  ierr = pTatinGetMaterialPoints(c,&materialpoint_db,NULL);CHKERRQ(ierr);
  sprintf(mp_file_prefix,"%s_mpoints",prefix);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  }
  */
  /* [3] Customized marker->cell viewer: Marker data is projected onto the velocity mesh. User defines specific fields */
  /*
  {
  const int                    nf = 3;
  const MaterialPointVariable  mp_prop_list[] = { MPV_viscosity, MPV_density, MPV_plastic_strain };

  ierr = pTatin3d_ModelOutput_MarkerCellFields(c,nf,mp_prop_list,prefix);CHKERRQ(ierr);
  }
  */

  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelSubductionObliqueCtx*)ctx;

  /* Free contents of structure */

  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_SubductionOblique(void)
{
  ModelSubductionObliqueCtx *data;
  pTatinModel      m;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelSubductionObliqueCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelSubductionObliqueCtx));CHKERRQ(ierr);

  /* set initial values for model parameters */
  data->param1 = 0.0;
  data->param2 = 0;

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"subduction_oblique");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_SubductionOblique);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
