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
#include "subduction_oblique_ctx.h"

#include "element_utils_q1.h"
#include "element_type_Q2.h"
#include "dmda_element_q2p1.h"

static const char MODEL_NAME_SO[] = "model_subduction_oblique_";

PetscErrorCode SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(DM da,PetscInt Nxp[],PetscReal perturb, PetscReal epsilon, PetscInt face_idx,DataBucket db);

PetscErrorCode ModelInitialize_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data;
  RheologyConstants         *rheology;
  DataField                 PField,PField_k,PField_Q;
  EnergyConductivityConst   *data_k;
  EnergySourceConst         *data_Q;
  DataBucket                materialconstants;
  EnergyMaterialConstants   *matconstants_e;
  PetscInt                  nn,region_idx;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscReal                 cm_per_yer2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 );
  PetscReal                 *preexpA,*Ascale,*entalpy,*Vmol,*nexp,*Tref;
  PetscReal                 *phi,*phi_inf,*Co,*Co_inf,*Tens_cutoff,*Hst_cutoff,*eps_min,*eps_max;
  PetscReal                 *beta,*alpha,*rho,*heat_source,*conductivity;
  PetscReal                 phi_rad,phi_inf_rad,Cp;
  PetscBool                 flg,found;
  char                      *option_name;
  PetscErrorCode            ierr;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelSubductionObliqueCtx*)ctx;
  
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);
  rheology->rheology_type = RHEOLOGY_VP_STD;
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);
  
  data->n_phases = 10;
  rheology->nphases_active = data->n_phases;
  rheology->apply_viscosity_cutoff_global = PETSC_TRUE;
  rheology->eta_upper_cutoff_global = 1.e+25;
  rheology->eta_lower_cutoff_global = 1.e+19;
  
  /* set the deffault values of the material constant for this particular model */
  /* scaling */
  data->length_bar    = 100.0 * 1.0e3;
  data->viscosity_bar = 1e22;
  data->velocity_bar  = 1.0e-10;
  /* cutoff */
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-apply_viscosity_cutoff_global",&rheology->apply_viscosity_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-eta_lower_cutoff_global",&rheology->eta_lower_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-eta_upper_cutoff_global",&rheology->eta_upper_cutoff_global,NULL);CHKERRQ(ierr);
  
  /* box geometry, [m] */
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
  
  data->y_continent[0] = -25.0e3; // Conrad
  data->y_continent[1] = -35.0e3; // Moho
  data->y_continent[2] = -120.0e3; // LAB
  data->y_ocean[0] = -1.0e3; // Top basement
  data->y_ocean[1] = -5.0e3; // Conrad
  data->y_ocean[2] = -10.0e3; // Moho
  data->y_ocean[3] = -80.0e3; // LAB
  
  data->y0 = 0.0; // depth of the first rock layer
  data->alpha_subd = 15.0; // angle of the trench
  data->theta_subd = 30.0; // angle of the subduction 
  data->wz = 20.0e3; // weak zone width
  
  /* Velocity */
  data->normV = 1.0;
  /* Angle of the velocity vector with the face on which it is applied */
  data->angle_v = 30.0;
  
  data->Ttop = 0.0;
  data->Tbottom = 1600.0;
  
  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_SO,"-y_continent",data->y_continent,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -y_continent. Found %d",nn);
    }
  }
  nn = 4;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_SO,"-y_ocean",data->y_ocean,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 4) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 4 values for -y_ocean. Found %d",nn);
    }
  }
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-normV",       &data->normV,NULL);CHKERRQ(ierr);  
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-angle_v",     &data->angle_v,NULL);CHKERRQ(ierr);  
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-alpha_trench",&data->alpha_subd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-theta_subd",  &data->theta_subd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-wz_width",    &data->wz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Ttop",        &data->Ttop,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Tbottom",     &data->Tbottom,NULL);CHKERRQ(ierr);
  
  PetscPrintf(PETSC_COMM_WORLD,"*************** Layering ***************\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"CONTINENT: Upper Crust Depth %+1.4e [m] \n", data->y_continent[0] );
  PetscPrintf(PETSC_COMM_WORLD,"           Lower Crust Depth %+1.4e [m] \n", data->y_continent[1] );
  PetscPrintf(PETSC_COMM_WORLD,"           LAB         Depth %+1.4e [m] \n", data->y_continent[2] );
  PetscPrintf(PETSC_COMM_WORLD,"OCEAN:     Upper Crust Depth %+1.4e [m] \n", data->y_ocean[0] );
  PetscPrintf(PETSC_COMM_WORLD,"           Lower Crust Depth %+1.4e [m] \n", data->y_ocean[1] );
  PetscPrintf(PETSC_COMM_WORLD,"           LAB         Depth %+1.4e [m] \n", data->y_ocean[2] );
  
  data->oblique_IC = PETSC_FALSE;
  data->oblique_BC = PETSC_FALSE;
  data->output_markers = PETSC_FALSE;
  data->subduction_temp_ic_steadystate_analytics = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-IC",&data->oblique_IC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-BC",&data->oblique_BC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-output_markers",&data->output_markers,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-ic_steady_state_analytics",&data->subduction_temp_ic_steadystate_analytics,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-is_2D",&data->is_2D,NULL);CHKERRQ(ierr);
  
  /* Material constants */
  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);
  
  /* Energy material constants */
  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  /* Conductivity */
  DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
  DataFieldGetEntries(PField_k,(void**)&data_k);
  
  /* Heat source */
  DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
  DataFieldGetEntries(PField_Q,(void**)&data_Q);
  
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
  ierr = PetscMalloc1(rheology->nphases_active,&heat_source);CHKERRQ(ierr);
  ierr = PetscMalloc1(rheology->nphases_active,&conductivity);CHKERRQ(ierr);
  /* Set default values for parameters */
  source_type[0] = ENERGYSOURCE_CONSTANT;
  Cp             = 800.0;
  for (region_idx=0;region_idx<rheology->nphases_active-1;region_idx++) {
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
    heat_source[region_idx] = 0.0;
    conductivity[region_idx] = 1.0;
  }
  for (region_idx=0;region_idx<rheology->nphases_active-1;region_idx++) {
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
    if (asprintf (&option_name, "-heat_source_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&heat_source[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-conductivity_%d", region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, option_name,&conductivity[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    
    /* Set energy params for region_idx */
    MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha[region_idx],beta[region_idx],rho[region_idx],Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
    MaterialConstantsSetValues_DensityBoussinesq(materialconstants,region_idx,rho[region_idx],alpha[region_idx],beta[region_idx]);
    EnergySourceConstSetField_HeatSource(&data_Q[region_idx],heat_source[region_idx]);
    EnergyConductivityConstSetField_k0(&data_k[region_idx],conductivity[region_idx]);
  }

  /* region_idx 9 --> Sticky Air */
  region_idx = 9;
  data->air_rho = 1.0;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-air_rho",&data->air_rho,NULL);CHKERRQ(ierr);  
  MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
  MaterialConstantsSetValues_ViscosityConst(materialconstants,region_idx,1.0e+19);
  MaterialConstantsSetValues_DensityConst(materialconstants,region_idx,data->air_rho);
  MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,0.0,0.0,1.0,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
  EnergyConductivityConstSetField_k0(&data_k[region_idx],1.0);
  EnergySourceConstSetField_HeatSource(&data_Q[region_idx],0.0);

  /* Report all material parameters values */
  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsPrintAll(materialconstants,region_idx);
    MaterialConstantsEnergyPrintAll(materialconstants,region_idx);
  }
  /* Compute additional scaling parameters */
  data->time_bar         = data->length_bar / data->velocity_bar;
  data->pressure_bar     = data->viscosity_bar/data->time_bar;
  data->density_bar      = data->pressure_bar * (data->time_bar*data->time_bar)/(data->length_bar*data->length_bar); // kg.m^-3
  data->acceleration_bar = data->length_bar / (data->time_bar*data->time_bar);
  //data->density_bar   = data->pressure_bar / data->length_bar; // This is not kg.m^-3, this is kg.m^-2.s^-2 or rho*g

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
  
  data->y_continent[0] = data->y_continent[0] / data->length_bar;
  data->y_continent[1] = data->y_continent[1] / data->length_bar;
  data->y_continent[2] = data->y_continent[2] / data->length_bar;
  data->y_ocean[0]     = data->y_ocean[0]     / data->length_bar;
  data->y_ocean[1]     = data->y_ocean[1]     / data->length_bar;
  data->y_ocean[2]     = data->y_ocean[2]     / data->length_bar;
  data->y_ocean[3]     = data->y_ocean[3]     / data->length_bar;
  data->y0             = data->y0             / data->length_bar;
  data->wz             = data->wz             / data->length_bar;
  data->alpha_subd     = data->alpha_subd * M_PI/180.0;
  data->theta_subd     = data->theta_subd * M_PI/180.0;
  
  /* Scale velocity */
  data->normV = data->normV*cm_per_yer2m_per_sec/data->velocity_bar;
  data->angle_v = data->angle_v*M_PI/180;
  
  /* scale material properties */
  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsScaleAll(materialconstants,region_idx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);
    MaterialConstantsEnergyScaleAll(materialconstants,region_idx,data->length_bar,data->time_bar,data->pressure_bar);
  }
    
  ierr = PetscFree(preexpA    );CHKERRQ(ierr);
  ierr = PetscFree(Ascale     );CHKERRQ(ierr);
  ierr = PetscFree(entalpy    );CHKERRQ(ierr);
  ierr = PetscFree(Vmol       );CHKERRQ(ierr);
  ierr = PetscFree(nexp       );CHKERRQ(ierr);
  ierr = PetscFree(Tref       );CHKERRQ(ierr);
  ierr = PetscFree(phi        );CHKERRQ(ierr);
  ierr = PetscFree(phi_inf    );CHKERRQ(ierr);
  ierr = PetscFree(Co         );CHKERRQ(ierr);
  ierr = PetscFree(Co_inf     );CHKERRQ(ierr);
  ierr = PetscFree(Tens_cutoff);CHKERRQ(ierr);
  ierr = PetscFree(Hst_cutoff );CHKERRQ(ierr);
  ierr = PetscFree(eps_min    );CHKERRQ(ierr);
  ierr = PetscFree(eps_max    );CHKERRQ(ierr);
  ierr = PetscFree(beta       );CHKERRQ(ierr);
  ierr = PetscFree(alpha      );CHKERRQ(ierr);
  ierr = PetscFree(rho        );CHKERRQ(ierr);
  ierr = PetscFree(heat_source);CHKERRQ(ierr);
  ierr = PetscFree(conductivity);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometry_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data = (ModelSubductionObliqueCtx*)ctx;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscInt         dir,npoints;
  PetscReal        *xref,*xnat;
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
  dir = 1; // 0 = x; 1 = y; 2 = z;
  npoints = 4;

  ierr = PetscMalloc1(npoints,&xref);CHKERRQ(ierr); 
  ierr = PetscMalloc1(npoints,&xnat);CHKERRQ(ierr); 

  xref[0] = 0.0;
  xref[1] = 0.28; //0.375;
  xref[2] = 0.65; //0.75;
  xref[3] = 1.0;

  xnat[0] = 0.0;
  xnat[1] = 0.8;
  xnat[2] = 0.935;//0.95;
  xnat[3] = 1.0;

  ierr = DMDACoordinateRefinementTransferFunction(dav,dir,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  
  PetscReal gvec[] = { 0.0, -9.8, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(c->stokes_ctx,1.0/data->acceleration_bar);CHKERRQ(ierr);

  ierr = PetscFree(xref);CHKERRQ(ierr);
  ierr = PetscFree(xnat);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* GEOMETRY: Obliquity through initial conditions */
PetscErrorCode ModelApplyInitialMaterialGeometry_ObliqueIC(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data = (ModelSubductionObliqueCtx*)ctx;
  DataBucket                db;
  DataField                 PField_std,PField_pls;
  PetscInt                  p,n_mp_points;
  PetscReal                 xc,zc;
  
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
    PetscReal     x_trench,x_subd;
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    int           region_idx;
    double        *position;
    float         pls;
    short         yield;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access using the getter function */
    MPntStdGetField_global_coord(material_point,&position);
    /* Set an initial small random noise on plastic strain */
    pls = ptatin_RandomNumberGetDouble(0.0,0.03);
    /* Set yield to none */
    yield = 0;
    /* Trench angle */
    x_trench =  (position[2] - zc      ) / tan(data->alpha_subd) + xc;
    /* Initial weak zone angle */
    x_subd   = -(position[1] - data->y0) / tan(data->theta_subd) + x_trench;
    
    if (position[0] <= x_subd && position[1] >= data->y_ocean[0]) {
      region_idx = 8; // Oceanic sediments
    } else if (position[0] <= x_subd && position[1] >= data->y_ocean[1]) {
      region_idx = 0; // Oceanic upper crust
    } else if (position[0] <= x_subd && position[1] >= data->y_ocean[2]) {
      region_idx = 1; // Oceanic lower crust
    } else if (position[0] <= (x_subd-data->wz) && position[1] >= data->y_ocean[3]) {
      region_idx = 4; // Oceanic lithosphere mantle
    } else if (position[0] >= (x_subd-data->wz) && 
               position[0] <= (x_subd+data->wz) &&
               position[1] < data->y_ocean[2]   && 
               position[1] >= data->y_continent[2]) {
      region_idx = 7; // Weak zone
    } else if (position[0] > x_subd && position[1] >= data->y_continent[0]) {
      region_idx = 2; // Continental upper crust
    } else if (position[0] > x_subd && position[1] >= data->y_continent[1]) {
      region_idx = 3; // Continental lower crust
    } else if (position[0] > x_subd && position[1] >= data->y_continent[2]) {
      region_idx = 5; // Continental lithosphere mante
    } else {
      region_idx = 6; // Asthenosphere
    }
    
    if (position[1] > data->y0) {
      region_idx = 9; // Sticky Air
    }
    
    MPntStdSetField_phase_index(material_point,region_idx);
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeIsostaticTopography(ModelSubductionObliqueCtx *data,PetscReal *dh)
{
  PetscReal      h_s,h_ouc,h_olc,h_uc,h_lc;
  PetscReal      d_s,d_ouc,d_olc,d_uc,d_lc,d_m,d_a;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  /* Compute thickness of the different layers */
  h_s   = data->y0 - data->y_ocean[0] ;
  h_ouc = data->y_ocean[0] - data->y_ocean[1];
  h_olc = data->y_ocean[1] - data->y_ocean[2];
  h_uc  = data->y0 - data->y_continent[0];
  h_lc  = data->y_continent[0] - data->y_continent[1];
  
  /* Get their densities from options */
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_0",&d_ouc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_1",&d_olc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_2",&d_uc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_3",&d_lc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_4",&d_m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_8",&d_s,NULL);CHKERRQ(ierr);
  /* Scale the densities */
  d_s = d_s/data->density_bar;
  d_ouc = d_ouc/data->density_bar;
  d_olc = d_olc/data->density_bar;
  d_uc = d_uc/data->density_bar;
  d_lc = d_lc/data->density_bar;
  d_m = d_m/data->density_bar;
  d_a = data->air_rho/data->density_bar;
  
  *dh = (h_ouc*(d_m-d_ouc) + h_olc*(d_m-d_olc) + h_s*(d_m-d_s) + h_uc*(d_uc-d_m) + h_lc*(d_lc-d_m))/(d_a - d_m);
  
  PetscFunctionReturn(0);
}

/* GEOMETRY: Obliquity through boundary conditions */
PetscErrorCode ModelApplyInitialMaterialGeometry_ObliqueBC(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data = (ModelSubductionObliqueCtx*)ctx;
  DataBucket                db;
  DataField                 PField_std,PField_pls;
  PetscInt                  p,n_mp_points;
  PetscReal                 xc,dy,x_trench;
  PetscReal                 d_ouc,d_olc,d_uc,d_lc,d_air,d_m;
  PetscReal                 h_s,h_ouc,h_olc,h_o;
  PetscReal                 h_uc,h_lc,h_c;
  PetscReal                 xa,ya,xb,yb,xd,yd,xf,yf,xg,yg,xi,yi;
  PetscErrorCode            ierr;
  
  PetscFunctionBegin;
  
  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));
  
  /* Get the densities from options */
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_0",&d_ouc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_1",&d_olc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_2",&d_uc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_3",&d_lc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO, "-rho_4",&d_m,NULL);CHKERRQ(ierr);
  /* Scale the densities */
  d_ouc = d_ouc/data->density_bar;
  d_olc = d_olc/data->density_bar;
  d_uc = d_uc/data->density_bar;
  d_lc = d_lc/data->density_bar;
  d_m = d_m/data->density_bar;
  d_air = data->air_rho/data->density_bar;
  
  xc = (data->Lx + data->Ox)/2.0;
  /* Trench located at centre of the model in x direction */
  x_trench =  xc;
  /* Compute the isostatic subsidence of the ocean */
  ierr = ComputeIsostaticTopography(data,&dy);CHKERRQ(ierr);
  
  /* Compute thickness of the different layers */
  /* Ocean */
  h_s   = data->y0 - data->y_ocean[0] ;
  h_ouc = data->y_ocean[0] - data->y_ocean[1];
  h_olc = data->y_ocean[1] - data->y_ocean[2];
  h_o   = h_s + h_ouc + h_olc;
  /* Continent */
  h_uc  = data->y0 - data->y_continent[0];
  h_lc  = data->y_continent[0] - data->y_continent[1];
  h_c   = h_uc + h_lc;
  /* Prepare intersection points between layers for isostatic equilibrium of the continental margin */
  /* Intersection point between the continental and oceanic plates d(xd,yd) */
  xd = -(-dy - data->y0) / tan(data->theta_subd) + x_trench;
  yd = data->y0 - dy;
  /* Intersection point between the centre of the weak zone and the bottom of the continental crust a(xa,ya) */
  xa = (h_c + data->y0)/tan(data->theta_subd) + x_trench;
  ya = data->y0 - h_c;
  /* Intersection point between the weak zone and the bottom of the continental crust b(xb,yb)*/
  xb = xa + data->wz;
  yb = ya;
  /* Intersection point between the weak zone and the top of the lower continental crust */
  xf = xb - h_lc/tan(data->theta_subd);
  yf = ya + h_lc;
  /* Upper right corner of the weak zone */
  xi = xf - (h_uc - data->y0 - h_o - dy)/tan(data->theta_subd);
  yi = data->y0 - h_o - dy;
  /* Intersection point between the centre of the weak zone, the bottom of the ocean and the continent */
  yg = yi;
  xg = xi - data->wz;
  
  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    PetscReal     x_subd,y_contact,h_contact_bot_cont,h_contact_uc;
    PetscReal     dy_bot_cont,dy_subd_uc,dy_margin;
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    int           region_idx;
    double        *position;
    float         pls;
    short         yield;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access using the getter function */
    MPntStdGetField_global_coord(material_point,&position);
    /* Set an initial small random noise on plastic strain */
    pls = ptatin_RandomNumberGetDouble(0.0,0.03);
    /* Set yield to none */
    yield = 0;
    
    /* Initial weak zone angle */
    x_subd   = -(position[1] - data->y0) / tan(data->theta_subd) + x_trench;
    
    if (position[0] <= x_subd && position[1] >= data->y_ocean[0]-dy) {
      region_idx = 8; // Oceanic sediments
    } else if (position[0] <= x_subd && position[1] >= data->y_ocean[1]-dy) {
      region_idx = 0; // Oceanic upper crust
    } else if (position[0] <= x_subd && position[1] >= data->y_ocean[2]-dy) {
      region_idx = 1; // Oceanic lower crust
    } else if (position[0] <= (x_subd-data->wz) && position[1] >= data->y_ocean[3]-dy) {
      region_idx = 4; // Oceanic lithosphere mantle
    } else if (position[0] >= (x_subd-data->wz)  && 
               position[0] <= (x_subd+data->wz)  &&
               position[1] < data->y_ocean[2]-dy && 
               position[1] >= data->y_continent[2]) {
      region_idx = 7; // Weak zone
    } else if (position[0] > x_subd && position[1] >= data->y_continent[0]) {
      region_idx = 2; // Continental upper crust
    } else if (position[0] > x_subd && position[1] >= data->y_continent[1]) {
      region_idx = 3; // Continental lower crust
    } else if (position[0] > x_subd && position[1] >= data->y_continent[2]) {
      region_idx = 5; // Continental lithosphere mante
    } else {
      region_idx = 6; // Asthenosphere
    }
    
    y_contact = data->y0 - (position[0] - x_trench - data->wz)*tan(data->theta_subd);
    h_contact_bot_cont = data->y_continent[0] - y_contact;
    h_contact_uc       = data->y0 - y_contact;
    dy_bot_cont = (h_contact_bot_cont*(d_m - d_lc) + h_lc*(d_lc - d_m))/(d_air - d_m);
    dy_subd_uc  = (h_contact_uc*(d_m - d_uc) + h_uc*(d_uc - d_m) + h_lc*(d_lc - d_m))/(d_air - d_m);
    dy_margin   = (h_o*(d_m - d_uc) + h_uc*(d_uc - d_m) + h_lc*(d_lc - d_m))/(d_air - d_m);
    
    if (position[0] <= x_subd && position[1] >= data->y0-dy) {
      region_idx = 9; // Sticky Air
    } 
    if (position[0] >= xf && position[0] <= xb && position[1] >= data->y0-dy_bot_cont) {
      region_idx = 9;
    }
    if (position[0] < xf && position[0] >= xi && position[1] >= data->y0-dy_subd_uc){
      region_idx = 9;
    }
    if (position[0] < xi && position[0] > x_subd && position[1] >= data->y0-dy_margin){
      region_idx = 9;
    }
    if (position[0] >= xb && position[1] > data->y0){
      region_idx = 9;
    }
    
    MPntStdSetField_phase_index(material_point,region_idx);
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data = (ModelSubductionObliqueCtx*)ctx;
  PetscErrorCode            ierr;
  
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
  ModelSubductionObliqueCtx                    *data;
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
  data = (ModelSubductionObliqueCtx*)ctx;
  
  stokes_pack = c->stokes_ctx->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  
  /* Velocity IC */
  vz = data->normV*PetscCosReal(data->angle_v);
  vx = PetscSqrtReal(data->normV*data->normV - vz*vz);

  /* Left face */  
  vxl = vx;
  vzl = vz;
  /* Right face */
  vxr = -vx;
  vzr = -vz;
  
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
    
    ierr = PetscOptionsGetBool(NULL,MODEL_NAME_SO,"-temperature_ic_from_file",&subduction_temperature_ic_from_file,NULL);CHKERRQ(ierr);
    if (subduction_temperature_ic_from_file) {
      /* Check if a file is provided */
      ierr = PetscOptionsGetString(NULL,MODEL_NAME_SO,"-temperature_file",temperature_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
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

static PetscErrorCode ModelApplyInitialStokesVariableMarkers_SubductionOblique(pTatinCtx user,Vec X,void *ctx)
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
/*  DataFieldGetAccess(PField);
  for (regionidx=0; regionidx<user->rheology_constants.nphases_active;regionidx++) {
    DataFieldAccessPoint(PField,regionidx,(void**)&truc);
    MaterialConst_MaterialTypeSetField_plastic_type(truc,PLASTIC_MISES);
  }
  DataFieldRestoreAccess(PField);
*/  
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
/*
  ierr = DMCompositeRestoreLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);
  for (regionidx=0; regionidx<user->rheology_constants.nphases_active;regionidx++) {
    DataBucketGetDataFieldByName(user->material_constants,MaterialConst_MaterialType_classname,&PField);
    DataFieldGetAccess(PField);
    for (regionidx=0; regionidx<user->rheology_constants.nphases_active;regionidx++) {
      DataFieldAccessPoint(PField,regionidx,(void**)&truc);
      MaterialConst_MaterialTypeSetField_plastic_type(truc,PLASTIC_DP);
    }
    DataFieldRestoreAccess(PField);
  }
*/  
  PetscFunctionReturn(0);
}

PetscBool ContinentalGeotherm_1DTS(PetscScalar coord[],PetscScalar *value,void *ctx)
{
  ModelSubductionObliqueCtx *data;
  PetscBool                 impose_dirichlet = PETSC_TRUE;
  
  data = (ModelSubductionObliqueCtx*)ctx;
  
  /* Analytical solution for a continental geotherm with heat production from Turcott and Schubert (2014) Geodynamics 3rd Edition p176*/
  *value = data->Ttop + data->qm*(-coord[1])/data->k + data->h_prod*pow(data->y_prod,2)/data->k * (1.0-exp(-coord[1]/data->y_prod));

  if (*value >= data->Tlitho) {
    *value = -((data->Tbottom-data->Tlitho)/(data->Oy-data->y_continent[2])) * (data->y_continent[2]-coord[1]) + data->Tlitho;
  }
  
  return impose_dirichlet;
}

PetscBool OceanicGeotherm_1DTS(PetscScalar coord[],PetscScalar *value,void *ctx)
{
  ModelSubductionObliqueCtx *data;
  PetscBool                 impose_dirichlet = PETSC_TRUE;
  
  data = (ModelSubductionObliqueCtx*)ctx;
  
  *value = (data->Ttop-data->Tbottom)*erfc(-coord[1]/(2.0*PetscSqrtReal(data->age)))+data->Tbottom;
  
  return impose_dirichlet;
}

PetscBool BCListEvaluator_Lithosphere(PetscScalar position[],PetscScalar *value,void *ctx)
{
  PetscBool      impose_dirichlet;
  BC_Lithosphere data_ctx = (BC_Lithosphere)ctx;
  
  PetscFunctionBegin;
  
  if (position[1] >= data_ctx->y_lab) {
    *value = data_ctx->v;
    impose_dirichlet = PETSC_TRUE;
  } else {
    impose_dirichlet = PETSC_FALSE;
  }
  
  return impose_dirichlet;
}

PetscErrorCode SubductionOblique_VelocityBC_Oblique(BCList bclist,DM dav,pTatinCtx c,ModelSubductionObliqueCtx *data)
{
  BC_Lithosphere bcdata;
  PetscReal      vx,vz,vxl,vxr,vzl,vzr;
  PetscReal      zero = 0.0;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  ierr = PetscMalloc(sizeof(struct _p_BC_Lithosphere),&bcdata);CHKERRQ(ierr);

  /* Computing of the 2 velocity component required to get a vector of norm normV and angle angle_v */
  //vx = data->normV*tan(data->angle_v)/sqrt(1+tan(data->angle_v)*tan(data->angle_v));
  //vz = sqrt(data->normV*data->normV-vx*vx);
  vz = data->normV*PetscCosReal(data->angle_v);
  vx = PetscSqrtReal(data->normV*data->normV - vz*vz);

  /* Left face */  
  vxl = vx;
  vzl = vz;
  /* Right face */
  vxr = -vx;
  vzr = -vz;

  bcdata->y_lab = data->y_ocean[3];
  bcdata->v = vxl;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = vzl;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&vxl);CHKERRQ(ierr);
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_constant,(void*)&vzl);CHKERRQ(ierr);
  
  
  bcdata->y_lab = data->y_continent[2];
  bcdata->v = vxr;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = vzr;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  if (data->is_2D) {
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  }
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  /* No slip bottom */
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  ierr = PetscFree(bcdata);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDABCMethod_ContinentalGeotherm(FVDA fv,
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
    PetscBool set = ContinentalGeotherm_1DTS((PetscScalar*)&coor[3*f],&value,ctx);
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    bcvalue[f] = value;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GetInitialContinentalGeotherm_FV(ModelSubductionObliqueCtx *data,
                                                       PetscErrorCode (**func)(FVDA,
                                                                               DACellFace,
                                                                               PetscInt,
                                                                               const PetscReal*,
                                                                               const PetscReal*,
                                                                               const PetscInt*,
                                                                               PetscReal,FVFluxType*,PetscReal*,void*))
{
  PetscFunctionBegin;
  
  /* assign function to use */
  *func = FVDABCMethod_ContinentalGeotherm;
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDABCMethod_OceanicGeotherm(FVDA fv,
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
    PetscBool set = OceanicGeotherm_1DTS((PetscScalar*)&coor[3*f],&value,ctx);
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    bcvalue[f] = value;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GetInitialOceanicGeotherm_FV(ModelSubductionObliqueCtx *data,
                                                   PetscErrorCode (**func)(FVDA,
                                                                           DACellFace,
                                                                           PetscInt,
                                                                           const PetscReal*,
                                                                           const PetscReal*,
                                                                           const PetscInt*,
                                                                           PetscReal,FVFluxType*,PetscReal*,void*))
{
  PetscFunctionBegin;
  
  /* assign function to use */
  *func = FVDABCMethod_OceanicGeotherm;
  
  PetscFunctionReturn(0);
}

static PetscErrorCode InitialSteadyStateGeotherm_BCFromAnalytics(pTatinCtx c,ModelSubductionObliqueCtx *data)
{
  PhysCompEnergyFV energy;
  PetscReal        val_T;
  PetscErrorCode   ierr;
  PetscErrorCode (*initial_continental_geotherm)(FVDA,DACellFace,PetscInt,const PetscReal*,const PetscReal*,const PetscInt*,PetscReal,FVFluxType*,PetscReal*,void*);
  PetscErrorCode (*initial_oceanic_geotherm)(FVDA,DACellFace,PetscInt,const PetscReal*,const PetscReal*,const PetscInt*,PetscReal,FVFluxType*,PetscReal*,void*);

  PetscFunctionBegin;
  
  ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
  
  /* Continental geotherm params */
  data->qm     = 20.0e-3;
  data->k      = 3.3;
  data->h_prod = 1.5e-6;
  data->y_prod = -40.0e3;
  data->Tlitho = 1300.0;
  
  /* Oceanic geotherm params */
  data->age = 100.0;
  
  /* Set from options */
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-heat_source_2",&data->h_prod,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-qm_init",      &data->qm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-k_init",       &data->k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-y_prod_init",  &data->y_prod,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-Tlitho_init",  &data->Tlitho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_SO,"-age_init",     &data->age,NULL);CHKERRQ(ierr);
  
  /* Scale values */
  data->qm     = data->qm / (data->pressure_bar*data->velocity_bar);
  data->k      = data->k / (data->pressure_bar * data->length_bar * data->length_bar / data->time_bar);
  data->h_prod = data->h_prod / (data->pressure_bar / data->time_bar);
  data->y_prod = data->y_prod / data->length_bar;
  data->age    = data->age*3.14e13 / data->time_bar;
  
  /* Get the 1D geotherms to apply as BCs */
  ierr = GetInitialContinentalGeotherm_FV(data,&initial_continental_geotherm);CHKERRQ(ierr);
  ierr = GetInitialOceanicGeotherm_FV(data,&initial_oceanic_geotherm);CHKERRQ(ierr);
  
  val_T = data->Tbottom;
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_S,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);
  
  val_T = data->Ttop;
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_N,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);

  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_E,PETSC_FALSE,0.0,initial_continental_geotherm,data);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_W,PETSC_FALSE,0.0,initial_oceanic_geotherm,data);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_F,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_B,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTimeDependantEnergy_BCs(pTatinCtx c,ModelSubductionObliqueCtx *data)
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

PetscErrorCode ModelApplyBoundaryCondition_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscBool        active_energy;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  data = (ModelSubductionObliqueCtx*)ctx;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Define velocity boundary conditions */
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = SubductionOblique_VelocityBC_Oblique(stokes->u_bclist,dav,c,data);CHKERRQ(ierr);
  /* Insert here the litho pressure functions */
  /* The faces with litho pressure BCs are:
     -  IMIN, IMAX below the lithosphere
     -  KMIN, KMAX on all face 
  */
  
  /* Define boundary conditions for any other physics */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    if (data->subduction_temp_ic_steadystate_analytics) {
      ierr = InitialSteadyStateGeotherm_BCFromAnalytics(c,data);CHKERRQ(ierr);
    } else {
      ierr = SetTimeDependantEnergy_BCs(c,data);CHKERRQ(ierr);
    }
  }
  
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
    ierr = SubductionOblique_VelocityBC_Oblique(bclist[n],dav[n],c,data);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyMaterialBoundaryCondition_SubductionOblique(pTatinCtx c,void *ctx)
{
  ModelSubductionObliqueCtx *data;
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

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelSubductionObliqueCtx*)ctx;
  
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(c,&material_point_db,NULL);CHKERRQ(ierr);

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
    Nxp[0]  = 1;
    Nxp[1]  = 1;
    perturb = 0.1;

    /* reset size */
    DataBucketSetSizes(material_point_face_db,0,-1);

    /* assign coords */
    epsilon = 1.0e-6;
    //ierr = SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d(dav,Nxp,perturb, face_list[f], material_point_face_db);CHKERRQ(ierr);
    ierr = SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(dav,Nxp,perturb,epsilon,face_list[f],material_point_face_db);CHKERRQ(ierr);

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

  /* Copy ALL values from nearest markers to newly inserted markers except (xi,xip,pid) */
  //ierr = MaterialPointRegionAssignment_v1(material_point_db,dav);CHKERRQ(ierr);
  ierr = MaterialPointRegionAssignment_KDTree(material_point_db,PETSC_TRUE);CHKERRQ(ierr);

  /* delete */
  DataBucketDestroy(&material_point_face_db);

  PetscFunctionReturn(0);
}
/* Currently not call in the ModelRegister function*/
PetscErrorCode ModelApplyUpdateMeshGeometry_SubductionOblique(pTatinCtx c,Vec X,void *ctx)
{
  PhysCompStokes  stokes;
  DM              stokes_pack,dav,dap;
  Vec             velocity,pressure;
  PetscInt        npoints,dir;
  PetscReal       step;
  PetscReal       *xref,*xnat;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* fully lagrangian update */
  ierr = pTatinGetTimestep(c,&step);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);

  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

 /* SURFACE REMESHING */
  ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,step);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
 
  /* Update Mesh Refinement */
  dir = 1; // 0 = x; 1 = y; 2 = z;
  npoints = 4;

  ierr = PetscMalloc1(npoints,&xref);CHKERRQ(ierr); 
  ierr = PetscMalloc1(npoints,&xnat);CHKERRQ(ierr); 

  xref[0] = 0.0;
  xref[1] = 0.28; //0.375;
  xref[2] = 0.65; //0.75;
  xref[3] = 1.0;

  xnat[0] = 0.0;
  xnat[1] = 0.8;
  xnat[2] = 0.935;//0.95;
  xnat[3] = 1.0;

  ierr = DMDACoordinateRefinementTransferFunction(dav,dir,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  
  ierr = PetscFree(xref);CHKERRQ(ierr);
  ierr = PetscFree(xnat);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_SubductionOblique(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  ModelSubductionObliqueCtx *data;
  PetscBool                 active_energy;
  DataBucket                materialpoint_db;
  PetscErrorCode            ierr;
  static PetscBool          been_here = PETSC_FALSE;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelSubductionObliqueCtx*)ctx;
  
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
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void))ModelApplyInitialStokesVariableMarkers_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))NULL);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_SubductionOblique);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_SubductionOblique);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(DM da,PetscInt Nxp[],PetscReal perturb, PetscReal epsilon, PetscInt face_idx,DataBucket db)
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