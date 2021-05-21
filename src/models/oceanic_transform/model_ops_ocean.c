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
**    filename:   model_ops_oceanic_transform.c
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

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Developped by Anthony Jourdon [jourdon.anthon@gmail.com] 
  Reproduction of T. Gerya (2013). Physics of the Earth and Planetary Interiors, 214
  without melt
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#include "petsc.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin3d_stokes.h"
#include "ptatin3d_energy.h"
#include "ptatin_models.h"
#include "model_template_ctx.h"

static const char MODEL_NAME_OT[] = "model_ocean_";

PetscErrorCode ModelInitialize_Ocean(pTatinCtx c,void *ctx)
{
  ModelOceanCtx     *data;
  RheologyConstants *rheology;
  PetscReal         crust_thickness;
  PetscReal         cm_per_yer2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 );
  int               source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelOceanCtx*)ctx;
  
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);
  rheology->rheology_type = RHEOLOGY_VP_STD;
  
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);
  
  data->n_phases = 3;
  rheology->nphases_active = data->n_phases;
  rheology->apply_viscosity_cutoff_global = PETSC_TRUE;
  rheology->eta_upper_cutoff_global = 1.0e22;
  rheology->eta_lower_cutoff_global = 1.0e18;
  
  /* set the deffault values of the material constant for this particular model */
  /* scaling */
  data->length_bar    = 1.0e4;
  data->viscosity_bar = 1.0e20;
  data->velocity_bar  = 1.0e-10;
  
  /* cutoff */
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_OT,"-apply_viscosity_cutoff_global",&rheology->apply_viscosity_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-eta_lower_cutoff_global",&rheology->eta_lower_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-eta_upper_cutoff_global",&rheology->eta_upper_cutoff_global,NULL);CHKERRQ(ierr);
  
  /* Box Geometry */
  data->Ox = 0.0;
  data->Lx = 98.0e3;
  data->Oy = -50.0e3;
  data->Ly = 0.0;
  data->Oz = 0.0;
  data->Lz = 98.0e3;
  
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-Ox",&data->Ox,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-Lx",&data->Lx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-Oy",&data->Oy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-Ly",&data->Ly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-Oz",&data->Oz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-Lz",&data->Lz,NULL);CHKERRQ(ierr);
  
  PetscPrintf(PETSC_COMM_WORLD,"******** Box Geometry ********\n");
  PetscPrintf(PETSC_COMM_WORLD,"Ox = %1.2e,     Lx = %1.2e [m]\n",data->Ox,data->Lx);
  PetscPrintf(PETSC_COMM_WORLD,"Oz = %1.2e,     Lz = %1.2e [m]\n",data->Oz,data->Lz);
  PetscPrintf(PETSC_COMM_WORLD,"Oy = %1.2e,     Ly = %1.2e [m]\n",data->Oy,data->Ly);
  
  /* Layering */
  data->ocean_floor = -5.0e3;
  crust_thickness = 7.0e3;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-ocean_floor",&data->ocean_floor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-crust_thickness",&crust_thickness,NULL);CHKERRQ(ierr);
  
  data->moho = data->ocean_floor - crust_thickness;
  
  PetscPrintf(PETSC_COMM_WORLD,"********* Layering *********\n");
  PetscPrintf(PETSC_COMM_WORLD,"Oceanic seafloor: %1.2e [m] \n",data->ocean_floor);
  PetscPrintf(PETSC_COMM_WORLD,"Oceanic Moho    : %1.2e [m] \n",data->moho);
  
  /* Velocity BCs */
  data->v_spreading = 3.8 * cm_per_yer2m_per_sec;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_OT,"-v_spreading",&data->v_spreading,NULL);CHKERRQ(ierr);
  model_thickness = data->Ly - data->Oy;
  rock_thickness  = model_thickness - (data->Ly - data->ocean_floor);
  data->v_top = data->v_spreading*(model_thickness - rock_thickness)/(data->Lx - data->Ox);
  data->v_bot = data->v_spreading*rock_thickness/(data->Lx - data->Ox);
  
  /* Temperature BCs */
  data->T_top = 0.0;
  data->T_bot = 1330.0;
  
  /* Compute additional scaling parameters */
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
  
  /* Scale length */
  data->Lx = data->Lx / data->length_bar;
  data->Ly = data->Ly / data->length_bar;
  data->Lz = data->Lz / data->length_bar;
  data->Ox = data->Ox / data->length_bar;
  data->Oy = data->Oy / data->length_bar;
  data->Oz = data->Oz / data->length_bar;
  
  data->ocean_floor = data->ocean_floor / data->length_bar;
  data->moho = data->moho / data->length_bar;
  
  /* Scale velocity */
  data->v_spreading = data->v_spreading / data->velocity_bar;
  data->v_top       = data->v_top       / data->velocity_bar;
  data->v_bot       = data->v_bot       / data->velocity_bar;
  
  /* Setup and scale rheological parameters */
  ierr = SetRheologicalParameters_Ocean(c,rheology,ctx);CHKERRQ(ierr);
  
  /* Scale viscosity cutoff */
  rheology->eta_lower_cutoff_global = rheology->eta_lower_cutoff_global / data->viscosity_bar;
  rheology->eta_upper_cutoff_global = rheology->eta_upper_cutoff_global / data->viscosity_bar;
  
  PetscFunctionReturn(0);
}

static PetscErrorCode SetRheologicalParameters_Ocean(pTatinCtx c,RheologyConstants rheology,void *ctx)
{
  ModelOceanCtx             *data;
  DataField                 PField,PField_k,PField_Q;
  EnergyConductivityConst   *data_k;
  EnergySourceConst         *data_Q;
  DataBucket                materialconstants;
  EnergyMaterialConstants   *matconstants_e;
  PetscInt                  region_idx;
  PetscReal                 *preexpA,*Ascale,*entalpy,*Vmol,*nexp,*Tref;
  PetscReal                 *phi,*phi_inf,*Co,*Co_inf,*Tens_cutoff,*Hst_cutoff,*eps_min,*eps_max;
  PetscReal                 *beta,*alpha,*rho,*heat_source,*conductivity;
  PetscReal                 phi_rad,phi_inf_rad,Cp;
  char                      *option_name;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelOceanCtx*)ctx;
  
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
  
  source_type[0] = ENERGYSOURCE_NONE;
  Cp             = 800.0;
  
  for (region_idx=0;region_idx<rheology->nphases_active-1;region_idx++) {
    /* Set material constitutive laws */
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_ARRHENIUS_2,PLASTIC_DP_TENS,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);

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

  /* region_idx 3 --> Water */
  region_idx = 3;
  MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
  MaterialConstantsSetValues_ViscosityConst(materialconstants,region_idx,rheology->eta_lower_cutoff_global);
  MaterialConstantsSetValues_DensityConst(materialconstants,region_idx,1000.0);
  MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,0.0,0.0,1.0,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
  EnergyConductivityConstSetField_k0(&data_k[region_idx],1.0);
  EnergySourceConstSetField_HeatSource(&data_Q[region_idx],0.0);

  /* Report all material parameters values */
  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsPrintAll(materialconstants,region_idx);
    MaterialConstantsEnergyPrintAll(materialconstants,region_idx);
  }
  
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

PetscErrorCode ModelApplyInitialMeshGeometry_Ocean(pTatinCtx c,void *ctx)
{
  ModelOceanCtx *data;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data = (ModelOceanCtx*)ctx;
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);CHKERRQ(ierr);
  
  PetscReal gvec[] = { 0.0, -9.8, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(c->stokes_ctx,data->acceleration_bar);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_Ocean(pTatinCtx c,void *ctx)
{
  ModelOceanCtx  *data;
  DataBucket     db;
  DataField      PField_std,PField_pls;
  PetscInt       p,n_mp_points;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelOceanCtx*)ctx;
  
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));
  
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
    
    if (position[1] <= data->Ly) {
      region_idx = 3; // Water
    }
    if (position[1] <= data->ocean_floor) {
      region_idx = 0; // Oceanic crust
    }
    if (position[1] < data->moho) {
      region_idx = 2; // Mantle
    }
    
    yield = 0;
    
    MPntStdSetField_phase_index(material_point,region_idx);
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialSolution_Ocean(pTatinCtx c,Vec X,void *ctx)
{
  /* ModelOceanCtx *data; */

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* data = (ModelOceanCtx*)ctx; */

  PetscFunctionReturn(0);
}

PetscErrorCode Ocean_VelocityBC(BCList bclist,DM dav,pTatinCtx c,ModelOceanCtx *data)
{
  ModelOceanCtx *data;
  PetscReal     vxl,vxr,zero = 0.0;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelOceanCtx*)ctx;
  
  vxl = -0.5*data->v_spreading;
  vxr =  0.5*data->v_spreading;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&vxl);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&vxr);CHKERRQ(ierr);
  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMAX_LOC,1,BCListEvaluator_constant,(void*)&data->v_top);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&data->v_bot);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryCondition_Ocean(pTatinCtx c,void *ctx)
{
  ModelOceanCtx    *data;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PhysCompEnergyFV energy;
  PetscReal        val_T;
  PetscInt         l;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  data = (ModelOceanCtx*)ctx;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  PetscPrintf(PETSC_COMM_WORLD,"param1 = %lf \n", data->param1 );

  /* Define velocity boundary conditions */
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = Ocean_VelocityBC(stokes->u_bclist,dav,c,data);CHKERRQ(ierr);

  /* Define boundary conditions for any other physics */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
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
    /*
    const DACellFace flist[] = { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_B, DACELL_FACE_F };
    for (l=0; l<sizeof(flist)/sizeof(DACellFace); l++) {
      ierr = FVSetDirichletFromInflow(energy->fv,energy->T,flist[l]);CHKERRQ(ierr);
    }
    */
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_Ocean(PetscInt nl,BCList bclist[],DM dav[],pTatinCtx c,void *ctx)
{
  ModelOceanCtx *data;
  PetscInt         n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data = (ModelOceanCtx*)ctx;
  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = Ocean_VelocityBC(bclist[n],dav[n],c,data);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyMaterialBoundaryCondition_Ocean(pTatinCtx c,void *ctx)
{
  /* ModelOceanCtx *data; */

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* data = (ModelOceanCtx*)ctx; */

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyUpdateMeshGeometry_Ocean(pTatinCtx c,Vec X,void *ctx)
{
  /* ModelOceanCtx *data; */

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* data = (ModelOceanCtx*)ctx; */

  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_Ocean(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  /* ModelOceanCtx *data; */

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* data = (ModelOceanCtx*)ctx; */


  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_Ocean(pTatinCtx c,void *ctx)
{
  ModelOceanCtx *data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelOceanCtx*)ctx;

  /* Free contents of structure */

  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_Ocean(void)
{
  ModelOceanCtx *data;
  pTatinModel      m;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelOceanCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelOceanCtx));CHKERRQ(ierr);

  /* set initial values for model parameters */
  data->param1 = 0.0;
  data->param2 = 0;

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"ocean");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_Ocean);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_Ocean);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
