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
 **    filename:   model_ops_gene3d.c
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
   Developed by Anthony Jourdon [anthony.jourdon@sorbonne-universite.fr]
 */

#include "petsc.h"

#include "ptatin3d.h"
#include "private/ptatin_impl.h"

#include "dmda_bcs.h"
#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPEnergy_def.h"
#include "cartgrid.h"
#include "material_point_popcontrol.h"

#include "ptatin3d_energy.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include <material_constants_energy.h>

#include "model_gene3d_ctx.h"

const char MODEL_NAME[] = "model_GENE3D_";

static PetscErrorCode ModelSetInitialGeometryFromOptions(ModelGENE3DCtx *data)
{
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Origin */
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-Ox",&data->O[0], &found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER,"Expected user to provide model origin -%sOx \n",MODEL_NAME); }
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-Oy",&data->O[1], &found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER,"Expected user to provide model origin -%sOy \n",MODEL_NAME); }
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-Oz",&data->O[2], &found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER,"Expected user to provide model origin -%sOz \n",MODEL_NAME); }

  /* Length */
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-Lx",&data->L[0], &found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER,"Expected user to provide model origin -%sLx \n",MODEL_NAME); }
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-Ly",&data->L[1], &found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER,"Expected user to provide model origin -%sLy \n",MODEL_NAME); }
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-Lz",&data->L[2], &found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER,"Expected user to provide model origin -%sLz \n",MODEL_NAME); }

  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"************* Box Geometry *************\n");
  PetscPrintf(PETSC_COMM_WORLD,"( Ox , Lx ) = ( %+1.4e [m], %+1.4e [m] )\n", data->O[0] ,data->L[0] );
  PetscPrintf(PETSC_COMM_WORLD,"( Oy , Ly ) = ( %+1.4e [m], %+1.4e [m] )\n", data->O[1] ,data->L[1] );
  PetscPrintf(PETSC_COMM_WORLD,"( Oz , Lz ) = ( %+1.4e [m], %+1.4e [m] )\n", data->O[2] ,data->L[2] );
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetRegionParametersFromOptions_Energy(DataBucket materialconstants, const int region_idx)
{
  DataField                 PField,PField_k,PField_Q;
  EnergyConductivityConst   *data_k;
  EnergySourceConst         *data_Q;
  EnergyMaterialConstants   *matconstants_e;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscReal                 alpha,beta,rho,heat_source,conductivity,Cp;
  char                      *option_name;
  PetscErrorCode            ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Energy material constants */
  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  /* Conductivity */
  DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
  DataFieldGetEntries(PField_k,(void**)&data_k);
  
  /* Heat source */
  DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
  DataFieldGetEntries(PField_Q,(void**)&data_Q);

  /* Set default values for parameters */
  source_type[0] = ENERGYSOURCE_CONSTANT;
  source_type[1] = ENERGYSOURCE_SHEAR_HEATING;

  /* Set material energy parameters from options file */
  Cp = 800.0;
  if (asprintf (&option_name, "-heatcapacity_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&Cp,NULL);CHKERRQ(ierr);
  free (option_name);

  alpha = 0.0;
  if (asprintf (&option_name, "-thermalexpension_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&alpha,NULL);CHKERRQ(ierr);
  free (option_name);

  beta = 0.0;
  if (asprintf (&option_name, "-compressibility_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&beta,NULL);CHKERRQ(ierr);
  free (option_name);

  rho = 1.0;
  if (asprintf (&option_name, "-density_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&rho,NULL);CHKERRQ(ierr);
  free (option_name);
  
  heat_source = 0.0;
  if (asprintf (&option_name, "-heat_source_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&heat_source,NULL);CHKERRQ(ierr);
  free (option_name);

  conductivity = 1.0;
  if (asprintf (&option_name, "-conductivity_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&conductivity,NULL);CHKERRQ(ierr);
  free (option_name);

  /* Set energy params for region_idx */
  MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha,beta,rho,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
  EnergySourceConstSetField_HeatSource(&data_Q[region_idx],heat_source);
  EnergyConductivityConstSetField_k0(&data_k[region_idx],conductivity);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParametersFromOptions(pTatinCtx ptatin, DataBucket materialconstants, ModelGENE3DCtx *data)
{
  PetscInt       n,region_idx;
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Get user mesh file */
  ierr = PetscSNPrintf(data->mesh_file,PETSC_MAX_PATH_LEN-1,"md.bin");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,"-mesh_file",data->mesh_file,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%smesh_file not found!\n",MODEL_NAME); }

  /* Get user regions file */
  ierr = PetscSNPrintf(data->region_file,PETSC_MAX_PATH_LEN-1,"region_cell.bin");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,"-regions_file",data->region_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%sregions_file not found!\n",MODEL_NAME); }

  /* Get the number of regions */
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-n_regions",&data->n_regions,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%sn_regions not found!\n",MODEL_NAME); }

  /* Allocate an array to hold the regions indices */
  ierr = PetscCalloc1(data->n_regions,&data->regions_table);CHKERRQ(ierr);
  /* Get user regions indices */
  nn = data->n_regions;
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,"-regions_list",data->regions_table,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != data->n_regions) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"n_regions (%d) and the number of entries in regions_list (%d) mismatch!\n",data->n_regions,nn);
    }
  }

  /* Set regions parameters */
  for (n = 0; n < data->nmaterials; n++) {
    /* get regions index */
    region_idx = data->regions_table[n];
    PetscPrintf(PETSC_COMM_WORLD,"[[ SETTING REGION ]]: %d\n",region_idx);
    ierr = MaterialConstantsSetFromOptions(materialconstants,MODEL_NAME,region_idx,PETSC_TRUE);CHKERRQ(ierr);
    ierr = ModelSetRegionParametersFromOptions_Energy(materialconstants,region_idx);CHKERRQ(ierr);
  }
  /* Report all material parameters values */
  for (n=0; n<data->nmaterials; n++) {
    /* get regions index */
    region_idx = data->regions_table[n];
    ierr = MaterialConstantsPrintAll(materialconstants,region_idx);CHKERRQ(ierr);
    ierr = MaterialConstantsEnergyPrintAll(materialconstants,region_idx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetViscosityCutoffFromOptions(ModelGENE3DCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->eta_cutoff = PETSC_TRUE;
  data->eta_max = 1.0e+25;
  data->eta_min = 1.0e+19;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-apply_viscosity_cutoff",&data->eta_cutoff,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-eta_lower_cutoff",      &data->eta_min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-eta_upper_cutoff",      &data->eta_max,NULL);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetSPMParametersFromOptions(ModelGENE3DCtx *data)
{
  PetscBool      found,flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->surface_diffusion = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-apply_surface_diffusion",&data->surface_diffusion,&found);CHKERRQ(ierr);
  if (found) {
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-diffusivity_spm",&data->diffusivity_spm,&flg);CHKERRQ(ierr);
    if (!flg) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface diffusion activated but no diffusivity provided. Use -%sdiffusivity_spm to set it.\n",MODEL_NAME); }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetPassiveMarkersSwarmParametersFromOptions(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  PSwarm         pswarm;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->passive_markers = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-apply_passive_markers",&data->passive_markers,&found);CHKERRQ(ierr);
  if (!found) { PetscFunctionReturn(0); }

  ierr = PSwarmCreate(PETSC_COMM_WORLD,&pswarm);CHKERRQ(ierr);
  ierr = PSwarmSetOptionsPrefix(pswarm,"passive_");CHKERRQ(ierr);
  ierr = PSwarmSetPtatinCtx(pswarm,ptatin);CHKERRQ(ierr);
  ierr = PSwarmSetTransportModeType(pswarm,PSWARM_TM_LAGRANGIAN);CHKERRQ(ierr);

  ierr = PSwarmSetFromOptions(pswarm);CHKERRQ(ierr);

  /* Copy reference into model data for later use in different functions */
  data->pswarm = pswarm;

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetScalingParametersFromOptions(ModelGENE3DCtx *data)
{
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Assume scaling factors based on typical length, viscosity and velocity of long-term geodynamic systems */
  data->length_bar     = 1.0e+5;
  data->viscosity_bar  = 1.0e+22;
  data->velocity_bar   = 1.0e-10;

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-length_scale",&data->length_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for length provided, assuming %1.4e. You can change it with the option -%slength_scale\n",data->length_bar,MODEL_NAME); }

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-viscosity_scale",&data->viscosity_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for viscosity provided, assuming %1.4e. You can change it with the option -%sviscosity_scale\n",data->viscosity_bar,MODEL_NAME); }

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-velocity_scale",&data->velocity_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for velocity provided, assuming %1.4e. You can change it with the option -%svelocity_scale\n",data->velocity_bar,MODEL_NAME); }

  /* Compute additional scaling parameters */
  data->time_bar         = data->length_bar / data->velocity_bar;
  data->pressure_bar     = data->viscosity_bar/data->time_bar;
  data->density_bar      = data->pressure_bar * (data->time_bar*data->time_bar)/(data->length_bar*data->length_bar); // kg.m^-3
  data->acceleration_bar = data->length_bar / (data->time_bar*data->time_bar);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelScaleParameters(DataBucket materialconstants, ModelGENE3DCtx *data)
{
  PetscInt  region_idx,i;
  PetscReal cm_per_year2m_per_sec,Myr2sec;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* scaling values */
  cm_per_year2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 );
  Myr2sec               = 1.0e6 * ( 365.0 * 24.0 * 3600.0 );
  
  /* Scale viscosity cutoff */
  data->eta_max /= data->viscosity_bar;
  data->eta_min /= data->viscosity_bar;
  /* Scale length */
  for (i=0; i<3; i++) { 
    data->L[i] /= data->length_bar;
    data->O[i] /= data->length_bar;
  }

  data->diffusivity_spm /= (data->length_bar*data->length_bar/data->time_bar);

  /* scale material properties */
  for (region_idx=0; region_idx<data->n_phases; region_idx++) {
    ierr = MaterialConstantsScaleAll(materialconstants,region_idx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);CHKERRQ(ierr);
    ierr = MaterialConstantsEnergyScaleAll(materialconstants,region_idx,data->length_bar,data->time_bar,data->pressure_bar);CHKERRQ(ierr);
  }

  PetscPrintf(PETSC_COMM_WORLD,"[Rift Nitsche Model]:  during the solve scaling is done using \n");
  PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n",       data->length_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n",  data->velocity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n",       data->time_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n",    data->viscosity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->density_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n",      data->pressure_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  a*    : %1.4e [m.s^-2]\n",  data->acceleration_bar );

  PetscFunctionReturn(0);
}
#if 0
static PetscErrorCode ModelSetBottomFlowFromOptions(ModelGENE3DCtx *data)
{
  PetscBool      found,flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->u_dot_n_flow = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-u_dot_n_bottomflow",&data->u_dot_n_flow,&found);CHKERRQ(ierr);
  if (!found) {
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-uy_bot",&data->u_bc[6*HEX_FACE_Neta + 1],&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"-uy_bot not found. You can provide either -%su_dot_n_bottomflow to automatically set the base velocity or -%suy_bot to set it directly.\n",MODEL_NAME,MODEL_NAME);
    }
  }

  PetscFunctionReturn(0);
}
#endif
PetscErrorCode ModelInitialize_Gene3D(pTatinCtx ptatin, void *ctx)
{
  ModelGENE3DCtx    *data = (ModelGENE3DCtx*)ctx;
  RheologyConstants *rheology;
  DataBucket        materialconstants;
  PetscBool         flg, found;
  char              *option_name;
  PetscInt          i,nphase;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinGetRheology(ptatin,&rheology);CHKERRQ(ierr);
  ierr = pTatinGetMaterialConstants(ptatin,&materialconstants);CHKERRQ(ierr);

  /* Box geometry */
  ierr = ModelSetInitialGeometryFromOptions(data);CHKERRQ(ierr);
  /* Material type */
  ierr = ModelSetMaterialParametersFromOptions(ptatin,materialconstants,data);CHKERRQ(ierr);
  ierr = ModelSetViscosityCutoffFromOptions(data);CHKERRQ(ierr);
  /* Surface processes */
  ierr = ModelSetSPMParametersFromOptions(data);CHKERRQ(ierr);
  /* Passive markers */
  ierr = ModelSetPassiveMarkersSwarmParametersFromOptions(ptatin,data);CHKERRQ(ierr);
  /* Scaling */
  ierr = ModelSetScalingParametersFromOptions(data);CHKERRQ(ierr);
  ierr = ModelScaleParameters(materialconstants,data);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}

static PetscErrorCode ModelApplyMeshRefinement(DM dav)
{
  PetscInt  d,ndir;
  PetscInt  *dir;
  PetscBool refine,found;
  PetscFunctionBegin;

  refine = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-apply_mesh_refinement",&refine,NULL);CHKERRQ(ierr);
  if (!refine) { PetscFunctionReturn(0); }

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ndir = 0;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-n_refinement_dir",&ndir,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"-%sn_refinement_dir not found!",MODEL_NAME); }
  if (ndir <= 0 || ndir > 3) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"n_refinement_dir cannot be 0 or more than 3. -%sn_refinement_dir = %d.",MODEL_NAME,ndir); }

  PetscPrintf(PETSC_COMM_WORLD,"Mesh is refined in %d directions.\n")

  ierr = PetscCalloc1(ndir,dir);CHKERRQ(ierr);
  nn = ndir;
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,"-refinement_dir",dir,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != ndir) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"n_refinement_dir (%d) and the number of entries in refinement_dir (%d) mismatch!\n",ndir,nn);
    }
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"-%srefinement_dir not found!",MODEL_NAME);
  }

  for (d=0; d<ndir; d++) {
    PetscInt  dim,npoints;
    PetscReal *xref,*xnat;
    char      option_name[PETSC_MAX_PATH_LEN];

    dim = dir[d];

    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-refinement_npoints_%d",dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&npoints,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name); }

    /* Allocate arrays for xref and xnat */
    ierr = PetscCalloc1(npoints,&xref);CHKERRQ(ierr); 
    ierr = PetscCalloc1(npoints,&xnat);CHKERRQ(ierr); 

    /* Get xref */
    nn = npoints;
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-refinement_xref_%d",dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,option_name,xref,&nn,&found);CHKERRQ(ierr);
    if (found) {
      if (nn != npoints) {
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"-refinement_npoints_%d (%d) and the number of entries in refinement_xref_%d (%d) mismatch!\n",dim,npoints,dim,nn);
      }
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name);
    }

    /* Get xnat */
    nn = npoints;
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-refinement_xnat_%d",dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,option_name,xnat,&nn,&found);CHKERRQ(ierr);
    if (found) {
      if (nn != npoints) {
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"-refinement_npoints_%d (%d) and the number of entries in refinement_xnat_%d (%d) mismatch!\n",dim,npoints,dim,nn);
      }
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name);
    }

    /* Apply mesh refinement */
    ierr = DMDACoordinateRefinementTransferFunction(dav,dim,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);

    ierr = PetscFree(xref);CHKERRQ(ierr);
    ierr = PetscFree(xnat);CHKERRQ(ierr);
  }

  ierr = PetscFree(dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometry_Gene3D(pTatinCtx ptatin,void *ctx)
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  PetscReal      gvec[] = { 0.0, -9.8, 0.0 };
  PetscInt       nn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = DMDASetUniformCoordinates(ptatin->stokes_ctx->dav, data->O[0], data->L[0], data->O[1], data->L[1], data->O[2], data->L[2]); CHKERRQ(ierr);
  ierr = ModelApplyMeshRefinement(ptatin->stokes_ctx->dav);CHKERRQ(ierr);
  
  /* Gravity */
  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL, MODEL_NAME,"-gravity_vec",gvec,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%sgravity_vec requires 3 arguments.",MODEL_NAME);
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]]: Option -%sgravity_vec not provided, assuming gravity = ( %1.4e, %1.4e, %1.4e )\n",MODEL_NAME,gvec[0],gvec[1],gvec[2]);
  }

  ierr = PhysCompStokesSetGravityVector(ptatin->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(ptatin->stokes_ctx,1.0/data->acceleration_bar);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}

static PetscErrorCode ModelApplySurfaceRemeshing(DM dav, PetscReal dt, ModelGENE3DCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dirichlet_xmin = PETSC_FALSE;
  dirichlet_xmax = PETSC_FALSE;
  dirichlet_zmin = PETSC_FALSE;
  dirichlet_zmax = PETSC_FALSE;

  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-spm_diffusion_dirichlet_xmin",dirichlet_xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-spm_diffusion_dirichlet_xmax",dirichlet_xmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-spm_diffusion_dirichlet_zmin",dirichlet_zmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-spm_diffusion_dirichlet_zmax",dirichlet_zmax,NULL);CHKERRQ(ierr);

  if ( !dirichlet_xmin && !dirichlet_xmax && !dirichlet_zmin && !dirichlet_zmax ) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"No boundary conditions provided for the surface diffusion (spm)! Use at least one of -%sspm_diffusion_dirichlet_{xmin,xmax,zmin,zmax}",MODEL_NAME);
  }

  /* Dirichlet velocity imposed on z normal faces so we do the same here */
  ierr = UpdateMeshGeometry_ApplyDiffusionJMAX(dav,data->diffusivity_spm,dt,dirichlet_xmin,dirichlet_xmax,dirichlet_zmin,dirichlet_zmax,PETSC_FALSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyUpdateMeshGeometry_Gene3D(pTatinCtx ptatin,Vec X,void *ctx)
{
  ModelRiftNitscheCtx *data;
  PhysCompStokes      stokes;
  DM                  stokes_pack,dav,dap;
  Vec                 velocity,pressure;
  PetscReal           dt;
  PetscErrorCode      ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftNitscheCtx*)ctx;
  
  /* fully lagrangian update */
  ierr = pTatinGetTimestep(ptatin,&dt);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);

  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /* SURFACE REMESHING */
  if (data->surface_diffusion) {
    ierr = ModelApplySurfaceRemeshing(dav,dt,data);CHKERRQ(ierr);
  }

  ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,dt);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
 
  /* Update Mesh Refinement */
  ierr = ModelSetMeshRefinement_RiftNitsche(dav,data);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);

  /* Passive markers update */
  if (data->passive_markers) { ierr = PSwarmFieldUpdateAll(data->pswarm);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}



















PetscErrorCode ModelApplyBoundaryCondition_Gene3D(pTatinCtx user,void *ctx)
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  PetscScalar zero = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  switch (data->boundary_conditon_type)
  {
    case GENEBC_FreeSlip:
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMIN_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMAX_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMIN_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMAX_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_KMIN_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_KMAX_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      break;

    case GENEBC_NoSlip:
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMIN_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMIN_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMIN_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMAX_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMAX_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMAX_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMIN_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMIN_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMIN_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMAX_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMAX_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMAX_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_KMIN_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_KMIN_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_KMIN_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_KMAX_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_KMAX_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_KMAX_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      break;

    case GENEBC_FreeSlipFreeSurface:
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMIN_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMAX_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMIN_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMAX_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      break;

    case GENEBC_NoSlipFreeSurface:
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMIN_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMIN_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMIN_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMAX_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMAX_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_IMAX_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMIN_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMIN_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMIN_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);

      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMAX_LOC, 0,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMAX_LOC, 1,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      ierr = DMDABCListTraverse3d(user->stokes_ctx->u_bclist,user->stokes_ctx->dav, DMDABCList_JMAX_LOC, 2,BCListEvaluator_constant, (void *) &zero);CHKERRQ(ierr);
      break;
    default:
      break;
  }

  /*
     {
     BCList flat;

     ierr = BCListFlattenedCreate(user->stokes_ctx->u_bclist,&flat);CHKERRQ(ierr);
     ierr = BCListDestroy(&user->stokes_ctx->u_bclist);CHKERRQ(ierr);
     user->stokes_ctx->u_bclist = flat;
     }
     */
  PetscFunctionReturn (0);
}

PetscErrorCode ModelAdaptMaterialPointResolution_Gene3D(pTatinCtx c,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);
  PetscPrintf(PETSC_COMM_WORLD, "  NO MARKER INJECTION ON FACES \n", PETSC_FUNCTION_NAME);
  /* Perform injection and cleanup of markers */
  ierr = MaterialPointPopulationControl_v1(c);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}



//=====================================================================================================================================

PetscErrorCode ModelSetMarkerIndexLayeredCake_Gene3D (pTatinCtx c,void *ctx)
  /* define phase index on material points from a map file extruded in z direction */
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  PetscInt i, nLayer;
  int p,n_mp_points;
  DataBucket db;
  DataField PField_std;
  int phase;
  PetscInt phaseLayer[LAYER_MAX];
  PetscScalar YLayer[LAYER_MAX + 1];
  char *option_name;
  PetscErrorCode ierr;
  PetscBool flg;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db, MPntStd_classname, &PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std, sizeof (MPntStd));
  DataBucketGetSizes(db, &n_mp_points, 0, 0);

  /* read layers from options */
  nLayer = 1;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME, "-nlayer", &nLayer, &flg);CHKERRQ(ierr);
  YLayer[0] = data->O[1];
  for (i=1; i<=nLayer; i++) {

    if (asprintf (&option_name, "-layer_y_%d", i) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name, &YLayer[i], &flg);CHKERRQ(ierr);
    if (flg == PETSC_FALSE) {
      /* NOTE - these error messages are useless if you don't include "model_Gene3D_" in the statement. I added &name[1] so that the "-" is skipped */
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER,"Expected user to provide value to option -model_Gene3D_%s \n",&option_name[1]);
    }
    free (option_name);

    if (YLayer[i] > YLayer[i-1]) {
      if (asprintf (&option_name, "-layer_phase_%d", i-1) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
      ierr = PetscOptionsGetInt(NULL,MODEL_NAME, option_name, &phaseLayer[i-1],&flg);CHKERRQ(ierr);

      if (flg == PETSC_FALSE) {
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP,"Expected user to provide value to option -model_Gene3D_%s \n",&option_name[1]);
      }
      free (option_name);
    } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,"Layers must be entered so that layer_y[i] is larger than layer_y[i-1]\n");
    }
  }

  for (i=0; i<nLayer; i++) {
    PetscPrintf(PETSC_COMM_WORLD,"layer [%D] :  y coord range [%1.2e -- %1.2e] : phase index [%D] \n", i,YLayer[i],YLayer[i+1],phaseLayer[i]);
  }

  for (p = 0; p < n_mp_points; p++) {
    MPntStd *material_point;
    double *pos;

    DataFieldAccessPoint(PField_std, p, (void **) &material_point);
    MPntStdGetField_global_coord(material_point,&pos);

    phase = -1;
    for (i=0; i<nLayer; i++) {
      if ( (pos[1] >= YLayer[i]) && (pos[1] <= YLayer[i+1]) ) {
        phase = phaseLayer[i];
        break;
      }
    }
    /* check if the break was never executed */
    if (i==nLayer) {
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Unable to detect layer containing the marker with coordinates (%1.4e,%1.4e,%1.4e)",pos[0],pos[1],pos[2]);
    }

    /* user the setters provided for you */
    MPntStdSetField_phase_index(material_point, phase);
  }

  DataFieldRestoreAccess(PField_std);
  PetscFunctionReturn (0);
}

//===============================================================================================================================
PetscErrorCode ModelSetMarkerIndexFromMap_Gene3D(pTatinCtx c,void *ctx)
  /* define phase index on material points from a map file extruded in z direction */
{
  PetscErrorCode ierr;
  CartGrid phasemap;
  PetscInt dir_0,dir_1,direction;
  DataBucket db;
  int p,n_mp_points;
  DataField PField_std;
  int phase_init, phase, phase_index;
  char map_file[PETSC_MAX_PATH_LEN], *name;
  PetscBool flg,phasefound;


  PetscFunctionBegin;
  PetscPrintf (PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = PetscOptionsGetString(NULL,MODEL_NAME,"-map_file",map_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (flg == PETSC_FALSE) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Expected user to provide a map file \n");
  }
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-extrude_dir",&direction,&flg);CHKERRQ(ierr);
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
           SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"-extrude_dir %d not valid",direction);
  }

  if (asprintf(&name,"./inputdata/%s.pmap",map_file) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = CartGridCreate(&phasemap);CHKERRQ(ierr);
  ierr = CartGridSetFilename(phasemap,map_file);CHKERRQ(ierr);
  ierr = CartGridSetUp(phasemap);CHKERRQ(ierr);
  free(name);

  if (asprintf(&name,"./inputdata/%s_phase_map.gp",map_file) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = CartGridViewPV(phasemap,name);CHKERRQ(ierr);
  free(name);


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

    MPntStdGetField_phase_index(material_point, &phase_init);

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

  PetscFunctionReturn (0);
}


//======================================================================================================================================

PetscErrorCode ModelSetInitialStokesVariableOnMarker_Gene3D(pTatinCtx c,void *ctx)
  /* define properties on material points */
{
  int p, n_mp_points;
  DataBucket db;
  DataField PField_std, PField_stokes;
  int phase_index;
  RheologyConstants *rheology;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  rheology = &c->rheology_constants;
  db = c->materialpoint_db;

  DataBucketGetDataFieldByName(db, MPntStd_classname, &PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std, sizeof (MPntStd));
  DataBucketGetDataFieldByName (db, MPntPStokes_classname, &PField_stokes);
  DataFieldGetAccess(PField_stokes);
  DataFieldVerifyAccess(PField_stokes, sizeof (MPntPStokes));

  DataBucketGetSizes(db, &n_mp_points, 0, 0);

  for (p = 0; p < n_mp_points; p++)
  {
    MPntStd *material_point;
    MPntPStokes *mpprop_stokes;

    DataFieldAccessPoint(PField_std, p, (void **) &material_point);
    DataFieldAccessPoint(PField_stokes, p, (void **) &mpprop_stokes);
    MPntStdGetField_phase_index (material_point, &phase_index);
    MPntPStokesSetField_eta_effective(mpprop_stokes,rheology->const_eta0[phase_index]);
    MPntPStokesSetField_density(mpprop_stokes,rheology->const_rho0[phase_index]);
  }

  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_stokes);

  PetscFunctionReturn (0);
}

//======================================================================================================================================

PetscErrorCode ModelGene3DInit(DataBucket db)
{
  int                p,n_mp_points;
  DataField          PField_std;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetSizes(db,&n_mp_points,0,0);

  for (p=0; p<n_mp_points; p++) {
    int phase_index;
    MPntStd *material_point;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);

    MPntStdGetField_phase_index(material_point,&phase_index);
    phase_index = -1;
    MPntStdSetField_phase_index(material_point,phase_index);
  }
  DataFieldRestoreAccess(PField_std);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelGene3DCheckPhase(DataBucket db,RheologyConstants *rheology)
{
  int                p,n_mp_points;
  DataField          PField_std;

  PetscFunctionBegin;
  DataBucketGetDataFieldByName (db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetSizes(db,&n_mp_points,0,0);

  for (p=0; p<n_mp_points; p++) {
    int phase_index;
    double *pos;
    MPntStd *material_point;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);

    MPntStdGetField_phase_index(material_point,&phase_index);
    MPntStdGetField_global_coord(material_point,&pos);
    if (phase_index<0) {
      PetscPrintf(PETSC_COMM_SELF,"Phase of marker %D is uninitialized. Marker coor (%1.4e,%1.4e,%1.4e)\n",p,pos[0],pos[1],pos[2]);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Marker phase is uninitialized");
    }
    if (phase_index>=rheology->nphases_active) {
      PetscPrintf(PETSC_COMM_SELF,"Phase of marker %D is larger than rheo->nphases_active = %D. Marker coor (%1.4e,%1.4e,%1.4e)\n",p,rheology->nphases_active, pos[0],pos[1],pos[2]);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Marker phase is undefined");
    }
  }
  DataFieldRestoreAccess(PField_std);

  PetscFunctionReturn(0);
}

/*

   These models are required to
   1) set the value c->rheology_constants->nphases_active => although this seems to be done in ModelInitialize_Gene3D()
   2) all markers are assigned a phase index between [0 -- nphases_active-1]

*/
PetscErrorCode ModelApplyInitialMaterialGeometry_Gene3D(pTatinCtx c,void *ctx)
{
  PetscErrorCode ierr;
  ModelGENE3DCtx *data = (ModelGENE3DCtx *)ctx;

  PetscFunctionBegin;

  /* initalize all phase indices to -1 */
  ierr = ModelGene3DInit(c->materialpoint_db);CHKERRQ(ierr);
  switch (data->initial_geom)
  {
    /*Layered cake */
    case 0:
      {
        ierr = ModelSetMarkerIndexLayeredCake_Gene3D(c,ctx);CHKERRQ(ierr);
      }
      break;
      /*Extrude from Map along Z */
    case 1:
      {
        ierr = ModelSetMarkerIndexFromMap_Gene3D(c,ctx);CHKERRQ(ierr);
      }
      break;
      /*Read from CAD file */
    case 2:
      {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Reading from CAD is not implemented yet \n");
      }
      break;
  }
  /* check all phase indices are between [0---rheo->max_phases-1] */
  ierr = ModelGene3DCheckPhase(c->materialpoint_db,&c->rheology_constants);CHKERRQ(ierr);

  ierr = ModelSetInitialStokesVariableOnMarker_Gene3D(c, ctx);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}


//======================================================================================================================================


PetscErrorCode ModelOutput_Gene3D(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  ierr = pTatin3d_ModelOutput_MPntStd(c,prefix); CHKERRQ(ierr);

  PetscFunctionReturn (0);
}

PetscErrorCode ModelDestroy_Gene3D(pTatinCtx c,void *ctx)
{
  ModelGENE3DCtx *data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelGENE3DCtx*)ctx;

  /* Free contents of structure */
  ierr = PetscFree(data->regions_table);
  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}

PetscErrorCode pTatinModelRegister_Gene3D(void)
{
  ModelGENE3DCtx *data;
  pTatinModel    m;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelGENE3DCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelGENE3DCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"Gene3D");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr =  pTatinModelSetFunctionPointer(m, PTATIN_MODEL_INIT,                  (void (*)(void)) ModelInitialize_Gene3D); CHKERRQ(ierr);
  ierr =  pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_BC,              (void (*)(void)) ModelApplyBoundaryCondition_Gene3D); CHKERRQ(ierr);
  ierr =  pTatinModelSetFunctionPointer(m, PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void)) ModelAdaptMaterialPointResolution_Gene3D);CHKERRQ(ierr);
  ierr =  pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void)) ModelApplyInitialMeshGeometry_Gene3D);CHKERRQ(ierr);
  ierr =  pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void)) ModelApplyInitialMaterialGeometry_Gene3D);CHKERRQ(ierr);
  ierr =  pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void)) ModelApplyUpdateMeshGeometry_Gene3D);CHKERRQ(ierr);
  ierr =  pTatinModelSetFunctionPointer(m, PTATIN_MODEL_OUTPUT,                (void (*)(void)) ModelOutput_Gene3D);CHKERRQ(ierr);
  ierr =  pTatinModelSetFunctionPointer(m, PTATIN_MODEL_DESTROY,               (void (*)(void)) ModelDestroy_Gene3D); CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m); CHKERRQ(ierr);

  PetscFunctionReturn (0);
}
