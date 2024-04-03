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
#include "dmda_remesh.h"
#include "dmda_iterator.h"
#include "dmda_element_q2p1.h"
#include "dmda_element_q1.h"
#include "mesh_update.h"
#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPStokesPl_def.h"
#include "MPntPEnergy_def.h"
#include "litho_pressure_PDESolve.h"
#include "material_point_std_utils.h"
#include "material_point_point_location.h"
#include "material_point_popcontrol.h"
#include "model_utils.h"
#include "output_material_points.h"
#include "output_material_points_p0.h"
#include "output_paraview.h"
#include "surface_constraint.h"
#include "ptatin_utils.h"

#include "ptatin3d_energy.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include <material_constants_energy.h>
#include "finite_volume/fvda_private.h"

#include "tinyexpr.h"
#include "model_gene3d_ctx.h"

const char MODEL_NAME[] = "model_GENE3D_";
static PetscLogEvent   PTATIN_MaterialPointPopulationControlRemove;

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
  PetscInt       nn,n,region_idx;
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
  for (n=0; n<data->n_regions; n++) {
    /* get regions index */
    region_idx = data->regions_table[n];
    PetscPrintf(PETSC_COMM_WORLD,"[[ SETTING REGION ]]: %d\n",region_idx);
    ierr = MaterialConstantsSetFromOptions(materialconstants,MODEL_NAME,region_idx,PETSC_TRUE);CHKERRQ(ierr);
    ierr = ModelSetRegionParametersFromOptions_Energy(materialconstants,region_idx);CHKERRQ(ierr);
  }
  /* Report all material parameters values */
  for (n=0; n<data->n_regions; n++) {
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

static PetscErrorCode ModelSetPoissonPressureParametersFromOptions_Gene3D(ModelGENE3DCtx *data)
{
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->poisson_pressure_active = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-poisson_pressure_active",&data->poisson_pressure_active,NULL);CHKERRQ(ierr);
  if (!data->poisson_pressure_active) { PetscFunctionReturn(0); }

  data->surface_pressure = 0.0;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-poisson_pressure_surface_p",&data->surface_pressure,&found);CHKERRQ(ierr);
  if (!found) {
    PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]]: No value provided for surface pressure to solve the poisson pressure. Assuming %1.6e [Pa]\n",data->surface_pressure);
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

static PetscErrorCode SurfaceConstraintSetFromOptions_Gene3D(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Create boundaries data */
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-bc_nsubfaces",&data->bc_nfaces,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%sbc_nsubfaces not found!\n",MODEL_NAME); }
  ierr = PetscCalloc1(data->bc_nfaces,&data->bc_tag_table);CHKERRQ(ierr);
  ierr = PetscCalloc1(data->bc_nfaces,&data->bc_sc);

  /* get the number of subfaces and their tag correspondance */
  nn = data->bc_nfaces;
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,"-bc_tag_list",data->bc_tag_table,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != data->bc_nfaces) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"bc_nsubfaces (%d) and the number of entries in bc_tag_list (%d) mismatch!\n",data->bc_nfaces,nn);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetScalingParametersFromOptions(ModelGENE3DCtx *data)
{
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Assume scaling factors based on typical length, viscosity and velocity of long-term geodynamic systems */
  data->scale->length_bar     = 1.0e+5;
  data->scale->viscosity_bar  = 1.0e+22;
  data->scale->velocity_bar   = 1.0e-10;

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-length_scale",&data->scale->length_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for length provided, assuming %1.4e. You can change it with the option -%slength_scale\n",data->scale->length_bar,MODEL_NAME); }

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-viscosity_scale",&data->scale->viscosity_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for viscosity provided, assuming %1.4e. You can change it with the option -%sviscosity_scale\n",data->scale->viscosity_bar,MODEL_NAME); }

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-velocity_scale",&data->scale->velocity_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for velocity provided, assuming %1.4e. You can change it with the option -%svelocity_scale\n",data->scale->velocity_bar,MODEL_NAME); }

  /* Compute additional scaling parameters */
  data->scale->time_bar         = data->scale->length_bar / data->scale->velocity_bar;
  data->scale->pressure_bar     = data->scale->viscosity_bar/data->scale->time_bar;
  data->scale->density_bar      = data->scale->pressure_bar * (data->scale->time_bar*data->scale->time_bar)/(data->scale->length_bar*data->scale->length_bar); // kg.m^-3
  data->scale->acceleration_bar = data->scale->length_bar / (data->scale->time_bar*data->scale->time_bar);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelScaleParameters(DataBucket materialconstants, ModelGENE3DCtx *data)
{
  PetscInt  region_idx,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* scaling values */
  data->scale->cm_per_year2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 );
  data->scale->Myr2sec               = 1.0e6 * ( 365.0 * 24.0 * 3600.0 );
  
  /* Scale viscosity cutoff */
  data->eta_max /= data->scale->viscosity_bar;
  data->eta_min /= data->scale->viscosity_bar;
  /* Scale length */
  for (i=0; i<3; i++) { 
    data->L[i] /= data->scale->length_bar;
    data->O[i] /= data->scale->length_bar;
  }

  data->diffusivity_spm /= (data->scale->length_bar*data->scale->length_bar/data->scale->time_bar);

  data->surface_pressure /= data->scale->pressure_bar;

  /* scale material properties */
  for (region_idx=0; region_idx<data->n_regions; region_idx++) {
    ierr = MaterialConstantsScaleAll(materialconstants,region_idx,data->scale->length_bar,data->scale->velocity_bar,data->scale->time_bar,data->scale->viscosity_bar,data->scale->density_bar,data->scale->pressure_bar);CHKERRQ(ierr);
    ierr = MaterialConstantsEnergyScaleAll(materialconstants,region_idx,data->scale->length_bar,data->scale->time_bar,data->scale->pressure_bar);CHKERRQ(ierr);
  }

  PetscPrintf(PETSC_COMM_WORLD,"[Rift Nitsche Model]:  during the solve scaling is done using \n");
  PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n",       data->scale->length_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n",  data->scale->velocity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n",       data->scale->time_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n",    data->scale->viscosity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->scale->density_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n",      data->scale->pressure_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  a*    : %1.4e [m.s^-2]\n",  data->scale->acceleration_bar );

  PetscFunctionReturn(0);
}

PetscErrorCode ModelInitialize_Gene3D(pTatinCtx ptatin, void *ctx)
{
  ModelGENE3DCtx    *data = (ModelGENE3DCtx*)ctx;
  RheologyConstants *rheology;
  DataBucket        materialconstants;
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
  /* Poisson pressure */
  ierr = ModelSetPoissonPressureParametersFromOptions_Gene3D(data);CHKERRQ(ierr);
  /* Surface constraint */
  ierr = SurfaceConstraintSetFromOptions_Gene3D(ptatin,data);CHKERRQ(ierr);
  /* Scaling */
  ierr = ModelSetScalingParametersFromOptions(data);CHKERRQ(ierr);
  ierr = ModelScaleParameters(materialconstants,data);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-output_markers",&data->output_markers,NULL);CHKERRQ(ierr);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}

/*
======================================================
=       Initial Mesh and Mesh Update functions       =
======================================================
*/

static PetscErrorCode ModelApplyMeshRefinement(DM dav)
{
  PetscInt       nn,d,ndir;
  PetscInt       *dir;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ndir = 0;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-n_refinement_dir",&ndir,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"-%sn_refinement_dir not found!",MODEL_NAME); }
  if (ndir <= 0 || ndir > 3) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"n_refinement_dir cannot be 0 or more than 3. -%sn_refinement_dir = %d.",MODEL_NAME,ndir); }

  PetscPrintf(PETSC_COMM_WORLD,"Mesh is refined in %d directions.\n",ndir);

  ierr = PetscCalloc1(ndir,&dir);CHKERRQ(ierr);
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
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name); }

    /* Allocate arrays for xref and xnat */
    ierr = PetscCalloc1(npoints,&xref);CHKERRQ(ierr); 
    ierr = PetscCalloc1(npoints,&xnat);CHKERRQ(ierr); 

    /* Get xref */
    nn = npoints;
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-refinement_xref_%d",dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME,option_name,xref,&nn,&found);CHKERRQ(ierr);
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
    ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME,option_name,xnat,&nn,&found);CHKERRQ(ierr);
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
  PetscBool      refine,found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = DMDASetUniformCoordinates(ptatin->stokes_ctx->dav, data->O[0], data->L[0], data->O[1], data->L[1], data->O[2], data->L[2]); CHKERRQ(ierr);
  
  refine = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-apply_mesh_refinement",&refine,NULL);CHKERRQ(ierr);
  if (refine) { 
    ierr = ModelApplyMeshRefinement(ptatin->stokes_ctx->dav);CHKERRQ(ierr);
  }
  /* Gravity */
  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL, MODEL_NAME,"-gravity_vec",gvec,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%sgravity_vec requires 3 arguments.",MODEL_NAME);
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]]: Option -%sgravity_vec not provided, assuming gravity = ( %1.4e, %1.4e, %1.4e )\n",MODEL_NAME,gvec[0],gvec[1],gvec[2]);
  }

  ierr = PhysCompStokesSetGravityVector(ptatin->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(ptatin->stokes_ctx,1.0/data->scale->acceleration_bar);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}

static PetscErrorCode ModelApplySurfaceRemeshing(DM dav, PetscReal dt, ModelGENE3DCtx *data)
{
  PetscBool      dirichlet_xmin,dirichlet_xmax,dirichlet_zmin,dirichlet_zmax;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dirichlet_xmin = PETSC_FALSE;
  dirichlet_xmax = PETSC_FALSE;
  dirichlet_zmin = PETSC_FALSE;
  dirichlet_zmax = PETSC_FALSE;

  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-spm_diffusion_dirichlet_xmin",&dirichlet_xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-spm_diffusion_dirichlet_xmax",&dirichlet_xmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-spm_diffusion_dirichlet_zmin",&dirichlet_zmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-spm_diffusion_dirichlet_zmax",&dirichlet_zmax,NULL);CHKERRQ(ierr);

  if ( !dirichlet_xmin && !dirichlet_xmax && !dirichlet_zmin && !dirichlet_zmax ) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"No boundary conditions provided for the surface diffusion (spm)! Use at least one of -%sspm_diffusion_dirichlet_{xmin,xmax,zmin,zmax}",MODEL_NAME);
  }

  /* Dirichlet velocity imposed on z normal faces so we do the same here */
  ierr = UpdateMeshGeometry_ApplyDiffusionJMAX(dav,data->diffusivity_spm,dt,dirichlet_xmin,dirichlet_xmax,dirichlet_zmin,dirichlet_zmax,PETSC_FALSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyUpdateMeshGeometry_Gene3D(pTatinCtx ptatin,Vec X,void *ctx)
{
  ModelGENE3DCtx      *data;
  PhysCompStokes      stokes;
  DM                  stokes_pack,dav,dap;
  Vec                 velocity,pressure;
  PetscReal           dt;
  PetscBool           refine;
  PetscErrorCode      ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelGENE3DCtx*)ctx;
  
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
  refine = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-apply_mesh_refinement",&refine,NULL);CHKERRQ(ierr);
  if (refine) { 
    ierr = ModelApplyMeshRefinement(ptatin->stokes_ctx->dav);CHKERRQ(ierr);
  }

  /* Passive markers update */
  if (data->passive_markers) { ierr = PSwarmFieldUpdateAll(data->pswarm);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

/*
======================================================
=              Initial Material geometry             =
======================================================
*/

static PetscErrorCode ModelApplyInitialPlasticStrain_FromExpr(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  DataBucket     material_points;
  DataField      PField_std,PField_pls;
  te_variable    *vars; 
  te_expr        **expression;
  PetscInt       n,n_wz,n_var;
  PetscScalar    coor[3];
  PetscBool      found;
  int            p,n_mp_points,err;
  char           opt_name[PETSC_MAX_PATH_LEN],wz_expr[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Get the number of weak zones to set */
  n_wz = 0;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-n_weak_zones",&n_wz,&found);CHKERRQ(ierr);
  if (!found || !n_wz) {
    PetscFunctionReturn(0);
  }
  /* Allocate an array of expression (1 for each weak zone) */
  ierr = PetscMalloc1(n_wz,&expression);CHKERRQ(ierr);
  /* Register variables for expression */
  n_var = 3; // 3 variables: x,y,z
  ierr = PetscCalloc1(n_var,&vars);CHKERRQ(ierr);
  vars[0].name = "x"; vars[0].address = &coor[0];
  vars[1].name = "y"; vars[1].address = &coor[1];
  vars[2].name = "z"; vars[2].address = &coor[2];

  for (n=0; n<n_wz; n++) {
    /* Evaluate expression of each weak zone */
    ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-wz_expression_%d",n);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,opt_name,wz_expr,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",opt_name); }
    PetscPrintf(PETSC_COMM_WORLD,"Weak zone %d, evaluating expression:\n\t%s\n",n,wz_expr);

    expression[n] = te_compile(wz_expr, vars, 3, &err);
    if (!expression[n]) {
      PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Weak zone %d, expression %s did not compile.",n,wz_expr);
    }
  }

  ierr = pTatinGetMaterialPoints(ptatin,&material_points,NULL);CHKERRQ(ierr);
  /* std variables */
  DataBucketGetDataFieldByName(material_points,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));
  /* Plastic strain variables */
  DataBucketGetDataFieldByName(material_points,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));
  
  DataBucketGetSizes(material_points,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    double        *position;
    float         pls;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access coordinates of the marker */
    MPntStdGetField_global_coord(material_point,&position);

    /* Scale for user expression (expression is expected to use SI) */
    coor[0] = position[0] * data->scale->length_bar;
    coor[1] = position[1] * data->scale->length_bar;
    coor[2] = position[2] * data->scale->length_bar;

    /* Background plastic strain */
    pls = ptatin_RandomNumberGetDouble(0.0,0.03);
    /* Evaluate expression of each weak zone */
    for (n=0; n<n_wz; n++) {
      pls += te_eval(expression[n]) * ptatin_RandomNumberGetDouble(0.0,1.0);
    }
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,0);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);

  /* Free expressions and variables */
  for (n=0; n<n_wz; n++) { te_free(expression[n]); }
  ierr = PetscFree(expression);CHKERRQ(ierr);
  ierr = PetscFree(vars);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_Gene3D(pTatinCtx ptatin, void *ctx)
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  PetscInt       method;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* 
  Point location method: 
    0: brute force
    1: partitionned bounding box
  */
  method = 1;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-mesh_point_location_method",&method,NULL);CHKERRQ(ierr);
  ierr = pTatin_MPntStdSetRegionIndexFromMesh(ptatin,data->mesh_file,data->region_file,method,data->scale->length_bar);CHKERRQ(ierr);
  /* Initial plastic strain */
  ierr = ModelApplyInitialPlasticStrain_FromExpr(ptatin,data);CHKERRQ(ierr);
  PetscFunctionReturn (0);
}

/*
======================================================
=              Initial Marker variables              =
======================================================
*/

PetscErrorCode ModelSetInitialStokesVariableOnMarker_Gene3D(pTatinCtx ptatin,Vec X,void *ctx)
{
  DM                         stokes_pack,dau,dap;
  PhysCompStokes             stokes;
  Vec                        Uloc,Ploc;
  PetscScalar                *LA_Uloc,*LA_Ploc;
  DataField                  PField;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  
  DataBucketGetDataFieldByName(ptatin->material_constants,MaterialConst_MaterialType_classname,&PField);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  ierr = DMCompositeScatter(stokes_pack,X,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  ierr = pTatin_EvaluateRheologyNonlinearities(ptatin,dau,LA_Uloc,dap,LA_Ploc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
======================================================
=                  Initial Solution                  =
======================================================
*/
PetscBool EvaluateVelocityFromExpression(PetscScalar position[], PetscScalar *value, void *ctx)
{
  ExpressionCtx *data = (ExpressionCtx*)ctx;
  PetscReal     val;
  PetscBool     impose = PETSC_TRUE;
  PetscFunctionBegin;

  /* Update expression variables values */
  *data->x = position[0] * data->scale->length_bar;
  *data->y = position[1] * data->scale->length_bar;
  *data->z = position[2] * data->scale->length_bar;
  /* Evaluate expression */
  val = te_eval(data->expression);
  *value = val / data->scale->velocity_bar;

  PetscFunctionReturn(impose);
}

static PetscErrorCode ModelSetInitialVelocityFromExpr(DM dav, Vec velocity, ModelGENE3DCtx *data)
{
  ExpressionCtx  ctx;
  te_variable    *vars;
  te_expr        *expression;
  PetscScalar    x,y,z,time;
  PetscInt       n,n_vars,ndir,nn;
  PetscInt       *dir;
  PetscBool      found;
  int            err;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Get the number of directions for which a function is passed */
  ndir = 0;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-v_init_n_dir",&ndir,&found);CHKERRQ(ierr);
  if (!found || !ndir) {
    PetscFunctionReturn(0);
  }
  /* Get the directions for which an expression is passed */
  ierr = PetscCalloc1(ndir,&dir);CHKERRQ(ierr);
  nn = ndir;
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,"-v_init_dir",dir,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != ndir) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"v_init_n_dir (%d) and the number of entries in v_init_dir (%d) mismatch!\n",ndir,nn);
    }
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"-%sv_init_dir not found!",MODEL_NAME);
  }

  /* Allocate and zero the expression variables data structure */
  n_vars = 4; // 4 variables x,y,z,t
  ierr = PetscCalloc1(n_vars,&vars);CHKERRQ(ierr);
  /* Attach variables */
  vars[0].name = "x"; vars[0].address = &x;
  vars[1].name = "y"; vars[1].address = &y;
  vars[2].name = "z"; vars[2].address = &z;
  vars[3].name = "t"; vars[3].address = &time;

  time = 0.0;
  for (n=0; n<ndir; n++) {
    char     opt_name[PETSC_MAX_PATH_LEN],v_expr[PETSC_MAX_PATH_LEN];
    PetscInt dim = dir[n];

    ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-v_init_expression_%d",dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,opt_name,v_expr,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",opt_name); }
    PetscPrintf(PETSC_COMM_WORLD,"Velocity component %d, evaluating expression:\n\t%s\n",dim,v_expr);

    expression = te_compile(v_expr, vars, n_vars, &err);
    if (!expression) {
      PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",v_expr);
    }
    /* Initialize the ExpressionCtx struct */
    ierr = PetscMemzero(&ctx,sizeof(ExpressionCtx));CHKERRQ(ierr);
    /* Attach variables to struct for the evaluating function */
    ctx.x = &x; ctx.y = &y; ctx.z = &z; ctx.t = &time;
    /* Attach expression */
    ctx.expression   = expression;
    ctx.scale        = data->scale;
    /* Set velocity */
    ierr = DMDAVecTraverse3d(dav,velocity,dim,EvaluateVelocityFromExpression,(void*)&ctx);CHKERRQ(ierr);
    te_free(expression);
  }

  ierr = PetscFree(vars);CHKERRQ(ierr);
  ierr = PetscFree(dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialHydrostaticPressureField_Gene3D(pTatinCtx ptatin, DM dau, DM dap, Vec pressure, ModelGENE3DCtx *data)
{
  PetscReal                                    MeshMin[3],MeshMax[3],domain_height;
  DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
  PetscErrorCode                               ierr;

  PetscFunctionBegin;

  /* Initialize pressure vector to zero */
  ierr = VecZeroEntries(pressure);CHKERRQ(ierr);
  ierr = DMGetBoundingBox(dau,MeshMin,MeshMax);CHKERRQ(ierr);
  domain_height = MeshMax[1] - MeshMin[1];

  /* Values for hydrostatic pressure computing */
  HPctx.surface_pressure = 0.0;
  HPctx.ref_height = domain_height;
  HPctx.ref_N      = ptatin->stokes_ctx->my-1;
  HPctx.grav       = 9.8 / data->scale->acceleration_bar;
  HPctx.rho        = 3300.0 / data->scale->density_bar;

  ierr = DMDAVecTraverseIJK(dap,pressure,0,DMDAVecTraverseIJK_HydroStaticPressure_v2,     (void*)&HPctx);CHKERRQ(ierr);
  ierr = DMDAVecTraverseIJK(dap,pressure,2,DMDAVecTraverseIJK_HydroStaticPressure_dpdy_v2,(void*)&HPctx);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelCreatePoissonPressure_Gene3D(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  PDESolveLithoP poisson_pressure;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create poisson pressure data struct and mesh */
  ierr = pTatinPhysCompActivate_LithoP(ptatin,PETSC_TRUE);CHKERRQ(ierr);
  ierr = pTatinGetContext_LithoP(ptatin,&poisson_pressure);CHKERRQ(ierr);
  /* Create the matrix to store the jacobian matrix */
  data->poisson_Jacobian = NULL;
  ierr = DMSetMatType(poisson_pressure->da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(poisson_pressure->da,&data->poisson_Jacobian);CHKERRQ(ierr);
  ierr = MatSetFromOptions(data->poisson_Jacobian);CHKERRQ(ierr);

  /* Initialize prev_step to step - 1 in case of restart */
  data->prev_step = -1;

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialSolution_Gene3D(pTatinCtx ptatin, Vec X, void *ctx)
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  DM             stokes_pack,dau,dap;
  Vec            velocity,pressure;
  PetscBool      active_energy;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  /* Access velocity and pressure vectors */
  stokes_pack = ptatin->stokes_ctx->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  ierr = ModelApplyInitialHydrostaticPressureField_Gene3D(ptatin,dau,dap,pressure,data);CHKERRQ(ierr);
  /* Initialize to zero the velocity vector */
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);
  ierr = ModelSetInitialVelocityFromExpr(dau,velocity,data);CHKERRQ(ierr);

  if (data->passive_markers) {
    /* Attach solution vector (u, p) to passive markers */
    ierr = PSwarmAttachStateVecVelocityPressure(data->pswarm,X);CHKERRQ(ierr);
  }
  /* Restore velocity and pressure vectors */
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /* Temperature IC */
  ierr = pTatinContextValid_EnergyFV(ptatin,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = pTatin_ModelLoadTemperatureInitialSolution_FromFile(ptatin,MODEL_NAME);CHKERRQ(ierr);
  }
  /* Create poisson pressure ctx and mesh */
  if (data->poisson_pressure_active) {
    ierr = ModelCreatePoissonPressure_Gene3D(ptatin,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
======================================================
=                Boundary conditions                 =
======================================================
*/
static PetscErrorCode ModelApplyBoundaryCondition_PoissonPressure(BCList bclist, DM da, ModelGENE3DCtx *data)
{
  PetscReal      val_P;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  val_P = data->surface_pressure;
  ierr = DMDABCListTraverse3d(bclist,da,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&val_P);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSolvePoissonPressure(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  PDESolveLithoP poisson_pressure;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinGetContext_LithoP(ptatin,&poisson_pressure);CHKERRQ(ierr);
  /* Update mesh coordinates from the velocity mesh */
  ierr = DMDAProjectCoordinatesQ2toOverlappingQ1_3d(ptatin->stokes_ctx->dav,poisson_pressure->da);CHKERRQ(ierr);
  ierr = ModelApplyBoundaryCondition_PoissonPressure(poisson_pressure->bclist,poisson_pressure->da,data);CHKERRQ(ierr);
  /* solve */
  ierr = SNESSolve_LithoPressure(poisson_pressure,data->poisson_Jacobian,poisson_pressure->X,poisson_pressure->F,ptatin);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode SurfaceConstraintCreateFromOptions_Gene3D(
  SurfBCList surf_bclist, 
  PetscInt tag, 
  SurfaceConstraint *bc_sc, 
  PetscBool insert_if_not_found, 
  ModelGENE3DCtx *data)
{
  SurfaceConstraint sc;
  PetscInt          sc_type;
  PetscBool         found;
  char              opt_name[PETSC_MAX_PATH_LEN],sc_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-sc_name_%d",tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,opt_name,sc_name,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Providing a name to -sc_name_%d is mandatory!",tag); }

  ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-sc_type_%d",tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,opt_name,&sc_type,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Providing a type to -sc_type_%d is mandatory!",tag); }

  /* force insertion if not a type of constraint that requires changing A11 */
  if (sc_type != SC_NITSCHE_GENERAL_SLIP || 
      sc_type != SC_NITSCHE_DIRICHLET    ||
      sc_type != SC_NITSCHE_NAVIER_SLIP ) { 
    insert_if_not_found = PETSC_TRUE; 
  }

  ierr = SurfBCListGetConstraint(surf_bclist,sc_name,&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surf_bclist,sc_name,&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,sc_type);CHKERRQ(ierr);
    } else { 
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint %d: %s not found",tag,sc_name); 
    }
  } else {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint %d: the name %s already given! Names must be unique.",tag,sc_name);
  }
  *bc_sc = sc;
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelMarkBoundaryFacets_Gene3D(SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  Mesh          mesh;
  MeshFacetInfo facet_info;
  MeshEntity    mesh_entity;
  PetscInt      method=1;
  PetscBool     found;
  char          opt_name[PETSC_MAX_PATH_LEN],meshfile[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Get mesh entity object */
  ierr = SurfaceConstraintGetFacets(sc,&mesh_entity);CHKERRQ(ierr);
  //if (!mesh_entity) { PetscFunctionReturn(0); }
  /* Check if facets for this surface constraint have already been marked */
  if (mesh_entity->empty == PETSC_FALSE) { PetscFunctionReturn(0); }
  
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]: tag %d\n",PETSC_FUNCTION_NAME,tag);

  /* read the facets mesh corresponding to tag */
  ierr = PetscSNPrintf(meshfile,PETSC_MAX_PATH_LEN-1,"facet_%d_mesh.bin",tag);CHKERRQ(ierr);
  ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-facet_mesh_file_%d",tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,opt_name,meshfile,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Options parsing the facet mesh for tag %d not found; Use %s",tag,opt_name); }

  /* get facet mesh */
  parse_mesh(meshfile,&mesh);
  /* mark facets */
  ierr = SurfaceConstraintGetMeshFacetInfo(sc,&facet_info);CHKERRQ(ierr);
  ierr = MeshFacetMarkFromMesh(mesh_entity,facet_info,mesh,method,data->scale->length_bar);CHKERRQ(ierr);
  /* clean up */
  MeshDestroy(&mesh);
  
  PetscFunctionReturn(0);
}

  /////////////
 // NEUMANN //
/////////////
PetscErrorCode UserSetTractionFromExpression(Facet facets, const PetscReal qp_coor[], PetscReal traction[], void *ctx)
{
  NeumannCtx     *data = (NeumannCtx*)ctx;
  MPntStd        point;
  PetscReal      el_pressure[Q1_NODES_PER_EL_3D],NiQ1[Q1_NODES_PER_EL_3D];
  PetscReal      tau=0.0;
  PetscReal      pressure_qp,tolerance;
  PetscInt       eidx,k,d,max_it;
  PetscBool      initial_guess,monitor;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  eidx = facets->cell_index;
  /* Get the lithostatic pressure in the element */
  ierr = DMDAEQ1_GetScalarElementField_3D(el_pressure,(PetscInt*)&data->elnidx[data->nen*eidx],data->pressure);CHKERRQ(ierr);
  
  /* Use point location to get local coords of quadrature points */

  /* Setup marker for point location */
  /* Initialize the MPntStd data structure memory */
  ierr = PetscMemzero(&point,sizeof(MPntStd));CHKERRQ(ierr);
  /* Copy quadrature point coords */
  ierr = PetscMemcpy(&point.coor,qp_coor,sizeof(double)*3);CHKERRQ(ierr);
  /* Set the element index */
  point.wil = eidx;
  /* Point location options */
  initial_guess = PETSC_TRUE;
  monitor       = PETSC_FALSE;
  max_it        = 10;
  tolerance     = 1.0e-10;
  InverseMappingDomain_3dQ2(tolerance,max_it,initial_guess,monitor,
                            (const PetscReal*)data->coor, // subdomain coordinates
                            (const PetscInt)data->m[0],(const PetscInt)data->m[1],(const PetscInt)data->m[2], // Q2 elements in each dir
                            (const PetscInt*)data->elnidx_q2,1,&point); 

  /* Evaluate basis function at quadrature point local coords */
  EvaluateBasis_Q1_3D(point.xi,NiQ1);
  /* Interpolate the pressure value to quadrature point */
  pressure_qp = 0.0;
  for (k=0;k<Q1_NODES_PER_EL_3D;k++) {
    pressure_qp += el_pressure[k]*NiQ1[k];
  }
  /* set variables values for expression and scale to SI */
  *data->expr_ctx->x = qp_coor[0]  * data->expr_ctx->scale->length_bar;
  *data->expr_ctx->y = qp_coor[1]  * data->expr_ctx->scale->length_bar;
  *data->expr_ctx->z = qp_coor[2]  * data->expr_ctx->scale->length_bar;
  *data->expr_ctx->p = pressure_qp * data->expr_ctx->scale->pressure_bar;
  /* Evaluate expression */
  tau = te_eval(data->expr_ctx->expression);
  tau /= data->expr_ctx->scale->pressure_bar;

  /* Set Traction = tau*n - p*n to the quadrature point */
  for (d=0; d<3; d++) {
    traction[d] = tau * facets->centroid_normal[d] - pressure_qp * facets->centroid_normal[d];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyNeumannConstraint(pTatinCtx ptatin, SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  PDESolveLithoP      poisson_pressure;
  NeumannCtx          data_neumann;
  ExpressionCtx       expression_ctx;
  te_variable         *vars;
  te_expr             *expression;
  DM                  cda;
  Vec                 P_local,gcoords;
  PetscInt            nel_p,nen_p,nel_u,nen_u,lmx,lmy,lmz,n_vars;
  const PetscInt      *elnidx_p,*elnidx_u;
  PetscReal           *LA_pressure_local;
  PetscReal           x,y,z,time,pp;
  PetscScalar         *LA_gcoords;
  PetscBool           found;
  int                 err;
  char                opt_name[PETSC_MAX_PATH_LEN],expr_tau[PETSC_MAX_PATH_LEN];
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  ierr = pTatinGetContext_LithoP(ptatin,&poisson_pressure);CHKERRQ(ierr);

  /* Get Q1 elements connectivity table */
  ierr = DMDAGetElements(poisson_pressure->da,&nel_p,&nen_p,&elnidx_p);CHKERRQ(ierr);
  /* Get elements number on local rank */
  ierr = DMDAGetLocalSizeElementQ2(sc->fi->dm,&lmx,&lmy,&lmz);CHKERRQ(ierr);
  /* Get Q2 elements connectivity table */
  ierr = DMDAGetElements_pTatinQ2P1(sc->fi->dm,&nel_u,&nen_u,&elnidx_u);CHKERRQ(ierr);
  /* Get coordinates */
  ierr = DMGetCoordinateDM(sc->fi->dm,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(sc->fi->dm,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* Initialize Neumann data structure */
  ierr = PetscMemzero(&data_neumann,sizeof(NeumannCtx));CHKERRQ(ierr);
  /* Attach element information to the data structure */
  data_neumann.nen       = nen_p;
  data_neumann.elnidx    = elnidx_p;
  data_neumann.elnidx_q2 = elnidx_u;
  data_neumann.m[0]      = lmx;
  data_neumann.m[1]      = lmy;
  data_neumann.m[2]      = lmz;
  /* Attach coords array to the data structure */
  data_neumann.coor = LA_gcoords;

  /* Get the values of the poisson pressure solution vector at local rank */
  ierr = DMGetLocalVector(poisson_pressure->da,&P_local);CHKERRQ(ierr);
  ierr = VecZeroEntries(P_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(poisson_pressure->da,poisson_pressure->X,INSERT_VALUES,P_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (poisson_pressure->da,poisson_pressure->X,INSERT_VALUES,P_local);CHKERRQ(ierr);
  ierr = VecGetArray(P_local,&LA_pressure_local);CHKERRQ(ierr);
  /* Attach the pressure array to the data structure */
  data_neumann.pressure = LA_pressure_local;

  /* Get time for expression */
  ierr = pTatinGetTime(ptatin,&time);CHKERRQ(ierr);
  time *= data->scale->time_bar;
  /* Create variables data structure */
  n_vars = 5; // 5 variables x,y,z,t,p
  ierr = PetscCalloc1(n_vars,&vars);CHKERRQ(ierr);
  /* Attach variables */
  vars[0].name = "x"; vars[0].address = &x;
  vars[1].name = "y"; vars[1].address = &y;
  vars[2].name = "z"; vars[2].address = &z;
  vars[3].name = "t"; vars[3].address = &time;
  vars[4].name = "p"; vars[4].address = &pp;
  /* Initialize Expression data structure */
  ierr = PetscMemzero(&expression_ctx,sizeof(ExpressionCtx));CHKERRQ(ierr);
  /* Get user expression */
  ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-dev_stress_%d",tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,opt_name,expr_tau,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression not found! Use %s to set it.",opt_name); }
  /* Compile expression */
  expression = te_compile(expr_tau, vars, n_vars, &err);
  if (!expression) {
    PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_tau);
  }
  PetscPrintf(PETSC_COMM_WORLD,"Boundary %s: Evaluating expression \n\t%s\n",sc->name,expr_tau);
  /* Attach variables to struct for the evaluating function */
  expression_ctx.x          = &x; 
  expression_ctx.y          = &y; 
  expression_ctx.z          = &z; 
  expression_ctx.t          = &time; 
  expression_ctx.p          = &pp;
  expression_ctx.scale      = data->scale;
  expression_ctx.expression = expression;
  /* Attach expression ctx to neumann ctx */
  data_neumann.expr_ctx = &expression_ctx;

  //SURFC_CHKSETVALS(SC_TRACTION,UserSetTractionFromExpression);
  ierr = SurfaceConstraintSetValues_TRACTION(sc,(SurfCSetValuesTraction)UserSetTractionFromExpression,(void*)&data_neumann);CHKERRQ(ierr);

  /* clean up */
  te_free(expression);
  ierr = VecRestoreArray(P_local,&LA_pressure_local);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(poisson_pressure->da,&P_local);CHKERRQ(ierr);
  ierr = DMDARestoreElements(poisson_pressure->da,&nel_p,&nen_p,&elnidx_p);CHKERRQ(ierr);
  ierr = PetscFree(vars);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetNeumann_VelocityBC(pTatinCtx ptatin, SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  PetscBool      active_poisson;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinContextValid_LithoP(ptatin,&active_poisson);CHKERRQ(ierr);
  if (!active_poisson) { 
    ierr = ModelCreatePoissonPressure_Gene3D(ptatin,data);CHKERRQ(ierr);
    data->poisson_pressure_active = PETSC_TRUE;
  }

  // TODO: check if we restrain the solve to once per time step or not
  ierr = ModelSolvePoissonPressure(ptatin,data);CHKERRQ(ierr);
  //ierr = ApplyPoissonPressureNeumannConstraint(ptatin,sc);CHKERRQ(ierr);
  ierr = ModelApplyNeumannConstraint(ptatin,sc,tag,data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

  ///////////////
 // DIRICHLET //
///////////////
static PetscErrorCode ModelSetDirichlet_VelocityBC(pTatinCtx ptatin, DM dav, BCList bclist, SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  ExpressionCtx  ctx;
  te_variable    *vars;
  PetscInt       d,n_vars;
  PetscReal      x,y,z,time;
  PetscBool      found[3];
  char           opt_name[PETSC_MAX_PATH_LEN];
  char           expr_ux[PETSC_MAX_PATH_LEN],expr_uy[PETSC_MAX_PATH_LEN],expr_uz[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* get ux */
  ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-ux_%d",tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,opt_name,expr_ux,PETSC_MAX_PATH_LEN-1,&found[0]);CHKERRQ(ierr);
  /* get uy */
  ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-uy_%d",tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,opt_name,expr_uy,PETSC_MAX_PATH_LEN-1,&found[1]);CHKERRQ(ierr);
  /* get uz */
  ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-uz_%d",tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,opt_name,expr_uz,PETSC_MAX_PATH_LEN-1,&found[2]);CHKERRQ(ierr);
  /* get time */
  ierr = pTatinGetTime(ptatin,&time);CHKERRQ(ierr);
  /* scale time for expression evaluation */
  time *= data->scale->time_bar;

  n_vars = 4; // 4 variables x,y,z,t
  ierr = PetscCalloc1(n_vars,&vars);CHKERRQ(ierr);
  /* Attach variables */
  vars[0].name = "x"; vars[0].address = &x;
  vars[1].name = "y"; vars[1].address = &y;
  vars[2].name = "z"; vars[2].address = &z;
  vars[3].name = "t"; vars[3].address = &time;

  /* iterate over the 3 spatial directions */
  for (d=0; d<3; d++) {
    te_expr *expression;
    int     err;

    /* If an expression was found for the the dof d */
    if (found[d]) {
      /* Initialize ExpressionCtx struct to zero */
      ierr = PetscMemzero(&ctx,sizeof(ExpressionCtx));CHKERRQ(ierr);
      /* Attach variables to struct for the evaluating function */
      ctx.x = &x; ctx.y = &y; ctx.z = &z; ctx.t = &time;
      ctx.scale = data->scale;
      switch (d) {
        case 0:
          PetscPrintf(PETSC_COMM_WORLD,"Velocity component 0, evaluating expression:\n\t%s\n",expr_ux);
          expression = te_compile(expr_ux, vars, n_vars, &err);
          if (!expression) {
            PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
            SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_ux);
          }
          /* Attach expression */
          ctx.expression = expression;
          break;
        case 1:
          PetscPrintf(PETSC_COMM_WORLD,"Velocity component 1, evaluating expression:\n\t%s\n",expr_uy);
          expression = te_compile(expr_uy, vars, n_vars, &err);
          if (!expression) {
            PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
            SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_uy);
          }
          /* Attach expression */
          ctx.expression = expression;
          break;
        case 2:
          PetscPrintf(PETSC_COMM_WORLD,"Velocity component 2, evaluating expression:\n\t%s\n",expr_uz);
          expression = te_compile(expr_uz, vars, n_vars, &err);
          if (!expression) {
            PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
            SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_uz);
          }
          /* Attach expression */
          ctx.expression = expression;
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"[[ %s ]] d = %d but should only be 0, 1 or 2",PETSC_FUNCTION_NAME,d);
          break;
      }
      /* Set velocity */
      ierr = DMDABCListTraverseFacets3d(bclist,dav,sc,d,EvaluateVelocityFromExpression,(void*)&ctx);CHKERRQ(ierr);
      te_free(expression);
    }
  }
  ierr = PetscFree(vars);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GeneralNavierSlipBC(
  Facet F,
  const PetscReal qp_coor[],
  PetscReal n_hat[],
  PetscReal t1_hat[],
  PetscReal epsS[],
  PetscReal H[],
  void *data)
{
  GenNavierSlipCtx *bc_data = (GenNavierSlipCtx*)data;
  PetscInt         i,j;

  PetscFunctionBegin;
  /* Fill the H tensor and the epsilon_s tensor */
  for (i=0;i<6;i++) {
    epsS[i] = bc_data->epsilon_s[i];
    H[i] = bc_data->mcal_H[i];
  }
  /* Fill the arbitrary normal and one tangent vectors */
  for (j=0;j<3;j++) {
    t1_hat[j] = bc_data->t1_hat[j];
    n_hat[j] = bc_data->n_hat[j];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGeneralNavierSlipBoundaryValuesFromOptions(PetscInt tag, ModelGENE3DCtx *data, GenNavierSlipCtx *bc_data)
{
  PetscInt       i,nn,dir;
  PetscReal      fac,bc_u[2];
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_navier_");CHKERRQ(ierr);

  nn = 2;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%su_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME,option_name,bc_u,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%s requires 2 entries, found %d.",option_name,nn);
    }
  } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%s not found!",option_name); }

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sderivative_dir_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&dir,&found);CHKERRQ(ierr);
  if (found) {
    if (dir != 0 || dir != 2) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"-%s can only be 0 or 2, found %d.",option_name,dir); }
  } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%s no found.",option_name); }

  bc_data->mcal_H[0] = 0; //H_00
  bc_data->mcal_H[1] = 1; //H_11
  bc_data->mcal_H[2] = 0; //H_22
  bc_data->mcal_H[3] = 1; //H_01
  bc_data->mcal_H[4] = 1; //H_02
  bc_data->mcal_H[5] = 1; //H_12
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%smathcal_H_%d",prefix,tag);CHKERRQ(ierr);
  nn   = 6;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME,option_name,bc_data->mcal_H,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 6) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"%s requires 6 entries, found %d.",option_name,nn);
    }
  }

  /* Scale values */
  for (i=0; i<2; i++) { bc_u[i] = bc_u[i]*data->scale->cm_per_year2m_per_sec / data->scale->velocity_bar; } 

  /* Set values of the user defined strain rate tensor */
  fac = 1.0 / (data->L[dir] - data->O[dir]);
  if (dir == 0) {
    /* Derivative in the x direction */
    bc_data->epsilon_s[0] = fac * 2.0 * bc_u[0]; // Exx
    bc_data->epsilon_s[1] = 0.0;                 // Eyy
    bc_data->epsilon_s[2] = 0.0;                 // Ezz

    bc_data->epsilon_s[3] = 0.0;                 // Exy
    bc_data->epsilon_s[4] = fac * bc_u[1];       // Exz
    bc_data->epsilon_s[5] = 0.0;                 // Eyz
  } else {
    /* Derivative in the z direction */
    bc_data->epsilon_s[0] = 0.0;                 // Exx
    bc_data->epsilon_s[1] = 0.0;                 // Eyy
    bc_data->epsilon_s[2] = fac * 2.0 * bc_u[1]; // Ezz

    bc_data->epsilon_s[3] = 0.0;                 // Exy
    bc_data->epsilon_s[4] = fac * bc_u[0];       // Exz
    bc_data->epsilon_s[5] = 0.0;                 // Eyz
  }
  /* Do not worry if the norm of these vectors is not 1, it is handled internally */
  /* Tangent vector 1 */
  bc_data->t1_hat[0] = bc_u[0];
  bc_data->t1_hat[1] = 0.0;
  bc_data->t1_hat[2] = bc_u[1];
  /* Normal vector */
  bc_data->n_hat[0] = -bc_u[1];
  bc_data->n_hat[1] = 0.0;
  bc_data->n_hat[2] = bc_u[0];

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetBoundaryValues_GeneralNavierSlip(SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  GenNavierSlipCtx bc_data;
  PetscReal        penalty;
  char             opt_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode   ierr;
  PetscFunctionBegin;

  /* penalty for the nitsche method */
  penalty = 1.0e3;
  ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-bc_navier_penalty_%d",tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,opt_name,&penalty,NULL);
  ierr = SurfaceConstraintNitscheGeneralSlip_SetPenalty(sc,penalty);CHKERRQ(ierr);
  /* Set values on boundary from options */
  ierr = ModelSetGeneralNavierSlipBoundaryValuesFromOptions(tag,data,&bc_data);CHKERRQ(ierr);
  ierr = SurfaceConstraintSetValuesStrainRate_NITSCHE_GENERAL_SLIP(sc,(SurfCSetValuesNitscheGeneralSlip)GeneralNavierSlipBC,(void*)&bc_data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetBoundaryValues_VelocityBC(
  pTatinCtx ptatin, 
  DM dav, 
  BCList bclist, 
  SurfaceConstraint sc,
  PetscInt tag, 
  ModelGENE3DCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (sc->type)
  {
    case SC_NONE:
      break;
    
    case SC_TRACTION:
      ierr = ModelSetNeumann_VelocityBC(ptatin,sc,tag,data);CHKERRQ(ierr);
      break;

    case SC_DEMO:
      break;

    case SC_FSSA:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"SC_FSSA not implemented for Gene3D.");
      break;

    case SC_NITSCHE_DIRICHLET:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"SC_NITSCHE_DIRICHLET not implemented for Gene3D.");
      break;

    case SC_NITSCHE_NAVIER_SLIP:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"SC_NITSCHE_NAVIER_SLIP not implemented for Gene3D.");
      break;

    case SC_NITSCHE_GENERAL_SLIP:
      //ierr = ModelSetBoundaryValues_GeneralNavierSlip(sc,tag,data);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"SC_NITSCHE_GENERAL_SLIP not implemented for Gene3D.");
      break;

    case SC_DIRICHLET:
      ierr = ModelSetDirichlet_VelocityBC(ptatin,dav,bclist,sc,tag,data);CHKERRQ(ierr);
      break;

    default:
      break;
  }
  

  PetscFunctionReturn(0);
}

/*
NOTES for time dependant BCs
  Possible approach:
  [1] get an option for the number of time values at which a change must occur e.g., 
      -ntime_changes 4
  [2] get time values of that interval e.g.,
      -time_bc_change t1,t2,t3,t4  
      with this t0 will always be 0.0 and t4 the last time value at which the bc change
  [3] provide an array containing the values to apply at each time change e.g.,
      -time_bc_val_%d u0,u1,u2,u3,u4
      expect 1 more value than ntime_changes for the interval t0->t1

      Problem ==> cannot change the type of the bc i.e., navier stays navier and dirichlet stays dirichlet
      need to find a way to interface with the option file the possibility to change the bc type
      but keep the marking of the facet and the tag (if mesh was deformed we do not want to mark again the facets)
*/

static PetscErrorCode ModelApplyBoundaryCondition_Velocity(pTatinCtx ptatin, DM dav, BCList bclist, SurfBCList surf_bclist, PetscBool insert_if_not_found, ModelGENE3DCtx *data)
{
  PetscInt       f;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  for (f=0; f<data->bc_nfaces; f++) {
    PetscInt tag;

    tag = data->bc_tag_table[f];
    if (!data->bc_sc[f]) {
      ierr = SurfaceConstraintCreateFromOptions_Gene3D(surf_bclist,tag,&(data->bc_sc[f]),insert_if_not_found,data);CHKERRQ(ierr);
      ierr = ModelMarkBoundaryFacets_Gene3D(data->bc_sc[f],tag,data);CHKERRQ(ierr);
    }
    /* Marking facet from mesh should never be done for another step than the very first one */
    //if (ptatin->step == 0) { ierr = ModelMarkBoundaryFacets_Gene3D(data->bc_sc[f],tag,data);CHKERRQ(ierr); }
    ierr = ModelSetBoundaryValues_VelocityBC(ptatin,dav,bclist,data->bc_sc[f],tag,data);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_EnergyFV(FVDA fv, ModelGENE3DCtx *data)
{
  DACellFace     face[] = { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_N, DACELL_FACE_F, DACELL_FACE_B };
  PetscInt       f;
  PetscReal      bc_T[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  PetscBool      found[6];
  PetscErrorCode ierr;  
  PetscFunctionBegin;

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-energy_bc_xmin",&bc_T[0],&found[0]);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-energy_bc_xmax",&bc_T[1],&found[1]);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-energy_bc_ymin",&bc_T[2],&found[2]);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-energy_bc_ymax",&bc_T[3],&found[3]);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-energy_bc_zmin",&bc_T[4],&found[4]);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,"-energy_bc_zmax",&bc_T[5],&found[5]);

  for (f=0; f<6; f++) {
    if (found[f]) {
      ierr = FVDAFaceIterator(fv,face[f],PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&bc_T[f]);CHKERRQ(ierr);
    } else {
      ierr = FVDAFaceIterator(fv,face[f],PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryCondition_Gene3D(pTatinCtx ptatin, void *ctx)
{
  ModelGENE3DCtx   *data = (ModelGENE3DCtx*)ctx;
  PhysCompStokes   stokes;
  PDESolveLithoP   poisson_pressure;
  PhysCompEnergyFV energy;
  PetscBool        active_poisson,active_energy;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = ModelApplyBoundaryCondition_Velocity(ptatin,stokes->dav,stokes->u_bclist,stokes->surf_bclist,PETSC_TRUE,data);CHKERRQ(ierr);

  ierr = pTatinContextValid_EnergyFV(ptatin,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = pTatinGetContext_EnergyFV(ptatin,&energy);CHKERRQ(ierr);
    ierr = ModelApplyBoundaryCondition_EnergyFV(energy->fv,data);CHKERRQ(ierr);
  }

  ierr = pTatinContextValid_LithoP(ptatin,&active_poisson);CHKERRQ(ierr);
  if (active_poisson) {
    ierr = pTatinGetContext_LithoP(ptatin,&poisson_pressure);CHKERRQ(ierr);
    ierr = ModelApplyBoundaryCondition_PoissonPressure(poisson_pressure->bclist,poisson_pressure->da,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn (0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_Gene3D(PetscInt nl,BCList bclist[],SurfBCList surf_bclist[],DM dav[], pTatinCtx ptatin,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*
======================================================
=               Material Point Resolution            =
======================================================
*/

static PetscErrorCode ModelApplyMaterialBoundaryCondition_Gene3D(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
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
  
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(ptatin,&material_point_db,NULL);CHKERRQ(ierr);

  /* create face storage for markers */
  DataBucketDuplicateFields(material_point_db,&material_point_face_db);
  
  n_face_list = 4;
  ierr = PetscMalloc1(n_face_list,&face_list);CHKERRQ(ierr);
  face_list[0] = 0;
  face_list[1] = 1;
  face_list[2] = 4;
  face_list[3] = 5;
  
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
  ierr = MaterialPointRegionAssignment_KDTree(material_point_db,PETSC_TRUE);CHKERRQ(ierr);

  /* delete */
  DataBucketDestroy(&material_point_face_db);
  
  ierr = PetscFree(face_list);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MaterialPointResolutionMask_BoundaryFaces(DM dav, pTatinCtx ctx, PetscBool *popctrl_mask)
{
  PetscInt        nel,nen,el;
  const PetscInt  *elnidx;
  PetscInt        mx,my,mz;
  PetscInt        esi,esj,esk,lmx,lmy,lmz,e;
  PetscInt        iel,kel,jel;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  /* Get Q2 elements information */ 
  ierr = DMDAGetElements_pTatinQ2P1(dav,&nel,&nen,&elnidx);CHKERRQ(ierr);
  ierr = DMDAGetSizeElementQ2(dav,&mx,&my,&mz);CHKERRQ(ierr);
  ierr = DMDAGetCornersElementQ2(dav,&esi,&esj,&esk,&lmx,&lmy,&lmz);CHKERRQ(ierr);

  /* Set all to TRUE */
  for (el=0; el<nel; el++) {
    popctrl_mask[el] = PETSC_TRUE;
  }
  
  esi = esi/2;
  esj = esj/2;
  esk = esk/2;

  /* max(x) face */
  if (esi + lmx == mx) { 
    iel = lmx-1;
    for (kel=0; kel<lmz; kel++) {
      for (jel=0; jel<lmy; jel++) {
        e = iel + jel*lmx + kel*lmx*lmy;
        popctrl_mask[e] = PETSC_FALSE;
      }
    }
  }
  
  /* min(x) face */
  if (esi == 0) {
    iel = 0;
    for (kel=0; kel<lmz; kel++) {
      for (jel=0; jel<lmy; jel++) {
        e = iel + jel*lmx + kel*lmx*lmy;
        popctrl_mask[e] = PETSC_FALSE;
      }
    }
  }

  /* max(z) face */
  if (esk + lmz == mz) {
    kel = lmz-1;
    for (jel=0; jel<lmy; jel++) {
      for (iel=0; iel<lmx; iel++) {  
        e = iel + jel*lmx + kel*lmx*lmy;
        popctrl_mask[e] = PETSC_FALSE;
      }
    }
  }

  /* min(z) face */
  if (esk == 0) {
    kel = 0;
    for (jel=0; jel<lmy; jel++) {
      for (iel=0; iel<lmx; iel++) {  
        e = iel + jel*lmx + kel*lmx*lmy;
        popctrl_mask[e] = PETSC_FALSE;
      }
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode MPPC_SimpleRemoval_Mask(PetscInt np_upper,DM da,DataBucket db,PetscBool reverse_order_removal, PetscBool *popctrl_mask)
{
  PetscInt        *cell_count,count;
  int             p32,npoints32;
  PetscInt        c,nel,nen;
  const PetscInt  *elnidx;
  DataField       PField;
  PetscLogDouble  t0,t1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PTATIN_MaterialPointPopulationControlRemove,0,0,0,0);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(da,&nel,&nen,&elnidx);CHKERRQ(ierr);

  ierr = PetscMalloc( sizeof(PetscInt)*(nel),&cell_count );CHKERRQ(ierr);
  ierr = PetscMemzero( cell_count, sizeof(PetscInt)*(nel) );CHKERRQ(ierr);

  DataBucketGetSizes(db,&npoints32,NULL,NULL);

  /* compute number of points per cell */
  DataBucketGetDataFieldByName(db, MPntStd_classname ,&PField);
  DataFieldGetAccess(PField);
  for (p32=0; p32<npoints32; p32++) {
    MPntStd *marker_p;

    DataFieldAccessPoint(PField,p32,(void**)&marker_p);
    if (marker_p->wil < 0) { continue; }

    cell_count[ marker_p->wil ]++;
  }
  DataFieldRestoreAccess(PField);

  count = 0;
  for (c=0; c<nel; c++) {
    if (cell_count[c] > np_upper) {
      count++;
    }
  }

  if (count == 0) {
    ierr = PetscFree(cell_count);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PTATIN_MaterialPointPopulationControlRemove,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscTime(&t0);

  if (!reverse_order_removal) {
    /* remove points from cells with excessive number */
    DataFieldGetAccess(PField);
    for (p32=0; p32<npoints32; p32++) {
      MPntStd *marker_p;
      int wil;

      DataFieldAccessPoint(PField,p32,(void**)&marker_p);
      wil = marker_p->wil;
      if (popctrl_mask[wil] == PETSC_TRUE) {
        if (cell_count[wil] > np_upper) {
          DataBucketRemovePointAtIndex(db,p32);

          DataBucketGetSizes(db,&npoints32,0,0); /* you need to update npoints as the list size decreases! */
          p32--; /* check replacement point */
          cell_count[wil]--;
        }
      }
    }
    DataFieldRestoreAccess(PField);
  }

  if (reverse_order_removal) {
    MPntStd *mp_std;
    int     wil;

    DataBucketGetDataFieldByName(db,MPntStd_classname,&PField);
    mp_std = PField->data;

    for (p32=npoints32-1; p32>=0; p32--) {

      wil = mp_std[p32].wil;
      if (wil < 0) { continue; }
  
      if (popctrl_mask[wil] == PETSC_TRUE) {
        if (cell_count[wil] > np_upper) {
          mp_std[p32].wil = -2;
          cell_count[wil]--;
        }
      }
    }

    for (p32=0; p32<npoints32; p32++) {
      wil = mp_std[p32].wil;
      if (wil == -2) {

        DataBucketRemovePointAtIndex(db,p32);
        DataBucketGetSizes(db,&npoints32,0,0); /* you need to update npoints as the list size decreases! */
        p32--; /* check replacement point */
        mp_std = PField->data;
      }
    }
  }

  PetscTime(&t1);

  ierr = PetscFree(cell_count);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PTATIN_MaterialPointPopulationControlRemove,0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode AdaptMaterialPointResolution_Mask(pTatinCtx ctx)
{
  PetscErrorCode ierr;
  PetscInt       np_lower,np_upper,patch_extent,nxp,nyp,nzp;
  PetscReal      perturb;
  PetscBool      flg;
  PetscBool      *popctrl_mask; 
  DataBucket     db;
  PetscBool      reverse_order_removal;
  PetscInt       nel,nen;
  const PetscInt *elnidx;
  MPI_Comm       comm;

  PetscFunctionBegin;

  /* options for control number of points per cell */
  np_lower = 0;
  np_upper = 60;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_np_lower",&np_lower,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_np_upper",&np_upper,&flg);CHKERRQ(ierr);

  /* options for injection of markers */
  nxp = 2;
  nyp = 2;
  nzp = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nxp",&nxp,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nyp",&nyp,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nzp",&nzp,&flg);CHKERRQ(ierr);

  perturb = 0.1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-mp_popctrl_perturb",&perturb,&flg);CHKERRQ(ierr);
  patch_extent = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_patch_extent",&patch_extent,&flg);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(ctx,&db,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ctx->stokes_ctx->dav,&comm);CHKERRQ(ierr);

  /* Get element number (nel)*/
  ierr = DMDAGetElements_pTatinQ2P1(ctx->stokes_ctx->dav,&nel,&nen,&elnidx);CHKERRQ(ierr);
  /* Allocate memory for the array */
  ierr = PetscMalloc1(nel,&popctrl_mask);CHKERRQ(ierr);
  
  ierr = MaterialPointResolutionMask_BoundaryFaces(ctx->stokes_ctx->dav,ctx,popctrl_mask);CHKERRQ(ierr);
  
  /* insertion */
  ierr = MPPC_NearestNeighbourPatch(np_lower,np_upper,patch_extent,nxp,nyp,nzp,perturb,ctx->stokes_ctx->dav,db);CHKERRQ(ierr);

  /* removal */
  if (np_upper != -1) {
    reverse_order_removal = PETSC_TRUE;
  ierr = MPPC_SimpleRemoval_Mask(np_upper,ctx->stokes_ctx->dav,db,reverse_order_removal,popctrl_mask);CHKERRQ(ierr);
  }

  ierr = PetscFree(popctrl_mask);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelAdaptMaterialPointResolution_Gene3D(pTatinCtx ptatin,void *ctx)
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  /* Particles injection on faces */
  ierr = ModelApplyMaterialBoundaryCondition_Gene3D(ptatin,data);CHKERRQ(ierr);
  /* Population control */
  ierr = AdaptMaterialPointResolution_Mask(ptatin);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}

/*
======================================================
=                       Output                       =
======================================================
*/

static PetscErrorCode ModelOutputMarkerFields_Gene3D(pTatinCtx ptatin,const char prefix[])
{
  DataBucket               materialpoint_db;
  int                      nf;
  const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl};
  char                     mp_file_prefix[256];
  PetscErrorCode           ierr;

  PetscFunctionBegin;

  nf = sizeof(mp_prop_list)/sizeof(mp_prop_list[0]);

  ierr = pTatinGetMaterialPoints(ptatin,&materialpoint_db,NULL);CHKERRQ(ierr);
  sprintf(mp_file_prefix,"%s_mpoints",prefix);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,ptatin->outputpath,mp_file_prefix);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelViewSurfaceConstraint_Gene3D(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  PhysCompStokes stokes;
  PetscInt       f;
  char           root[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s/",ptatin->outputpath);CHKERRQ(ierr);
  for (f=0; f<data->bc_nfaces; f++) {
    if (data->bc_sc[f]) { ierr = SurfaceConstraintViewParaview(data->bc_sc[f],root,data->bc_sc[f]->name);CHKERRQ(ierr); }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputPassiveMarkers_Gene3D(ModelGENE3DCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PSwarmView(data->pswarm,PSW_VT_SINGLETON);CHKERRQ(ierr);
  ierr = PSwarmViewInfo(data->pswarm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputEnergyFV_Gene3D(pTatinCtx ptatin, const char prefix[], PetscBool been_here, ModelGENE3DCtx *data)
{
  PhysCompEnergyFV energy;
  char             root[PETSC_MAX_PATH_LEN],pvoutputdir[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
  char             pvdfilename[PETSC_MAX_PATH_LEN],vtkfilename[PETSC_MAX_PATH_LEN];
  char             stepprefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  ierr = pTatinGetContext_EnergyFV(ptatin,&energy);CHKERRQ(ierr);
  /* PVD file */
  PetscSNPrintf(pvdfilename,PETSC_MAX_PATH_LEN-1,"%s/timeseries_T_fv.pvd",ptatin->outputpath);
  if (prefix) { PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "%s_T_fv.pvts",prefix);
  } else {      PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "T_fv.pvts");           }
  
  PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%D",ptatin->step);
  if (!been_here) { /* new file */
    ierr = ParaviewPVDOpen(pvdfilename);CHKERRQ(ierr);
    ierr = ParaviewPVDAppend(pvdfilename,ptatin->time,vtkfilename,stepprefix);CHKERRQ(ierr);
  } else {
    ierr = ParaviewPVDAppend(pvdfilename,ptatin->time,vtkfilename,stepprefix);CHKERRQ(ierr);
  }
  
  ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",ptatin->outputpath);CHKERRQ(ierr);
  ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",root,ptatin->step);CHKERRQ(ierr);
  
  /* PetscVec */
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_energy",prefix);CHKERRQ(ierr);
  ierr = FVDAView_JSON(energy->fv,pvoutputdir,fname);CHKERRQ(ierr); /* write meta data abour fv mesh, its DMDA and the coords */
  ierr = FVDAView_Heavy(energy->fv,pvoutputdir,fname);CHKERRQ(ierr);  /* write cell fields */
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s_energy_T",pvoutputdir,prefix);CHKERRQ(ierr);
  ierr = PetscVecWriteJSON(energy->T,0,fname);CHKERRQ(ierr); /* write cell temperature */
  
  if (data->output_markers) {
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s-Tfv",pvoutputdir,prefix);
    ierr = FVDAView_CellData(energy->fv,energy->T,PETSC_TRUE,fname);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_Gene3D(pTatinCtx ptatin,Vec X,const char prefix[],void *ctx)
{
  ModelGENE3DCtx              *data = (ModelGENE3DCtx*)ctx;
  const MaterialPointVariable mp_prop_list[] = { MPV_region, MPV_viscosity, MPV_density, MPV_plastic_strain };
  static PetscBool            been_here = PETSC_FALSE;
  PetscBool                   active_energy;
  PetscErrorCode              ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  /* Output Velocity and pressure */
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(ptatin,X,prefix);CHKERRQ(ierr);
  /* Output markers cell fields (for production runs) */
  ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(ptatin,PETSC_FALSE,sizeof(mp_prop_list)/sizeof(MaterialPointVariable),mp_prop_list,prefix);CHKERRQ(ierr);
  
  /* Output raw markers and vtu velocity and pressure (for testing and debugging) */
  if (data->output_markers) {
    ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(ptatin,X,prefix);CHKERRQ(ierr);
    ierr = ModelOutputMarkerFields_Gene3D(ptatin,prefix);CHKERRQ(ierr);
    ierr = ModelViewSurfaceConstraint_Gene3D(ptatin,data);CHKERRQ(ierr);
  }
  /* Output passive markers */
  if (data->passive_markers) { ierr = ModelOutputPassiveMarkers_Gene3D(data);CHKERRQ(ierr); }

  /* Output temperature (FV) */
  ierr = pTatinContextValid_EnergyFV(ptatin,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = ModelOutputEnergyFV_Gene3D(ptatin,prefix,been_here,data);CHKERRQ(ierr);
  }

  /* Output poisson pressure */
  if (data->poisson_pressure_active) {
    PetscBool vts = PETSC_FALSE;
    if (data->output_markers) { vts = PETSC_TRUE; }
    ierr = PoissonPressureOutput(ptatin,prefix,vts,been_here);CHKERRQ(ierr);
  }
  been_here = PETSC_TRUE;

  PetscFunctionReturn (0);
}











































//======================================================================================================================================



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
//======================================================================================================================================





















PetscErrorCode ModelDestroy_Gene3D(pTatinCtx ptatin,void *ctx)
{
  ModelGENE3DCtx *data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelGENE3DCtx*)ctx;

  /* Free contents of structure */
  ierr = PetscFree(data->regions_table);
  ierr = PetscFree(data->bc_tag_table);CHKERRQ(ierr);
  ierr = PetscFree(data->bc_sc);CHKERRQ(ierr);

  if (data->poisson_pressure_active) {
    ierr = MatDestroy(&data->poisson_Jacobian);
  }
  if (data->scale) { ierr = PetscFree(data->scale);CHKERRQ(ierr); }
  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn (0);
}

static PetscErrorCode ModelCreateScalingCtx(ScalingCtx *s)
{
  PetscErrorCode ierr;
  ScalingCtx     scaling;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_ScalingCtx),&scaling);CHKERRQ(ierr);
  ierr = PetscMemzero(scaling,sizeof(struct _p_ScalingCtx));CHKERRQ(ierr);
  *s = scaling;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_Gene3D(void)
{
  ModelGENE3DCtx *data;
  ScalingCtx     scale;
  pTatinModel    m;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = ModelCreateScalingCtx(&scale);CHKERRQ(ierr);
  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelGENE3DCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelGENE3DCtx));CHKERRQ(ierr);
  data->scale = scale;

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"Gene3D");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_INIT,                  (void (*)(void)) ModelInitialize_Gene3D); CHKERRQ(ierr); 
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void)) ModelApplyInitialMeshGeometry_Gene3D);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_OUTPUT,                (void (*)(void)) ModelOutput_Gene3D);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void)) ModelApplyInitialMaterialGeometry_Gene3D);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void)) ModelSetInitialStokesVariableOnMarker_Gene3D);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void)) ModelApplyInitialSolution_Gene3D);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_BC,              (void (*)(void)) ModelApplyBoundaryCondition_Gene3D); CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_DESTROY,               (void (*)(void)) ModelDestroy_Gene3D); CHKERRQ(ierr);
/* 
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_BCMG,            (void (*)(void)) ModelApplyBoundaryConditionMG_Gene3D);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void)) ModelAdaptMaterialPointResolution_Gene3D);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void)) ModelApplyUpdateMeshGeometry_Gene3D);CHKERRQ(ierr);
*/
  /* Insert model into list */
  ierr = pTatinModelRegister(m); CHKERRQ(ierr);

  PetscFunctionReturn (0);
}
