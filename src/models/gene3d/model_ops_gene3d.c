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

static PetscErrorCode ModelSetMeshTypeFromOptions(ModelGENE3DCtx *data)
{
  PetscInt       mtype;
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data->mesh_type = MESH_EULERIAN;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-mesh_type",&mtype,&found);CHKERRQ(ierr);
  if (found) {
    if (mtype < 0 || mtype > 1) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown mesh type. Can only be 0: Eulerian, or 1: ALE. Found %d\n",mtype); }
    data->mesh_type = (GeneMeshType)mtype;
  }

  PetscPrintf(PETSC_COMM_WORLD,"[[ Mesh Type ]]: %s\n",data->mesh_type == MESH_EULERIAN ? "Eulerian" : "ALE");
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetRegionHeatSourceTypeFromOptions(const int region_idx, const char model_name[], int source_type[])
{
  PetscInt       i,nn=7;
  PetscBool      found;
  int            type[] = {0, 0, 0, 0, 0, 0, 0};
  char           option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-heat_source_type_%d",region_idx);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,model_name,option_name,type,&nn,&found);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Found %d source types\n",nn);
  if (nn > 7) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"[ Region %d ]: Too many heat source types, maximum is 7, found %d\n",region_idx,nn);}

  for (i=0; i<nn; i++) {
    switch (type[i]) {
      case 0:
        source_type[i] = ENERGYSOURCE_NONE;
        break;
      
      case 1:
        source_type[i] = ENERGYSOURCE_USE_MATERIALPOINT_VALUE;
        break;
      
      case 2:
        source_type[i] = ENERGYSOURCE_CONSTANT;
        break;
      
      case 3:
        source_type[i] = ENERGYSOURCE_SHEAR_HEATING;
        break;
      
      case 4:
        source_type[i] = ENERGYSOURCE_DECAY;
        break;
      
      case 5:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"[ENERGYSOURCE_ADIABATIC] Not supported with Gene3D (because not supported with FV)");
        break;
      
      case 6:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"[ENERGYSOURCE_ADIABATIC_ADVECTION] Not supported with Gene3D (because not supported with FV)");
        break;
      
      default:
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"[ Region %d ]: Unknown heat source type %d\n",region_idx,type[i]);
        break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MaterialConstantsEnergySetFromOptions_SourceConstant(DataBucket materialconstants, const char model_name[], const int region_idx)
{
  DataField         PField;
  EnergySourceConst *data_Q;
  PetscReal         heat_source;
  char              option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  /* Heat source */
  DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField);
  DataFieldGetEntries(PField,(void**)&data_Q);
  heat_source = 0.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-heat_source_%d",region_idx);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,model_name,option_name,&heat_source,NULL);CHKERRQ(ierr);
  EnergySourceConstSetField_HeatSource(&data_Q[region_idx],heat_source);
  PetscFunctionReturn(0);
}

static PetscErrorCode MaterialConstantsEnergySetFromOptions_SourceDecay(DataBucket materialconstants, const char model_name[], const int region_idx)
{
  DataField         PField;
  EnergySourceDecay *data_Q;
  PetscReal         heat_source_ref,half_life;
  char              option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(materialconstants,EnergySourceDecay_classname,&PField);
  DataFieldGetEntries(PField,(void**)&data_Q);

  heat_source_ref = 0.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-heat_source_ref_%d",region_idx);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,model_name,option_name,&heat_source_ref,NULL);CHKERRQ(ierr);

  half_life = 0.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-heat_source_half_life_%d",region_idx);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,model_name,option_name,&half_life,NULL);CHKERRQ(ierr);

  EnergySourceDecaySetField_HeatSourceRef(&data_Q[region_idx],heat_source_ref);
  EnergySourceDecaySetField_HalfLife(&data_Q[region_idx],half_life);
  PetscFunctionReturn(0);
}

static PetscErrorCode MaterialConstantsEnergySetFromOptions_Source(DataBucket materialconstants, const char model_name[], int source_type[], const int region_idx)
{
  int            i,nsource = 7;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  for (i=0; i<nsource; i++) {
    switch (source_type[i]) {
      case ENERGYSOURCE_NONE: // nothing to do
        break;

      case ENERGYSOURCE_USE_MATERIALPOINT_VALUE: // done in ModelApplyInitialMaterialGeometry_Gene3D
        break;
      
      case ENERGYSOURCE_CONSTANT:
        ierr = MaterialConstantsEnergySetFromOptions_SourceConstant(materialconstants,model_name,region_idx);CHKERRQ(ierr);
        break;
      
      case ENERGYSOURCE_SHEAR_HEATING: // nothing to do
        break;

      case ENERGYSOURCE_DECAY:
        ierr = MaterialConstantsEnergySetFromOptions_SourceDecay(materialconstants,model_name,region_idx);CHKERRQ(ierr);
        break;
      
      case ENERGYSOURCE_ADIABATIC:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"[ENERGYSOURCE_ADIABATIC] Not supported with Gene3D (because not supported with FV)");
        break;
      
      case ENERGYSOURCE_ADIABATIC_ADVECTION:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"[ENERGYSOURCE_ADIABATIC_ADVECTION] Not supported with Gene3D (because not supported with FV)");
        break;
      
      default:
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"[ Region %d ]: Unknown heat source type %d\n",region_idx,source_type[i]);
        break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetRegionParametersFromOptions_Energy(DataBucket materialconstants, const int region_idx)
{
  DataField                 PField,PField_k;
  EnergyConductivityConst   *data_k;
  EnergyMaterialConstants   *matconstants_e;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscReal                 alpha,beta,rho,conductivity,Cp;
  char                      option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode            ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Energy material constants */
  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  /* Conductivity */
  DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
  DataFieldGetEntries(PField_k,(void**)&data_k);

  /* Set material energy parameters from options file */
  Cp = 800.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-heatcapacity_%d",region_idx);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&Cp,NULL);CHKERRQ(ierr);

  alpha = 0.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-thermalexpension_%d",region_idx);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&alpha,NULL);CHKERRQ(ierr);

  beta = 0.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-compressibility_%d",region_idx);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&beta,NULL);CHKERRQ(ierr);

  rho = 1.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-density_%d",region_idx);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&rho,NULL);CHKERRQ(ierr);
  
  conductivity = 1.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-conductivity_%d",region_idx);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME, option_name,&conductivity,NULL);CHKERRQ(ierr);

  ierr = ModelSetRegionHeatSourceTypeFromOptions(region_idx,MODEL_NAME,source_type);CHKERRQ(ierr);
  for (int k=0; k<7; k++) {
    PetscPrintf(PETSC_COMM_WORLD,"[ Region %d ]: Heat source type %d\n",region_idx,source_type[k]);
  }
  /* Set energy params for region_idx */
  MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha,beta,rho,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
  EnergyConductivityConstSetField_k0(&data_k[region_idx],conductivity);
  ierr = MaterialConstantsEnergySetFromOptions_Source(materialconstants,MODEL_NAME,source_type,region_idx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParametersFromOptions(pTatinCtx ptatin, DataBucket materialconstants, ModelGENE3DCtx *data)
{
  PetscInt       nn,n,region_idx;
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"regions_");CHKERRQ(ierr);

  /* Get the number of regions */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%snregions",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&data->n_regions,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!\n",option_name); }

  /* Allocate an array to hold the regions indices */
  ierr = PetscCalloc1(data->n_regions,&data->regions_table);CHKERRQ(ierr);
  /* Get user regions indices */
  nn = data->n_regions;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%slist",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,option_name,data->regions_table,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != data->n_regions) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"regions_nregions (%d) and the number of entries in regions_list (%d) mismatch!\n",data->n_regions,nn);
    }
  } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!\n",option_name); }

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
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->eta_cutoff = PETSC_TRUE;
  data->eta_max = 1.0e+25;
  data->eta_min = 1.0e+19;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"viscosity_cutoff_");CHKERRQ(ierr);

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sapply",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&data->eta_cutoff,NULL);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%seta_lower",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&data->eta_min,NULL);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%seta_upper",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&data->eta_max,NULL);CHKERRQ(ierr);
  /* Report if activated */
  if (data->eta_cutoff) { PetscPrintf(PETSC_COMM_WORLD,"[[ Viscosity Cutoff ]]: eta_min = %1.4e [Pa.s] eta_max = %1.4e [Pa.s]\n",data->eta_min,data->eta_max); }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetSPMParametersFromOptions(ModelGENE3DCtx *data)
{
  PetscBool      found,flg;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"spm_");CHKERRQ(ierr);

  data->surface_diffusion = PETSC_FALSE;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sapply_surface_diffusion",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&data->surface_diffusion,&found);CHKERRQ(ierr);
  if (found) {
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdiffusivity",prefix);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&data->diffusivity_spm,&flg);CHKERRQ(ierr);
    if (!flg) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface diffusion activated but no diffusivity provided. Use %s to set it.\n",option_name); }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetPoissonPressureParametersFromOptions_Gene3D(ModelGENE3DCtx *data)
{
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"poisson_pressure_");CHKERRQ(ierr);

  data->poisson_pressure_active = PETSC_FALSE;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sapply",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&data->poisson_pressure_active,NULL);CHKERRQ(ierr);
  if (!data->poisson_pressure_active) { PetscFunctionReturn(0); }

  data->surface_pressure = 0.0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%ssurface_p",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&data->surface_pressure,&found);CHKERRQ(ierr);
  if (!found) {
    PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]]: No value provided for surface pressure to solve the poisson pressure. Assuming %1.6e [Pa]\n",data->surface_pressure);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetPassiveMarkersSwarmParametersFromOptions(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  PSwarm         pswarm;
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->passive_markers = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-passive_pswarm_apply",&data->passive_markers,&found);CHKERRQ(ierr);
  if (!found) { PetscFunctionReturn(0); }

  ierr = PSwarmCreate(PETSC_COMM_WORLD,&pswarm);CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"%spassive_",MODEL_NAME);CHKERRQ(ierr);
  ierr = PSwarmSetOptionsPrefix(pswarm,prefix);CHKERRQ(ierr);
  ierr = PSwarmSetPtatinCtx(pswarm,ptatin);CHKERRQ(ierr);
  ierr = PSwarmSetTransportModeType(pswarm,PSWARM_TM_LAGRANGIAN);CHKERRQ(ierr);

  ierr = PSwarmSetFromOptions(pswarm);CHKERRQ(ierr);

  /* Copy reference into model data for later use in different functions */
  data->pswarm = pswarm;

  PetscFunctionReturn(0);
}

static PetscErrorCode RegisterMeshFacets(char prefix[], ModelGENE3DCtx *data)
{
  PetscInt       f;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  for (f=0; f<data->bc_nfaces; f++) {
    PetscInt tag = data->bc_tag_table[f];
    char     option_name[PETSC_MAX_PATH_LEN],meshfile[PETSC_MAX_PATH_LEN];

    /* read the facets mesh corresponding to tag */
    ierr = PetscSNPrintf(meshfile,PETSC_MAX_PATH_LEN-1,"facet_%d_mesh.bin",tag);CHKERRQ(ierr);
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sfacet_mesh_file_%d",prefix,tag);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,meshfile,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Options parsing the facet mesh for tag %d not found; Use %s",tag,option_name); }

    /* get facet mesh */
    parse_mesh(PETSC_COMM_WORLD,meshfile,&(data->mesh_facets[f]));
    if (!data->mesh_facets[f]) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Failed to read the mesh file %s\n",meshfile); }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode SurfaceConstraintSetFromOptions_Gene3D(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_");CHKERRQ(ierr);

  /* Create boundaries data */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%snsubfaces",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&data->bc_nfaces,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!\n",option_name); }
  ierr = PetscCalloc1(data->bc_nfaces,&data->bc_tag_table);CHKERRQ(ierr);
  ierr = PetscCalloc1(data->bc_nfaces,&data->mesh_facets);CHKERRQ(ierr);

  /* get the number of subfaces and their tag correspondance */
  nn = data->bc_nfaces;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%stag_list",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,option_name,data->bc_tag_table,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != data->bc_nfaces) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"bc_nsubfaces (%d) and the number of entries in bc_tag_list (%d) mismatch!\n",data->bc_nfaces,nn);
    }
  } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"Option %s not found!",option_name); }

  if (data->mesh_type == MESH_EULERIAN) { ierr = RegisterMeshFacets(prefix,data);CHKERRQ(ierr); }

  data->bc_debug = PETSC_FALSE;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdebug",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&data->bc_debug,NULL);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetScalingParametersFromOptions(ModelGENE3DCtx *data)
{
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"scaling_");CHKERRQ(ierr);

  /* Assume scaling factors based on typical length, viscosity and velocity of long-term geodynamic systems */
  data->scale->length_bar     = 1.0e+5;
  data->scale->viscosity_bar  = 1.0e+22;
  data->scale->velocity_bar   = 1.0e-10;

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%slength",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&data->scale->length_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for length provided, assuming %1.4e. You can change it with the option %s\n",data->scale->length_bar,option_name); }

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sviscosity",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&data->scale->viscosity_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for viscosity provided, assuming %1.4e. You can change it with the option %s\n",data->scale->viscosity_bar,option_name); }

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%svelocity",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&data->scale->velocity_bar,&found);CHKERRQ(ierr);
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No scaling factor for velocity provided, assuming %1.4e. You can change it with the option %s\n",data->scale->velocity_bar,option_name); }

  /* Compute additional scaling parameters */
  data->scale->time_bar         = data->scale->length_bar / data->scale->velocity_bar;
  data->scale->pressure_bar     = data->scale->viscosity_bar/data->scale->time_bar;
  data->scale->density_bar      = data->scale->pressure_bar * (data->scale->time_bar*data->scale->time_bar)/(data->scale->length_bar*data->scale->length_bar); // kg.m^-3
  data->scale->acceleration_bar = data->scale->length_bar / (data->scale->time_bar*data->scale->time_bar);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelScaleParameters(DataBucket materialconstants, ModelGENE3DCtx *data)
{
  PetscInt       i;
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
  for (i=0; i<data->n_regions; i++) {
    PetscInt region_idx;

    region_idx = data->regions_table[i];
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

static PetscErrorCode ModelSetIsostaticTopographyParameters(ModelGENE3DCtx *data)
{
  PetscReal      isostatic_density_ref,isostatic_depth;
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN],option_value[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"isostatic_");CHKERRQ(ierr);

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdensity_ref",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&isostatic_density_ref,&found);CHKERRQ(ierr);
  if (found) {
    isostatic_density_ref = isostatic_density_ref / data->scale->density_bar;
    ierr = PetscSNPrintf(option_value,PETSC_MAX_PATH_LEN-1,"%1.5e",isostatic_density_ref);CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-isostatic_density_ref_adim",option_value);CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdepth",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&isostatic_depth,&found);CHKERRQ(ierr);
  if (found) {
    isostatic_depth = isostatic_depth / data->scale->length_bar;
    PetscSNPrintf(option_value,PETSC_MAX_PATH_LEN-1,"%1.5e",isostatic_depth);
    ierr = PetscOptionsSetValue(NULL,"-isostatic_compensation_depth_adim",option_value);CHKERRQ(ierr);
  }
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
  /* Set the rheology to visco-plastic temperature dependant */
  rheology->rheology_type = RHEOLOGY_VP_STD;
  /* Box geometry */
  ierr = ModelSetInitialGeometryFromOptions(data);CHKERRQ(ierr);
  /* Mesh type */
  ierr = ModelSetMeshTypeFromOptions(data);CHKERRQ(ierr);
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
  /* Isostatic initial topography options */
  ierr = ModelSetIsostaticTopographyParameters(data);CHKERRQ(ierr);

  /* Fetch scaled values for the viscosity cutoff */
  rheology->apply_viscosity_cutoff_global = data->eta_cutoff;
  rheology->eta_upper_cutoff_global       = data->eta_max;
  rheology->eta_lower_cutoff_global       = data->eta_min;

  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-output_markers",&data->output_markers,NULL);CHKERRQ(ierr);CHKERRQ(ierr);

  /* Initialize prev_step to step - 1 in case of restart */
  data->prev_step = -1;
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
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"refinement_");CHKERRQ(ierr);

  ndir = 0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sndir",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&ndir,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name); }
  if (ndir <= 0 || ndir > 3) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"refinement_ndir cannot be 0 or more than 3. Found -refinement_ndir = %d!",ndir); }

  PetscPrintf(PETSC_COMM_WORLD,"Mesh is refined in %d directions.\n",ndir);

  ierr = PetscCalloc1(ndir,&dir);CHKERRQ(ierr);
  nn = ndir;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdir",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,option_name,dir,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != ndir) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"refinement_ndir (%d) and the number of entries in refinement_dir (%d) mismatch!\n",ndir,nn);
    }
  } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name); }

  for (d=0; d<ndir; d++) {
    PetscInt  dim,npoints;
    PetscReal *xref,*xnat;
    char      option_name[PETSC_MAX_PATH_LEN];

    dim = dir[d];

    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%snpoints_%d",prefix,dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&npoints,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name); }

    /* Allocate arrays for xref and xnat */
    ierr = PetscCalloc1(npoints,&xref);CHKERRQ(ierr); 
    ierr = PetscCalloc1(npoints,&xnat);CHKERRQ(ierr); 

    /* Get xref */
    nn = npoints;
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sxref_%d",prefix,dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME,option_name,xref,&nn,&found);CHKERRQ(ierr);
    if (found) {
      if (nn != npoints) {
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"-refinement_npoints_%d (%d) and the number of entries in refinement_xref_%d (%d) mismatch!\n",dim,npoints,dim,nn);
      }
    } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name); }

    /* Get xnat */
    nn = npoints;
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sxnat_%d",prefix,dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME,option_name,xnat,&nn,&found);CHKERRQ(ierr);
    if (found) {
      if (nn != npoints) {
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"-refinement_npoints_%d (%d) and the number of entries in refinement_xnat_%d (%d) mismatch!\n",dim,npoints,dim,nn);
      }
    } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"%s not found!",option_name); }

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
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-refinement_apply",&refine,NULL);CHKERRQ(ierr);
  if (refine) { 
    ierr = ModelApplyMeshRefinement(ptatin->stokes_ctx->dav);CHKERRQ(ierr);
    ierr = DMDABilinearizeQ2Elements(ptatin->stokes_ctx->dav);CHKERRQ(ierr);
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

  /* Passive markers setup */
  if (data->passive_markers) { ierr = PSwarmSetUp(data->pswarm);CHKERRQ(ierr); }
  PetscFunctionReturn (0);
}

static PetscErrorCode ModelApplySurfaceRemeshing(DM dav, PetscReal dt, ModelGENE3DCtx *data)
{
  PetscBool      dirichlet_xmin,dirichlet_xmax,dirichlet_zmin,dirichlet_zmax;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"spm_diffusion_");CHKERRQ(ierr);

  dirichlet_xmin = PETSC_FALSE;
  dirichlet_xmax = PETSC_FALSE;
  dirichlet_zmin = PETSC_FALSE;
  dirichlet_zmax = PETSC_FALSE;

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdirichlet_xmin",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&dirichlet_xmin,NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdirichlet_xmax",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&dirichlet_xmax,NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdirichlet_zmin",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&dirichlet_zmin,NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdirichlet_zmax",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&dirichlet_zmax,NULL);CHKERRQ(ierr);

  if ( !dirichlet_xmin && !dirichlet_xmax && !dirichlet_zmin && !dirichlet_zmax ) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"No boundary conditions provided for the surface diffusion (spm)! Use at least one of -%sspm_diffusion_dirichlet_{xmin,xmax,zmin,zmax}",MODEL_NAME);
  }

  /* Dirichlet velocity imposed on z normal faces so we do the same here */
  ierr = UpdateMeshGeometry_ApplyDiffusionJMAX(dav,data->diffusivity_spm,dt,dirichlet_xmin,dirichlet_xmax,dirichlet_zmin,dirichlet_zmax,PETSC_FALSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelUpdateMeshGeometryALE(pTatinCtx ptatin, DM dav, Vec velocity, PetscReal dt, ModelGENE3DCtx *data)
{
  Vec            velocity_ale;
  PetscInt       d;
  PetscReal      MeshMin[3],MeshMax[3];
  PetscBool      found = PETSC_FALSE;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"mesh_");CHKERRQ(ierr);
  
  /* Copy the fluid velocity */
  ierr = DMGetGlobalVector(dav,&velocity_ale);CHKERRQ(ierr);
  ierr = VecCopy(velocity,velocity_ale);CHKERRQ(ierr);

  /* Remove component requested by user, by default all components are passed to the mesh update function */
  for (d=0; d<3; d++) {
    found = PETSC_FALSE;
    ierr  = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sale_rm_component_%d",prefix,d);CHKERRQ(ierr);
    ierr  = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&found,NULL);CHKERRQ(ierr);
    if (found) { ierr = VecStrideSet(velocity_ale,d,0.0);CHKERRQ(ierr); /* zero d component */ }
  }
  ierr = UpdateMeshGeometry_DecoupledHorizontalVerticalMeshMovement(dav,velocity_ale,dt) ;
  /* Update model data */
  ierr = DMGetBoundingBox(dav,MeshMin,MeshMax);CHKERRQ(ierr);
  for (d=0; d<3; d++) {
    data->O[d] = MeshMin[d];
    data->L[d] = MeshMax[d];
  }
  /* Restore ale velocity vector */
  ierr = DMRestoreGlobalVector(dav,&velocity_ale);CHKERRQ(ierr);
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
  
  ierr = pTatinGetTimestep(ptatin,&dt);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);

  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /* SURFACE REMESHING */
  if (data->surface_diffusion) {
    ierr = ModelApplySurfaceRemeshing(dav,dt,data);CHKERRQ(ierr);
  }

  switch (data->mesh_type) {
    case MESH_EULERIAN:
      /* Resample nodes vertically to adapt with free surface motion */
      ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,dt);CHKERRQ(ierr);
      break;
    
    case MESH_ALE:
      ierr = ModelUpdateMeshGeometryALE(ptatin,dav,velocity,dt,data);CHKERRQ(ierr);
      break;
    
    default:
      /* Resample nodes vertically to adapt with free surface motion */
      ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,dt);CHKERRQ(ierr);
      break;
  }
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
 
  /* Update Mesh Refinement */
  refine = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,"-refinement_apply",&refine,NULL);CHKERRQ(ierr);
  if (refine) { 
    ierr = ModelApplyMeshRefinement(ptatin->stokes_ctx->dav);CHKERRQ(ierr);
    ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
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
static PetscErrorCode ModelApplyInitialVariables_FromExpr(pTatinCtx ptatin, ModelGENE3DCtx *data)
{
  DataBucket     material_points;
  DataField      PField_std,PField_pls,PField_stokes,PField_energy;
  te_variable    *vars; 
  te_expr        **expression_plastic,**expression_heat_source;
  PetscInt       n,n_wz,n_hs,n_var;
  PetscScalar    coor[3];
  PetscBool      found,energy_active;
  int            p,n_mp_points,err;
  char           prefix_wz[PETSC_MAX_PATH_LEN],prefix_hs[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  char           wz_expr[PETSC_MAX_PATH_LEN],hs_expr[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = PetscSNPrintf(prefix_wz,PETSC_MAX_PATH_LEN-1,"wz_");CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix_hs,PETSC_MAX_PATH_LEN-1,"heat_source_");CHKERRQ(ierr);

  /* Get the number of weak zones expressions to evaluate */
  n_wz = 0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%snwz",prefix_wz);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&n_wz,&found);CHKERRQ(ierr);
  /* Get the number of heat sources expressions to evaluate */
  n_hs = 0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%snhs",prefix_hs);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&n_hs,&found);CHKERRQ(ierr);
  /* If there are no weak zones and no heat source on marker to set, return */
  if ((!n_wz && !n_hs)) { PetscFunctionReturn(0); }

  /* Allocate arrays of expression - 1 for each weak zone */
  ierr = PetscMalloc1(n_wz,&expression_plastic);CHKERRQ(ierr);
  /* 1 for each source */
  ierr = PetscMalloc1(n_hs,&expression_heat_source);CHKERRQ(ierr);

  /* Register variables for expression */
  n_var = 3; // 3 variables: x,y,z
  ierr = PetscCalloc1(n_var,&vars);CHKERRQ(ierr);
  vars[0].name = "x"; vars[0].address = &coor[0];
  vars[1].name = "y"; vars[1].address = &coor[1];
  vars[2].name = "z"; vars[2].address = &coor[2];

  for (n=0; n<n_wz; n++) {
    /* Evaluate expression of each weak zone */
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sexpression_%d",prefix_wz,n);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,wz_expr,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
    PetscPrintf(PETSC_COMM_WORLD,"Weak zone %d, evaluating expression:\n\t%s\n",n,wz_expr);

    expression_plastic[n] = te_compile(wz_expr, vars, 3, &err);
    if (!expression_plastic[n]) {
      PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Weak zone %d, expression %s did not compile.",n,wz_expr);
    }
  }

  for (n=0; n<n_hs; n++) {
    /* Evaluate expression of heat source */
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sexpression_%d",prefix_hs,n);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,hs_expr,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
    PetscPrintf(PETSC_COMM_WORLD,"Heat source %d, evaluating expression:\n\t%s\n",n,hs_expr);

    expression_heat_source[n] = te_compile(hs_expr, vars, 3, &err);
    if (!expression_heat_source[n]) {
      PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Heat source %d, expression %s did not compile.",n,hs_expr);
    }
  }

  ierr = pTatinGetMaterialPoints(ptatin,&material_points,NULL);CHKERRQ(ierr);
  /* std variables */
  DataBucketGetDataFieldByName(material_points,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));
  /* stokes variables */
  DataBucketGetDataFieldByName(material_points,MPntPStokes_classname,&PField_stokes);
  DataFieldGetAccess(PField_stokes);
  DataFieldVerifyAccess(PField_stokes,sizeof(MPntPStokes));
  /* Plastic strain variables */
  DataBucketGetDataFieldByName(material_points,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));
  /* Energy variables */
  ierr = pTatinContextValid_EnergyFV(ptatin,&energy_active);CHKERRQ(ierr);
  if (energy_active) {
    DataBucketGetDataFieldByName(material_points,MPntPEnergy_classname,&PField_energy);
    DataFieldGetAccess(PField_energy);
    DataFieldVerifyAccess(PField_energy,sizeof(MPntPEnergy));
  }

  DataBucketGetSizes(material_points,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    MPntPStokes   *mpprop_stokes;
    MPntPStokesPl *mpprop_pls;
    MPntPEnergy   *mpprop_energy;
    double        *position,eta0;
    float         pls;

    DataFieldAccessPoint(PField_std,   p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,   p,(void**)&mpprop_pls);
    DataFieldAccessPoint(PField_stokes,p,(void**)&mpprop_stokes);
    if (energy_active) { DataFieldAccessPoint(PField_energy,p,(void**)&mpprop_energy); }

    /* Set an initial non-zero viscosity */
    eta0 = 1.0e+22 / data->scale->viscosity_bar;
    MPntPStokesSetField_eta_effective(mpprop_stokes,eta0);

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
      pls += te_eval(expression_plastic[n]) * ptatin_RandomNumberGetDouble(0.0,1.0);
    }
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,0);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);

    /* Evaluate expression of heat source */
    if (energy_active) {
      PetscReal heat_source = 0.0;
      for (n=0; n<n_hs; n++) {
        heat_source += te_eval(expression_heat_source[n]);
      }
      heat_source /= (data->scale->pressure_bar / data->scale->time_bar);
      if (heat_source > 1.0e3) {
        PetscPrintf(PETSC_COMM_WORLD,"Heat source: %1.4e\n",heat_source);
      }
      MPntPEnergySetField_heat_source_init(mpprop_energy,heat_source);
    }
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);
  DataFieldRestoreAccess(PField_stokes);
  if (energy_active) { DataFieldRestoreAccess(PField_energy); }

  /* Free expressions and variables */
  for (n=0; n<n_wz; n++) { te_free(expression_plastic[n]); }
  for (n=0; n<n_hs; n++) { te_free(expression_heat_source[n]); }
  ierr = PetscFree(expression_plastic);CHKERRQ(ierr);
  ierr = PetscFree(expression_heat_source);CHKERRQ(ierr);
  ierr = PetscFree(vars);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_Gene3D(pTatinCtx ptatin, void *ctx)
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  PetscInt       method;
  PetscBool      found;
  char           mesh_file[PETSC_MAX_PATH_LEN],region_file[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

    /* Get user mesh file */
  ierr = PetscSNPrintf(mesh_file,PETSC_MAX_PATH_LEN-1,"md.bin");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,"-mesh_file",mesh_file,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%smesh_file not found!\n",MODEL_NAME); }

  /* Get user regions file */
  ierr = PetscSNPrintf(region_file,PETSC_MAX_PATH_LEN-1,"region_cell.bin");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,"-regions_file",region_file,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -%sregions_file not found!\n",MODEL_NAME); }

  /* 
  Point location method: 
    0: brute force
    1: partitionned bounding box
  */
  method = 1;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,"-mesh_point_location_method",&method,NULL);CHKERRQ(ierr);
  ierr = pTatin_MPntStdSetRegionIndexFromMesh(ptatin,mesh_file,region_file,method,data->scale->length_bar);CHKERRQ(ierr);
  /* Initial plastic strain */
  ierr = ModelApplyInitialVariables_FromExpr(ptatin,data);CHKERRQ(ierr);
  
  /* Last thing done (should always be the last thing done) */
  ierr = LagrangianAdvectionFromIsostaticDisplacementVector(ptatin);CHKERRQ(ierr);

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
  ierr = DMCompositeRestoreLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);
  
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
  PetscScalar    x,y,z,time,O[3],L[3];
  PetscInt       d,n,n_vars,ndir,nn;
  PetscInt       *dir;
  PetscBool      found;
  int            err;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"ic_velocity_");CHKERRQ(ierr);

  /* Get the number of directions for which a function is passed */
  ndir = 0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sndir",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&ndir,&found);CHKERRQ(ierr);
  if (!found || !ndir) { PetscFunctionReturn(0); }
  /* Get the directions for which an expression is passed */
  ierr = PetscCalloc1(ndir,&dir);CHKERRQ(ierr);
  nn = ndir;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sdir",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,option_name,dir,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != ndir) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"ic_velocity_ndir (%d) and the number of entries in ic_velocity_dir (%d) mismatch!\n",ndir,nn);
    }
  } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }

  /* Set time to 0 and scale model domain for expression evaluation */
  time = 0.0;
  for (d=0; d<3; d++) { 
    O[d] = data->O[d] * data->scale->length_bar; 
    L[d] = data->L[d] * data->scale->length_bar; 
  }

  /* Allocate and zero the expression variables data structure */
  n_vars = 10; // 10 variables x,y,z,t,Ox,Oy,Oz,Lx,Ly,Lz
  ierr = PetscCalloc1(n_vars,&vars);CHKERRQ(ierr);
  /* Attach variables */
  vars[0].name = "x";  vars[0].address = &x;
  vars[1].name = "y";  vars[1].address = &y;
  vars[2].name = "z";  vars[2].address = &z;
  vars[3].name = "t";  vars[3].address = &time;
  vars[4].name = "Ox"; vars[4].address = &O[0];
  vars[5].name = "Oy"; vars[5].address = &O[1];
  vars[6].name = "Oz"; vars[6].address = &O[2];
  vars[7].name = "Lx"; vars[7].address = &L[0];
  vars[8].name = "Ly"; vars[8].address = &L[1];
  vars[9].name = "Lz"; vars[9].address = &L[2];

  for (n=0; n<ndir; n++) {
    char     v_expr[PETSC_MAX_PATH_LEN];
    PetscInt dim = dir[n];

    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sexpression_%d",prefix,dim);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,v_expr,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
    if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"Velocity component %d, evaluating expression:\n\t%s\n",dim,v_expr); }

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
  const char sc_name[],
  PetscBool insert_if_not_found, 
  ModelGENE3DCtx *data)
{
  SurfaceConstraint sc;
  PetscInt          sc_type;
  PetscBool         found;
  char              option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-bc_sc_type_%d",tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&sc_type,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Providing a type to -bc_sc_type_%d is mandatory!",tag); }

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
    } else { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint %d: %s not found",tag,sc_name); }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelMarkBoundaryFaces_Gene3D(PetscInt tag, MeshEntity mesh_entity, MeshFacetInfo facet_info, ModelGENE3DCtx *data)
{
  HexElementFace face[] = { tag };
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Check if facets for this surface constraint have already been marked, if it is the case, pass */
  if (mesh_entity->set_values_called) { PetscFunctionReturn(0); }
  ierr = MeshFacetMarkDomainFaces(mesh_entity,facet_info,1,face);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelMarkBoundaryFacets_Gene3D(Mesh mesh, PetscInt tag, SurfaceConstraint sc, ModelGENE3DCtx *data)
{
  MeshFacetInfo     facet_info;
  MeshEntity        mesh_entity;
  PetscInt          method=0;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* Get mesh entity object */
  ierr = SurfaceConstraintGetFacets(sc,&mesh_entity);CHKERRQ(ierr);
  /* mark facets */
  ierr = SurfaceConstraintGetMeshFacetInfo(sc,&facet_info);CHKERRQ(ierr);

  switch (data->mesh_type) {
    case MESH_EULERIAN:
      ierr = MeshFacetMarkFromMesh(mesh_entity,facet_info,mesh,method,data->scale->length_bar);CHKERRQ(ierr);
      break;
    
    case MESH_ALE:
      if (tag < 0 || tag > 5) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid boundary tag. For mesh type MESH_LAGRANGIAN can only be 0, 1, 2, 3, 4 or 5, found: %d",tag); }
      ierr = ModelMarkBoundaryFaces_Gene3D(tag,mesh_entity,facet_info,data);CHKERRQ(ierr);
      break;
    
    default:
      ierr = MeshFacetMarkFromMesh(mesh_entity,facet_info,mesh,method,data->scale->length_bar);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

  ///////////////
 // DIRICHLET //
///////////////
static PetscErrorCode ModelSetDirichlet_VelocityBC_Constant(pTatinCtx ptatin, DM dav, BCList bclist, SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  PetscInt       d;
  PetscReal      u_bc[] = {0.0, 0.0, 0.0};
  PetscBool      found[3];
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_dirichlet_");CHKERRQ(ierr);
  /* get ux */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sux_%d",prefix,tag);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&u_bc[0],&found[0]);CHKERRQ(ierr);
  /* get uy */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%suy_%d",prefix,tag);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&u_bc[1],&found[1]);CHKERRQ(ierr);
  /* get uz */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%suz_%d",prefix,tag);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&u_bc[2],&found[2]);CHKERRQ(ierr);

  for (d=0; d<3; d++) {
    /* Set velocity */
    if (found[d]) { ierr = DMDABCListTraverseFacets3d(bclist,dav,sc,d,BCListEvaluator_constant,(void*)&u_bc[d]);CHKERRQ(ierr); }
  } 
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetDirichlet_VelocityBC_Expression(pTatinCtx ptatin, DM dav, BCList bclist, SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  ExpressionCtx  ctx;
  te_variable    *vars;
  PetscInt       d,n_vars;
  PetscReal      x,y,z,time,O[3],L[3];
  PetscBool      found[3];
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  char           expr_ux[PETSC_MAX_PATH_LEN],expr_uy[PETSC_MAX_PATH_LEN],expr_uz[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_dirichlet_");CHKERRQ(ierr);

  /* get ux */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sux_%d",prefix,tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr_ux,PETSC_MAX_PATH_LEN-1,&found[0]);CHKERRQ(ierr);
  /* get uy */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%suy_%d",prefix,tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr_uy,PETSC_MAX_PATH_LEN-1,&found[1]);CHKERRQ(ierr);
  /* get uz */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%suz_%d",prefix,tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr_uz,PETSC_MAX_PATH_LEN-1,&found[2]);CHKERRQ(ierr);
  /* get time */
  ierr = pTatinGetTime(ptatin,&time);CHKERRQ(ierr);
  /* scale time for expression evaluation */
  time *= data->scale->time_bar;
  /* scale model domain for expression evaluation */
  for (d=0; d<3; d++) {
    O[d] = data->O[d] * data->scale->length_bar;
    L[d] = data->L[d] * data->scale->length_bar;
  }

  n_vars = 10; // 10 variables x,y,z,t,Ox,Oy,Oz,Lx,Ly,Lz
  ierr = PetscCalloc1(n_vars,&vars);CHKERRQ(ierr);
  /* Attach variables */
  vars[0].name = "x";  vars[0].address = &x;
  vars[1].name = "y";  vars[1].address = &y;
  vars[2].name = "z";  vars[2].address = &z;
  vars[3].name = "t";  vars[3].address = &time;
  vars[4].name = "Ox"; vars[4].address = &O[0];
  vars[5].name = "Oy"; vars[5].address = &O[1];
  vars[6].name = "Oz"; vars[6].address = &O[2];
  vars[7].name = "Lx"; vars[7].address = &L[0];
  vars[8].name = "Ly"; vars[8].address = &L[1];
  vars[9].name = "Lz"; vars[9].address = &L[2];

  /* iterate over the 3 spatial directions */
  for (d=0; d<3; d++) {
    te_expr *expression;
    int     err;

    /* Initialize ExpressionCtx struct to zero */
    ierr = PetscMemzero(&ctx,sizeof(ExpressionCtx));CHKERRQ(ierr);
    /* If an expression was found for the dof d */
    if (found[d]) {
      if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"Found expression for component %d for boundary tag %d\n",d,tag); }
      /* Attach variables to struct for the evaluating function */
      ctx.x = &x; ctx.y = &y; ctx.z = &z; ctx.t = &time;
      ctx.scale = data->scale;
      switch (d) {
        case 0:
          if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"Velocity component 0, evaluating expression:\n\t%s\n",expr_ux); }
          expression = te_compile(expr_ux, vars, n_vars, &err);
          if (!expression) {
            PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
            SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_ux);
          }
          /* Attach expression */
          ctx.expression = expression;
          break;
        case 1:
          if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"Velocity component 1, evaluating expression:\n\t%s\n",expr_uy); }
          expression = te_compile(expr_uy, vars, n_vars, &err);
          if (!expression) {
            PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
            SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_uy);
          }
          /* Attach expression */
          ctx.expression = expression;
          break;
        case 2:
          if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"Velocity component 2, evaluating expression:\n\t%s\n",expr_uz); }
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

static PetscErrorCode ModelSetDirichlet_VelocityBC_BottomFlowUdotN(pTatinCtx ptatin, DM dav, BCList bclist, SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  PhysCompStokes stokes;
  DM             dms;
  Vec            X,velocity,pressure;
  PetscReal      int_u_dot_n[HEX_EDGES];
  PetscReal      uy;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = pTatinPhysCompGetData_Stokes(ptatin,&X);CHKERRQ(ierr); 
  ierr = PhysCompStokesGetDMComposite(stokes,&dms);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(dms,X,&velocity,&pressure);CHKERRQ(ierr);  
  
  ierr = StokesComputeVdotN(stokes,velocity,int_u_dot_n);CHKERRQ(ierr);
  if (data->bc_debug) {
    PetscPrintf(PETSC_COMM_WORLD,"imin: %+1.4e\n",int_u_dot_n[ WEST_FACE  -1]);
    PetscPrintf(PETSC_COMM_WORLD,"imax: %+1.4e\n",int_u_dot_n[ EAST_FACE  -1]);
    PetscPrintf(PETSC_COMM_WORLD,"jmin: %+1.4e\n",int_u_dot_n[ SOUTH_FACE -1]);
    PetscPrintf(PETSC_COMM_WORLD,"jmax: [free surface] %+1.4e\n",int_u_dot_n[ NORTH_FACE -1]);
    PetscPrintf(PETSC_COMM_WORLD,"kmin: %+1.4e\n",int_u_dot_n[ BACK_FACE  -1]);
    PetscPrintf(PETSC_COMM_WORLD,"kmax: %+1.4e\n",int_u_dot_n[ FRONT_FACE -1]);
  }
  ierr = DMCompositeRestoreAccess(dms,X,&velocity,&pressure);CHKERRQ(ierr);
  
  /* Compute the uy velocity based on faces inflow/outflow except the top free surface */
  /* At step 0, use the user provided initial velocity */
  if (ptatin->step == 0) {
    PetscBool found;
    char uy_expr[PETSC_MAX_PATH_LEN];
    
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,"-ic_velocity_expression_1",uy_expr,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    /* use the user provided expression for the initial condition if it exists */
    if (found) {
      ExpressionCtx  ctx;
      te_variable    *vars;
      te_expr        *expression;
      PetscScalar    x,y,z,time,O[3],L[3];
      PetscInt       d,n_vars;
      int            err;

      if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"Step 0, vertical velocity at bottom, evaluating expression:\n\t%s\n",uy_expr); }

      /* get time */
      ierr = pTatinGetTime(ptatin,&time);CHKERRQ(ierr);
      /* scale time for expression evaluation */
      time *= data->scale->time_bar;
      /* scale model domain for expression evaluation */
      for (d=0; d<3; d++) { 
        O[d] = data->O[d] * data->scale->length_bar; 
        L[d] = data->L[d] * data->scale->length_bar; 
      }
      /* Allocate and zero the expression variables data structure */
      n_vars = 10; // 10 variables x,y,z,t,Ox,Oy,Oz,Lx,Ly,Lz
      ierr = PetscCalloc1(n_vars,&vars);CHKERRQ(ierr);
      /* Attach variables */
      vars[0].name = "x";  vars[0].address = &x;
      vars[1].name = "y";  vars[1].address = &y;
      vars[2].name = "z";  vars[2].address = &z;
      vars[3].name = "t";  vars[3].address = &time;
      vars[4].name = "Ox"; vars[4].address = &O[0];
      vars[5].name = "Oy"; vars[5].address = &O[1];
      vars[6].name = "Oz"; vars[6].address = &O[2];
      vars[7].name = "Lx"; vars[7].address = &L[0];
      vars[8].name = "Ly"; vars[8].address = &L[1];
      vars[9].name = "Lz"; vars[9].address = &L[2];

      expression = te_compile(uy_expr, vars, n_vars, &err);
      if (!expression) {
        PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",uy_expr);
      }
      /* Initialize the ExpressionCtx struct */
      ierr = PetscMemzero(&ctx,sizeof(ExpressionCtx));CHKERRQ(ierr);
      /* Attach variables to struct for the evaluating function */
      ctx.x = &x; ctx.y = &y; ctx.z = &z; ctx.t = &time;
      /* Attach expression */
      ctx.expression   = expression;
      ctx.scale        = data->scale;
      /* Set velocity */
      ierr = DMDABCListTraverseFacets3d(bclist,dav,sc,1,EvaluateVelocityFromExpression,(void*)&ctx);CHKERRQ(ierr);
      te_free(expression);
      ierr = PetscFree(vars);CHKERRQ(ierr);
    } else { 
      /* if not found, set uy = 0 */
      uy = 0.0; 
      ierr = DMDABCListTraverseFacets3d(bclist,dav,sc,1,BCListEvaluator_constant,(void*)&uy);CHKERRQ(ierr);
    }
  } else {
    uy = (int_u_dot_n[WEST_FACE-1]+int_u_dot_n[EAST_FACE-1]+int_u_dot_n[BACK_FACE-1]+int_u_dot_n[FRONT_FACE-1])/((data->L[0] - data->O[0])*(data->L[2] - data->O[2]));
    PetscPrintf(PETSC_COMM_WORLD,"Computed bottom velocity uy = %+1.4e\n",uy);
    ierr = DMDABCListTraverseFacets3d(bclist,dav,sc,1,BCListEvaluator_constant,(void*)&uy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelSetDirichlet_VelocityBC(pTatinCtx ptatin, DM dav, BCList bclist, SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  PetscBool      constant,bot_udotn;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_dirichlet_");CHKERRQ(ierr);

  constant  = PETSC_FALSE;
  bot_udotn = PETSC_FALSE;

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sconstant_%d",prefix,tag);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&constant,NULL);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sbot_u.n_%d",prefix,tag);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&bot_udotn,NULL);CHKERRQ(ierr);

  if (constant) {
    ierr = ModelSetDirichlet_VelocityBC_Constant(ptatin,dav,bclist,sc,tag,data);CHKERRQ(ierr);
  } else if (bot_udotn) {
    ierr = ModelSetDirichlet_VelocityBC_BottomFlowUdotN(ptatin,dav,bclist,sc,tag,data);CHKERRQ(ierr);
  } else {
    ierr = ModelSetDirichlet_VelocityBC_Expression(ptatin,dav,bclist,sc,tag,data);CHKERRQ(ierr);
  }
  
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
  PetscInt            d,nel_p,nen_p,nel_u,nen_u,lmx,lmy,lmz,n_vars;
  const PetscInt      *elnidx_p,*elnidx_u;
  PetscReal           *LA_pressure_local;
  PetscReal           x,y,z,time,pp,O[3],L[3];
  PetscScalar         *LA_gcoords;
  PetscBool           found;
  int                 err;
  char                option_name[PETSC_MAX_PATH_LEN],expr_tau[PETSC_MAX_PATH_LEN];
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
  /* Get model domain for expression */
  for (d=0; d<3; d++) {
    O[d] = data->O[d] * data->scale->length_bar;
    L[d] = data->L[d] * data->scale->length_bar;
  }
  /* Create variables data structure */
  n_vars = 11; // 11 variables x,y,z,t,p,Ox,Oy,Oz,Lx,Ly,Lz
  ierr = PetscCalloc1(n_vars,&vars);CHKERRQ(ierr);
  /* Attach variables */
  vars[0].name  = "x";  vars[0].address  = &x;
  vars[1].name  = "y";  vars[1].address  = &y;
  vars[2].name  = "z";  vars[2].address  = &z;
  vars[3].name  = "t";  vars[3].address  = &time;
  vars[4].name  = "p";  vars[4].address  = &pp;
  vars[5].name  = "Ox"; vars[5].address  = &O[0];
  vars[6].name  = "Oy"; vars[6].address  = &O[1];
  vars[7].name  = "Oz"; vars[7].address  = &O[2];
  vars[8].name  = "Lx"; vars[8].address  = &L[0];
  vars[9].name  = "Ly"; vars[9].address  = &L[1];
  vars[10].name = "Lz"; vars[10].address = &L[2];

  /* Initialize Expression data structure */
  ierr = PetscMemzero(&expression_ctx,sizeof(ExpressionCtx));CHKERRQ(ierr);
  /* Get user expression */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-bc_neumann_dev_stress_%d",tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr_tau,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression not found! Use %s to set it.",option_name); }
  /* Compile expression */
  expression = te_compile(expr_tau, vars, n_vars, &err);
  if (!expression) {
    PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_tau);
  }
  if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"Boundary %s: Evaluating expression \n\t%s\n",sc->name,expr_tau); }
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
  PetscBool      active_poisson,found;
  char           option_name[PETSC_MAX_PATH_LEN],expr[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinContextValid_LithoP(ptatin,&active_poisson);CHKERRQ(ierr);
  if (!active_poisson) { 
    ierr = ModelCreatePoissonPressure_Gene3D(ptatin,data);CHKERRQ(ierr);
    data->poisson_pressure_active = PETSC_TRUE;
  }

  /* 
  Solve the poisson pressure at each BC call.
  When the density is of type Boussinesq, the poisson pressure needs to be
  solved at each Stokes non-linear iteration due to the Stokes pressure changes.
  */
  ierr = ModelSolvePoissonPressure(ptatin,data);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-bc_neumann_dev_stress_%d",tag);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (found) { ierr = ModelApplyNeumannConstraint(ptatin,sc,tag,data);CHKERRQ(ierr); }
  else       { ierr = ApplyPoissonPressureNeumannConstraint(ptatin,sc);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

  /////////////////////////
 // GENERAL NAVIER-SLIP //
/////////////////////////
static PetscErrorCode GeneralNavierSlipBC_Constant(
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

static PetscErrorCode ModelSetGeneralNavierSlipBoundaryValuesFromOptions_Constant(PetscInt tag, ModelGENE3DCtx *data, GenNavierSlipCtx *bc_data)
{
  PetscInt       nn;
  PetscReal      duxdx,duxdz,duzdx,duzdz;
  PetscReal      uL[] = {0.0,0.0};
  char           option_name[PETSC_MAX_PATH_LEN],prefix[PETSC_MAX_PATH_LEN];
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_navier_");CHKERRQ(ierr);

  /* Get velocity vector at (Lx,Lz), magnitude does not matter we only use it for orientation */
  nn = 2;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%suL_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME,option_name,uL,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s requires 2 entries, found %d.",option_name,nn);
    }
  } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }

  /* Get derivatives */
  /* dux/dx */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sduxdx_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&duxdx,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
  /* dux/dz */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sduxdz_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&duxdz,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
  /* duz/dx */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sduzdx_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&duzdx,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
  /* duz/dz */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sduzdz_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&duzdz,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }

  /* Scale by 1/t_bar ==> du / (1/t_bar) = du * t_bar */
  duxdx *= data->scale->time_bar;
  duxdz *= data->scale->time_bar;
  duzdx *= data->scale->time_bar;
  duzdz *= data->scale->time_bar;

  /* Imposed strain-rate */
  bc_data->epsilon_s[0] = duxdx; // Exx
  bc_data->epsilon_s[1] = 0.0;   // Eyy             
  bc_data->epsilon_s[2] = duzdz; // Ezz 
  
  bc_data->epsilon_s[3] = 0.0;                   // Exy                
  bc_data->epsilon_s[4] = 0.5*( duxdz + duzdx ); // Exz
  bc_data->epsilon_s[5] = 0.0;                   // Eyz
  
  /* 
  Tangent vector 1, the scaling is not necessary because it is an orientation 
  better if user provides a normalized vector
  */
  bc_data->t1_hat[0] = uL[0];
  bc_data->t1_hat[1] = 0.0;
  bc_data->t1_hat[2] = uL[1];
  /* Normal vector */
  bc_data->n_hat[0] = -bc_data->t1_hat[2];
  bc_data->n_hat[1] = 0.0;
  bc_data->n_hat[2] = bc_data->t1_hat[0];

  /* 
  Set which component of the strain rate tensor in the nhat, that coord system
  is constrained (1) and which is left unknown (0) 
  */
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
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGeneralNavierSlipBoundaryValuesFromOptions_Expression(pTatinCtx ptatin,PetscInt tag, ModelGENE3DCtx *data, GenNavierSlipCtx *bc_data)
{
  te_variable    *vars;
  te_expr        **expression;
  PetscReal      time,duxdx,duxdz,duzdx,duzdz,O[3],L[3];
  PetscReal      uL[] = {0.0,0.0};
  PetscInt       n,n_vars,n_expression,nn;
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  char           expr_duxdx[PETSC_MAX_PATH_LEN],expr_duxdz[PETSC_MAX_PATH_LEN];
  char           expr_duzdx[PETSC_MAX_PATH_LEN],expr_duzdz[PETSC_MAX_PATH_LEN];
  int            err;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* 
  For a time dependant function there is a problem to define the orientation vector t1_hat 
  because this vector should never be the null vector, it is an orientation not a velocity
  therefore,
  the user can pass a math expression for the derivative
  but,
  if the orientation is described by a math expression that at some point evaluate to 0 it will break the 
  formulation as t1_hat is normalized (thus divided by its norm, thus divided by 0 in that case).
  */

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_navier_");CHKERRQ(ierr);
  if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"Found expressions for Navier-slip BCs for boundary tag %d\n",tag); }

  /* Get velocity vector at (Lx,Lz), magnitude does not matter we only use it for orientation */
  nn = 2;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%suL_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME,option_name,uL,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s requires 2 entries, found %d.",option_name,nn);
    }
  } else { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }


  /* get time */
  ierr = pTatinGetTime(ptatin,&time);CHKERRQ(ierr);
  time *= data->scale->time_bar; // scale to SI
  /* Scale model domain for expression evaluation */
  for (n=0; n<3; n++) { 
    O[n] = data->O[n] * data->scale->length_bar; 
    L[n] = data->L[n] * data->scale->length_bar; 
  }
  /* 
  Create variables data structure 
  For now only time dependant expression is supported.
  This choice comes from the fact that if the derivatives contain spatial variables (x,y,z)
  it means that the choosen velocity is non-linear (bi- and tri-linear count as non-linear)
  I don't think that chosing a non-linear velocity to prescribe the stress is a good choice, 
  but it can change in the future.
  */
  n_vars = 7; // 7 variable -> time,Ox,Oy,Oz,Lx,Ly,Lz
  ierr = PetscCalloc1(n_vars,&vars);CHKERRQ(ierr); // Allocate and zero the expression variables data structure
  /* Attach variables */
  vars[0].name = "t";  vars[0].address = &time;
  vars[1].name = "Ox"; vars[1].address = &O[0];
  vars[2].name = "Oy"; vars[2].address = &O[1];
  vars[3].name = "Oz"; vars[3].address = &O[2];
  vars[4].name = "Lx"; vars[4].address = &L[0];
  vars[5].name = "Ly"; vars[5].address = &L[1];
  vars[6].name = "Lz"; vars[6].address = &L[2];

  /* Get user expression */
  n_expression = 4; // 4 expression dux/dx, dux/dz, duz/dx, duz/dz
  ierr = PetscCalloc1(n_expression,&expression);CHKERRQ(ierr); // Allocate and zero the expression data structure
  /* dux/dx */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sduxdx_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr_duxdx,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
  if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"dux/dx, evaluating expression:\n\t%s\n",expr_duxdx); }
  /* dux/dz */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sduxdz_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr_duxdz,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
  if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"dux/dz, evaluating expression:\n\t%s\n",expr_duxdz); }
  /* duz/dx */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sduzdx_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr_duzdx,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
  if (data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"duz/dx, evaluating expression:\n\t%s\n",expr_duzdx); }
  /* duz/dz */
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sduzdz_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,expr_duzdz,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Option %s not found!",option_name); }
  if(data->bc_debug) { PetscPrintf(PETSC_COMM_WORLD,"duz/dz, evaluating expression:\n\t%s\n",expr_duzdz); }

  /* Compile expression */
  expression[0] = te_compile(expr_duxdx, vars, n_vars, &err);
  if (!expression[0]) {
    PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_duxdx);
  }
  expression[1] = te_compile(expr_duxdz, vars, n_vars, &err);
  if (!expression[1]) {
    PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_duxdz);
  }
  expression[2] = te_compile(expr_duzdx, vars, n_vars, &err);
  if (!expression[2]) {
    PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_duzdx);
  }
  expression[3] = te_compile(expr_duzdz, vars, n_vars, &err);
  if (!expression[3]) {
    PetscPrintf(PETSC_COMM_WORLD,"\t%*s^\nError near here", err-1, "");
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Expression %s did not compile.",expr_duzdz);
  }

  /* Evaluate expressions */
  duxdx = te_eval(expression[0]);
  duxdz = te_eval(expression[1]);
  duzdx = te_eval(expression[2]);
  duzdz = te_eval(expression[3]);

  /* Imposed strain-rate */
  bc_data->epsilon_s[0] = duxdx * data->scale->time_bar; // Exx
  bc_data->epsilon_s[1] = 0.0;                           // Eyy             
  bc_data->epsilon_s[2] = duzdz * data->scale->time_bar; // Ezz 
  
  bc_data->epsilon_s[3] = 0.0;                                           // Exy                
  bc_data->epsilon_s[4] = 0.5*( duxdz + duzdx ) * data->scale->time_bar; // Exz
  bc_data->epsilon_s[5] = 0.0;                                           // Eyz

  bc_data->t1_hat[0] = uL[0];
  bc_data->t1_hat[1] = 0.0;
  bc_data->t1_hat[2] = uL[1];
  /* Normal vector */
  bc_data->n_hat[0] = -bc_data->t1_hat[2];
  bc_data->n_hat[1] = 0.0;
  bc_data->n_hat[2] = bc_data->t1_hat[0];

  /* 
  Set which component of the strain rate tensor in the nhat, that coord system
  is constrained (1) and which is left unknown (0) 
  */
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

  for (n=0; n<n_expression; n++) { te_free(expression[n]); }
  ierr = PetscFree(vars);CHKERRQ(ierr);
  ierr = PetscFree(expression);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetBoundaryValues_GeneralNavierSlip(pTatinCtx ptatin, SurfaceConstraint sc, PetscInt tag, ModelGENE3DCtx *data)
{
  GenNavierSlipCtx bc_data;
  PetscReal        penalty;
  PetscBool        expr=PETSC_FALSE;
  char             prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode   ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_navier_");CHKERRQ(ierr);

  /* penalty for the nitsche method */
  penalty = 1.0e3;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%spenalty_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&penalty,NULL);
  ierr = SurfaceConstraintNitscheGeneralSlip_SetPenalty(sc,penalty);CHKERRQ(ierr);
  /* Set values on boundary from options */
  ierr = PetscMemzero(&bc_data,sizeof(GenNavierSlipCtx));CHKERRQ(ierr);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sexpression_%d",prefix,tag);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME,option_name,&expr,NULL);CHKERRQ(ierr);

  if (expr) { ierr = ModelSetGeneralNavierSlipBoundaryValuesFromOptions_Expression(ptatin,tag,data,&bc_data);CHKERRQ(ierr); }
  else      { ierr = ModelSetGeneralNavierSlipBoundaryValuesFromOptions_Constant(tag,data,&bc_data);CHKERRQ(ierr); }
  ierr = SurfaceConstraintSetValuesStrainRate_NITSCHE_GENERAL_SLIP(sc,(SurfCSetValuesNitscheGeneralSlip)GeneralNavierSlipBC_Constant,(void*)&bc_data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetBoundaryValues_VelocityBC(
  pTatinCtx ptatin, 
  DM dav, 
  BCList bclist, 
  PetscInt tag, 
  SurfaceConstraint sc,
  ModelGENE3DCtx *data)
{
  PetscErrorCode    ierr;
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
      ierr = ModelSetBoundaryValues_GeneralNavierSlip(ptatin,sc,tag,data);CHKERRQ(ierr);
      break;

    case SC_DIRICHLET:
      ierr = ModelSetDirichlet_VelocityBC(ptatin,dav,bclist,sc,tag,data);CHKERRQ(ierr);
      break;

    default:
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_Velocity(pTatinCtx ptatin, DM dav, BCList bclist, SurfBCList surf_bclist, PetscBool insert_if_not_found, ModelGENE3DCtx *data)
{
  PetscInt       f;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  for (f=0; f<data->bc_nfaces; f++) {
    SurfaceConstraint sc;
    Mesh              mesh;
    PetscInt          tag;
    char              option_name[PETSC_MAX_PATH_LEN],sc_name[PETSC_MAX_PATH_LEN];

    tag  = data->bc_tag_table[f];
    mesh = data->mesh_facets[f]; 
    /* Get sc name */
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-bc_sc_name_%d",tag);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,sc_name,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Providing a name to -bc_sc_name_%d is mandatory!",tag); }

    ierr = SurfaceConstraintCreateFromOptions_Gene3D(surf_bclist,tag,sc_name,insert_if_not_found,data);CHKERRQ(ierr);

    /* Querying sc after CreateFromOptions should be safe */
    ierr = SurfBCListGetConstraint(surf_bclist,sc_name,&sc);CHKERRQ(ierr);
    if (!sc) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"sc object for name %s and tag %d does not exist!\n",sc_name,tag); }
    /* mark facets */
    if (data->prev_step != ptatin->step) {
      ierr = ModelMarkBoundaryFacets_Gene3D(mesh,tag,sc,data);CHKERRQ(ierr);
    }
    /* Apply BCs */
    ierr = ModelSetBoundaryValues_VelocityBC(ptatin,dav,bclist,tag,sc,data);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_EnergyFV(FVDA fv, ModelGENE3DCtx *data)
{
  DACellFace     face[] = { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_N, DACELL_FACE_F, DACELL_FACE_B };
  PetscInt       f;
  PetscReal      bc_T[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  PetscBool      found[6];
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;  
  PetscFunctionBegin;

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_energy_");CHKERRQ(ierr);

  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sxmin",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&bc_T[0],&found[0]);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sxmax",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&bc_T[1],&found[1]);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%symin",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&bc_T[2],&found[2]);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%symax",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&bc_T[3],&found[3]);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%szmin",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&bc_T[4],&found[4]);
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%szmax",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME,option_name,&bc_T[5],&found[5]);

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

PetscErrorCode ModelApplyBoundaryConditionMG_Gene3D(PetscInt nl,BCList bclist[],SurfBCList surf_bclist[],DM dav[], pTatinCtx ptatin, void *ctx)
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  PetscInt       n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = ModelApplyBoundaryCondition_Velocity(ptatin,dav[n],bclist[n],surf_bclist[n],PETSC_FALSE,data);CHKERRQ(ierr);
  }
  /* Every BC function has been called at least once, we can update prev_step */
  data->prev_step = ptatin->step;
  PetscFunctionReturn(0);
}

/*
======================================================
=               Material Point Resolution            =
======================================================
*/
static PetscErrorCode ModelApplyMaterialBoundaryCondition_Gene3D(pTatinCtx ptatin, PetscInt n_face_list, PetscInt face_list[], ModelGENE3DCtx *data)
{
  PhysCompStokes  stokes;
  MPAccess        mpX;
  DataBucket      material_point_db,material_point_face_db;
  DM              stokes_pack,dav,dap;
  PetscInt        f,Nxp[2];
  PetscReal       perturb, epsilon;
  int             p,n_mp_points;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  if (n_face_list == 0) { 
    PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No markers injection on faces\n");
    PetscFunctionReturn(0); 
  }
  
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(ptatin,&material_point_db,NULL);CHKERRQ(ierr);

  /* create face storage for markers */
  DataBucketDuplicateFields(material_point_db,&material_point_face_db);
  
  /* traverse */
  for (f=0; f<n_face_list; f++) {
    PetscPrintf(PETSC_COMM_WORLD,"Markers injection on face %d\n",face_list[f]);
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
  /* Clean */
  DataBucketDestroy(&material_point_face_db);

  PetscFunctionReturn(0);
}

static PetscErrorCode MaterialPointResolutionMask_BoundaryFaces(pTatinCtx ptatin, DM dav, PetscInt n_face_list, PetscInt face_list[], PetscBool *popctrl_mask)
{
  PetscInt        nel,nen,el;
  const PetscInt  *elnidx;
  PetscInt        mx,my,mz;
  PetscInt        esi,esj,esk,lmx,lmy,lmz,e;
  PetscInt        iel,kel,jel,f;
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
  /* If no faces are provided apply pop control everywhere */
  if (n_face_list == 0) { PetscFunctionReturn(0); }

  esi = esi/2;
  esj = esj/2;
  esk = esk/2;

  for (f=0; f<n_face_list; f++) {
    PetscInt face = face_list[f];
    switch (face) {
      case 0: // east = xmax = imax = Pxi
        if (esi + lmx == mx) { 
          iel = lmx-1;
          for (kel=0; kel<lmz; kel++) {
            for (jel=0; jel<lmy; jel++) {
              e = iel + jel*lmx + kel*lmx*lmy;
              popctrl_mask[e] = PETSC_FALSE;
            }
          }
        }
        break;
      
      case 1: // west = xmin = imin = Nxi
        if (esi == 0) {
          iel = 0;
          for (kel=0; kel<lmz; kel++) {
            for (jel=0; jel<lmy; jel++) {
              e = iel + jel*lmx + kel*lmx*lmy;
              popctrl_mask[e] = PETSC_FALSE;
            }
          }
        }
        break;
      
      case 2: // north = ymax = jmax = Peta
        if (esj + lmy == my) { 
          jel = lmy-1;
          for (kel=0; kel<lmz; kel++) {
            for (iel=0; iel<lmx; iel++) {
              e = iel + jel*lmx + kel*lmx*lmy;
              popctrl_mask[e] = PETSC_FALSE;
            }
          }
        }
        break;
      
      case 3: // south = ymin = jmin = Neta
        if (esj == 0) {
          jel = 0;
          for (kel=0; kel<lmz; kel++) {
            for (iel=0; iel<lmx; iel++) {
              e = iel + jel*lmx + kel*lmx*lmy;
              popctrl_mask[e] = PETSC_FALSE;
            }
          }
        }
        break;
      
      case 4: // front = zmax = kmax = Pzeta
        if (esk + lmz == mz) {
          kel = lmz-1;
          for (jel=0; jel<lmy; jel++) {
            for (iel=0; iel<lmx; iel++) {  
              e = iel + jel*lmx + kel*lmx*lmy;
              popctrl_mask[e] = PETSC_FALSE;
            }
          }
        }
        break;
      
      case 5: // back = zmin = kmin = Nzeta
        if (esk == 0) {
          kel = 0;
          for (jel=0; jel<lmy; jel++) {
            for (iel=0; iel<lmx; iel++) {  
              e = iel + jel*lmx + kel*lmx*lmy;
              popctrl_mask[e] = PETSC_FALSE;
            }
          }
        }
        break;
      
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Face %d does not exist can only be in [0,5]",face);
        break;
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

static PetscErrorCode AdaptMaterialPointResolution_Mask(pTatinCtx ptatin, PetscInt n_face_list, PetscInt face_list[])
{
  DataBucket     db;
  PetscInt       np_lower,np_upper,patch_extent,nxp,nyp,nzp;
  PetscInt       nel,nen;
  const PetscInt *elnidx;
  PetscReal      perturb;
  PetscBool      found;
  PetscBool      *popctrl_mask;
  PetscBool      reverse_order_removal;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* options for control number of points per cell */
  np_lower = 0;
  np_upper = 60;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_np_lower",&np_lower,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_np_upper",&np_upper,&found);CHKERRQ(ierr);

  /* options for injection of markers */
  nxp = 2;
  nyp = 2;
  nzp = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nxp",&nxp,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nyp",&nyp,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nzp",&nzp,&found);CHKERRQ(ierr);

  perturb = 0.1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-mp_popctrl_perturb",&perturb,&found);CHKERRQ(ierr);
  patch_extent = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_patch_extent",&patch_extent,&found);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(ptatin,&db,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ptatin->stokes_ctx->dav,&comm);CHKERRQ(ierr);

  /* Get element number (nel)*/
  ierr = DMDAGetElements_pTatinQ2P1(ptatin->stokes_ctx->dav,&nel,&nen,&elnidx);CHKERRQ(ierr);
  /* Allocate memory for the array */
  ierr = PetscMalloc1(nel,&popctrl_mask);CHKERRQ(ierr);
  /* Mark faces to remove them from the cleaning */
  ierr = MaterialPointResolutionMask_BoundaryFaces(ptatin,ptatin->stokes_ctx->dav,n_face_list,face_list,popctrl_mask);CHKERRQ(ierr);
  
  /* insertion */
  ierr = MPPC_NearestNeighbourPatch(np_lower,np_upper,patch_extent,nxp,nyp,nzp,perturb,ptatin->stokes_ctx->dav,db);CHKERRQ(ierr);

  /* removal */
  if (np_upper != -1) {
    reverse_order_removal = PETSC_TRUE;
  ierr = MPPC_SimpleRemoval_Mask(np_upper,ptatin->stokes_ctx->dav,db,reverse_order_removal,popctrl_mask);CHKERRQ(ierr);
  }
  ierr = PetscFree(popctrl_mask);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelAdaptMaterialPointResolution_Gene3D(pTatinCtx ptatin,void *ctx)
{
  ModelGENE3DCtx *data = (ModelGENE3DCtx*)ctx;
  PetscInt       n_face_list,nn;
  PetscInt       *face_list;
  PetscBool      found;
  char           prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"bc_marker_");CHKERRQ(ierr);

  /* TODO: think about what to do by default */
  n_face_list = 0;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%snfaces",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME,option_name,&n_face_list,&found);
  /* Allocate memory for the face_list */
  ierr = PetscCalloc1(n_face_list,&face_list);CHKERRQ(ierr);
  /* 
  Faces numbering:
    0: east  = xmax = imax = Pxi
    1: west  = xmin = imin = Nxi
    2: north = ymax = jmax = Peta
    3: south = ymin = jmin = Neta
    4: front = zmax = kmax = Pzeta
    5: back  = zmin = kmin = Nzeta
  */
  nn = n_face_list;
  ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sfaces_list",prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,MODEL_NAME,option_name,face_list,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != n_face_list) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Found %d entries to -bc_marker_faces_list, while expected %d from -bc_marker_nfaces",nn,n_face_list); }
  }
  /* Particles injection on faces */
  ierr = ModelApplyMaterialBoundaryCondition_Gene3D(ptatin,n_face_list,face_list,data);CHKERRQ(ierr);
  /* Population control */
  ierr = AdaptMaterialPointResolution_Mask(ptatin,n_face_list,face_list);CHKERRQ(ierr);

  ierr = PetscFree(face_list);CHKERRQ(ierr);
  PetscFunctionReturn (0);
}

/*
======================================================
=                       Output                       =
======================================================
*/
static PetscErrorCode ModelOutputMarkerFields_Gene3D(pTatinCtx ptatin,const char prefix[])
{
  DataBucket         materialpoint_db;
  MaterialPointField *mp_prop_list;
  char               mp_file_prefix[PETSC_MAX_PATH_LEN];
  int                nf;
  PetscBool          energy_active;
  PetscErrorCode     ierr;

  PetscFunctionBegin;

  ierr = pTatinContextValid_EnergyFV(ptatin,&energy_active);CHKERRQ(ierr);
  if (energy_active) { nf = 4; } 
  else               { nf = 3; }
  ierr = PetscCalloc1(nf,&mp_prop_list);CHKERRQ(ierr);
  mp_prop_list[0] = MPField_Std;
  mp_prop_list[1] = MPField_Stokes;
  mp_prop_list[2] = MPField_StokesPl;
  if (energy_active) { mp_prop_list[3] = MPField_Energy; }

  ierr = pTatinGetMaterialPoints(ptatin,&materialpoint_db,NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(mp_file_prefix,PETSC_MAX_PATH_LEN-1,"%s_mpoints",prefix);CHKERRQ(ierr);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,ptatin->outputpath,mp_file_prefix);CHKERRQ(ierr);
  
  ierr = PetscFree(mp_prop_list);CHKERRQ(ierr);
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
    SurfaceConstraint sc;
    PetscInt          tag;
    PetscBool         found;
    char              option_name[PETSC_MAX_PATH_LEN],sc_name[PETSC_MAX_PATH_LEN];

    tag = data->bc_tag_table[f];
    /* Get sc name */
    ierr = PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-bc_sc_name_%d",tag);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,MODEL_NAME,option_name,sc_name,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Providing a name to -bc_sc_name_%d is mandatory!",tag); }

    /* Get sc object */
    ierr = SurfBCListGetConstraint(stokes->surf_bclist,sc_name,&sc);CHKERRQ(ierr);
    if (sc) { ierr = SurfaceConstraintViewParaview(sc,root,sc->name);CHKERRQ(ierr); }
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
#if 0
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
#endif
PetscErrorCode ModelDestroy_Gene3D(pTatinCtx ptatin,void *ctx)
{
  ModelGENE3DCtx *data;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelGENE3DCtx*)ctx;

  /* Free contents of structure */
  ierr = PetscFree(data->regions_table);
  ierr = PetscFree(data->bc_tag_table);CHKERRQ(ierr);
  for (f=0; f<data->bc_nfaces; f++) {
    if (data->mesh_facets[f]) {MeshDestroy(&(data->mesh_facets[f])); }
  }
  ierr = PetscFree(data->mesh_facets);CHKERRQ(ierr);

  if (data->poisson_pressure_active) {
    ierr = MatDestroy(&data->poisson_Jacobian);
    data->poisson_Jacobian = NULL;
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
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_BCMG,            (void (*)(void)) ModelApplyBoundaryConditionMG_Gene3D);CHKERRQ(ierr);
  
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void)) ModelAdaptMaterialPointResolution_Gene3D);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m, PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void)) ModelApplyUpdateMeshGeometry_Gene3D);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m); CHKERRQ(ierr);

  PetscFunctionReturn (0);
}
