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
 **    filename:   material_constants.c
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
#include "data_bucket.h"
#include "rheology.h"
#include "material_constants.h"
#include "phase_map.h"
#include "material_constants_energy.h"



PetscErrorCode MaterialConstantsCreate(DataBucket *_db)
{
  DataBucket db;

  DataBucketCreate(&db);
  *_db = db;
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsStokesInitialize(DataBucket db)
{

  PetscFunctionBegin;

  DataBucketRegisterField(db,MaterialConst_MaterialType_classname,           sizeof(MaterialConst_MaterialType),           NULL);

  DataBucketRegisterField(db,MaterialConst_ViscosityConst_classname,         sizeof(MaterialConst_ViscosityConst),         NULL);
  DataBucketRegisterField(db,MaterialConst_ViscosityZ_classname,             sizeof(MaterialConst_ViscosityZ),             NULL);
  DataBucketRegisterField(db,MaterialConst_ViscosityArrh_classname,          sizeof(MaterialConst_ViscosityArrh),          NULL);
  DataBucketRegisterField(db,MaterialConst_ViscosityArrh_DislDiff_classname, sizeof(MaterialConst_ViscosityArrh_DislDiff), NULL);
  DataBucketRegisterField(db,MaterialConst_ViscosityFK_classname,            sizeof(MaterialConst_ViscosityFK),            NULL);

  DataBucketRegisterField(db,MaterialConst_PlasticMises_classname,           sizeof(MaterialConst_PlasticMises),           NULL);
  DataBucketRegisterField(db,MaterialConst_PlasticDP_classname,              sizeof(MaterialConst_PlasticDP),              NULL);

  DataBucketRegisterField(db,MaterialConst_DensityConst_classname,           sizeof(MaterialConst_DensityConst),           NULL);
  DataBucketRegisterField(db,MaterialConst_DensityBoussinesq_classname,      sizeof(MaterialConst_DensityBoussinesq),      NULL);
  DataBucketRegisterField(db,MaterialConst_DensityTable_classname,           sizeof(MaterialConst_DensityTable),           NULL);

  DataBucketRegisterField(db,MaterialConst_SoftLin_classname,                sizeof(MaterialConst_SoftLin),                NULL);
  DataBucketRegisterField(db,MaterialConst_SoftExpo_classname,               sizeof(MaterialConst_SoftExpo),               NULL);

  DataBucketFinalize(db);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsInitialize(DataBucket db)
{
  PetscErrorCode ierr;
  ierr = MaterialConstantsStokesInitialize(db);CHKERRQ(ierr);
  ierr = MaterialConstantsEnergyInitialize(db);CHKERRQ(ierr);

  DataBucketFinalize(db);

  DataBucketSetInitialSizes(db,200,0);

  DataBucketView(PETSC_COMM_WORLD,db,"Material Constants",DATABUCKET_VIEW_STDOUT);

  ierr = MaterialConstantsSetDefaults(db);CHKERRQ(ierr);
  ierr = MaterialConstantsEnergySetDefaults(db);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* MaterialType */
PetscErrorCode MaterialConstantsSetDefault_MaterialType(DataBucket db)
{
  int                        r,nregions;
  DataField                  PField;
  MaterialConst_MaterialType *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_MaterialType_classname,&PField);

  data = (MaterialConst_MaterialType*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].visc_type      = VISCOUS_CONSTANT;
    data[r].plastic_type   = PLASTIC_NONE;
    data[r].softening_type = SOFTENING_NONE;
    data[r].density_type   = DENSITY_CONSTANT;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_MaterialType(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_MaterialType   *data;
  PetscBool                    found;
  PetscInt                     value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_MaterialType_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* visc_type */
  value = 1; /* default - not required is nothing happens if option not found */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_MaterialType_member_names[0],region_id);
  ierr = PetscOptionsGetInt(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_MaterialTypeSetField_visc_type(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_MaterialType_member_names[0],region_id);CHKERRQ(ierr);
  }

  /* plastic_type */
  value = 1; /* default - not required is nothing happens if option not found */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_MaterialType_member_names[1],region_id);
  ierr = PetscOptionsGetInt(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_MaterialTypeSetField_plastic_type(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_MaterialType_member_names[1],region_id);CHKERRQ(ierr);
  }

  /* softening_type */
  value = 1; /* default - not required is nothing happens if option not found */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_MaterialType_member_names[2],region_id);
  ierr = PetscOptionsGetInt(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_MaterialTypeSetField_softening_type(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_MaterialType_member_names[2],region_id);CHKERRQ(ierr);
  }

  /* density_type */
  value = 1; /* default - not required is nothing happens if option not found */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_MaterialType_member_names[3],region_id);
  ierr = PetscOptionsGetInt(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_MaterialTypeSetField_density_type(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_MaterialType_member_names[3],region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsPrintValues_MaterialType(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_MaterialType   *data;
  int                          value;


  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_MaterialType_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_MaterialType_member_names[0],region_id);
  MaterialConst_MaterialTypeGetField_visc_type(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %d  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_MaterialType_member_names[1],region_id);
  MaterialConst_MaterialTypeGetField_plastic_type(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %d  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_MaterialType_member_names[2],region_id);
  MaterialConst_MaterialTypeGetField_softening_type(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %d  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_MaterialType_member_names[3],region_id);
  MaterialConst_MaterialTypeGetField_density_type(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %d  \n", opt_name ,value);


  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsSetValues_MaterialType(DataBucket db,const int region_id,PetscInt visc_t,PetscInt plast_t,PetscInt soft_t,PetscInt dens_t)
{
  DataField                    PField;
  MaterialConst_MaterialType   *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_MaterialType_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* visc_type */
  if (visc_t != PETSC_DEFAULT) {
    MaterialConst_MaterialTypeSetField_visc_type(data,visc_t);
  }
  /* plastic_type */
  if (plast_t != PETSC_DEFAULT) {
    MaterialConst_MaterialTypeSetField_plastic_type(data,plast_t);
  }
  /* softening_type */
  if (soft_t != PETSC_DEFAULT) {
    MaterialConst_MaterialTypeSetField_softening_type(data,soft_t);
  }
  /* density_type */
  if (dens_t != PETSC_DEFAULT) {
    MaterialConst_MaterialTypeSetField_density_type(data,dens_t);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

/* DensityConst */
PetscErrorCode MaterialConstantsSetDefault_DensityConst(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_DensityConst   *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_DensityConst_classname,&PField);

  data = (MaterialConst_DensityConst*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].density = 1.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_DensityConst(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_DensityConst   *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityConst_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  value = 1.0; /* default - not required is nothing happens if option not found */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityConst_member_names[0],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_DensityConstSetField_density(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_DensityConst_member_names[0],region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsPrintValues_DensityConst(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_DensityConst   *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityConst_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);


  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityConst_member_names[0],region_id);
  MaterialConst_DensityConstGetField_density(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}
PetscErrorCode MaterialConstantsSetValues_DensityConst(DataBucket db,const int region_id,PetscReal density)
{
  DataField                    PField;
  MaterialConst_DensityConst   *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityConst_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);
  if (density != PETSC_DEFAULT) {
    MaterialConst_DensityConstSetField_density(data,density);
  }
  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsScaleValues_DensityConst(DataBucket db,const int region_id,PetscReal rho_star)
{
  DataField                    PField;
  MaterialConst_DensityConst   *data;
  PetscReal                    density;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityConst_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_DensityConstGetField_density(data,&density);
  density=density/rho_star;
  MaterialConst_DensityConstSetField_density(data,density);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


/* DensityBoussinesq */
PetscErrorCode MaterialConstantsSetDefault_DensityBoussinesq(DataBucket db)
{
  int                            r,nregions;
  DataField                      PField;
  MaterialConst_DensityBoussinesq *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_DensityBoussinesq_classname,&PField);

  data = (MaterialConst_DensityBoussinesq*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].density = 1.0;
    data[r].alpha = 0.0;
    data[r].beta = 0.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_DensityBoussinesq(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                            opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                       PField;
  MaterialConst_DensityBoussinesq *data;
  PetscBool                       found;
  PetscReal                       value;
  PetscErrorCode                  ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityBoussinesq_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityBoussinesq_member_names[0],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_DensityBoussinesqSetField_density(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_DensityBoussinesq_member_names[0],region_id);CHKERRQ(ierr);
  }

  /* alpha */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityBoussinesq_member_names[1],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_DensityBoussinesqSetField_thermalexpension(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_DensityBoussinesq_member_names[1],region_id);CHKERRQ(ierr);
  }

  /* beta */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityBoussinesq_member_names[2],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_DensityBoussinesqSetField_compressibility(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_DensityBoussinesq_member_names[2],region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsPrintValues_DensityBoussinesq(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_DensityBoussinesq   *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityBoussinesq_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityBoussinesq_member_names[0],region_id);
  MaterialConst_DensityBoussinesqGetField_density(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityBoussinesq_member_names[1],region_id);
  MaterialConst_DensityBoussinesqGetField_thermalexpension(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityBoussinesq_member_names[2],region_id);
  MaterialConst_DensityBoussinesqGetField_compressibility(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}
PetscErrorCode MaterialConstantsSetValues_DensityBoussinesq(DataBucket db,const int region_id,PetscReal density,PetscReal alpha,PetscReal beta)
{
  DataField                       PField;
  MaterialConst_DensityBoussinesq *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityBoussinesq_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);
  if (density != PETSC_DEFAULT) {
    MaterialConst_DensityBoussinesqSetField_density(data,density);
  }
  if (alpha != PETSC_DEFAULT) {
    MaterialConst_DensityBoussinesqSetField_thermalexpension(data,alpha);
  }
    if (beta != PETSC_DEFAULT) {
    MaterialConst_DensityBoussinesqSetField_compressibility(data,beta);
  }
  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsScaleValues_DensityBoussinesq(DataBucket db,const int region_id,PetscReal rho_star,PetscReal sigma_star)
{
  DataField                       PField;
  MaterialConst_DensityBoussinesq *data;
  PetscReal                       density,beta;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityBoussinesq_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* scaling for eta0 */
  MaterialConst_DensityBoussinesqGetField_density(data,&density);
  density=density/rho_star;
  MaterialConst_DensityBoussinesqSetField_density(data,density);

  /* no scaling for alpha */

  /* scaling for beta */
  MaterialConst_DensityBoussinesqGetField_compressibility(data,&beta);
  beta=beta*sigma_star;
  MaterialConst_DensityBoussinesqSetField_compressibility(data,beta);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


/*############################################################################*/

/* DensityTable */
PetscErrorCode MaterialConstantsSetDefault_DensityTable(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_DensityTable   *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_DensityTable_classname,&PField);

  data = (MaterialConst_DensityTable*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].density = 1.0;
  }
 PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_DensityTable(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                            opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                       PField;
  MaterialConst_DensityTable *data;
  PetscBool                       found;
  PetscReal                       value;
  char                            fname[PETSC_MAX_PATH_LEN-1];
 
  PetscErrorCode                  ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityTable_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* rho0 */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityTable_member_names[0],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    
    MaterialConst_DensityTableSetField_density(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_DensityTable_member_names[0],region_id);CHKERRQ(ierr);
  }

  /* mapname */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityTable_member_names[1],region_id);
  ierr = PetscOptionsGetString(NULL,model_name,opt_name,fname,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
  if (found) {
    PhaseMap phasemap;
    PhaseMapLoadFromFile(fname,&phasemap);

    MaterialConst_DensityTableSetField_map(data,phasemap);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_DensityTable_member_names[1],region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsPrintValues_DensityTable(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_DensityTable   *data;
  PetscReal                    value;
  char                         fname[PETSC_MAX_PATH_LEN-1];
  PhaseMap                     phasemap; 
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityTable_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityTable_member_names[0],region_id);
  MaterialConst_DensityTableGetField_density(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_DensityTable_member_names[1],region_id);
  MaterialConst_DensityTableGetField_map(data,&phasemap);
  PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_%d.gp","map",region_id);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   writen in file %s    \n", opt_name ,fname);
  PhaseMapViewGnuplot(fname,phasemap);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetValues_DensityTable(DataBucket db,const int region_id,PetscReal density,PhaseMap map)
{
  DataField                       PField;
  MaterialConst_DensityTable       *data;
  
  
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityTable_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);
  if (density != PETSC_DEFAULT) {
    MaterialConst_DensityTableSetField_density(data,density);
  }
  //if (map != PETSC_DEFAULT) {
    // I don't know what that does but it issues a warning 
    // impossible to pass PETSC_DEFAULT
   MaterialConst_DensityTableSetField_map(data,map);
  //}
  
  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsScaleValues_DensityTable(DataBucket db,const int region_id,PetscReal rho_star,PetscReal sigma_star)
{
  DataField                       PField;
  MaterialConst_DensityTable *data;
  PetscReal                       density;
  PhaseMap                        phasemap; 

  //char                            fname[PETSC_MAX_PATH_LEN-1];
  PetscErrorCode                  ierr;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_DensityTable_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* scaling for rho0 */
  MaterialConst_DensityTableGetField_density(data,&density);
  density=density/rho_star;
  MaterialConst_DensityTableSetField_density(data,density);

  /* scaling density table */
  MaterialConst_DensityTableGetField_map(data,&phasemap);
 /*  sprintf(mapname,"%s_%d","density",region_id);
  ierr = pTatinCtxGetPhaseMap(ctx,&phasemap, mapname); CHKERRQ(ierr); */

  ierr = pTatinScalePhaseMap(phasemap,rho_star,sigma_star,1.0);CHKERRQ(ierr);
 /*  sprintf(fname,"%s_%d_scaled.gp","map",region_id);
  PhaseMapViewGnuplot(fname,phasemap); */
  MaterialConst_DensityTableSetField_map(data,phasemap);
  //ierr = pTatinCtxAttachPhaseMap(ctx,phasemap, mapname); CHKERRQ(ierr);
  
  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}










/*############################################################################*/



/* ViscosityConst */
PetscErrorCode MaterialConstantsSetDefault_ViscosityConst(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_ViscosityConst *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityConst_classname,&PField);

  data = (MaterialConst_ViscosityConst*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].eta0 = 1.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_ViscosityConst(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_ViscosityConst *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityConst_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  value = 1.0; /* default - not required is nothing happens if option not found */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityConst_member_names[0],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityConstSetField_eta0(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityConst_member_names[0],region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsPrintValues_ViscosityConst(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_ViscosityConst *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityConst_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityConst_member_names[0],region_id);
  MaterialConst_ViscosityConstGetField_eta0(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsSetValues_ViscosityConst(DataBucket db,const int region_id,PetscReal viscosity)
{
  DataField                    PField;
  MaterialConst_ViscosityConst *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityConst_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);
  if (viscosity != PETSC_DEFAULT) {
    MaterialConst_ViscosityConstSetField_eta0(data,viscosity);
  }
  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsScaleValues_ViscosityConst(DataBucket db,const int region_id,PetscReal eta_star)
{
  DataField                    PField;
  MaterialConst_ViscosityConst *data;
  PetscReal                    viscosity;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityConst_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);
  MaterialConst_ViscosityConstGetField_eta0(data,&viscosity);
  viscosity=viscosity/eta_star;
  MaterialConst_ViscosityConstSetField_eta0(data,viscosity);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

/* ViscosityZ */
PetscErrorCode MaterialConstantsSetDefault_ViscosityZ(DataBucket db)
{
  int                      r,nregions;
  DataField                PField;
  MaterialConst_ViscosityZ *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityZ_classname,&PField);

  data = (MaterialConst_ViscosityZ*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].eta0 = 1.0;
    data[r].zeta = 1.0;
    data[r].zref = 0.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_ViscosityZ(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_ViscosityZ     *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityZ_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  value = 1.0; /* default - not required is nothing happens if option not found */

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityZ_member_names[0],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityZSetField_eta0(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityZ_member_names[0],region_id);CHKERRQ(ierr);
  }

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityZ_member_names[1],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityZSetField_zeta(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityZ_member_names[1],region_id);CHKERRQ(ierr);
  }

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityZ_member_names[2],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityZSetField_zref(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityZ_member_names[2],region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}
PetscErrorCode MaterialConstantsPrintValues_ViscosityZ(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_ViscosityZ     *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityZ_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityZ_member_names[0],region_id);
  MaterialConst_ViscosityZGetField_eta0(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityZ_member_names[1],region_id);
  MaterialConst_ViscosityZGetField_zeta(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityZ_member_names[2],region_id);
  MaterialConst_ViscosityZGetField_zref(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}
PetscErrorCode MaterialConstantsSetValues_ViscosityZ(DataBucket db,const int region_id,PetscReal eta0,PetscReal zeta,PetscReal zref)
{
  DataField                    PField;
  MaterialConst_ViscosityZ     *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityZ_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  if (eta0 != PETSC_DEFAULT) {
    MaterialConst_ViscosityZSetField_eta0(data,eta0);
  }
  if (zeta != PETSC_DEFAULT) {
    MaterialConst_ViscosityZSetField_zeta(data,zeta);
  }
  if (zref != PETSC_DEFAULT) {
    MaterialConst_ViscosityZSetField_zref(data,zref);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsScaleValues_ViscosityZ(DataBucket db,const int region_id,PetscReal eta_star,PetscReal L_star)
{
  DataField                    PField;
  MaterialConst_ViscosityZ     *data;
  PetscReal                    eta0,zeta,zref;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityZ_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_ViscosityZGetField_eta0(data,&eta0);
  eta0=eta0/eta_star;
  MaterialConst_ViscosityZSetField_eta0(data,eta0);

  MaterialConst_ViscosityZGetField_zeta(data,&zeta);
  zeta=zeta/L_star;
  MaterialConst_ViscosityZSetField_zeta(data,zeta);

  MaterialConst_ViscosityZGetField_zref(data,&zref);
  zref=zref/L_star;
  MaterialConst_ViscosityZSetField_zref(data,zref);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

/* Viscosity Arrhenius */
PetscErrorCode MaterialConstantsSetDefault_ViscosityArrh(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_ViscosityArrh  *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_classname,&PField);

  data = (MaterialConst_ViscosityArrh*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].preexpA   = 6.5e-6;
    data[r].Ascale    = 1.0e6;
    data[r].entalpy   = 150.0e3;
    data[r].Vmol      = 0.0;
    data[r].nexp      = 3.0;
    data[r].Tref      = 273.0;
    data[r].Eta_scale = 1.0;
    data[r].P_scale   = 1.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_ViscosityArrh(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_ViscosityArrh  *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  value = 1.0; /* default - not required is nothing happens if option not found */

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[0],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityArrhSetField_preexpA(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityArrh_member_names[0],region_id);CHKERRQ(ierr);
  }

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[1],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityArrhSetField_Ascale(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityArrh_member_names[1],region_id);CHKERRQ(ierr);
  }

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[2],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityArrhSetField_entalpy(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityArrh_member_names[2],region_id);CHKERRQ(ierr);
  }

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[3],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityArrhSetField_Vmol(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityArrh_member_names[3],region_id);CHKERRQ(ierr);
  }

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[4],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityArrhSetField_nexp(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityArrh_member_names[4],region_id);CHKERRQ(ierr);
  }

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[5],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityArrhSetField_Tref(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityArrh_member_names[5],region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}
PetscErrorCode MaterialConstantsPrintValues_ViscosityArrh(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_ViscosityArrh  *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[0],region_id);
  MaterialConst_ViscosityArrhGetField_preexpA(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[1],region_id);
  MaterialConst_ViscosityArrhGetField_Ascale(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[2],region_id);
  MaterialConst_ViscosityArrhGetField_entalpy(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[3],region_id);
  MaterialConst_ViscosityArrhGetField_Vmol(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[4],region_id);
  MaterialConst_ViscosityArrhGetField_nexp(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[5],region_id);
  MaterialConst_ViscosityArrhGetField_Tref(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[6],region_id);
  MaterialConst_ViscosityArrhGetField_Eta_scale(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_member_names[7],region_id);
  MaterialConst_ViscosityArrhGetField_P_scale(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}
PetscErrorCode MaterialConstantsSetValues_ViscosityArrh(DataBucket db,const int region_id,PetscReal preexpA,PetscReal Ascale,PetscReal entalpy,PetscReal Vmol,PetscReal nexp,PetscReal Tref)
{
  DataField                    PField;
  MaterialConst_ViscosityArrh  *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  if (preexpA != PETSC_DEFAULT) {
    MaterialConst_ViscosityArrhSetField_preexpA(data,preexpA);
  }

  if (Ascale != PETSC_DEFAULT) {
    MaterialConst_ViscosityArrhSetField_Ascale(data,Ascale);
  }

  if (entalpy != PETSC_DEFAULT) {
    MaterialConst_ViscosityArrhSetField_entalpy(data,entalpy);
  }

  if (Vmol != PETSC_DEFAULT) {
    MaterialConst_ViscosityArrhSetField_Vmol(data,Vmol);
  }

  if (nexp != PETSC_DEFAULT) {
    MaterialConst_ViscosityArrhSetField_nexp(data,nexp);
  }

  if (Tref != PETSC_DEFAULT) {
    MaterialConst_ViscosityArrhSetField_Tref(data,Tref);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsScaleValues_ViscosityArrh(DataBucket db,const int region_id,PetscReal eta_star,PetscReal sigma_star)
{
  DataField                    PField;
  MaterialConst_ViscosityArrh  *data;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_ViscosityArrhSetField_Eta_scale(data,eta_star);

  MaterialConst_ViscosityArrhSetField_P_scale(data,sigma_star);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

/* Viscosity Arrhenius Dislocation-Diffusion */
PetscErrorCode MaterialConstantsSetDefault_ViscosityArrh_DislDiff(DataBucket db)
{
  int                                   r,nregions;
  DataField                             PField;
  MaterialConst_ViscosityArrh_DislDiff  *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_DislDiff_classname,&PField);

  data = (MaterialConst_ViscosityArrh_DislDiff*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].preexpA_disl = 1.1e5;
    data[r].Ascale_disl  = 1.0e6;
    data[r].entalpy_disl = 530.0e3;
    data[r].Vmol_disl    = 1.8e-5;
    data[r].nexp_disl    = 3.5;
    data[r].preexpA_diff = 1.5e9;
    data[r].Ascale_diff  = 1.0e6;
    data[r].entalpy_diff = 375.0e3;
    data[r].Vmol_diff    = 1.0e-5;
    data[r].pexp_diff    = 3.0;
    data[r].gsize        = 1.0e-2;
    data[r].Tref         = 273.0;
    data[r].Eta_scale    = 1.0;
    data[r].P_scale      = 1.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_ViscosityArrh_DislDiff(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                                  opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                             PField;
  MaterialConst_ViscosityArrh_DislDiff  *data;
  PetscBool                             found;
  PetscReal                             value;
  void                                  (*disl_diff_setter[12])(MaterialConst_ViscosityArrh_DislDiff*,double);
  PetscInt                              i,n_members;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_DislDiff_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* Assign functions to the array of functions */
  disl_diff_setter[ 0] = MaterialConst_ViscosityArrh_DislDiffSetField_preexpA_disl;
  disl_diff_setter[ 1] = MaterialConst_ViscosityArrh_DislDiffSetField_Ascale_disl;
  disl_diff_setter[ 2] = MaterialConst_ViscosityArrh_DislDiffSetField_entalpy_disl;
  disl_diff_setter[ 3] = MaterialConst_ViscosityArrh_DislDiffSetField_Vmol_disl;
  disl_diff_setter[ 4] = MaterialConst_ViscosityArrh_DislDiffSetField_nexp_disl;
  disl_diff_setter[ 5] = MaterialConst_ViscosityArrh_DislDiffSetField_preexpA_diff;
  disl_diff_setter[ 6] = MaterialConst_ViscosityArrh_DislDiffSetField_Ascale_diff;
  disl_diff_setter[ 7] = MaterialConst_ViscosityArrh_DislDiffSetField_entalpy_diff;
  disl_diff_setter[ 8] = MaterialConst_ViscosityArrh_DislDiffSetField_Vmol_diff;
  disl_diff_setter[ 9] = MaterialConst_ViscosityArrh_DislDiffSetField_pexp_diff;
  disl_diff_setter[10] = MaterialConst_ViscosityArrh_DislDiffSetField_gsize;
  disl_diff_setter[11] = MaterialConst_ViscosityArrh_DislDiffSetField_Tref;

  /* number of parameters that will be set */
  n_members = 12;

  /* insert options */
  value = 1.0; /* default - not required is nothing happens if option not found */

  for (i=0; i<n_members; i++) {
    PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_DislDiff_member_names[i],region_id);
    ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
    if (found) {
      disl_diff_setter[i](data,value);
    } else if ( (!found) && (essential) ) {
      ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityArrh_DislDiff_member_names[i],region_id);CHKERRQ(ierr);
    }
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}
PetscErrorCode MaterialConstantsPrintValues_ViscosityArrh_DislDiff(DataBucket db,const int region_id)
{
  char                                  opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                             PField;
  MaterialConst_ViscosityArrh_DislDiff  *data;
  void                                  (*disl_diff_getter[14])(MaterialConst_ViscosityArrh_DislDiff*,double*);
  PetscInt                              i,n_members;
  PetscReal                             value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_DislDiff_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  disl_diff_getter[ 0] = MaterialConst_ViscosityArrh_DislDiffGetField_preexpA_disl;
  disl_diff_getter[ 1] = MaterialConst_ViscosityArrh_DislDiffGetField_Ascale_disl;
  disl_diff_getter[ 2] = MaterialConst_ViscosityArrh_DislDiffGetField_entalpy_disl;
  disl_diff_getter[ 3] = MaterialConst_ViscosityArrh_DislDiffGetField_Vmol_disl;
  disl_diff_getter[ 4] = MaterialConst_ViscosityArrh_DislDiffGetField_nexp_disl;
  disl_diff_getter[ 5] = MaterialConst_ViscosityArrh_DislDiffGetField_preexpA_diff;
  disl_diff_getter[ 6] = MaterialConst_ViscosityArrh_DislDiffGetField_Ascale_diff;
  disl_diff_getter[ 7] = MaterialConst_ViscosityArrh_DislDiffGetField_entalpy_diff;
  disl_diff_getter[ 8] = MaterialConst_ViscosityArrh_DislDiffGetField_Vmol_diff;
  disl_diff_getter[ 9] = MaterialConst_ViscosityArrh_DislDiffGetField_pexp_diff;
  disl_diff_getter[10] = MaterialConst_ViscosityArrh_DislDiffGetField_gsize;
  disl_diff_getter[11] = MaterialConst_ViscosityArrh_DislDiffGetField_Tref;
  disl_diff_getter[12] = MaterialConst_ViscosityArrh_DislDiffGetField_Eta_scale;
  disl_diff_getter[13] = MaterialConst_ViscosityArrh_DislDiffGetField_P_scale;

  n_members = 14;

  for (i=0; i<n_members; i++) {
    PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityArrh_DislDiff_member_names[i],region_id);
    disl_diff_getter[i](data,&value);
    PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}
PetscErrorCode MaterialConstantsSetValues_ViscosityArrh_DislDiff(DataBucket db,const int region_id,
                                                                 PetscReal preexpA_disl,
                                                                 PetscReal Ascale_disl,
                                                                 PetscReal entalpy_disl,
                                                                 PetscReal Vmol_disl,
                                                                 PetscReal nexp_disl,
                                                                 PetscReal preexpA_diff,
                                                                 PetscReal Ascale_diff,
                                                                 PetscReal entalpy_diff,
                                                                 PetscReal Vmol_diff,
                                                                 PetscReal pexp_diff,
                                                                 PetscReal gsize,
                                                                 PetscReal Tref)
{
  DataField                             PField;
  MaterialConst_ViscosityArrh_DislDiff  *data;
  void                                  (*disl_diff_setter[12])(MaterialConst_ViscosityArrh_DislDiff*,double);
  PetscReal                             parameters[12];
  PetscInt                              i,n_members=12;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_DislDiff_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* Assign parameter to the array of parameters */
  parameters[ 0] = preexpA_disl;
  parameters[ 1] = Ascale_disl;
  parameters[ 2] = entalpy_disl;
  parameters[ 3] = Vmol_disl;
  parameters[ 4] = nexp_disl;
  parameters[ 5] = preexpA_diff;
  parameters[ 6] = Ascale_diff;
  parameters[ 7] = entalpy_diff;
  parameters[ 8] = Vmol_diff;
  parameters[ 9] = pexp_diff;
  parameters[10] = gsize;
  parameters[11] = Tref;

  /* Assign functions to the array of functions */
  disl_diff_setter[ 0] = MaterialConst_ViscosityArrh_DislDiffSetField_preexpA_disl;
  disl_diff_setter[ 1] = MaterialConst_ViscosityArrh_DislDiffSetField_Ascale_disl;
  disl_diff_setter[ 2] = MaterialConst_ViscosityArrh_DislDiffSetField_entalpy_disl;
  disl_diff_setter[ 3] = MaterialConst_ViscosityArrh_DislDiffSetField_Vmol_disl;
  disl_diff_setter[ 4] = MaterialConst_ViscosityArrh_DislDiffSetField_nexp_disl;
  disl_diff_setter[ 5] = MaterialConst_ViscosityArrh_DislDiffSetField_preexpA_diff;
  disl_diff_setter[ 6] = MaterialConst_ViscosityArrh_DislDiffSetField_Ascale_diff;
  disl_diff_setter[ 7] = MaterialConst_ViscosityArrh_DislDiffSetField_entalpy_diff;
  disl_diff_setter[ 8] = MaterialConst_ViscosityArrh_DislDiffSetField_Vmol_diff;
  disl_diff_setter[ 9] = MaterialConst_ViscosityArrh_DislDiffSetField_pexp_diff;
  disl_diff_setter[10] = MaterialConst_ViscosityArrh_DislDiffSetField_gsize;
  disl_diff_setter[11] = MaterialConst_ViscosityArrh_DislDiffSetField_Tref;

  for (i=0; i<n_members; i++) {
    if (parameters[i] != PETSC_DEFAULT) {
      disl_diff_setter[i](data,parameters[i]);
    }
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsScaleValues_ViscosityArrh_DislDiff(DataBucket db,const int region_id,PetscReal eta_star,PetscReal sigma_star)
{
  DataField                             PField;
  MaterialConst_ViscosityArrh_DislDiff  *data;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityArrh_DislDiff_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_ViscosityArrh_DislDiffSetField_Eta_scale(data,eta_star);

  MaterialConst_ViscosityArrh_DislDiffSetField_P_scale(data,sigma_star);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

/* ViscosityFK */
PetscErrorCode MaterialConstantsSetDefault_ViscosityFK(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_ViscosityFK    *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityFK_classname,&PField);

  data = (MaterialConst_ViscosityFK*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].eta0  = 1.0;
    data[r].theta = 1.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_ViscosityFK(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_ViscosityFK     *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityFK_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  value = 1.0; /* default - not required is nothing happens if option not found */

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityFK_member_names[0],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityFKSetField_eta0(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityFK_member_names[0],region_id);CHKERRQ(ierr);
  }

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityFK_member_names[1],region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_ViscosityFKSetField_theta(data,value);
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,MaterialConst_ViscosityFK_member_names[1],region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsPrintValues_ViscosityFK(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1];
  DataField                    PField;
  MaterialConst_ViscosityFK    *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityFK_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* eta0 */
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityFK_member_names[0],region_id);
  MaterialConst_ViscosityFKGetField_eta0(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",MaterialConst_ViscosityFK_member_names[1],region_id);
  MaterialConst_ViscosityFKGetField_theta(data,&value);
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetValues_ViscosityFK(DataBucket db,const int region_id,PetscReal eta0,PetscReal theta)
{
  DataField                    PField;
  MaterialConst_ViscosityFK    *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityFK_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  if (eta0 != PETSC_DEFAULT) {
    MaterialConst_ViscosityFKSetField_eta0(data,eta0);
  }
  if (theta != PETSC_DEFAULT) {
    MaterialConst_ViscosityFKSetField_theta(data,theta);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsScaleValues_ViscosityFK(DataBucket db,const int region_id,PetscReal eta_star)
{
  DataField                    PField;
  MaterialConst_ViscosityFK    *data;
  PetscReal                    eta0;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_ViscosityFK_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_ViscosityFKGetField_eta0(data,&eta0);
  eta0=eta0/eta_star;
  MaterialConst_ViscosityFKSetField_eta0(data,eta0);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


/*##########################################################################*/

/* PlasticMises */
PetscErrorCode MaterialConstantsSetDefault_PlasticMises(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_PlasticMises   *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_PlasticMises_classname,&PField);

  data = (MaterialConst_PlasticMises*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].tau_yield     = 1.0e32;
    data[r].tau_yield_inf = 1.0e32;
  }

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsSetFromOptions_PlasticMises(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1],*field_name;
  DataField                    PField;
  MaterialConst_PlasticMises   *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_PlasticMises_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* tau_yield */
  value      = 1.0; /* default - not required is nothing happens if option not found */
  field_name = (char*)MaterialConst_PlasticMises_member_names[0];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_PlasticMisesSetField_yield_stress(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  /* tau_yield_inf */
  value      = 1.0; /* default - not required is nothing happens if option not found */
  field_name = (char*)MaterialConst_PlasticMises_member_names[1];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_PlasticMisesSetField_yield_stress_inf(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsPrintValues_PlasticMises(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1],*field_name;
  DataField                    PField;
  MaterialConst_PlasticMises   *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_PlasticMises_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* tau_yield */
  field_name = (char*)MaterialConst_PlasticMises_member_names[0];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_PlasticMisesGetField_yield_stress(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  field_name = (char*)MaterialConst_PlasticMises_member_names[1];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_PlasticMisesGetField_yield_stress_inf(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetValues_PlasticMises(DataBucket db,const int region_id,PetscReal yield_stress,PetscReal yield_stress_inf)
{
  DataField                    PField;
  MaterialConst_PlasticMises   *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_PlasticMises_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  if (yield_stress != PETSC_DEFAULT) {
    MaterialConst_PlasticMisesSetField_yield_stress(data,yield_stress); /* use setter */
  }
  if (yield_stress_inf != PETSC_DEFAULT) {
    MaterialConst_PlasticMisesSetField_yield_stress_inf(data,yield_stress_inf); /* use setter */
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsScaleValues_PlasticMises(DataBucket db,const int region_id,PetscReal stress_star)
{
  DataField                    PField;
  MaterialConst_PlasticMises   *data;
  PetscReal                    yield_stress, yield_stress_inf;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_PlasticMises_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_PlasticMisesGetField_yield_stress(data,&yield_stress);
  yield_stress = yield_stress / stress_star;
  MaterialConst_PlasticMisesSetField_yield_stress(data,yield_stress);

  MaterialConst_PlasticMisesGetField_yield_stress_inf(data,&yield_stress_inf);
  yield_stress_inf = yield_stress_inf / stress_star;
  MaterialConst_PlasticMisesSetField_yield_stress_inf(data,yield_stress_inf);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


/*DP*/
PetscErrorCode MaterialConstantsSetDefault_PlasticDP(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_PlasticDP      *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_PlasticDP_classname,&PField);

  data = (MaterialConst_PlasticDP*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].Co      = 1.0e32;
    data[r].Co_inf  = 1.0e32;
    data[r].phi     = 0.6;
    data[r].phi_inf = 0.6;
    data[r].tens_cutoff = 1.0;
    data[r].hst_cutoff = 1.0e32;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_PlasticDP(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1],*field_name;
  DataField                    PField;
  MaterialConst_PlasticDP      *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_PlasticDP_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* friction */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_PlasticDP_member_names[0];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_PlasticDPSetField_friction(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  /* friction_inf */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_PlasticDP_member_names[2];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_PlasticDPSetField_friction_inf(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }
  
  /* Cohesion */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_PlasticDP_member_names[1];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_PlasticDPSetField_cohesion(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  /* cohesion_inf */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_PlasticDP_member_names[3];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_PlasticDPSetField_cohesion_inf(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  /* tension_cutoff */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_PlasticDP_member_names[4];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_PlasticDPSetField_tens_cutoff(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  /* high stress cutoff */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_PlasticDP_member_names[5];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_PlasticDPSetField_hst_cutoff(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsPrintValues_PlasticDP(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1],*field_name;
  DataField                    PField;
  MaterialConst_PlasticDP      *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_PlasticDP_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);


  /* friction */
  field_name = (char*)MaterialConst_PlasticDP_member_names[0];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_PlasticDPGetField_friction(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  /* friction_inf */
  field_name = (char*)MaterialConst_PlasticDP_member_names[2];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_PlasticDPGetField_friction_inf(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);
  /* Cohesion */
  field_name = (char*)MaterialConst_PlasticDP_member_names[1];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_PlasticDPGetField_cohesion(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);
  /* cohesion_inf */
  field_name = (char*)MaterialConst_PlasticDP_member_names[3];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_PlasticDPGetField_cohesion_inf(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  /* tension_cutoff */
  field_name = (char*)MaterialConst_PlasticDP_member_names[4];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_PlasticDPGetField_tens_cutoff(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  /* high stress cutoff */
  field_name = (char*)MaterialConst_PlasticDP_member_names[5];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_PlasticDPGetField_hst_cutoff(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetValues_PlasticDP(DataBucket db,const int region_id,PetscReal friction,PetscReal friction_inf,PetscReal cohesion,PetscReal cohesion_inf,PetscReal tens_cutoff,PetscReal hst_cutoff)
{
  DataField                    PField;
  MaterialConst_PlasticDP      *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_PlasticDP_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  if (friction != PETSC_DEFAULT) {
    if ( (friction > 2.0 * M_PI) || (friction < 0.0) ) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"The DP phi parameter is assumed to be given as an angle in radians - your value : \"friction = %1.6e\" \n",friction);
    }
    if ( (friction_inf > 2.0 * M_PI) || (friction_inf < 0.0) ) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"The DP phi_inf parameter is assumed to be given as an angle in radians - your value : \"friction_inf = %1.6e\" \n",friction_inf);
    }
    if ( friction > 45.0 * M_PI/180.0) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"The DP phi parameter is using a friction angle greater than 45 degress - your value : \"friction = %1.6e\" \n",friction);
    }

    MaterialConst_PlasticDPSetField_friction(data,friction); /* use setter */
  }
  if (friction_inf != PETSC_DEFAULT) {
    MaterialConst_PlasticDPSetField_friction_inf(data,friction_inf); /* use setter */
  }
  if (cohesion != PETSC_DEFAULT) {
    MaterialConst_PlasticDPSetField_cohesion(data,cohesion); /* use setter */
  }
  if (cohesion_inf != PETSC_DEFAULT) {
    MaterialConst_PlasticDPSetField_cohesion_inf(data,cohesion_inf); /* use setter */
  }
  if (tens_cutoff != PETSC_DEFAULT) {
    MaterialConst_PlasticDPSetField_tens_cutoff(data,tens_cutoff); /* use setter */
  }
  if (hst_cutoff != PETSC_DEFAULT) {
    MaterialConst_PlasticDPSetField_hst_cutoff(data,hst_cutoff); /* use setter */
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsScaleValues_PlasticDP(DataBucket db,const int region_id, PetscReal stress_star)
{
  DataField                    PField;
  MaterialConst_PlasticDP      *data;
  PetscReal                    cohesion,cohesion_inf,tens_cutoff,hst_cutoff;
  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_PlasticDP_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_PlasticDPGetField_cohesion(data,&cohesion); /* use setter */
  cohesion = cohesion/stress_star;
  MaterialConst_PlasticDPSetField_cohesion(data,cohesion); /* use setter */

  MaterialConst_PlasticDPGetField_cohesion_inf(data,&cohesion_inf); /* use setter */
  cohesion_inf = cohesion_inf/stress_star;
  MaterialConst_PlasticDPSetField_cohesion_inf(data,cohesion_inf); /* use setter */

  MaterialConst_PlasticDPGetField_tens_cutoff(data,&tens_cutoff); /* use setter */
  tens_cutoff = tens_cutoff/stress_star;
  MaterialConst_PlasticDPSetField_tens_cutoff(data,tens_cutoff); /* use setter */

  MaterialConst_PlasticDPGetField_hst_cutoff(data,&hst_cutoff); /* use setter */
  hst_cutoff = hst_cutoff/stress_star;
  MaterialConst_PlasticDPSetField_hst_cutoff(data,hst_cutoff); /* use setter */

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

/*######################       SOFTENING LAWS        ########################*/
/*  NONE OF THEM NEED SCALING FOR THE MOMENT  BUT I AM THINKING ABOUT ONE    */

PetscErrorCode MaterialConstantsSetDefault_SoftLin(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_SoftLin        *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_SoftLin_classname,&PField);

  data = (MaterialConst_SoftLin*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].eps_min      = 0.0;
    data[r].eps_max      = 1.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_SoftLin(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1],*field_name;
  DataField                    PField;
  MaterialConst_SoftLin        *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_SoftLin_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* minimum strain */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_SoftLin_member_names[0];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_SoftLinSetField_eps_min(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  /* maximum strain */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_SoftLin_member_names[1];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_SoftLinSetField_eps_max(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }


  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsPrintValues_SoftLin(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1],*field_name;
  DataField                    PField;
  MaterialConst_SoftLin        *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_SoftLin_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* minimum strain */
  field_name = (char*)MaterialConst_SoftLin_member_names[0];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_SoftLinGetField_eps_min(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  /* exponential fold */
  field_name = (char*)MaterialConst_SoftLin_member_names[1];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_SoftLinGetField_eps_max(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetValues_SoftLin(DataBucket db,const int region_id,PetscReal emin,PetscReal emax)
{
  DataField                    PField;
  MaterialConst_SoftLin      *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_SoftLin_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  if (emin  != PETSC_DEFAULT) {
    MaterialConst_SoftLinSetField_eps_min(data,emin); /* use setter */
  }
  if (emax != PETSC_DEFAULT) {
    MaterialConst_SoftLinSetField_eps_max(data,emax); /* use setter */
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetDefault_SoftExpo(DataBucket db)
{
  int                          r,nregions;
  DataField                    PField;
  MaterialConst_SoftExpo      *data;

  PetscFunctionBegin;

  DataBucketGetSizes(db,&nregions,NULL,NULL);
  DataBucketGetDataFieldByName(db,MaterialConst_SoftExpo_classname,&PField);

  data = (MaterialConst_SoftExpo*)PField->data; /* should write a function to do this */
  for (r=0; r<nregions; r++) {
    data[r].eps_min      = 0.0;
    data[r].eps_fold     = 1.0;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions_SoftExpo(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1],*field_name;
  DataField                    PField;
  MaterialConst_SoftExpo      *data;
  PetscBool                    found;
  PetscReal                    value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_SoftExpo_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  /* insert options */
  /* minimum strain */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_SoftExpo_member_names[0];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_SoftExpoSetField_eps_min(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }

  /* exponential fold */
  value      = 1.0; /* default - not required as nothing happens if option not found */
  field_name = (char*)MaterialConst_SoftExpo_member_names[1];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
  if (found) {
    MaterialConst_SoftExpoSetField_eps_fold(data,value); /* use setter */
  } else if ( (!found)  && (essential) ) {
    ierr = MaterialConstantsReportParseError(model_name,field_name,region_id);CHKERRQ(ierr);
  }


  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}


PetscErrorCode MaterialConstantsPrintValues_SoftExpo(DataBucket db,const int region_id)
{
  char                         opt_name[PETSC_MAX_PATH_LEN-1],*field_name;
  DataField                    PField;
  MaterialConst_SoftExpo      *data;
  PetscReal                    value;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_SoftExpo_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);


  /* minimum strain */
  field_name = (char*)MaterialConst_SoftExpo_member_names[0];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_SoftExpoGetField_eps_min(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  /* exponential fold */
  field_name = (char*)MaterialConst_SoftExpo_member_names[1];
  PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-%s_%d",field_name,region_id);
  MaterialConst_SoftExpoGetField_eps_fold(data,&value); /* use setter */
  PetscPrintf(PETSC_COMM_WORLD,"Current Value %s   :  %1.4e  \n", opt_name ,value);

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetValues_SoftExpo(DataBucket db,const int region_id,PetscReal emin,PetscReal efold)
{
  DataField                    PField;
  MaterialConst_SoftExpo      *data;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_SoftExpo_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  if (emin != PETSC_DEFAULT) {
    MaterialConst_SoftExpoSetField_eps_min(data,emin); /* use setter */
  }
  if (efold != PETSC_DEFAULT) {
    MaterialConst_SoftExpoSetField_eps_fold(data,efold); /* use setter */
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

/*###################### GENERAL STUFF FOR CONSTANTS ########################*/

PetscErrorCode MaterialConstantsReportParseError(const char model_name[],const char field_name[],const int region)
{
  PetscFunctionBegin;
  if (model_name) {
    PetscPrintf(PETSC_COMM_WORLD,"Expected user to provide option for field (%s) in region (%D) via -%s%s_%D \n",field_name,region,  model_name,field_name,region);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Expected user to provide option for field (%s) in region (%D) via -%s_%D \n",field_name,region,    field_name,region);
  }
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Option not found");
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetDefaults(DataBucket db)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = MaterialConstantsSetDefault_MaterialType(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_ViscosityConst(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_ViscosityZ(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_ViscosityFK(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_ViscosityArrh(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_ViscosityArrh_DislDiff(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_PlasticMises(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_PlasticDP(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_DensityConst(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_DensityBoussinesq(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_DensityTable(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_SoftLin(db);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefault_SoftExpo(db);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsSetFromOptions(DataBucket db,const char model_name[],const int region_id,PetscBool essential)
{
  DataField                    PField;
  MaterialConst_MaterialType   *data;
  int                          value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  ierr = MaterialConstantsSetFromOptions_MaterialType(db,model_name,region_id,essential);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(db,MaterialConst_MaterialType_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_MaterialTypeGetField_visc_type(data,&value);
  switch (value) {
    case VISCOUS_CONSTANT:
      ierr = MaterialConstantsSetFromOptions_ViscosityConst(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case VISCOUS_FRANKK:
      ierr = MaterialConstantsSetFromOptions_ViscosityFK(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case VISCOUS_Z:
      ierr = MaterialConstantsSetFromOptions_ViscosityZ(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS:
      ierr = MaterialConstantsSetFromOptions_ViscosityArrh(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS_2:
      ierr = MaterialConstantsSetFromOptions_ViscosityArrh(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS_DISLDIFF:
      ierr = MaterialConstantsSetFromOptions_ViscosityArrh_DislDiff(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"VISCOUS TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_plastic_type(data,&value);
  switch (value) {
    case PLASTIC_NONE:
      break;
    case PLASTIC_MISES:
      ierr = MaterialConstantsSetFromOptions_PlasticMises(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case PLASTIC_MISES_H:
      ierr = MaterialConstantsSetFromOptions_PlasticMises(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case PLASTIC_DP:
      ierr = MaterialConstantsSetFromOptions_PlasticDP(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case PLASTIC_DP_H:
      ierr = MaterialConstantsSetFromOptions_PlasticDP(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"PLASTIC TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_softening_type(data,&value);
  switch (value) {
    case SOFTENING_NONE:

      break;
    case SOFTENING_LINEAR:
      ierr = MaterialConstantsSetFromOptions_SoftLin(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case SOFTENING_EXPONENTIAL:
      ierr = MaterialConstantsSetFromOptions_SoftExpo(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"SOFTENING TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_density_type(data,&value);
  switch (value) {
    case DENSITY_CONSTANT:
      ierr = MaterialConstantsSetFromOptions_DensityConst(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case DENSITY_BOUSSINESQ:
      ierr = MaterialConstantsSetFromOptions_DensityBoussinesq(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    case DENSITY_TABLE:
      ierr = MaterialConstantsSetFromOptions_DensityTable(db,model_name,region_id,essential);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"DENSITY TYPE UNDEFINED");
      break;
  }

  DataFieldRestoreAccess(PField);


  PetscFunctionReturn(0);

}

PetscErrorCode MaterialConstantsScaleAll(DataBucket db,const int region_id,PetscReal L_star, PetscReal U_star,PetscReal t_star,PetscReal eta_star,PetscReal rho_star,PetscReal P_star)
{
  DataField                    PField;
  MaterialConst_MaterialType   *data;
  int                          value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_MaterialType_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_MaterialTypeGetField_visc_type(data,&value);
  switch (value) {
    case VISCOUS_CONSTANT:
      ierr = MaterialConstantsScaleValues_ViscosityConst(db,region_id,eta_star);CHKERRQ(ierr);
      break;
    case VISCOUS_FRANKK:
      ierr = MaterialConstantsScaleValues_ViscosityFK(db,region_id,eta_star);CHKERRQ(ierr);
      break;
    case VISCOUS_Z:
      ierr = MaterialConstantsScaleValues_ViscosityZ(db,region_id,eta_star,L_star);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS:
      ierr = MaterialConstantsScaleValues_ViscosityArrh(db,region_id,eta_star,P_star);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS_2:
      ierr = MaterialConstantsScaleValues_ViscosityArrh(db,region_id,eta_star,P_star);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS_DISLDIFF:
      ierr = MaterialConstantsScaleValues_ViscosityArrh_DislDiff(db,region_id,eta_star,P_star);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"VISCOUS TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_plastic_type(data,&value);
  switch (value) {
    case PLASTIC_NONE:
      break;
    case PLASTIC_MISES:
      ierr = MaterialConstantsScaleValues_PlasticMises(db,region_id,P_star);CHKERRQ(ierr);
      break;
    case PLASTIC_MISES_H:
      ierr = MaterialConstantsScaleValues_PlasticMises(db,region_id,P_star);CHKERRQ(ierr);
      break;
    case PLASTIC_DP:
      ierr = MaterialConstantsScaleValues_PlasticDP(db,region_id,P_star);CHKERRQ(ierr);
      break;
    case PLASTIC_DP_H:
      ierr = MaterialConstantsScaleValues_PlasticDP(db,region_id,P_star);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"PLASTIC TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_softening_type(data,&value);
  switch (value) {
    case SOFTENING_NONE:
      break;
    case SOFTENING_LINEAR:
      break;
    case SOFTENING_EXPONENTIAL:
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"SOFTENING TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_density_type(data,&value);
  switch (value) {
    case DENSITY_CONSTANT:
      ierr = MaterialConstantsScaleValues_DensityConst(db,region_id,rho_star);CHKERRQ(ierr);
      break;
    case DENSITY_BOUSSINESQ:
      ierr = MaterialConstantsScaleValues_DensityBoussinesq(db,region_id,rho_star,P_star);CHKERRQ(ierr);
      break;
    case DENSITY_TABLE:
      ierr = MaterialConstantsScaleValues_DensityTable(db,region_id,rho_star,P_star);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"DENSITY TYPE UNDEFINED");
      break;
  }

  DataFieldRestoreAccess(PField);


  PetscFunctionReturn(0);
}

PetscErrorCode MaterialConstantsPrintAll(DataBucket db,const int region_id)
{
  DataField                    PField;
  MaterialConst_MaterialType   *data;
  int                          value;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  DataBucketGetDataFieldByName(db,MaterialConst_MaterialType_classname,&PField);

  DataFieldGetAccess(PField);
  DataFieldAccessPoint(PField,region_id,(void**)&data);

  MaterialConst_MaterialTypeGetField_visc_type(data,&value);
  switch (value) {
    case VISCOUS_CONSTANT:
      ierr = MaterialConstantsPrintValues_ViscosityConst(db,region_id);CHKERRQ(ierr);
      break;
    case VISCOUS_FRANKK:
      ierr = MaterialConstantsPrintValues_ViscosityFK(db,region_id);CHKERRQ(ierr);
      break;
    case VISCOUS_Z:
      ierr = MaterialConstantsPrintValues_ViscosityZ(db,region_id);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS:
      ierr = MaterialConstantsPrintValues_ViscosityArrh(db,region_id);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS_2:
      ierr = MaterialConstantsPrintValues_ViscosityArrh(db,region_id);CHKERRQ(ierr);
      break;
    case VISCOUS_ARRHENIUS_DISLDIFF:
      ierr = MaterialConstantsPrintValues_ViscosityArrh_DislDiff(db,region_id);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"VISCOUS TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_plastic_type(data,&value);
  switch (value) {
    case PLASTIC_NONE:

      break;
    case PLASTIC_MISES:
      ierr = MaterialConstantsPrintValues_PlasticMises(db,region_id);CHKERRQ(ierr);
      break;
    case PLASTIC_MISES_H:
      ierr = MaterialConstantsPrintValues_PlasticMises(db,region_id);CHKERRQ(ierr);
      break;
    case PLASTIC_DP:
      ierr = MaterialConstantsPrintValues_PlasticDP(db,region_id);CHKERRQ(ierr);
      break;
    case PLASTIC_DP_H:
      ierr = MaterialConstantsPrintValues_PlasticDP(db,region_id);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"PLASTIC TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_softening_type(data,&value);
  switch (value) {
    case SOFTENING_NONE:
      break;
    case SOFTENING_LINEAR:
      ierr = MaterialConstantsPrintValues_SoftLin(db,region_id);CHKERRQ(ierr);
      break;
    case SOFTENING_EXPONENTIAL:
      ierr = MaterialConstantsPrintValues_SoftExpo(db,region_id);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"DENSITY TYPE UNDEFINED");
      break;
  }

  MaterialConst_MaterialTypeGetField_density_type(data,&value);
  switch (value) {
    case DENSITY_CONSTANT:
      ierr = MaterialConstantsPrintValues_DensityConst(db,region_id);CHKERRQ(ierr);
      break;
    case DENSITY_BOUSSINESQ:
      ierr = MaterialConstantsPrintValues_DensityBoussinesq(db,region_id);CHKERRQ(ierr);
      break;
    case DENSITY_TABLE:
      ierr = MaterialConstantsPrintValues_DensityTable(db,region_id);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"DENSITY TYPE UNDEFINED");
      break;
  }

  DataFieldRestoreAccess(PField);

  PetscFunctionReturn(0);
}

