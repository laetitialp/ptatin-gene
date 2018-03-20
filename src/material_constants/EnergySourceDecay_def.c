/*
  Auto generated by version 0.0 of material_constant_generator.py
  on eduroam-nw-dock-1-133.ethz.ch, at 2015-09-21 13:23:20.589804 by gduclaux
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <petsc.h>

#include "material_constants/EnergySourceDecay_def.h"

PetscErrorCode MaterialConstantsReportParseError(const char model_name[],const char field_name[],const int region);

const char EnergySourceDecay_classname[] = "EnergySourceDecay";

const int EnergySourceDecay_nmembers = 2;

const size_t EnergySourceDecay_member_sizes[] = {
  1 * sizeof(double),
  1 * sizeof(double)
};

const char *EnergySourceDecay_member_names[] = {
  "HeatSourceRef",
  "HalfLife"
};


/* ================================================================= */
/*   Getters for EnergySourceDecay */
/* ================================================================= */
void EnergySourceDecayGetField_HeatSourceRef(EnergySourceDecay *point,double *data) 
{
  *data = point->H0;
}

void EnergySourceDecayGetField_HalfLife(EnergySourceDecay *point,double *data) 
{
  *data = point->lambda;
}


/* ================================================================= */
/*   Setters for EnergySourceDecay */
/* ================================================================= */
void EnergySourceDecaySetField_HeatSourceRef(EnergySourceDecay *point,double data) 
{
  point->H0 = data;
}

void EnergySourceDecaySetField_HalfLife(EnergySourceDecay *point,double data) 
{
  point->lambda = data;
}


/* ================================================================= */
/*   C-viewer for EnergySourceDecay */
/* ================================================================= */
void EnergySourceDecayView(EnergySourceDecay *point)
{
  {
    double data;
    EnergySourceDecayGetField_HeatSourceRef(point,&data);
    printf("field: HeatSourceRef = %1.6e; [size %zu; type double; variable_name H0]\n",data, EnergySourceDecay_member_sizes[0] );
  }
  {
    double data;
    EnergySourceDecayGetField_HalfLife(point,&data);
    printf("field: HalfLife = %1.6e; [size %zu; type double; variable_name lambda]\n",data, EnergySourceDecay_member_sizes[1] );
  }
}


/* ================================================================= */
/*   Getters for default parameters (EnergySourceDecay) */
/* ================================================================= */
void EnergySourceDecayGetDefault_HeatSourceRef(double *data) 
{
  *data = (double)0.0;
}

void EnergySourceDecayGetDefault_HalfLife(double *data) 
{
  *data = (double)0.0;
}

void MaterialConstantsSetDefaultAll_SourceDecay( 
    int nr,EnergySourceDecay _data[])
{
  int r; 

  for (r=0; r<nr; r++) {
    { double value;
      EnergySourceDecayGetDefault_HeatSourceRef((double*)&value);
      EnergySourceDecaySetField_HeatSourceRef(&_data[r],(double)value);
    }

    { double value;
      EnergySourceDecayGetDefault_HalfLife((double*)&value);
      EnergySourceDecaySetField_HalfLife(&_data[r],(double)value);
    }

  }

} 

#undef __FUNCT__
#define __FUNCT__ "MaterialConstantsSetFromOptions_SourceDecay"
PetscErrorCode MaterialConstantsSetFromOptions_SourceDecay(const char model_name[],const int region_id,EnergySourceDecay _data[],PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN];
  PetscBool                    found;
  PetscErrorCode               ierr;

  EnergySourceDecay *data = &_data[region_id];
  /* options for HeatSourceRef ==>> H0 */
  sprintf(opt_name,"-HeatSourceRef_%d",region_id);
  { PetscReal value;
    ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
    if (found) { EnergySourceDecaySetField_HeatSourceRef(data,(double)value); }
    else if ( (!found)  && (essential) ) {
      ierr = MaterialConstantsReportParseError(model_name,"HeatSourceRef",region_id);CHKERRQ(ierr);
  }}

  /* options for HalfLife ==>> lambda */
  sprintf(opt_name,"-HalfLife_%d",region_id);
  { PetscReal value;
    ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
    if (found) { EnergySourceDecaySetField_HalfLife(data,(double)value); }
    else if ( (!found)  && (essential) ) {
      ierr = MaterialConstantsReportParseError(model_name,"HalfLife",region_id);CHKERRQ(ierr);
  }}

  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MaterialConstantsPrintValues_SourceDecay"
PetscErrorCode MaterialConstantsPrintValues_SourceDecay(const char model_name[],const int region_id,EnergySourceDecay _data[]) 
{
  EnergySourceDecay *data = &_data[region_id];
  char   opt_name[PETSC_MAX_PATH_LEN];

  PetscPrintf(PETSC_COMM_WORLD,"------------------------------------------------------------------------------------------------\n");
  PetscPrintf(PETSC_COMM_WORLD,"  MaterialView(SourceDecay): RegionIndex[%d]\n", region_id);
  /* options for HeatSourceRef ==>> H0 */
  sprintf(opt_name,"-%s_HeatSourceRef_%d", model_name,region_id);
  { double value;
    EnergySourceDecayGetField_HeatSourceRef(data,(double*)&value);
    PetscPrintf(PETSC_COMM_WORLD,"    HeatSourceRef = %1.4e (%s) \n", value,opt_name); 
  }

  /* options for HalfLife ==>> lambda */
  sprintf(opt_name,"-%s_HalfLife_%d", model_name,region_id);
  { double value;
    EnergySourceDecayGetField_HalfLife(data,(double*)&value);
    PetscPrintf(PETSC_COMM_WORLD,"    HalfLife = %1.4e (%s) \n", value,opt_name); 
  }

  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MaterialConstantsSetValues_SourceDecay"
PetscErrorCode MaterialConstantsSetValues_SourceDecay(const int region_id,EnergySourceDecay _data[],
    double H0,
    double lambda)
{
  EnergySourceDecay *data = &_data[region_id];
  data->H0 =  H0;
  data->lambda =  lambda;
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MaterialConstantsScaleValues_SourceDecay"
PetscErrorCode MaterialConstantsScaleValues_SourceDecay(const int region_id,EnergySourceDecay _data[],
    double H0,
    double lambda)
{
  EnergySourceDecay *data = &_data[region_id];

  { double value;
    EnergySourceDecayGetField_HeatSourceRef(data,(double*)&value);
    value = value / H0;
    EnergySourceDecaySetField_HeatSourceRef(data,(double)value);
  }

  { double value;
    EnergySourceDecayGetField_HalfLife(data,(double*)&value);
    value = value / lambda;
    EnergySourceDecaySetField_HalfLife(data,(double)value);
  }

  PetscFunctionReturn(0);
} 

