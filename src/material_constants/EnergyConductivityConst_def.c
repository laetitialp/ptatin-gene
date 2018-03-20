/*
  Auto generated by version 0.0 of material_constant_generator.py
  on geop-318.ethz.ch, at 2015-09-22 11:27:29.366433 by dmay
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <petsc.h>

#include "material_constants/EnergyConductivityConst_def.h"


PetscErrorCode MaterialConstantsReportParseError(const char model_name[],const char field_name[],const int region);

const char EnergyConductivityConst_classname[] = "EnergyConductivityConst";

const int EnergyConductivityConst_nmembers = 1;

const size_t EnergyConductivityConst_member_sizes[] = {
  1 * sizeof(double)
};

const char *EnergyConductivityConst_member_names[] = {
  "k0"
};


/* ================================================================= */
/*   Getters for EnergyConductivityConst */
/* ================================================================= */
void EnergyConductivityConstGetField_k0(EnergyConductivityConst *point,double *data) 
{
  *data = point->k0;
}


/* ================================================================= */
/*   Setters for EnergyConductivityConst */
/* ================================================================= */
void EnergyConductivityConstSetField_k0(EnergyConductivityConst *point,double data) 
{
  point->k0 = data;
}


/* ================================================================= */
/*   C-viewer for EnergyConductivityConst */
/* ================================================================= */
void EnergyConductivityConstView(EnergyConductivityConst *point)
{
  {
    double data;
    EnergyConductivityConstGetField_k0(point,&data);
    printf("field: k0 = %1.6e; [size %zu; type double; variable_name k0]\n",data, EnergyConductivityConst_member_sizes[0] );
  }
}


/* ================================================================= */
/*   Getters for default parameters (EnergyConductivityConst) */
/* ================================================================= */
void EnergyConductivityConstGetDefault_k0(double *data) 
{
  *data = (double)0.0;
}

void MaterialConstantsSetDefaultAll_ConductivityConst( 
    int nr,EnergyConductivityConst _data[])
{
  int r; 

  for (r=0; r<nr; r++) {
    { double value;
      EnergyConductivityConstGetDefault_k0((double*)&value);
      EnergyConductivityConstSetField_k0(&_data[r],(double)value);
    }

  }

} 

#undef __FUNCT__
#define __FUNCT__ "MaterialConstantsSetFromOptions_ConductivityConst"
PetscErrorCode MaterialConstantsSetFromOptions_ConductivityConst(const char model_name[],const int region_id,EnergyConductivityConst _data[],PetscBool essential)
{
  char                         opt_name[PETSC_MAX_PATH_LEN];
  PetscBool                    found;
  PetscErrorCode               ierr;

  EnergyConductivityConst *data = &_data[region_id];
  /* options for k0 ==>> k0 */
  sprintf(opt_name,"-k0_%d",region_id);
  { PetscReal value;
    ierr = PetscOptionsGetReal(NULL,model_name,opt_name,&value,&found);CHKERRQ(ierr);
    if (found) {
      data->k0 = (double)value;
    }
    else if ( (!found)  && (essential) ) {
      ierr = MaterialConstantsReportParseError(model_name,"k0",region_id);CHKERRQ(ierr);
  }}

  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MaterialConstantsPrintValues_ConductivityConst"
PetscErrorCode MaterialConstantsPrintValues_ConductivityConst(const char model_name[],const int region_id,EnergyConductivityConst _data[]) 
{
  EnergyConductivityConst *data = &_data[region_id];
  char   opt_name[PETSC_MAX_PATH_LEN];

  PetscPrintf(PETSC_COMM_WORLD,"------------------------------------------------------------------------------------------------\n");
  PetscPrintf(PETSC_COMM_WORLD,"  MaterialView(ConductivityConst): RegionIndex[%d]\n", region_id);
  /* options for k0 ==>> k0 */
  sprintf(opt_name,"-%s_k0_%d", model_name,region_id);
  { double value;
    EnergyConductivityConstGetField_k0(data,(double*)&value);
    PetscPrintf(PETSC_COMM_WORLD,"    k0 = %1.4e (%s) \n", value,opt_name); 
  }

  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MaterialConstantsSetValues_ConductivityConst"
PetscErrorCode MaterialConstantsSetValues_ConductivityConst(const int region_id,EnergyConductivityConst _data[],
    double k0)
{
  EnergyConductivityConst *data = &_data[region_id];
  data->k0 =  k0;
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MaterialConstantsScaleValues_ConductivityConst"
PetscErrorCode MaterialConstantsScaleValues_ConductivityConst(const int region_id,EnergyConductivityConst _data[],
    double k0)
{
  EnergyConductivityConst *data = &_data[region_id];

  { double value;
    EnergyConductivityConstGetField_k0(data,(double*)&value);
    value = value / k0;
    EnergyConductivityConstSetField_k0(data,(double)value);
  }

  PetscFunctionReturn(0);
} 

