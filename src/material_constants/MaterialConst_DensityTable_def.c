
/*
  Auto generated by version 0.0 of swarm_class_generator.py
  on mesu2, at 2024-05-03 08:23:40.484017 by lepourh
*/

#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "material_constants/MaterialConst_DensityTable_def.h"


const char MaterialConst_DensityTable_classname[] = "MaterialConst_DensityTable";

const int MaterialConst_DensityTable_nmembers = 2;

const size_t MaterialConst_DensityTable_member_sizes[] = {
  1  * sizeof(double),
  1  * sizeof(PhaseMap)
};

const char *MaterialConst_DensityTable_member_names[] = {
  "density",
  "map"
};

MPI_Datatype MPI_MATERIALCONST_DENSITYTABLE;


/* ===================================== */
/* Getters for MaterialConst_DensityTable */
/* ===================================== */
void MaterialConst_DensityTableGetField_density(MaterialConst_DensityTable *point,double *data) 
{
  *data = point->density;
}

void MaterialConst_DensityTableGetField_map(MaterialConst_DensityTable *point,PhaseMap *data) 
{
  *data = point->map;
}


/* ===================================== */
/* Setters for MaterialConst_DensityTable */
/* ===================================== */
void MaterialConst_DensityTableSetField_density(MaterialConst_DensityTable *point,double data) 
{
  point->density = data;
  PetscPrintf(PETSC_COMM_WORLD," %f \n", point->density); 
}
void MaterialConst_DensityTableSetField_map(MaterialConst_DensityTable *point,PhaseMap data) 
{
  point->map = data;
}


/* ===================================== */
/* C-viewer for MaterialConst_DensityTable */
/* ===================================== */
void MaterialConst_DensityTableView(MaterialConst_DensityTable *point)
{
  {
    double data;
    MaterialConst_DensityTableGetField_density(point,&data);
    printf("field: density = %1.6e; [size %zu; type double; variable_name density]\n",data, MaterialConst_DensityTable_member_sizes[0] );
  }
}


