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
 **    filename:   MPntPEnergy_def.h
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
  Auto generated by version 0.0 of swarm_class_generator.py
  on lifou, at 2024-07-08 16:37:56.412428 by jourdon
*/

#ifndef __MPntPEnergy_DEF_H__
#define __MPntPEnergy_DEF_H__

#include <mpi.h>

typedef struct {
 double diffusivity;
 double heat_source;
 double heat_source_init;
} MPntPEnergy;

typedef enum {
  MPPEgy_diffusivity = 0,
  MPPEgy_heat_source,
  MPPEgy_heat_source_init
} MPntPEnergyTypeName;

extern const char MPntPEnergy_classname[];

extern const int MPntPEnergy_nmembers;

extern const size_t MPntPEnergy_member_sizes[];

extern const char *MPntPEnergy_member_names[];

extern MPI_Datatype MPI_MPNTPENERGY;

/* prototypes */
void MPntPEnergyGetField_diffusivity(MPntPEnergy *point,double *data);
void MPntPEnergyGetField_heat_source(MPntPEnergy *point,double *data);
void MPntPEnergyGetField_heat_source_init(MPntPEnergy *point,double *data);
void MPntPEnergySetField_diffusivity(MPntPEnergy *point,double data);
void MPntPEnergySetField_heat_source(MPntPEnergy *point,double data);
void MPntPEnergySetField_heat_source_init(MPntPEnergy *point,double data);
void MPntPEnergyView(MPntPEnergy *point);
void MPntPEnergyVTKWriteAsciiAllFields(FILE *vtk_fp,const int N,const MPntPEnergy points[]);
void MPntPEnergyPVTUWriteAllPPointDataFields(FILE *vtk_fp);
void MPntPEnergyVTKWriteBinaryAppendedHeaderAllFields(FILE *vtk_fp,int *offset,const int N,const MPntPEnergy points[]);
void MPntPEnergyVTKWriteBinaryAppendedDataAllFields(FILE *vtk_fp,const int N,const MPntPEnergy points[]);
int MPntPEnergyCreateMPIDataType(MPI_Datatype *ptype);

#endif
