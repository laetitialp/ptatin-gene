/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2012, 
 **        Dave A. May [dave.may@erdw.ethz.ch]
 **        Geophysical Fluid Dynamics, 
 **        Department of Earth Sciences,
 **        ETH Zürich,
 **        Sonneggstrasse 5,
 **        CH-8092 Zurich,
 **        Switzerland
 **
 **    Project:       pTatin3d
 **    Filename:      QPntVolCoefStokes_def.c
 **
 **
 **    pTatin3d is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published by
 **    the Free Software Foundation, either version 3 of the License, or
 **    (at your option) any later version.
 **
 **    pTatin3d is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 **    GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with pTatin3d.  If not, see <http://www.gnu.org/licenses/>.
 **
 **
 **    $Id$
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@*/



/*
  Auto generated by version 0.0 of swarm_class_generator.py
  on geop-043.ethz.ch, at 2012-02-14 21:01:55.587403 by dmay
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "QPntVolCoefStokes_def.h"


const char QPntVolCoefStokes_classname[] = "QPntVolCoefStokes";

const int QPntVolCoefStokes_nmembers = 4;

const size_t QPntVolCoefStokes_member_sizes[] = {
  1 * sizeof(double),
  1 * sizeof(double),
  3 * sizeof(double),
  1 * sizeof(double)
} ;

const char *QPntVolCoefStokes_member_names[] = {
  "eta_effective",
  "rho_effective",
  "momentum_rhs",
  "continuity_rhs"
} ;


/* ===================================== */
/* Getters for QPntVolCoefStokes */
/* ===================================== */
void QPntVolCoefStokesGetField_eta_effective(QPntVolCoefStokes *point,double *data) 
{
  *data = point->eta;
}

void QPntVolCoefStokesGetField_rho_effective(QPntVolCoefStokes *point,double *data) 
{
  *data = point->rho;
}

void QPntVolCoefStokesGetField_momentum_rhs(QPntVolCoefStokes *point,double *data[]) 
{
  *data = point->Fu;
}

void QPntVolCoefStokesGetField_continuity_rhs(QPntVolCoefStokes *point,double *data) 
{
  *data = point->Fp;
}


/* ===================================== */
/* Setters for QPntVolCoefStokes */
/* ===================================== */
void QPntVolCoefStokesSetField_eta_effective(QPntVolCoefStokes *point,double data) 
{
  point->eta = data;
}

void QPntVolCoefStokesSetField_rho_effective(QPntVolCoefStokes *point,double data) 
{
  point->rho = data;
}

void QPntVolCoefStokesSetField_momentum_rhs(QPntVolCoefStokes *point,double data[]) 
{
  memcpy( &point->Fu[0], data, sizeof(double)*3 );
}

void QPntVolCoefStokesSetField_continuity_rhs(QPntVolCoefStokes *point,double data) 
{
  point->Fp = data;
}


/* ===================================== */
/* C-viewer for QPntVolCoefStokes */
/* ===================================== */
void QPntVolCoefStokesView(QPntVolCoefStokes *point)
{
  {
    double data;
    QPntVolCoefStokesGetField_eta_effective(point,&data);
    printf("field: eta_effective = %1.6e; [size %zu; type double; variable_name eta]\n",data, QPntVolCoefStokes_member_sizes[0] );
  }
  {
    double data;
    QPntVolCoefStokesGetField_rho_effective(point,&data);
    printf("field: rho_effective = %1.6e; [size %zu; type double; variable_name rho]\n",data, QPntVolCoefStokes_member_sizes[1] );
  }
  {
    double *data;
    QPntVolCoefStokesGetField_momentum_rhs(point,&data);
    printf("field: momentum_rhs[0] = %1.6e; [size %zu; type double; variable_name Fu]\n",data[0], QPntVolCoefStokes_member_sizes[2] );
    printf("field: momentum_rhs[1] = %1.6e; [size %zu; type double; variable_name Fu]\n",data[1], QPntVolCoefStokes_member_sizes[2] );
    printf("field: momentum_rhs[2] = %1.6e; [size %zu; type double; variable_name Fu]\n",data[2], QPntVolCoefStokes_member_sizes[2] );
  }
  {
    double data;
    QPntVolCoefStokesGetField_continuity_rhs(point,&data);
    printf("field: continuity_rhs = %1.6e; [size %zu; type double; variable_name Fp]\n",data, QPntVolCoefStokes_member_sizes[3] );
  }
}


/* ===================================== */
/* VTK viewer for QPntVolCoefStokes */
/* ===================================== */
void QPntVolCoefStokesVTKWriteAsciiAllFields(FILE *vtk_fp,const int N,const QPntVolCoefStokes points[]) 
{
  int p;
  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Float64\" Name=\"eta\" format=\"ascii\">\n");
  for(p=0;p<N;p++) {
    fprintf( vtk_fp,"\t\t\t\t\t%lf\n",(double)points[p].eta);
  }
  fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n");
  for(p=0;p<N;p++) {
    fprintf( vtk_fp,"\t\t\t\t\t%lf\n",(double)points[p].rho);
  }
  fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Float64\" Name=\"Fp\" format=\"ascii\">\n");
  for(p=0;p<N;p++) {
    fprintf( vtk_fp,"\t\t\t\t\t%lf\n",(double)points[p].Fp);
  }
  fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
}


/* ===================================== */
/* PVTU viewer for QPntVolCoefStokes */
/* ===================================== */
void QPntVolCoefStokesPVTUWriteAllPPointDataFields(FILE *vtk_fp) 
{
  fprintf(vtk_fp, "\t\t\t<PDataArray type=\"Float64\" Name=\"eta\" NumberOfComponents=\"1\"/>\n");
  fprintf(vtk_fp, "\t\t\t<PDataArray type=\"Float64\" Name=\"rho\" NumberOfComponents=\"1\"/>\n");
  fprintf(vtk_fp, "\t\t\t<PDataArray type=\"Float64\" Name=\"Fp\" NumberOfComponents=\"1\"/>\n");
}


/* ===================================== */
/* VTK binary (appended header) viewer for QPntVolCoefStokes */
/* ===================================== */
void QPntVolCoefStokesVTKWriteBinaryAppendedHeaderAllFields(FILE *vtk_fp,int *offset,const int N,const QPntVolCoefStokes points[]) 
{
  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Float64\" Name=\"eta\" format=\"appended\"  offset=\"%d\" />\n",*offset);
  *offset = *offset + sizeof(int) + N * sizeof(double);

  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Float64\" Name=\"rho\" format=\"appended\"  offset=\"%d\" />\n",*offset);
  *offset = *offset + sizeof(int) + N * sizeof(double);

  /* Warning: swarm_class_generator.py is ignoring multi-component field Fu[] */

  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Float64\" Name=\"Fp\" format=\"appended\"  offset=\"%d\" />\n",*offset);
  *offset = *offset + sizeof(int) + N * sizeof(double);

}


/* ================================================== */
/* VTK binary (appended data) viewer for QPntVolCoefStokes */
/* ==================================================== */
void QPntVolCoefStokesVTKWriteBinaryAppendedDataAllFields(FILE *vtk_fp,const int N,const QPntVolCoefStokes points[]) 
{
  int p,length;
  size_t atomic_size;

  atomic_size = sizeof(double);
  length = (int)( atomic_size * ((size_t)N) );
  fwrite( &length,sizeof(int),1,vtk_fp);
  for(p=0;p<N;p++) {
    fwrite( &points[p].eta,atomic_size,1,vtk_fp);
  }

  atomic_size = sizeof(double);
  length = (int)( atomic_size * ((size_t)N) );
  fwrite( &length,sizeof(int),1,vtk_fp);
  for(p=0;p<N;p++) {
    fwrite( &points[p].rho,atomic_size,1,vtk_fp);
  }

  /* Warning: swarm_class_generator.py is ignoring multi-component field Fu[] */

  atomic_size = sizeof(double);
  length = (int)( atomic_size * ((size_t)N) );
  fwrite( &length,sizeof(int),1,vtk_fp);
  for(p=0;p<N;p++) {
    fwrite( &points[p].Fp,atomic_size,1,vtk_fp);
  }

}

