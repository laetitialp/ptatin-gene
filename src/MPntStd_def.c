


/*
  Auto generated by version 0.0 of swarm_class_generator.py
  on otsu.local, at 2012-02-13 00:08:18.106151 by dmay
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "MPntStd_def.h"


const char MPntStd_classname[] = "MPntStd";

const int MPntStd_nmembers = 5;

const size_t MPntStd_member_sizes[] = {
  1 * sizeof(long int),
  3 * sizeof(double),
  3 * sizeof(double),
  1 * sizeof(int),
  1 * sizeof(int)
} ;

const char *MPntStd_member_names[] = {
  "point_index",
  "global_coord",
  "local_coord",
  "phase_index",
  "local_element_index"
} ;


/* ===================================== */
/* Getters for MPntStd */
/* ===================================== */
void MPntStdGetField_point_index(MPntStd *point,long int *data) 
{
  *data = point->pid;
}

void MPntStdGetField_global_coord(MPntStd *point,double *data[]) 
{
  *data = point->coor;
}

void MPntStdGetField_local_coord(MPntStd *point,double *data[]) 
{
  *data = point->xi;
}

void MPntStdGetField_phase_index(MPntStd *point,int *data) 
{
  *data = point->phase;
}

void MPntStdGetField_local_element_index(MPntStd *point,int *data) 
{
  *data = point->wil;
}


/* ===================================== */
/* Setters for MPntStd */
/* ===================================== */
void MPntStdSetField_point_index(MPntStd *point,long int data) 
{
  point->pid = data;
}

void MPntStdSetField_global_coord(MPntStd *point,double data[]) 
{
  memcpy( &point->coor[0], data, sizeof(double)*3 );
}

void MPntStdSetField_local_coord(MPntStd *point,double data[]) 
{
  memcpy( &point->xi[0], data, sizeof(double)*3 );
}

void MPntStdSetField_phase_index(MPntStd *point,int data) 
{
  point->phase = data;
}

void MPntStdSetField_local_element_index(MPntStd *point,int data) 
{
  point->wil = data;
}


/* ===================================== */
/* C-viewer for MPntStd */
/* ===================================== */
void MPntStdView(MPntStd *point)
{
  {
    long int data;
    MPntStdGetField_point_index(point,&data);
    printf("field: point_index = %ld; [size %zu; type long int; variable_name pid]\n",data, MPntStd_member_sizes[0] );
  }
  {
    double *data;
    MPntStdGetField_global_coord(point,&data);
    printf("field: global_coord[0] = %1.6e; [size %zu; type double; variable_name coor]\n",data[0], MPntStd_member_sizes[1] );
    printf("field: global_coord[1] = %1.6e; [size %zu; type double; variable_name coor]\n",data[1], MPntStd_member_sizes[1] );
    printf("field: global_coord[2] = %1.6e; [size %zu; type double; variable_name coor]\n",data[2], MPntStd_member_sizes[1] );
  }
  {
    double *data;
    MPntStdGetField_local_coord(point,&data);
    printf("field: local_coord[0] = %1.6e; [size %zu; type double; variable_name xi]\n",data[0], MPntStd_member_sizes[2] );
    printf("field: local_coord[1] = %1.6e; [size %zu; type double; variable_name xi]\n",data[1], MPntStd_member_sizes[2] );
    printf("field: local_coord[2] = %1.6e; [size %zu; type double; variable_name xi]\n",data[2], MPntStd_member_sizes[2] );
  }
  {
    int data;
    MPntStdGetField_phase_index(point,&data);
    printf("field: phase_index = %d; [size %zu; type int; variable_name phase]\n",data, MPntStd_member_sizes[3] );
  }
  {
    int data;
    MPntStdGetField_local_element_index(point,&data);
    printf("field: local_element_index = %d; [size %zu; type int; variable_name wil]\n",data, MPntStd_member_sizes[4] );
  }
}


/* ===================================== */
/* VTK viewer for MPntStd */
/* ===================================== */
void MPntStdVTKWriteAsciiAllFields(FILE *vtk_fp,const int N,const MPntStd points[]) 
{
  int p;
  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int64\" Name=\"pid\" format=\"ascii\">\n");
  for(p=0;p<N;p++) {
    fprintf( vtk_fp,"\t\t\t\t\t%ld\n",(long int)points[p].pid);
  }
  fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int32\" Name=\"phase\" format=\"ascii\">\n");
  for(p=0;p<N;p++) {
    fprintf( vtk_fp,"\t\t\t\t\t%d\n",(int)points[p].phase);
  }
  fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int32\" Name=\"wil\" format=\"ascii\">\n");
  for(p=0;p<N;p++) {
    fprintf( vtk_fp,"\t\t\t\t\t%d\n",(int)points[p].wil);
  }
  fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
}


/* ===================================== */
/* PVTU viewer for MPntStd */
/* ===================================== */
void MPntStdPVTUWriteAllPPointDataFields(FILE *vtk_fp) 
{
  fprintf(vtk_fp, "\t\t\t<PDataArray type=\"Int64\" Name=\"pid\" NumberOfComponents=\"1\"/>\n");
  fprintf(vtk_fp, "\t\t\t<PDataArray type=\"Int32\" Name=\"phase\" NumberOfComponents=\"1\"/>\n");
  fprintf(vtk_fp, "\t\t\t<PDataArray type=\"Int32\" Name=\"wil\" NumberOfComponents=\"1\"/>\n");
}


/* ===================================== */
/* VTK binary (appended header) viewer for MPntStd */
/* ===================================== */
void MPntStdVTKWriteBinaryAppendedHeaderAllFields(FILE *vtk_fp,int *offset,const int N,const MPntStd points[]) 
{
  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int64\" Name=\"pid\" format=\"appended\"  offset=\"%d\" />\n",*offset);
  *offset = *offset + sizeof(int) + N * sizeof(long int);

  /* Warning: swarm_class_generator.py is ignoring multi-component field coor[] */

  /* Warning: swarm_class_generator.py is ignoring multi-component field xi[] */

  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int32\" Name=\"phase\" format=\"appended\"  offset=\"%d\" />\n",*offset);
  *offset = *offset + sizeof(int) + N * sizeof(int);

  fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int32\" Name=\"wil\" format=\"appended\"  offset=\"%d\" />\n",*offset);
  *offset = *offset + sizeof(int) + N * sizeof(int);

}


/* ================================================== */
/* VTK binary (appended data) viewer for MPntStd */
/* ==================================================== */
void MPntStdVTKWriteBinaryAppendedDataAllFields(FILE *vtk_fp,const int N,const MPntStd points[]) 
{
  int p,length;
  size_t atomic_size;

  atomic_size = sizeof(long int);
  length = (int)( atomic_size * ((size_t)N) );
  fwrite( &length,sizeof(int),1,vtk_fp);
  for(p=0;p<N;p++) {
    fwrite( &points[p].pid,atomic_size,1,vtk_fp);
  }

  /* Warning: swarm_class_generator.py is ignoring multi-component field coor[] */

  /* Warning: swarm_class_generator.py is ignoring multi-component field xi[] */

  atomic_size = sizeof(int);
  length = (int)( atomic_size * ((size_t)N) );
  fwrite( &length,sizeof(int),1,vtk_fp);
  for(p=0;p<N;p++) {
    fwrite( &points[p].phase,atomic_size,1,vtk_fp);
  }

  atomic_size = sizeof(int);
  length = (int)( atomic_size * ((size_t)N) );
  fwrite( &length,sizeof(int),1,vtk_fp);
  for(p=0;p<N;p++) {
    fwrite( &points[p].wil,atomic_size,1,vtk_fp);
  }

}

