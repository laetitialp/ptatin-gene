#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <mpi.h>

#include "MPntPDepositionTime_def.h"

const char MPntPDepositionTime_classname[] = "MPntPDepositionTime";

const int MPntPDepositionTime_nmembers = 1;

const size_t MPntPDepositionTime_member_sizes[] = {
  sizeof(double)
};

const char *MPntPDepositionTime_member_names[] = {
  "deposition_time"
};

MPI_Datatype MPI_MPNTPDEPOSITIONTIME;


/* ===================================== */
/* Getter                                */
/* ===================================== */

void MPntPDepositionTimeGetField_deposition_time(
        MPntPDepositionTime *point,
        double *data)
{
  *data = point->deposition_time;
}


/* ===================================== */
/* Setter                                */
/* ===================================== */

void MPntPDepositionTimeSetField_deposition_time(
        MPntPDepositionTime *point,
        double data)
{
  point->deposition_time = data;
}


/* ===================================== */
/* Viewer                                */
/* ===================================== */

void MPntPDepositionTimeView(
        MPntPDepositionTime *point)
{
  printf(
      "field: deposition_time = %1.6e "
      "[size %zu]\n",
      point->deposition_time,
      MPntPDepositionTime_member_sizes[0]);
}


/* ===================================== */
/* ASCII VTK                             */
/* ===================================== */

void MPntPDepositionTimeVTKWriteAsciiAllFields(
        FILE *vtk_fp,
        const int N,
        const MPntPDepositionTime points[])
{
  int p;

  fprintf(vtk_fp,
      "\t\t\t\t<DataArray "
      "type=\"Float64\" "
      "Name=\"deposition_time\" "
      "format=\"ascii\">\n");

  for (p=0; p<N; p++) {
    fprintf(vtk_fp,
        "\t\t\t\t\t%lf\n",
        points[p].deposition_time);
  }

  fprintf(vtk_fp,
      "\t\t\t\t</DataArray>\n");
}


/* ===================================== */
/* PVTU                                  */
/* ===================================== */

void MPntPDepositionTimePVTUWriteAllPPointDataFields(
        FILE *vtk_fp)
{
  fprintf(vtk_fp,
      "\t\t\t<PDataArray "
      "type=\"Float64\" "
      "Name=\"deposition_time\" "
      "NumberOfComponents=\"1\"/>\n");
}


/* ===================================== */
/* Binary Header                         */
/* ===================================== */

void MPntPDepositionTimeVTKWriteBinaryAppendedHeaderAllFields(
        FILE *vtk_fp,
        int *offset,
        const int N,
        const MPntPDepositionTime points[])
{
  (void)points;

  fprintf(vtk_fp,
      "\t\t\t\t<DataArray "
      "type=\"Float64\" "
      "Name=\"deposition_time\" "
      "format=\"appended\" "
      "offset=\"%d\" />\n",
      *offset);

  *offset += sizeof(int) + N*sizeof(double);
}


/* ===================================== */
/* Binary Data                           */
/* ===================================== */

void MPntPDepositionTimeVTKWriteBinaryAppendedDataAllFields(
        FILE *vtk_fp,
        const int N,
        const MPntPDepositionTime points[])
{
  int p;
  int length;
  size_t atomic_size;

  atomic_size = sizeof(double);

  length = (int)(atomic_size*((size_t)N));

  fwrite(&length,sizeof(int),1,vtk_fp);

  for (p=0; p<N; p++) {
    fwrite(
        &points[p].deposition_time,
        atomic_size,
        1,
        vtk_fp);
  }
}


/* ===================================== */
/* MPI datatype                          */
/* ===================================== */

int MPntPDepositionTimeCreateMPIDataType(
        MPI_Datatype *ptype)
{
  MPI_Datatype newtype;

  MPI_Datatype types[1];
  int blocklens[1];

  MPI_Aint loc[2];
  MPI_Aint disp[1];

  MPntPDepositionTime dummy;

  types[0] = MPI_DOUBLE;
  blocklens[0] = 1;

  MPI_Get_address(&dummy,&loc[0]);
  MPI_Get_address(&dummy.deposition_time,&loc[1]);

  disp[0] = loc[1] - loc[0];

  MPI_Type_create_struct(
      1,
      blocklens,
      disp,
      types,
      &newtype);

  MPI_Type_commit(&newtype);

  *ptype = newtype;

  return 0;
}
