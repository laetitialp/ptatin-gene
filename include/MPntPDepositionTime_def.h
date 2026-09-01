#ifndef __MPntPDepositionTime_DEF_H__
#define __MPntPDepositionTime_DEF_H__

#include <mpi.h>

typedef struct {
  double deposition_time;
} MPntPDepositionTime;

typedef enum {
  MPPDepTime_deposition_time = 0
} MPntPDepositionTimeTypeName;

extern const char MPntPDepositionTime_classname[];

extern const int MPntPDepositionTime_nmembers;

extern const size_t MPntPDepositionTime_member_sizes[];

extern const char *MPntPDepositionTime_member_names[];

extern MPI_Datatype MPI_MPNTPDEPOSITIONTIME;

/* prototypes */

void MPntPDepositionTimeGetField_deposition_time(
        MPntPDepositionTime *point,
        double *data);

void MPntPDepositionTimeSetField_deposition_time(
        MPntPDepositionTime *point,
        double data);

void MPntPDepositionTimeView(
        MPntPDepositionTime *point);

void MPntPDepositionTimeVTKWriteAsciiAllFields(
        FILE *vtk_fp,
        const int N,
        const MPntPDepositionTime points[]);

void MPntPDepositionTimePVTUWriteAllPPointDataFields(
        FILE *vtk_fp);

void MPntPDepositionTimeVTKWriteBinaryAppendedHeaderAllFields(
        FILE *vtk_fp,
        int *offset,
        const int N,
        const MPntPDepositionTime points[]);

void MPntPDepositionTimeVTKWriteBinaryAppendedDataAllFields(
        FILE *vtk_fp,
        const int N,
        const MPntPDepositionTime points[]);

int MPntPDepositionTimeCreateMPIDataType(
        MPI_Datatype *ptype);

#endif
