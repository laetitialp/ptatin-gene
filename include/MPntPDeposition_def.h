#ifndef __MPntPDeposition_DEF_H__
#define __MPntPDeposition_DEF_H__

#include <mpi.h>

typedef struct {

  double deposition_time;

} MPntPDeposition;

typedef enum {

  MPPDep_deposition_time = 0

} MPntPDepositionTypeName;

extern const char MPntPDeposition_classname[];

extern const int MPntPDeposition_nmembers;

extern const size_t MPntPDeposition_member_sizes[];

extern const char *MPntPDeposition_member_names[];

extern MPI_Datatype MPI_MPNTPDEPOSITION;

/* prototypes */

void MPntPDepositionGetField_deposition_time(
        MPntPDeposition *point,
        double *data);

void MPntPDepositionSetField_deposition_time(
        MPntPDeposition *point,
        double data);

void MPntPDepositionView(
        MPntPDeposition *point);

void MPntPDepositionVTKWriteAsciiAllFields(
        FILE *vtk_fp,
        const int N,
        const MPntPDeposition points[]);

void MPntPDepositionPVTUWriteAllPPointDataFields(
        FILE *vtk_fp);

void MPntPDepositionVTKWriteBinaryAppendedHeaderAllFields(
        FILE *vtk_fp,
        int *offset,
        const int N,
        const MPntPDeposition points[]);

void MPntPDepositionVTKWriteBinaryAppendedDataAllFields(
        FILE *vtk_fp,
        const int N,
        const MPntPDeposition points[]);

int MPntPDepositionCreateMPIDataType(
        MPI_Datatype *ptype);

#endif
