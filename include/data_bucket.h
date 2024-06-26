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
 **    filename:   data_bucket.h
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

#ifndef __PTATIN_DATA_BUCKET_H__
#define __PTATIN_DATA_BUCKET_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "data_bucket_namespace.h"

#define PTATIN_DEFAULT -32654789

#define PTATIN_DATAFIELD_POINT_ACCESS_GUARD

/* Logging flag */
#define PTAT3D_LOG_DATA_BUCKET


typedef enum { BFALSE=0, BTRUE } BTruth;
typedef enum { DATABUCKET_VIEW_STDOUT=0, DATABUCKET_VIEW_BINARY, DATABUCKET_VIEW_NATIVE } DataBucketViewType;

typedef struct _p_DataField* DataField;
typedef struct _p_DataBucket* DataBucket;


struct _p_DataField {
  char   *registration_function;
  int    L;
  BTruth active;
  size_t atomic_size;
  char   *name; /* what are they called */
  void   *data; /* the data - an array of structs */
};

struct _p_DataBucket {
  int L; /* number in use */
  int buffer; /* memory buffer used for re-allocation */
  int allocated;  /* number allocated, this will equal datafield->L */
  BTruth finalised;
  int nfields; /* how many fields of this type */
  DataField *field; /* the data */
};

#define ERROR() {\
printf("ERROR: %s() from line %d in %s !!\n", __func__, __LINE__, __FILE__);\
exit(EXIT_FAILURE);\
}

#define MPI_ERROR_CHECK(comm,ierr) {\
  if (ierr != MPI_SUCCESS) { \
    printf("MPI ERROR: %s() from line %d in %s !! Aborting.\n", __func__, __LINE__, __FILE__);\
    MPI_Abort(comm,ierr);\
  }\
}

#define __DATATFIELD_point_access(data,index,atomic_size) (void*)((char*)(data) + (index)*(atomic_size))
#define __DATATFIELD_point_access_offset(data,index,atomic_size,offset) (void*)((char*)(data) + (index)*(atomic_size) + (offset))



void DataFieldCreate( const char registration_function[], const char name[], const size_t size, const int L, DataField *DF );
void DataFieldDestroy( DataField *DF );
void DataBucketCreate( DataBucket *DB );
void DataBucketDestroy( DataBucket *DB );
void _DataBucketRegisterField(
                              DataBucket db,
                              const char registration_function[],
                              const char field_name[],
                              size_t atomic_size, DataField *_gfield );


#define DataBucketRegisterField(db,name,size,k) {\
  char *location;\
  if (asprintf(&location,"Registered by %s() at line %d within file %s", __func__, __LINE__, __FILE__) < 0) {printf("asprintf() failed. Exiting ungracefully.\n"); exit(1);}\
  _DataBucketRegisterField( (db), location, (name), (size), (k) );\
  free(location);\
}

void DataFieldGetNumEntries(DataField df, int *sum);
void DataFieldSetSize( DataField df, const int new_L );
void DataFieldZeroBlock( DataField df, const int start, const int end );
void DataFieldGetAccess( const DataField gfield );
void DataFieldAccessPoint( const DataField gfield, const int pid, void **ctx_p );
void DataFieldAccessPointOffset( const DataField gfield, const size_t offset, const int pid, void **ctx_p );
void DataFieldRestoreAccess( DataField gfield );
void DataFieldVerifyAccess( const DataField gfield, const size_t size);
void DataFieldGetAtomicSize(const DataField gfield,size_t *size);

void DataFieldGetEntries(const DataField gfield,void **data);
void DataFieldRestoreEntries(const DataField gfield,void **data);

void DataFieldInsertPoint( const DataField field, const int index, const void *ctx );
void DataFieldCopyPoint( const int pid_x, const DataField field_x,
                        const int pid_y, const DataField field_y );
void DataFieldZeroPoint( const DataField field, const int index );

void DataBucketGetDataFieldByName(DataBucket db,const char name[],DataField *gfield);
void DataBucketQueryDataFieldByName(DataBucket db,const char name[],BTruth *found);
void DataBucketFinalize(DataBucket db);
void DataBucketSetInitialSizes( DataBucket db, const int L, const int buffer );
void DataBucketSetSizes( DataBucket db, const int L, const int buffer );
void DataBucketGetSizes( DataBucket db, int *L, int *buffer, int *allocated );
void DataBucketGetGlobalSizes(MPI_Comm comm, DataBucket db, long int *L, long int *buffer, long int *allocated );
void DataBucketGetDataFields( DataBucket db, int *L, DataField *fields[] );

void DataBucketCopyPoint( const DataBucket xb, const int pid_x,
                         const DataBucket yb, const int pid_y );
void DataBucketCreateFromSubset( DataBucket DBIn, const int N, const int list[], DataBucket *DB );
void DataBucketZeroPoint( const DataBucket db, const int index );

void DataBucketLoadFromFile(MPI_Comm comm,const char filename[], DataBucketViewType type, DataBucket *db);
void DataBucketView(MPI_Comm comm,DataBucket db,const char filename[],DataBucketViewType type);
void DataBucketLoadRedundantFromFile(MPI_Comm comm,const char filename[], DataBucketViewType type, DataBucket *db);

void DataBucketAddPoint( DataBucket db );
void DataBucketRemovePoint( DataBucket db );
void DataBucketRemovePointAtIndex( const DataBucket db, const int index );

void DataBucketDuplicateFields(DataBucket dbA,DataBucket *dbB);
void DataBucketInsertValues(DataBucket db1,DataBucket db2);

/* helpers for parallel send/recv */
void DataBucketCreatePackedArray(DataBucket db,size_t *bytes,void **buf);
void DataBucketDestroyPackedArray(DataBucket db,void **buf);
void DataBucketFillPackedArray(DataBucket db,const int index,void *buf);
void DataBucketInsertPackedArray(DataBucket db,const int idx,void *data);

void DataBucketView_NATIVE(MPI_Comm comm,DataBucket db,const char prefix[]);
void DataBucketLoad_NATIVE(MPI_Comm comm,const char jfilename[],DataBucket *_db);
void DataBucketLoadRedundant_NATIVE(MPI_Comm comm,const char jfilename[],DataBucket *_db);

void DataBucketGetEntriesdByName(DataBucket db,const char name[],void *data[]);
void DataBucketRestoreEntriesdByName(DataBucket db,const char name[],void *data[]);
void DataBucketRegister_double(DataBucket db,const char name[],int blocksize);
void DataBucketGetArray_double(DataBucket db,const char name[],int *blocksize,double *data[]);
void DataBucketRestoreArray_double(DataBucket db,const char name[],double *data[]);

#endif

