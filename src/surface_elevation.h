#ifndef __SURFACE_ELEVATION_H__
#define __SURFACE_ELEVATION_H__

#include "ptatin3d.h"
#include "petscdm.h"
#include "petscdmda.h"
#include "data_bucket.h"

typedef struct {

  PetscInt mx;
  PetscInt mz;

  PetscReal *old;
  PetscReal *current;
  PetscReal *dh;

} SurfaceElevation;

PetscErrorCode SurfaceElevationCreate(
        DM da,
        SurfaceElevation *surf);

PetscErrorCode SurfaceElevationDestroy(
        SurfaceElevation *surf);

PetscErrorCode SurfaceElevationRecord(
        DM da,
        SurfaceElevation *surf,
        PetscBool record_old);

PetscErrorCode SurfaceElevationAddDepositionMarkers(
        DM da,
        DataBucket db,
        SurfaceElevation *surf);
#endif
