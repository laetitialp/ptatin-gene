#include "surface_elevation.h"
#include "material_point_std_utils.h"
#include "data_bucket.h"

PetscErrorCode SurfaceElevationCreate(
        DM dav,
        SurfaceElevation *surf)
{
  PetscErrorCode ierr;
  PetscInt mx,my,mz;
  PetscInt n;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(dav,
                     NULL,
                     &mx,&my,&mz,
                     NULL,NULL,NULL,
                     NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  surf->mx = mx;
  surf->mz = mz;

  n = mx * mz;

  ierr = PetscMalloc1(n,&surf->old);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&surf->current);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&surf->dh);CHKERRQ(ierr);

  ierr = PetscMemzero(surf->old,n*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(surf->current,n*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(surf->dh,n*sizeof(PetscReal));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode SurfaceElevationDestroy(
        SurfaceElevation *surf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscFree(surf->old);CHKERRQ(ierr);
  ierr = PetscFree(surf->current);CHKERRQ(ierr);
  ierr = PetscFree(surf->dh);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode SurfaceElevationRecord(
        DM da,
        SurfaceElevation *surf,
        PetscBool record_old)
{
  PetscErrorCode ierr;
  DM             cda;
  Vec            coord;
  DMDACoor3d     ***LA_coord;
  PetscInt       mx,my,mz;
  PetscInt       xs,ys,zs;
  PetscInt       xm,ym,zm;
  PetscInt       i,j,k;
  PetscReal      *surface;

  PetscFunctionBegin;

  if (record_old) {
    surface = surf->old;
  } else {
    surface = surf->current;
  }

  ierr = DMDAGetInfo(da,
                     NULL,
                     &mx,
                     &my,
                     &mz,
                     NULL,NULL,NULL,
                     NULL,NULL,
                     NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,
                        &xs,&ys,&zs,
                        &xm,&ym,&zm);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(da,&coord);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(cda,coord,&LA_coord);CHKERRQ(ierr);

  /* top surface */
  j = my - 1;

  for (k=zs; k<zs+zm; k++) {
    for (i=xs; i<xs+xm; i++) {

      surface[k*surf->mx + i] = LA_coord[k][j][i].y;

    }
  }

  ierr = DMDAVecRestoreArray(cda,coord,&LA_coord);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode SurfaceElevationDifference(
        SurfaceElevation *surf,
        PetscReal *dhmin,
        PetscReal *dhmax)
{
  PetscInt  i,n;
  PetscReal dh;
  PetscInt  positive = 0;
  PetscInt  negative = 0;

  PetscFunctionBegin;

  n = surf->mx * surf->mz;

  *dhmin = PETSC_MAX_REAL;
  *dhmax = PETSC_MIN_REAL;

  for (i=0; i<n; i++) {

    surf->dh[i] = surf->current[i] - surf->old[i];
    dh = surf->dh[i];

    if (dh < *dhmin) *dhmin = dh;
    if (dh > *dhmax) *dhmax = dh;

    if (dh > 0.0) positive++;
    if (dh < 0.0) negative++;
  }

  PetscPrintf(PETSC_COMM_WORLD,
              "Surface elevation change:\n");
  PetscPrintf(PETSC_COMM_WORLD,
              "  Maximum deposition : %12.6e\n",
              (double)(*dhmax));
  PetscPrintf(PETSC_COMM_WORLD,
              "  Maximum erosion    : %12.6e\n",
              (double)(*dhmin));
  PetscPrintf(PETSC_COMM_WORLD,
              "  Positive nodes     : %D\n",
              positive);
  PetscPrintf(PETSC_COMM_WORLD,
              "  Negative nodes     : %D\n",
              negative);

  PetscFunctionReturn(0);
}


PetscErrorCode SurfaceElevationGenerateSedimentMarkers(
        SurfaceElevation *surf,
        DM dav,
        PetscReal h_min)
{
  PetscInt i,k;
  PetscInt mx = surf->mx;
  PetscInt mz = surf->mz;
  PetscReal dh;
  PetscReal x,z;
  PetscReal x0,z0,dx,dz;

  PetscFunctionBegin;

  DMDAGetCorners(dav,&i,&k,NULL,&mx,&mz,NULL);
  DMDAGetInfo(dav,NULL,
              &mx,&mz,NULL,
              NULL,NULL,NULL,
              NULL,NULL,NULL,NULL,NULL,NULL);

  dx = 1.0;
  dz = 1.0;

  for (k=0; k<surf->mz; k++) {
    for (i=0; i<surf->mx; i++) {

      dh = surf->dh[k*surf->mx + i];

      if (dh <= h_min) continue;

      /* surface coordinate (simplified mapping) */
      x = i * dx;
      z = k * dz;

      /* number of markers */
      PetscInt nmark = (PetscInt)(dh / h_min);

      for (PetscInt m=0; m<nmark; m++) {

        PetscReal y = - (m+1) * h_min;

        /* ===== HERE YOU INSERT MARKER ===== */
        /* IMPORTANT: phase = 20 */

        /* pseudo-call (we will replace next step) */
        /*
        MPntStd marker;
        MPntStdSetField_position(&marker, x, y, z);
        MPntStdSetField_phase_index(&marker, 20);
        AddMarkerToSwarm(...);
        */
      }
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceElevationAddDepositionMarkers(
        DM dav,
        DataBucket db,
        SurfaceElevation *surf,
	PetscReal deposition_time)
{
  PetscErrorCode ierr;

  DM          cda;
  Vec         coord;
  DMDACoor3d  ***LA_coord;

  PetscInt mx,my,mz;
  PetscInt xs,ys,zs;
  PetscInt xm,ym,zm;

  PetscInt i,k;

  PetscInt before,after;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(dav,
                     NULL,
                     &mx,&my,&mz,
                     NULL,NULL,NULL,
                     NULL,NULL,
                     NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dav,
                        &xs,&ys,&zs,
                        &xm,&ym,&zm);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(dav,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dav,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&LA_coord);CHKERRQ(ierr);

  DataBucketGetSizes(db,&before,NULL,NULL);

  /* top surface */
  PetscInt j = my-1;

  for (k=zs;k<zs+zm;k++) {

    for (i=xs;i<xs+xm;i++) {

      PetscReal dh;

      dh = surf->current[k*surf->mx+i]
         - surf->old[k*surf->mx+i];

      if (dh <= 0.0) continue;

      PetscReal x,y,z;

      x = LA_coord[k][j][i].x;
      y = surf->old[k*surf->mx+i] + 0.5*dh;
      z = LA_coord[k][j][i].z;

      PetscPrintf(PETSC_COMM_WORLD,
          "Insert sediment marker (%g %g %g), dh=%g\n",
          (double)x,
          (double)y,
          (double)z,
          (double)dh);

      ierr = SwarmMPntStd_InsertSedimentMarker(db,dav,x,y,z,20,deposition_time);CHKERRQ(ierr);
    }
  }

  ierr = DMDAVecRestoreArray(cda,coord,&LA_coord);CHKERRQ(ierr);

  DataBucketGetSizes(db,&after,NULL,NULL);

  PetscPrintf(PETSC_COMM_WORLD,
      "Markers before insertion = %d\n",(int)before);

  PetscPrintf(PETSC_COMM_WORLD,
      "Markers after insertion  = %d\n",(int)after);

  PetscPrintf(PETSC_COMM_WORLD,
      "Markers added            = %d\n",
      (int)(after-before));

  PetscFunctionReturn(0);
}
