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

  PetscBool      owns_top;

  PetscFunctionBegin;

  if (record_old) {
    surface = surf->old;
  } else {
    surface = surf->current;
  }

  ierr = DMDAGetInfo(
      da,
      NULL,
      &mx,&my,&mz,
      NULL,NULL,NULL,
      NULL,NULL,
      NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAGetCorners(
      da,
      &xs,&ys,&zs,
      &xm,&ym,&zm);CHKERRQ(ierr);

  /*
   * Global index of the top surface.
   */
  j = my - 1;

  /*
   * Check whether this MPI rank actually owns the top layer.
   *
   * Owned y indices on this rank are:
   *
   *     ys <= j < ys + ym
   */
  owns_top = PETSC_FALSE;

  if ((j >= ys) && (j < ys + ym)) {
    owns_top = PETSC_TRUE;
  }

  /*
   * Ranks which do not own the top layer must NOT access
   * LA_coord[k][my-1][i].
   *
   * Their local SurfaceElevation arrays simply remain unchanged
   * for this call.
   */
  if (!owns_top) {
    PetscFunctionReturn(0);
  }

  ierr = DMGetCoordinateDM(
      da,
      &cda);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(
      da,
      &coord);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(
      cda,
      coord,
      &LA_coord);CHKERRQ(ierr);

  /*
   * Only process owned i-k nodes on this rank.
   */
  for (k=zs; k<zs+zm; k++) {

    for (i=xs; i<xs+xm; i++) {

      surface[k*surf->mx + i] =
          LA_coord[k][j][i].y;
    }
  }

  ierr = DMDAVecRestoreArray(
      cda,
      coord,
      &LA_coord);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode SurfaceElevationDifference(
        SurfaceElevation *surf,
        PetscReal *dhmin,
        PetscReal *dhmax)
{
  PetscErrorCode ierr;

  PetscInt  i,n;
  PetscReal dh;

  PetscReal local_dhmin;
  PetscReal local_dhmax;

  PetscReal global_dhmin;
  PetscReal global_dhmax;

  PetscInt local_positive;
  PetscInt local_negative;

  PetscInt global_positive;
  PetscInt global_negative;

  PetscFunctionBegin;

  n = surf->mx * surf->mz;

  /*
   * Each MPI rank contains valid elevation values only for the
   * top-surface nodes owned by that rank.
   *
   * All other entries remain zero.
   */
  local_dhmin = 0.0;
  local_dhmax = 0.0;

  local_positive = 0;
  local_negative = 0;

  for (i=0; i<n; i++) {

    /*
     * Compute the elevation change stored locally on this rank.
     */
    surf->dh[i] =
        surf->current[i]
      - surf->old[i];

    dh = surf->dh[i];

    /*
     * Zero-valued entries belonging to other MPI ranks do not
     * affect deposition/erosion counts.
     */
    if (dh > 0.0) {

      local_positive++;

      if (dh > local_dhmax) {
        local_dhmax = dh;
      }

    } else if (dh < 0.0) {

      local_negative++;

      if (dh < local_dhmin) {
        local_dhmin = dh;
      }
    }
  }

  /*
   * Combine results from all MPI ranks.
   *
   * Deposition:
   *     largest positive dh anywhere in the global domain.
   *
   * Erosion:
   *     most negative dh anywhere in the global domain.
   */
  ierr = MPI_Allreduce(
      &local_dhmax,
      &global_dhmax,
      1,
      MPIU_REAL,
      MPI_MAX,
      PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = MPI_Allreduce(
      &local_dhmin,
      &global_dhmin,
      1,
      MPIU_REAL,
      MPI_MIN,
      PETSC_COMM_WORLD);CHKERRQ(ierr);

  /*
   * Sum numbers of positive/negative surface nodes over all ranks.
   */
  ierr = MPI_Allreduce(
      &local_positive,
      &global_positive,
      1,
      MPIU_INT,
      MPI_SUM,
      PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = MPI_Allreduce(
      &local_negative,
      &global_negative,
      1,
      MPIU_INT,
      MPI_SUM,
      PETSC_COMM_WORLD);CHKERRQ(ierr);

  /*
   * Return the global values to the caller.
   */
  *dhmin = global_dhmin;
  *dhmax = global_dhmax;

  /*
   * PetscPrintf prints once on PETSC_COMM_WORLD,
   * so these are now GLOBAL MPI statistics.
   */
  ierr = PetscPrintf(
      PETSC_COMM_WORLD,
      "Surface elevation change:\n");CHKERRQ(ierr);

  ierr = PetscPrintf(
      PETSC_COMM_WORLD,
      "  Maximum deposition : %12.6e\n",
      (double)(*dhmax));CHKERRQ(ierr);

  ierr = PetscPrintf(
      PETSC_COMM_WORLD,
      "  Maximum erosion    : %12.6e\n",
      (double)(*dhmin));CHKERRQ(ierr);

  ierr = PetscPrintf(
      PETSC_COMM_WORLD,
      "  Positive nodes     : %D\n",
      global_positive);CHKERRQ(ierr);

  ierr = PetscPrintf(
      PETSC_COMM_WORLD,
      "  Negative nodes     : %D\n",
      global_negative);CHKERRQ(ierr);

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
  PetscInt j;
  PetscInt idx;

  PetscInt  max_candidates;
  PetscInt  ncandidates;
  PetscReal *candidate_coords = NULL;

  PetscMPIInt rank;
  PetscBool   owns_top;

  PetscFunctionBegin;


  ierr = DMDAGetInfo(
      dav,
      NULL,
      &mx,&my,&mz,
      NULL,NULL,NULL,
      NULL,NULL,
      NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAGetCorners(
      dav,
      &xs,&ys,&zs,
      &xm,&ym,&zm);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(
      PetscObjectComm((PetscObject)dav),
      &rank);CHKERRQ(ierr);

  /*
   * Global top-surface index.
   */
  j = my - 1;

  /*
   * Determine whether this rank owns j = my-1.
   */
  owns_top = PETSC_FALSE;

  if ((j >= ys) && (j < ys + ym)) {
    owns_top = PETSC_TRUE;
  }

  /*
   * Default for ranks which do not own the top surface.
   */
  ncandidates    = 0;
  max_candidates = 0;

  /*
   * Only top-surface ranks access LA_coord[k][j][i]
   * and generate sediment candidates.
   */
  if (owns_top) {

    ierr = DMGetCoordinateDM(
        dav,
        &cda);CHKERRQ(ierr);

    ierr = DMGetCoordinatesLocal(
        dav,
        &coord);CHKERRQ(ierr);

    ierr = DMDAVecGetArray(
        cda,
        coord,
        &LA_coord);CHKERRQ(ierr);

    /*
     * One candidate maximum per owned top-surface node.
     */
    max_candidates = xm * zm;

    if (max_candidates > 0) {

      ierr = PetscMalloc1(
          3 * max_candidates,
          &candidate_coords);CHKERRQ(ierr);
    }

    /*
     * Only owned i-k nodes.
     */
    for (k=zs; k<zs+zm; k++) {

      for (i=xs; i<xs+xm; i++) {

        PetscReal dh;
        PetscReal x,y,z;

        idx = k * surf->mx + i;

        /*
         * dh was already computed by
         * SurfaceElevationDifference().
         */
        dh = surf->dh[idx];

        if (dh <= 0.0) {
          continue;
        }

        /*
         * Candidate marker position.
         */
        x = LA_coord[k][j][i].x;

        y = surf->old[idx]
          + 0.5 * dh;

        z = LA_coord[k][j][i].z;

        candidate_coords[3*ncandidates + 0] = x;
        candidate_coords[3*ncandidates + 1] = y;
        candidate_coords[3*ncandidates + 2] = z;

        ncandidates++;
      }
    }

    ierr = DMDAVecRestoreArray(
        cda,
        coord,
        &LA_coord);CHKERRQ(ierr);
  }

  /*
   * Every MPI rank reaches this point.
   *
   * This is important because the future parallel insertion routine
   * will contain MPI collectives.
   */
  ierr = PetscSynchronizedPrintf(
      PetscObjectComm((PetscObject)dav),
      "[rank %d] ys=%d ym=%d top_j=%d owns_top=%d "
      "xs=%d xm=%d zs=%d zm=%d candidates=%d\n",
      (int)rank,
      (int)ys,
      (int)ym,
      (int)j,
      (int)owns_top,
      (int)xs,
      (int)xm,
      (int)zs,
      (int)zm,
      (int)ncandidates);CHKERRQ(ierr);

  ierr = PetscSynchronizedFlush(PetscObjectComm((PetscObject)dav),PETSC_STDOUT);CHKERRQ(ierr);

/*
 * Insert deposition markers in parallel.
 *
 * IMPORTANT:
 *
 * Every MPI rank must call this routine, including ranks that have
 * ncandidates == 0, because the routine contains MPI collective
 * operations.
 */
  ierr = SwarmMPntStd_InsertSedimentMarkersParallel(db,dav,ncandidates,candidate_coords,20,deposition_time);CHKERRQ(ierr);
  ierr = PetscFree(candidate_coords);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
