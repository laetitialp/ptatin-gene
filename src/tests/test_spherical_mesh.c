#include "petsc.h"
#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "dmda_bcs.h"
#include "element_utils_q1.h"
#include "dmda_element_q1.h"
#include "dmda_element_q2p1.h"
#include "quadrature.h"
#include "dmda_checkpoint.h"
#include "data_bucket.h"
#include "dmdae.h"
#include "fvda_private.h"
#include "ptatin_log.h"
#include "dmda_update_coords.h"
#include "mesh_update.h"


PetscErrorCode CartesianToSphericalCoords(DM da)
{
  DM              cda;
  Vec             coord;
  PetscScalar     *LA_coords;
  PetscReal       r,theta,phi;
  PetscInt        M,N,P,nx,ny,nz,si,sj,sk,i,j,k,nidx;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = VecGetArray(coord,&LA_coords);CHKERRQ(ierr);

  for (k=0; k<nz; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {

        nidx = i + j*nx + k*nx*ny;

        r     = LA_coords[3*nidx + 1];
        theta = LA_coords[3*nidx + 0];
        phi   = LA_coords[3*nidx + 2];

        LA_coords[3*nidx + 0] = r * cos(theta);
        LA_coords[3*nidx + 1] = r * sin(theta) * cos(phi);
        LA_coords[3*nidx + 2] = r * sin(theta) * sin(phi);
      }
    }
  }
  ierr = VecRestoreArray(coord,&LA_coords);CHKERRQ(ierr);

  ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDASetUniformCartesianCoordinatesFromSpherical(DM da, PetscReal thetamin, PetscReal thetamax, PetscReal rmin, PetscReal rmax, PetscReal phimin, PetscReal phimax)
{
  MPI_Comm       comm;
  DM             cda;
  Vec            coord;
  PetscScalar   *LA_coords;
  PetscReal      h_theta, h_r, h_phi, r, theta, phi;
  PetscInt       i, j, k, M, N, P, istart, isize, jstart, jsize, kstart, ksize, dim, cnt;
  PetscErrorCode ierr;

  ierr = DMDAGetInfo(da, &dim, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

  /*
  if (dim != 3) {
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Cannot create uniform coordinates for this dimension %" PetscInt_FMT, dim);
  }
  */
  ierr = PetscObjectGetComm((PetscObject)da, &comm);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, &istart, &jstart, &kstart, &isize, &jsize, &ksize);CHKERRQ(ierr); 
  ierr = DMGetCoordinateDM(da, &cda);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(cda, &coord);CHKERRQ(ierr);

  h_theta = (thetamax - thetamin) / (M - 1);
  h_r     = (rmax - rmin)         / (N - 1);
  h_phi   = (phimax - phimin)     / (P - 1);

  ierr = VecGetArray(coord, &LA_coords);CHKERRQ(ierr);
  cnt = 0;
  for (k = 0; k < ksize; k++) {
    for (j = 0; j < jsize; j++) {
      for (i = 0; i < isize; i++) {

        r     = rmin     + h_r     * (j + jstart);
        theta = thetamax - h_theta * (i + istart); // keeps IMAX face as xmax and IMIN face as xmin
        phi   = phimin   + h_phi   * (k + kstart);

        LA_coords[cnt++] = r * PetscCosReal(theta);
        LA_coords[cnt++] = r * PetscSinReal(theta) * PetscCosReal(phi);
        LA_coords[cnt++] = r * PetscSinReal(theta) * PetscSinReal(phi);
      }
    }
  }
  ierr = VecRestoreArray(coord, &LA_coords);CHKERRQ(ierr);
  ierr = DMSetCoordinates(da, coord);CHKERRQ(ierr);
  ierr = VecDestroy(&coord);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GenerateSphericalMesh()
{
  Vec X;
  IS        *is_stokes_field;
  Vec       velocity,pressure;
  pTatinCtx      ptatin = NULL;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscReal      O[3],L[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatin3dCreateContext(&ptatin);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(ptatin);CHKERRQ(ierr);

  O[0] = M_PI / 3.0; //0.0;
  O[1] = 6375.0e3-1200e3;//650.0e3;
  O[2] = -M_PI / 6.0; //0.0; 

  L[0] = M_PI / 1.5; //M_PI;
  L[1] = 6375.0e3;
  L[2] = M_PI / 6.0; //2.0*M_PI;

  PetscPrintf(PETSC_COMM_WORLD,"Box: Ox, Lx = [ %f, %f ]\n",O[0],L[0]);
  PetscPrintf(PETSC_COMM_WORLD,"Box: Oy, Ly = [ %f, %f ]\n",O[1],L[1]);
  PetscPrintf(PETSC_COMM_WORLD,"Box: Oz, Lz = [ %f, %f ]\n",O[2],L[2]);

  ptatin->mx = 32;
  ptatin->my = 32;
  ptatin->mz = 32;
  // Create a Q2 mesh 
  ierr = pTatin3d_PhysCompStokesCreate(ptatin);CHKERRQ(ierr);

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  //ierr = DMDASetUniformCoordinates(dav,O[0],L[0],O[1],L[1],O[2],L[2]);CHKERRQ(ierr);
  //ierr = CartesianToSphericalCoords(dav);CHKERRQ(ierr);
  ierr = DMDASetUniformCartesianCoordinatesFromSpherical(dav,O[0],L[0],O[1],L[1],O[2],L[2]);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);

  {
    Vec X;

    ierr = DMGetGlobalVector(stokes_pack,&X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(stokes_pack,&X);CHKERRQ(ierr);
  }
  ierr = DMCompositeGetGlobalISs(stokes_pack,&is_stokes_field);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(stokes_pack,&X);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);

  {
    PetscViewer viewer;
    char        fname[256];

    sprintf(fname,"spherical_domain.vts");

    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(velocity,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(stokes_pack,&X);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = GenerateSphericalMesh();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}