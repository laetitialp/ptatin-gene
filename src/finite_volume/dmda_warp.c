
#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <dmda_update_coords.h>


PetscErrorCode DMDAWarpCoordinates_SinJMax(DM da,PetscReal amp,PetscReal omega[])
{
  PetscErrorCode ierr;
  PetscInt       si,sj,sk,nx,ny,nz,i,j,k,MY;
  DM             cda;
  Vec            coord;
  DMDACoor3d     ***_coord;
  PetscReal      y_height,dy;
  PetscReal      Gmin[3],Gmax[3];
  PetscBool      flg;

  
  PetscFunctionBegin;
  
  {
    PetscReal tmp;
    
    ierr = PetscOptionsGetReal(NULL,NULL,"-warp_sin_amp",&tmp,&flg);CHKERRQ(ierr);
    if (flg) { amp = tmp; }
    ierr = PetscOptionsGetReal(NULL,NULL,"-warp_sin_omega_i",&tmp,&flg);CHKERRQ(ierr);
    if (flg) { omega[0] = tmp; }
    ierr = PetscOptionsGetReal(NULL,NULL,"-warp_sin_omega_j",&tmp,&flg);CHKERRQ(ierr);
    if (flg) { omega[1] = tmp; }
  }
  
  ierr = DMDAGetBoundingBox(da,Gmin,Gmax);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  
  ierr = DMDAGetInfo(da,NULL,NULL,&MY,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);
  
  for (j=sj; j<sj+ny; j++) {
    for (k=sk; k<sk+nz; k++) {
      for (i=si; i<si+nx; i++) {
        PetscReal xn,zn;
        
        xn = _coord[k][j][i].x;
        zn = _coord[k][j][i].z;
        
        /* constant spacing */
        y_height = amp * PetscSinReal(PETSC_PI * omega[0] * xn) * PetscSinReal(PETSC_PI * omega[1] * zn);
        dy = (y_height+1.0)/(PetscReal)(MY-1);
        
        _coord[k][j][i].y = -1.0 + j * dy;
      }
    }
  }
  /* rescale */
  for (j=sj; j<sj+ny; j++) {
    for (k=sk; k<sk+nz; k++) {
      for (i=si; i<si+nx; i++) {
        PetscReal xn,yn,zn;
        
        xn = _coord[k][j][i].x;
        yn = _coord[k][j][i].y;
        zn = _coord[k][j][i].z;
        
        _coord[k][j][i].x = (Gmax[0]-Gmin[0])*(xn + 1.0)/2.0 + Gmin[0];
        _coord[k][j][i].y = (Gmax[1]-Gmin[1])*(yn + 1.0)/2.0 + Gmin[1];
        _coord[k][j][i].z = (Gmax[2]-Gmin[2])*(zn + 1.0)/2.0 + Gmin[2];
      }
    }
  }
  
  ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);
  
  /* update ghost coords */
  {
    Vec gcoords;
    
    ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(cda,coord,INSERT_VALUES,gcoords);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(cda,coord,INSERT_VALUES,gcoords);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAWarpCoordinates_ExpJMax(DM da,PetscReal amp,PetscReal lambda[])
{
  PetscErrorCode ierr;
  PetscInt       si,sj,sk,nx,ny,nz,i,j,k,MY;
  DM             cda;
  Vec            coord;
  DMDACoor3d     ***_coord;
  PetscReal      y_height,dy;
  PetscReal      Gmin[3],Gmax[3];
  PetscBool      flg;
  
  
  PetscFunctionBegin;
  
  {
    PetscReal tmp;
    
    ierr = PetscOptionsGetReal(NULL,NULL,"-warp_exp_amp",&tmp,&flg);CHKERRQ(ierr);
    if (flg) { amp = tmp; }
    ierr = PetscOptionsGetReal(NULL,NULL,"-warp_exp_lambda_i",&tmp,&flg);CHKERRQ(ierr);
    if (flg) { lambda[0] = tmp; }
    ierr = PetscOptionsGetReal(NULL,NULL,"-warp_exp_lambda_j",&tmp,&flg);CHKERRQ(ierr);
    if (flg) { lambda[1] = tmp; }
  }
  
  ierr = DMDAGetBoundingBox(da,Gmin,Gmax);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  
  ierr = DMDAGetInfo(da,NULL,NULL,&MY,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);
  
  for (j=sj; j<sj+ny; j++) {
    for (k=sk; k<sk+nz; k++) {
      for (i=si; i<si+nx; i++) {
        PetscReal xn,zn;
        
        xn = _coord[k][j][i].x;
        zn = _coord[k][j][i].z;
        
        /* constant spacing */
        y_height = amp * PetscExpReal(lambda[0] * xn) * PetscExpReal(lambda[1] * zn);
        dy = (y_height+1.0)/(PetscReal)(MY-1);
        
        _coord[k][j][i].y = -1.0 + j * dy;
      }
    }
  }
  /* rescale */
  for (j=sj; j<sj+ny; j++) {
    for (k=sk; k<sk+nz; k++) {
      for (i=si; i<si+nx; i++) {
        PetscReal xn,yn,zn;
        
        xn = _coord[k][j][i].x;
        yn = _coord[k][j][i].y;
        zn = _coord[k][j][i].z;
        
        _coord[k][j][i].x = (Gmax[0]-Gmin[0])*(xn + 1.0)/2.0 + Gmin[0];
        _coord[k][j][i].y = (Gmax[1]-Gmin[1])*(yn + 1.0)/2.0 + Gmin[1];
        _coord[k][j][i].z = (Gmax[2]-Gmin[2])*(zn + 1.0)/2.0 + Gmin[2];
      }
    }
  }
  
  ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);
  
  /* update ghost coords */
  {
    Vec gcoords;
    
    ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(cda,coord,INSERT_VALUES,gcoords);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(cda,coord,INSERT_VALUES,gcoords);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
