
/* 
Deform mesh based on analytic functions.
Typically just used to test robustness of multigrid solver w.r.t to aspect ratio,
grid deformation etc.
*/

#include "dmda_update_coords.h"
#include "mesh_deformation.h"


/* 
 Apply a bump to the surface 
 Domain is set to be [-1,1]^3
 "bump" is defined in y-direction and given by the function

 gbump_amp * exp(gbump_lambda*(x*x+z*z))+1
 
 */
PetscErrorCode MeshDeformation_GaussianBump_YMAX(DM da)
{
	PetscErrorCode ierr;
	PetscReal gbump_amp,gbump_lambda;
	PetscInt si,sj,sk,nx,ny,nz,i,j,k,MY;
	DM cda;
	Vec coord;
	DMDACoor3d ***_coord;
	PetscReal y_height,dy;
	
	PetscFunctionBegin;
	gbump_amp    = -1.1;
	gbump_lambda = -5.6;
	ierr = PetscOptionsGetReal(PETSC_NULL,"-gbump_amp",&gbump_amp,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(PETSC_NULL,"-gbump_lambda",&gbump_lambda,PETSC_NULL);CHKERRQ(ierr);
	
	
	ierr = DMDASetUniformCoordinates(da,-1.0,1.0, -1.0,1.0, -1.0,1.0);CHKERRQ(ierr);

	ierr = DMDAGetInfo(da,0,0,&MY,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
	ierr = DMDAGetCorners( da, &si,&sj,&sk, &nx,&ny,&nz );CHKERRQ(ierr);
	ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
	ierr = DMDAGetCoordinates(da,&coord);CHKERRQ(ierr);
	ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);
	
	for( j=sj; j<sj+ny; j++ ) {
		for( k=sk; k<sk+nz; k++ ) {
			for( i=si; i<si+nx; i++ ) {
				PetscReal xn,yn,zn;
				
				xn = _coord[k][j][i].x;
				yn = _coord[k][j][i].y;
				zn = _coord[k][j][i].z;
				
				/* scale amplitude with depth */
				//fac_scale = (yn-(-1.0))/2.0;

				/* constant spacing */
				y_height = gbump_amp * exp(gbump_lambda*(xn*xn+zn*zn))+1.0;
				dy = y_height/(PetscReal)(MY-1);
				
				_coord[k][j][i].y = j * dy;
			}
		}
	}
	ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);
	
	/* update */
	ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);
	

	PetscFunctionReturn(0);
}