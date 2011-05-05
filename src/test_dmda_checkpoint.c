
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define _GNU_SOURCE

#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>

#include "dmda_update_coords.h"
#include "dmda_view_petscvtk.h"
#include "dmda_checkpoint.h"


#undef __FUNCT__  
#define __FUNCT__ "test_dmda_checkpoint_pack"
PetscErrorCode test_dmda_checkpoint_pack(void) 
{
	DM  da;
	PetscInt nx,ny,nz;
	Vec x,coords;
	PetscViewer v;
	PetscReal val;
	PetscScalar max;
	PetscReal x0,y0,z0,x1,y1,z1;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;

	/* create the da */
	nx = ny = nz = 10;
	PetscOptionsGetInt( PETSC_NULL, "-mx", &nx, 0 );
	PetscOptionsGetInt( PETSC_NULL, "-my", &ny, 0 );
	PetscOptionsGetInt( PETSC_NULL, "-mz", &nz, 0 );
	
	ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_NONPERIODIC,DMDA_STENCIL_BOX,nx,ny,nz, PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 6,1, 0,0,0,&da);CHKERRQ(ierr);
	
	x0 = y0 = z0 = -1.0;
	x1 = y1 = z1 = 1.0;
	ierr = DMDASetUniformCoordinates(da, x0,x1, y0,y1, z0,z1);CHKERRQ(ierr);
	
	ierr = DMDAGetCoordinates(da,&coords);CHKERRQ(ierr);
	
	ierr = VecStrideScale(coords,0,10.0);CHKERRQ(ierr);
	ierr = VecStrideScale(coords,1,20.0);CHKERRQ(ierr);
	ierr = VecStrideScale(coords,2,30.0);CHKERRQ(ierr);
	
	ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);
	
	
	/* create a field */
	ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
	ierr = VecSetRandom( x, PETSC_NULL );CHKERRQ(ierr);
	ierr = VecNorm( x, NORM_1, &val );	PetscPrintf( PETSC_COMM_WORLD, "|x| = %1.5e \n", val );CHKERRQ(ierr);
	ierr = VecNorm( x, NORM_2, &val ); PetscPrintf( PETSC_COMM_WORLD, "|x|_2 = %1.5e \n", val );CHKERRQ(ierr);
	ierr = VecMin( x, 0, &max ); PetscPrintf( PETSC_COMM_WORLD, "min(x) = %1.5e \n", max );CHKERRQ(ierr);
	ierr = VecMax( x, 0, &max ); PetscPrintf( PETSC_COMM_WORLD, "max(x) = %1.5e \n", max );CHKERRQ(ierr);
	
	
	/* dump field to vtk */
	ierr = DMDAViewPetscVTK(da, x, "dmda_checkpoint_1.vtk");CHKERRQ(ierr);
	
	/* dump field to disk */
	ierr = PetscViewerBinaryOpen( ((PetscObject)da)->comm, "dmda_checkpoint_stressfield.dat", FILE_MODE_WRITE, &v );CHKERRQ(ierr);
	ierr = VecView( x, v );CHKERRQ(ierr);
	ierr = PetscViewerDestroy(v);CHKERRQ(ierr);

	/* dump coords to disk */
	/*
	ierr = DMDAGetCoordinates(da,&coords);CHKERRQ(ierr);
	ierr = PetscViewerBinaryOpen( ((PetscObject)da)->comm, "coord-data.dat", FILE_MODE_WRITE, &v );CHKERRQ(ierr);
	ierr = VecView( coords, v );CHKERRQ(ierr);
	ierr = PetscViewerDestroy(v);CHKERRQ(ierr);
	*/
	 
	/* dump dm to disk */
	ierr = DMDAPackDataToFile( da, "dmda_checkpoint_output.dat" );CHKERRQ(ierr);
	
	ierr = DMDestroy(da);CHKERRQ(ierr);
	ierr = VecDestroy( x );CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "test_dmda_checkpoint_load"
PetscErrorCode test_dmda_checkpoint_load( void ) 
{
	DM  da;
	Vec x,coords;
	PetscReal val;
	PetscScalar max;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	ierr = DMDACreateFromPackDataToFile( PETSC_COMM_WORLD, "dmda_checkpoint_output.dat",&da );CHKERRQ(ierr);
	ierr = DMView( da, PETSC_VIEWER_STDOUT_WORLD );CHKERRQ(ierr);
	
	ierr = DMDALoadGlobalVectorFromFile( da, "dmda_checkpoint_stressfield.dat", &x );CHKERRQ(ierr);
	/*
	 VecNorm( x, NORM_1, &val );	PetscPrintf( PETSC_COMM_WORLD, "|x| = %1.5e \n", val );
	 VecNorm( x, NORM_2, &val ); PetscPrintf( PETSC_COMM_WORLD, "|x|_2 = %1.5e \n", val );
	 VecMin( x, 0, &max ); PetscPrintf( PETSC_COMM_WORLD, "min(x) = %1.5e \n", max );
	 VecMax( x, 0, &max ); PetscPrintf( PETSC_COMM_WORLD, "max(x) = %1.5e \n", max );
	 */	
	
	//ierr = DMDALoadCoordinatesFromFile(da,"coord-data.dat");CHKERRQ(ierr);
	
	/* dump field to vtk */
	ierr = DMDAViewPetscVTK(da, x, "dmda_checkpoint_2.vtk");CHKERRQ(ierr);
	
	ierr = DMDestroy(da);CHKERRQ(ierr);
	ierr = VecDestroy(x);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "test_DMDACheckPoint"
PetscErrorCode test_DMDACheckPoint(void)
{
	PetscErrorCode ierr;
	PetscBool restart, checkpoint,flg;
	
	PetscFunctionBegin;
	checkpoint = PETSC_FALSE;
	PetscOptionsGetBool( PETSC_NULL, "-checkpoint", &checkpoint, &flg );
	if( checkpoint == PETSC_TRUE ) {
		ierr = test_dmda_checkpoint_pack();CHKERRQ(ierr);
	}
	
	restart = PETSC_FALSE;
	PetscOptionsGetBool( PETSC_NULL, "-restart", &restart, &flg );
	if( restart == PETSC_TRUE ) {
		ierr = test_dmda_checkpoint_load();CHKERRQ(ierr);
	}
	
	PetscFunctionReturn(0);
}


int main( int argc,char **argv )
{
	PetscErrorCode ierr;
	PetscInt mx,my,mz;
	
	PetscInitialize(&argc,&argv,(char *)0,0);
	
	ierr = test_DMDACheckPoint();CHKERRQ(ierr);

	ierr = PetscFinalize();CHKERRQ(ierr);
	return 0;
}
