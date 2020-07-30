
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda.h>
#include <fvda_utils.h>


PetscErrorCode bcset_west(FVDA fv,
                          DACellFace face,
                          PetscInt nfaces,
                          const PetscReal coor[],
                          const PetscReal normal[],
                          const PetscInt cell[],
                          PetscReal time,
                          FVFluxType flux[],
                          PetscReal bcvalue[],
                          void *ctx)
{
  PetscInt f;
  
  for (f=0; f<nfaces; f++) {
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    bcvalue[f] = 0.3;
  }
  for (f=0; f<nfaces; f++) {
    PetscReal r2=0;
    //r2 += coor[3*f+0]*coor[3*f+0];
    r2 += coor[3*f+1]*coor[3*f+1];
    r2 += coor[3*f+2]*coor[3*f+2];
    if (r2 <= 0.4*0.4) {
      bcvalue[f] = 1.3;
    }
    
    //bcvalue[f] = 0.3;
    //if (fabs(coor[3*f+1]) < 0.4) {
    //  bcvalue[f] = 1.3;
    //}
    
  }
  PetscFunctionReturn(0);
}

PetscErrorCode t6(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 12;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,Xk,F;
  Mat            J;
  DM             dm;
  SNES           snes;
  PetscInt       nt,max;
  PetscReal      *dt = NULL;
  
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_TRUE,FVDA_HYPERBOLIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  ierr = FVDASetup_TimeDep(fv);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr);
  {
    PetscInt        f,nfaces;
    const PetscReal *face_centroid,*face_normal;
    PetscReal       *vdotn;
    const PetscReal velocity[] = { 1.0, 0.0, 0.0 }; /* imposed velocity field */
    
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,0,&vdotn);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      vdotn[f] = velocity[0] * face_normal[3*f+0] + velocity[1] * face_normal[3*f+1] + velocity[2] * face_normal[3*f+2];
    }
  }
  
  /* set boundary value at intitial time */
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_west,NULL);CHKERRQ(ierr);
  
  dm = fv->dm_fv;
  ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);
  
  ierr = SNESSetFunction(snes,F,fvda_eval_F_timedep,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,fvda_eval_J_timedep,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  ierr = FVDAAccessData_TimeDep(fv,&dt,&Xk);CHKERRQ(ierr);
  
  max = 30;
  *dt = 0.1;

  /* Set the initial condition to be consistent with the ambient background Dirichlet BC */
  ierr = VecSet(Xk,0.3);CHKERRQ(ierr);
  
  /* View the initial condition */
  nt = 0;
  ierr = VecCopy(Xk,X);CHKERRQ(ierr);
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"step%.4d-x.vts",nt);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  for (nt=1; nt<max; nt++) {
    PetscViewer viewer;
    char        fname[256];
    
    /* Push current state into old state */
    ierr = VecCopy(X,Xk);CHKERRQ(ierr);
    
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  
    sprintf(fname,"step%.4d-x.vts",nt);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = t6();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
