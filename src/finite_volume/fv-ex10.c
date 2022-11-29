
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
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

PetscErrorCode bcset_default(FVDA fv,
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
    if (coor[3*f + 1] >= -0.5 && coor[3*f + 1] <= 0.5) {
      bcvalue[f] = 1.3;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode bcset_default_neumann(FVDA fv,
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
    flux[f] = FVFLUX_NEUMANN_CONSTRAINT;
    bcvalue[f] = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscBool initial_thermal_field(PetscScalar coords[],PetscScalar vals[], void *data)
{
  PetscBool impose = PETSC_TRUE;

  if (coords[1] >= -0.5 && coords[1] <= 0.5) {
    vals[0] = 1.3;
  } else {
    vals[0] = 0.3;
  }

  PetscFunctionReturn(impose);
}

PetscErrorCode t10(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 32*2+1;//65;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,F;
  Mat            J;
  DM             dm;
  SNES           snes;
  
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_ADV_DIFF,0,0);CHKERRQ(ierr);
  //ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_PARABOLIC,0,0);CHKERRQ(ierr);

  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  //DMDASetInterpolationType(fv->dm_fv,DMDA_Q1);
  //DMDASetRefinementFactor(fv->dm_fv,3,3,3);
  
  {
    Vec gcoor;
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"k",1);CHKERRQ(ierr);
  {
    PetscInt  f,nfaces;
    PetscReal *k;
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,0,&k);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      k[f] = 1.0;
    }
  }
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr);
  {
    PetscInt        f,nfaces;
    const PetscReal *face_centroid,*face_normal;
    PetscReal       *vdotn;
    const PetscReal velocity[] = { 10.0e1, 0.0, 0.0 }; /* imposed velocity field */
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,1,&vdotn);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      vdotn[f] = velocity[0] * face_normal[3*f+0]
      + velocity[1] * face_normal[3*f+1]
      + velocity[2] * face_normal[3*f+2];
    }
  }
  
  /* set boundary value at intitial time */
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_default_neumann,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,bcset_default_neumann,NULL);CHKERRQ(ierr);

  ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  
  ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  
  dm = fv->dm_fv;
  ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);

  ierr = FVDAVecTraverse(fv,X,0.0,0,initial_thermal_field,NULL);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);
  
  ierr = SNESSetFunction(snes,F,eval_F,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,eval_J,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"x_adv-diff.vts");
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
  ierr = t10();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
