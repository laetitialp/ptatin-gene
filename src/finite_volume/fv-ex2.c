
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
  }
  
  for (f=0; f<nfaces; f++) {
    PetscReal r2=0;
    r2 += coor[3*f+1]*coor[3*f+1];
    bcvalue[f] = 1.3 * PetscExpReal(-r2/0.01) + 0.3;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode bcset_def(FVDA fv,
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
  PetscFunctionReturn(0);
}

PetscErrorCode t2(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 24;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,F;
  Mat            J;
  DM             dm;
  SNES           snes;
  
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_HYPERBOLIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
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
      //printf("vdotn[f] %+1.4e\n",vdotn[f]);
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
  
  ierr = SNESSetFunction(snes,F,eval_F,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,eval_J,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  /*
   ierr = eval_J(snes,X,J,J,NULL);CHKERRQ(ierr);
   MatView(J,PETSC_VIEWER_STDOUT_WORLD);
   exit(1);
   */
  
  //ierr = SNESComputeJacobian(snes,X,J,J);CHKERRQ(ierr);
  //MatView(J,PETSC_VIEWER_STDOUT_WORLD);
  
  ierr = VecSet(X,1.0);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"x.vts");
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

PetscErrorCode t2_hr(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 24;
  const PetscInt m[] = {mx,mx,3};
  FVDA           fv;
  Vec            X,F;
  Mat            J;
  DM             dm;
  SNES           snes;
  
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_HYPERBOLIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-0.1,0.1);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr);
  {
    PetscInt        f,nfaces;
    const PetscReal *face_centroid,*face_normal;
    PetscReal       *vdotn;
    const PetscReal velocity[] = { 1.0, 0.1, 0.0 }; /* imposed velocity field */
    
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,0,&vdotn);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      vdotn[f] = velocity[0] * face_normal[3*f+0] + velocity[1] * face_normal[3*f+1] + velocity[2] * face_normal[3*f+2];
      //printf("vdotn[f] %+1.4e\n",vdotn[f]);
    }
  }
  
  /* set boundary value at intitial time */
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_west,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,bcset_def,NULL);CHKERRQ(ierr);
  
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

  ierr = SNESSetFunction(snes,F,eval_F_hr,NULL);CHKERRQ(ierr);
  //ierr = SNESSetFunction(snes,F,eval_F,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,eval_J,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  ierr = VecSet(X,0.3);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

#if 0
  //ierr = SNESSetFunction(snes,F,eval_F_hr,NULL);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,eval_F,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,eval_J,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = VecSet(X,0.3);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,F,eval_F_hr,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,eval_J,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
#endif
  
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"x.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = FVDAView_CellData(fv,X,PETSC_TRUE,"xcell");CHKERRQ(ierr);

  {
    DM          dmf;
    Vec         Xv;
    PetscViewer viewer;
    char        fname[256];
    
    ierr = FVDAFieldSetUpProjectToVertex_Q1(fv,&dmf,&Xv);CHKERRQ(ierr);
    ierr = FVDAFieldProjectReconstructionToVertex_Q1(fv,X,0.3,1.3,dmf,Xv);CHKERRQ(ierr);
    
    sprintf(fname,"xv.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(Xv,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    
    ierr = VecDestroy(&Xv);CHKERRQ(ierr);
    ierr = DMDestroy(&dmf);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode t2_ssp(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 128;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,F;
  DM             dm;
  const PetscReal range[]={0.3,1.3};
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_HYPERBOLIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
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
    const PetscReal velocity[] = { 1.0, 0.1, 0.0 }; /* imposed velocity field */
    
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,0,&vdotn);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      vdotn[f] = velocity[0] * face_normal[3*f+0] + velocity[1] * face_normal[3*f+1] + velocity[2] * face_normal[3*f+2];
    }
  }
  
  /* set boundary value at intitial time */
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_west,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,bcset_def,NULL);CHKERRQ(ierr);
  
  dm = fv->dm_fv;
  
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  
  ierr = VecSet(X,0.3);CHKERRQ(ierr);
  
  PetscReal dt = 0.1 * 0.03125/PetscSqrtReal(1.0*1.0 + 0.1*0.1);
  PetscReal time;
  PetscInt k;
  PetscLogDouble t0,t1;
  
  PetscTime(&t0);
  dt = (2.0 / ((PetscReal)mx)) * (PetscSqrtReal(2.0)/2.0) * PetscSqrtReal(1.0*1.0 + 0.1*0.1);
  time = dt;
  for (k=1; k<10; k++) {
    PetscPrintf(PETSC_COMM_WORLD,"[step %D]\n",k);
    //ierr = FVDAStep_FEuler(fv,time,dt,X,F);CHKERRQ(ierr);
    ierr = FVDAStep_RK2SSP(fv,range,time,dt,X,F);CHKERRQ(ierr);
    ierr = VecCopy(F,X);CHKERRQ(ierr);
    
    /*
    if (k%25==0) {
      PetscViewer viewer;
      char        fname[256];
      
      sprintf(fname,"x-%d.vts",k);
      ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
      ierr = VecView(X,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      
      sprintf(fname,"xcell-%d",k);
      ierr = FVDAView_CellData(fv,X,PETSC_TRUE,fname);CHKERRQ(ierr);
    }
    */
  }
  PetscTime(&t1);
  printf("time %1.2e\n",t1-t0);
  
  
  
  ierr = FVDAView_CellData(fv,X,PETSC_TRUE,"xcell");CHKERRQ(ierr);
  
  {
    DM          dmf;
    Vec         Xv;
    PetscViewer viewer;
    char        fname[256];
    
    ierr = FVDAFieldSetUpProjectToVertex_Q1(fv,&dmf,&Xv);CHKERRQ(ierr);
    ierr = FVDAFieldProjectReconstructionToVertex_Q1(fv,X,0.3,1.3,dmf,Xv);CHKERRQ(ierr);
    
    sprintf(fname,"xv.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(Xv,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    
    ierr = VecDestroy(&Xv);CHKERRQ(ierr);
    ierr = DMDestroy(&dmf);CHKERRQ(ierr);
  }
  
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       tid = 0;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-tid",&tid,NULL);CHKERRQ(ierr);
  switch (tid) {
    case 0:
      ierr = t2();CHKERRQ(ierr);
      break;
    case 1:
      ierr = t2_hr();CHKERRQ(ierr);
      break;
    case 2:
      ierr = t2_ssp();CHKERRQ(ierr);
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Valid values for -tid {0,1,2}");
      break;
  }
  ierr = PetscFinalize();
  return ierr;
}
