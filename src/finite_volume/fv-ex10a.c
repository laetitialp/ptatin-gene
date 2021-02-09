
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_utils.h>

static PetscErrorCode FVSetDirichletFromNeighbour(FVDA fv,Vec T,DACellFace face)
{
  PetscInt       f,len,s,e;
  const PetscInt *indices;
  PetscInt       cell;
  Vec            Tl;
  DM             dm;
  PetscScalar    *LA_T;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Get rank-local T values
  dm = fv->dm_fv;

  ierr = DMGetLocalVector(dm,&Tl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,T,INSERT_VALUES,Tl);CHKERRQ(ierr);
  ierr = VecGetArray(Tl,&LA_T);CHKERRQ(ierr);

  //ierr = VecGetArray(T,&LA_T);CHKERRQ(ierr);
  // Get boundary cells faces indices
  ierr = FVDAGetBoundaryFaceIndicesRead(fv,face,&len,&indices);CHKERRQ(ierr);
  ierr = FVDAGetBoundaryFaceIndicesOwnershipRange(fv,face,&s,&e);CHKERRQ(ierr);

  for (f=0; f<len; f++) {
    PetscInt fvid = indices[f];
    cell = fv->face_element_map[2*fvid + 0];
	if (face == DACELL_FACE_W){
      fv->boundary_value[s + f] = LA_T[cell];
      fv->boundary_flux[s + f] = FVFLUX_DIRICHLET_CONSTRAINT;
	  //PetscPrintf(PETSC_COMM_SELF,"fid=indices[%d]=%d, cell[%d]=face_element_map[%d]=%d \n",f,indices[f],f,2*fvid + 0,cell);
    } else {
      fv->boundary_value[s + f] = 0.3;
      fv->boundary_flux[s + f] = FVFLUX_DIRICHLET_CONSTRAINT;
    }
	/* This may not be required in solved problems since the BC insertions are embeded inside the matrix operations */
    //ierr = VecSetValue(T,cell,fv->boundary_value[s + f],INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecSetValue(Tl,cell,fv->boundary_value[s + f],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobal(dm,Tl,INSERT_VALUES,T);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Tl);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Tl);CHKERRQ(ierr);  
  ierr = VecRestoreArray(Tl,&LA_T);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Tl);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode FVSetDirichletFromNeighbourVdotN(FVDA fv,Vec T,DACellFace face)
{
  PetscInt       f,len,s,e;
  const PetscInt *indices;
  const PetscReal *vdotn;
  PetscInt       cell;
  Vec            Tl;
  DM             dm;
  PetscScalar    *LA_T;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Get rank-local T values
  dm = fv->dm_fv;

  ierr = DMGetLocalVector(dm,&Tl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,T,INSERT_VALUES,Tl);CHKERRQ(ierr);
  ierr = VecGetArray(Tl,&LA_T);CHKERRQ(ierr);

  //ierr = VecGetArray(T,&LA_T);CHKERRQ(ierr);
  // Get boundary cells faces indices
  ierr = FVDAGetBoundaryFaceIndicesRead(fv,face,&len,&indices);CHKERRQ(ierr);
  ierr = FVDAGetBoundaryFaceIndicesOwnershipRange(fv,face,&s,&e);CHKERRQ(ierr);

  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"v.n",&vdotn);CHKERRQ(ierr);
  
  for (f=0; f<len; f++) {
    PetscInt fvid = indices[f];
    cell = fv->face_element_map[2*fvid + 0];
	if (vdotn[fvid] < 0.0){
      fv->boundary_value[s + f] = LA_T[cell];
      fv->boundary_flux[s + f] = FVFLUX_DIRICHLET_CONSTRAINT;
	  //PetscPrintf(PETSC_COMM_SELF,"fid=indices[%d]=%d, cell[%d]=face_element_map[%d]=%d \n",f,indices[f],f,2*fvid + 0,cell);
    } else {
      fv->boundary_value[s + f] = 0.3;
      fv->boundary_flux[s + f] = FVFLUX_DIRICHLET_CONSTRAINT;
    }
	/* This may not be required in solved problems since the BC insertions are embeded inside the matrix operations */
    //ierr = VecSetValue(T,cell,fv->boundary_value[s + f],INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecSetValue(Tl,cell,fv->boundary_value[s + f],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobal(dm,Tl,INSERT_VALUES,T);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Tl);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Tl);CHKERRQ(ierr);  
  ierr = VecRestoreArray(Tl,&LA_T);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Tl);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode PrintBoundaryValues(FVDA fv,Vec T,DACellFace face)
{
  PetscInt       f,len,s,e;
  const PetscInt  *indices;
  PetscInt        cell;
  const PetscReal *LA_T;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  // Get rank-local T values
  ierr = VecGetArrayRead(T,&LA_T);CHKERRQ(ierr);
  
  ierr = FVDAGetBoundaryFaceIndicesRead(fv,face,&len,&indices);CHKERRQ(ierr);
  ierr = FVDAGetBoundaryFaceIndicesOwnershipRange(fv,face,&s,&e);CHKERRQ(ierr);
  
  for (f=0; f<len; f++) {
    PetscInt fvid = indices[f];
    cell = fv->face_element_map[2*fvid + 0];
	PetscPrintf(PETSC_COMM_SELF,"LA_T[%d] = %f, BC_val[%d] = %f \n",cell,LA_T[cell],s+f,fv->boundary_value[s + f]);
  }

  ierr = VecRestoreArrayRead(T,&LA_T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscBool iterator_initial_thermal_field(PetscScalar coor[],PetscScalar *val,void *ctx)
{
  PetscBool impose=PETSC_TRUE;
  
  *val = coor[2]*2.0;

  return impose;
}

PetscErrorCode t10a(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 8; //32*2+1;//65;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            T;
  DM             dm;
 
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_PARABOLIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
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
    const PetscReal velocity[] = { 1.0e1, 1.0e1, 0.0 }; /* imposed velocity field */
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,1,&vdotn);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      vdotn[f] = velocity[0] * face_normal[3*f+0]
      + velocity[1] * face_normal[3*f+1]
      + velocity[2] * face_normal[3*f+2];
    }
  }
  
  dm = fv->dm_fv;
  ierr = DMCreateGlobalVector(dm,&T);CHKERRQ(ierr);
  ierr = FVDAVecTraverse(fv,T,0.0,0,iterator_initial_thermal_field,NULL);CHKERRQ(ierr);
  //ierr = VecSet(T,1.0);CHKERRQ(ierr);

  const DACellFace flist[] = { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_N, DACELL_FACE_B, DACELL_FACE_F };
  PetscInt l;
  for (l=0; l<sizeof(flist)/sizeof(DACellFace); l++) {
    //ierr = FVSetDirichletFromNeighbour(fv,T,flist[l]);CHKERRQ(ierr);
	ierr = FVSetDirichletFromNeighbourVdotN(fv,T,flist[l]);CHKERRQ(ierr);
	//ierr = PrintBoundaryValues(fv,T,flist[l]);CHKERRQ(ierr);
  }
  VecAssemblyBegin(T);
  VecAssemblyEnd(T); 
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"T.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(T,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&T);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = t10a();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
