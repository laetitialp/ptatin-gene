
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>


PetscErrorCode FVDAGetCellPropertyInfo(FVDA fv,PetscInt *len,const char ***name)
{
  PetscFunctionBegin;
  if (len) { *len = fv->ncoeff_cell; }
  if (name) { *name = (const char**)fv->cell_coeff_name; }
  PetscFunctionReturn(0);
}

PetscErrorCode FVDARegisterCellProperty(FVDA fv,const char name[],PetscInt blocksize)
{
  PetscErrorCode ierr;
  PetscInt       index = -1;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  {
    PetscInt c;
    for (c=0; c<fv->ncoeff_cell; c++) {
      PetscBool match = PETSC_FALSE;
      ierr = PetscStrcmp(name,fv->cell_coeff_name[c],&match);CHKERRQ(ierr);
      if (match) { index = c; break; }
    }
  }
  if (index != -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"A cell property with name \"%s\" has already been registered (index %D). Textual names are required to be unique",name,index);
  
  index = fv->ncoeff_cell;
  ierr = PetscRealloc(sizeof(char**)*(fv->ncoeff_cell+1),&fv->cell_coeff_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&fv->cell_coeff_name[index]);CHKERRQ(ierr);
  
  ierr = PetscRealloc(sizeof(PetscInt*)*(fv->ncoeff_cell+1),&fv->cell_coeff_size);CHKERRQ(ierr);
  fv->cell_coeff_size[index] = blocksize * fv->ncells;
  
  ierr = PetscRealloc(sizeof(PetscReal**)*(fv->ncoeff_cell+1),&fv->cell_coefficient);CHKERRQ(ierr);
  ierr = PetscCalloc1(fv->ncells * blocksize,&fv->cell_coefficient[index]);CHKERRQ(ierr);
  //printf("<mem> cell_coefficient %1.2e (MB)\n",sizeof(PetscReal)*blocksize*fv->ncells * 1.0e-6);
  
  fv->ncoeff_cell++;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDACellPropertyGetInfo(FVDA fv,const char name[],PetscInt *index,PetscInt *len,PetscInt *bs)
{
  PetscErrorCode ierr;
  PetscInt       c,i=-1,l,b;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  for (c=0; c<fv->ncoeff_cell; c++) {
    PetscBool match = PETSC_FALSE;
    ierr = PetscStrcmp(name,fv->cell_coeff_name[c],&match);CHKERRQ(ierr);
    if (match) { i = c; break; }
  }
  if (i < 0 || i >= fv->ncoeff_cell) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided cell property index %D is not valid, Must be less < %D. Name \"%s\" not registered as cell property",index,fv->ncoeff_cell,name);
  
  l = fv->cell_coeff_size[i];
  b = l / fv->ncells;
  l = fv->cell_coeff_size[i] / b;
  
  if (index) { *index = i; }
  if (len) { *len = l; }
  if (bs) { *bs = b; }
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetCellPropertyArray(FVDA fv,PetscInt index,PetscReal *data[])
{
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  if (index < 0 || index >= fv->ncoeff_cell) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided cell property index %D is not valid, Must be less < %D",index,fv->ncoeff_cell);
  *data = fv->cell_coefficient[index];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetCellPropertyArrayRead(FVDA fv,PetscInt index,const PetscReal *data[])
{
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  if (index < 0 || index >= fv->ncoeff_cell) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided cell property index %D is not valid, Must be less < %D",index,fv->ncoeff_cell);
  *data = (const PetscReal*)fv->cell_coefficient[index];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetCellPropertyByNameArrayRead(FVDA fv,const char name[],const PetscReal *data[])
{
  PetscInt       index=-1,c;
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  for (c=0; c<fv->ncoeff_cell; c++) {
    PetscBool match = PETSC_FALSE;
    ierr = PetscStrcmp(name,fv->cell_coeff_name[c],&match);CHKERRQ(ierr);
    if (match) { index = c; break; }
  }
  if (index < 0 || index >= fv->ncoeff_cell) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided cell property index %D is not valid, Must be less < %D. Name \"%s\" not registered as cell property",index,fv->ncoeff_cell,name);
  *data = (const PetscReal*)fv->cell_coefficient[index];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetCellPropertyByNameArray(FVDA fv,const char name[],PetscReal *data[])
{
  PetscInt       index=-1,c;
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  for (c=0; c<fv->ncoeff_cell; c++) {
    PetscBool match = PETSC_FALSE;
    ierr = PetscStrcmp(name,fv->cell_coeff_name[c],&match);CHKERRQ(ierr);
    if (match) { index = c; break; }
  }
  if (index < 0 || index >= fv->ncoeff_cell) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided cell property index %D is not valid, Must be less < %D. Name \"%s\" not registered as cell property",index,fv->ncoeff_cell,name);
  *data = (PetscReal*)fv->cell_coefficient[index];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetFacePropertyInfo(FVDA fv,PetscInt *len,const char ***name)
{
  PetscFunctionBegin;
  if (len) { *len = fv->ncoeff_face; }
  if (name) { *name = (const char**)fv->face_coeff_name; }
  PetscFunctionReturn(0);
}

PetscErrorCode FVDARegisterFaceProperty(FVDA fv,const char name[],PetscInt blocksize)
{
  PetscErrorCode ierr;
  PetscInt       index = -1;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  {
    PetscInt c;
    for (c=0; c<fv->ncoeff_face; c++) {
      PetscBool match = PETSC_FALSE;
      ierr = PetscStrcmp(name,fv->face_coeff_name[c],&match);CHKERRQ(ierr);
      if (match) { index = c; break; }
    }
  }
  if (index != -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"A face property with name \"%s\" has already been registered (index %D). Textual names are required to be unique",name,index);
  
  index = fv->ncoeff_face;
  ierr = PetscRealloc(sizeof(char**)*(fv->ncoeff_face+1),&fv->face_coeff_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&fv->face_coeff_name[index]);CHKERRQ(ierr);
  
  ierr = PetscRealloc(sizeof(PetscInt*)*(fv->ncoeff_face+1),&fv->face_coeff_size);CHKERRQ(ierr);
  fv->face_coeff_size[index] = blocksize * fv->nfaces;
  
  ierr = PetscRealloc(sizeof(PetscReal**)*(fv->ncoeff_face+1),&fv->face_coefficient);CHKERRQ(ierr);
  ierr = PetscCalloc1(fv->nfaces * blocksize,&fv->face_coefficient[index]);CHKERRQ(ierr);
  //printf("<mem> face_coefficient %1.2e (MB)\n",sizeof(PetscReal)*blocksize*fv->nfaces * 1.0e-6);
  
  fv->ncoeff_face++;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAFacePropertyGetInfo(FVDA fv,const char name[],PetscInt *index,PetscInt *len,PetscInt *bs)
{
  PetscErrorCode ierr;
  PetscInt       c,i=-1,l,b;
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  for (c=0; c<fv->ncoeff_face; c++) {
    PetscBool match = PETSC_FALSE;
    ierr = PetscStrcmp(name,fv->face_coeff_name[c],&match);CHKERRQ(ierr);
    if (match) { i = c; break; }
  }
  if (i < 0 || i >= fv->ncoeff_face) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided face property index %D is not valid, Must be less < %D. Name \"%s\" not registered as face property",index,fv->ncoeff_face,name);
  
  l = fv->face_coeff_size[i];
  b = l / fv->nfaces;
  l = fv->face_coeff_size[i] / b;
  
  if (index) { *index = i; }
  if (len) { *len = l; }
  if (bs) { *bs = b; }
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetFacePropertyArray(FVDA fv,PetscInt index,PetscReal *data[])
{
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  if (index < 0 || index >= fv->ncoeff_face) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided face property index %D is not valid, Must be less < %D",index,fv->ncoeff_face);
  *data = fv->face_coefficient[index];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetFacePropertyArrayRead(FVDA fv,PetscInt index,const PetscReal *data[])
{
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  if (index < 0 || index >= fv->ncoeff_face) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided face property index %D is not valid, Must be less < %D",index,fv->ncoeff_face);
  *data = (const PetscReal*)fv->face_coefficient[index];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetFacePropertyByNameArrayRead(FVDA fv,const char name[],const PetscReal *data[])
{
  PetscInt       index=-1,c;
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  for (c=0; c<fv->ncoeff_face; c++) {
    PetscBool match = PETSC_FALSE;
    ierr = PetscStrcmp(name,fv->face_coeff_name[c],&match);CHKERRQ(ierr);
    if (match) { index = c; break; }
  }
  if (index < 0 || index >= fv->ncoeff_face) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided face property index %D is not valid, Must be less < %D. Name \"%s\" not registered as face property",index,fv->ncoeff_face,name);
  *data = (const PetscReal*)fv->face_coefficient[index];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetFacePropertyByNameArray(FVDA fv,const char name[],PetscReal *data[])
{
  PetscInt       index=-1,c;
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  for (c=0; c<fv->ncoeff_face; c++) {
    PetscBool match = PETSC_FALSE;
    ierr = PetscStrcmp(name,fv->face_coeff_name[c],&match);CHKERRQ(ierr);
    if (match) { index = c; break; }
  }
  if (index < 0 || index >= fv->ncoeff_face) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Provided face property index %D is not valid, Must be less < %D. Name \"%s\" not registered as face property",index,fv->ncoeff_face,name);
  *data = (PetscReal*)fv->face_coefficient[index];
  PetscFunctionReturn(0);
}
