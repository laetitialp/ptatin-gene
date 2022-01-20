
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petsc/private/dmdaimpl.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_private.h>
#include <fvda_utils.h>


static PetscErrorCode DMDAGetElementOwnershipRanges2d(DM dm,PetscInt *_li[],PetscInt *_lj[]);
static PetscErrorCode DMDAGetElementOwnershipRanges3d(DM dm,PetscInt *_li[],PetscInt *_lj[],PetscInt *_lk[]);

static PetscErrorCode private_FVDACreateFaceLabels(DM dm,PetscInt *ne,PetscInt *e2e[],DACellFace *ft[],DACellFaceLocation *fl[]);

static PetscErrorCode private_FVDASetUpLocalFVIndices2d(FVDA fv,DM dmfv);
static PetscErrorCode private_FVDASetUpLocalFVIndices3d(FVDA fv,DM dmfv);

/*static PetscErrorCode private_FVDACreateElementFaceLabels3d(FVDA fv);*/

static PetscErrorCode private_FVDASetUpBoundaryLabels(FVDA fv);
static PetscErrorCode private_FVDAUpdateGeometry3d(FVDA fv);


PetscErrorCode FVDACreateFromDMDA(DM vertex_layout,FVDA *_fv)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  FVDA           fv;
  PetscInt       mi[]={0,0,0},Mi[]={0,0,0},nel,nen;
  const PetscInt *e;
  Vec            coor;
  PetscBool      isda = PETSC_FALSE;
  
  
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)vertex_layout,DMDA,&isda);CHKERRQ(ierr);
  if (!isda) SETERRQ(PetscObjectComm((PetscObject)vertex_layout),PETSC_ERR_SUP,"Only valid for input DMs of type DMDA");
  ierr = DMGetDimension(vertex_layout,&dim);CHKERRQ(ierr);
  ierr = FVDACreate(PetscObjectComm((PetscObject)vertex_layout),&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,dim);CHKERRQ(ierr);
  ierr = DMDACreateCompatibleDMDA(vertex_layout,dim,&fv->dm_geometry);CHKERRQ(ierr);
  ierr = DMDASetElementType(fv->dm_geometry,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
  ierr = DMDAGetElements(fv->dm_geometry,&nel,&nen,&e);CHKERRQ(ierr);
  ierr = DMDAGetElementsSizes(fv->dm_geometry,&mi[0],&mi[1],&mi[2]);CHKERRQ(ierr);
  ierr = DMDAGetInfo(fv->dm_geometry,NULL,&Mi[0],&Mi[1],&Mi[2],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  Mi[0]--;
  Mi[1]--;
  Mi[2]--;
  ierr = FVDASetSizes(fv,(const PetscInt*)mi,(const PetscInt*)Mi);CHKERRQ(ierr);
  
  /* copy coordinates */
  ierr = DMCreateGlobalVector(fv->dm_geometry,&fv->vertex_coor_geometry);CHKERRQ(ierr);
  ierr = DMGetCoordinates(vertex_layout,&coor);CHKERRQ(ierr);
  if (!coor) SETERRQ(PetscObjectComm((PetscObject)vertex_layout),PETSC_ERR_SUP,"Require the provided DMDA (arg 1) has coordinates set");
  ierr = VecCopy(coor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  
  *_fv = fv;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDACreate2d(MPI_Comm comm,PetscInt Mi[],FVDA *_fv)
{
  PetscErrorCode ierr;
  FVDA           fv;
  
  
  PetscFunctionBegin;
  ierr = FVDACreate(comm,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,2);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,(const PetscInt*)Mi);CHKERRQ(ierr);
  *_fv = fv;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDACreate3d(MPI_Comm comm,PetscInt Mi[],FVDA *_fv)
{
  PetscErrorCode ierr;
  FVDA           fv;
  
  
  PetscFunctionBegin;
  ierr = FVDACreate(comm,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,(const PetscInt*)Mi);CHKERRQ(ierr);
  *_fv = fv;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDACreate(MPI_Comm comm,FVDA *_fv)
{
  PetscErrorCode ierr;
  FVDA           fv;
  
  
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_FVDA),&fv);CHKERRQ(ierr);
  ierr = PetscMemzero(fv,sizeof(struct _p_FVDA));CHKERRQ(ierr);
  fv->comm = comm;
  fv->dim = -1;
  fv->mi[0] = PETSC_DECIDE;
  fv->mi[1] = PETSC_DECIDE;
  fv->mi[2] = PETSC_DECIDE;
  fv->Mi[0] = PETSC_DECIDE;
  fv->Mi[1] = PETSC_DECIDE;
  fv->Mi[2] = PETSC_DECIDE;
  
  ierr = PetscCalloc1(1,&fv->cell_coeff_name);CHKERRQ(ierr);
  ierr = PetscCalloc1(1,&fv->face_coeff_name);CHKERRQ(ierr);
  fv->cell_coeff_name[0] = NULL;
  fv->face_coeff_name[0] = NULL;
  ierr = PetscCalloc1(1,&fv->cell_coefficient);CHKERRQ(ierr);
  ierr = PetscCalloc1(1,&fv->face_coefficient);CHKERRQ(ierr);
  fv->cell_coefficient[0] = NULL;
  fv->face_coefficient[0] = NULL;
  ierr = PetscCalloc1(1,&fv->cell_coeff_size);CHKERRQ(ierr);
  ierr = PetscCalloc1(1,&fv->face_coeff_size);CHKERRQ(ierr);
  fv->cell_coeff_size[0] = 0;
  fv->face_coeff_size[0] = 0;
  fv->ctx = NULL;
  fv->ctx_destroy = NULL;
  
  fv->setup = PETSC_FALSE;
  *_fv = fv;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDADestroy(FVDA *_fv)
{
  PetscErrorCode ierr;
  PetscInt       c;
  FVDA           fv;
  
  
  PetscFunctionBegin;
  if (!_fv) PetscFunctionReturn(0);
  if (!*_fv) PetscFunctionReturn(0);
  fv = *_fv;
  
  ierr = DMDestroy(&fv->dm_geometry);CHKERRQ(ierr);
  ierr = VecDestroy(&fv->vertex_coor_geometry);CHKERRQ(ierr);
  ierr = DMDestroy(&fv->dm_fv);CHKERRQ(ierr);
  /*ierr = DMDestroy(&fv->dm_fv_2);CHKERRQ(ierr);*/
  /*ierr = VecDestroy(&fv->cell_coor_geometry_local);CHKERRQ(ierr);*/
  ierr = PetscFree(fv->cell_ownership_i);CHKERRQ(ierr);
  ierr = PetscFree(fv->cell_ownership_j);CHKERRQ(ierr);
  ierr = PetscFree(fv->cell_ownership_k);CHKERRQ(ierr);
  if (fv->cell_coeff_name) {
    for (c=0; c<fv->ncoeff_cell; c++) {
      ierr = PetscFree(fv->cell_coeff_name[c]);CHKERRQ(ierr);
      ierr = PetscFree(fv->cell_coefficient[c]);CHKERRQ(ierr);
    }
    ierr = PetscFree(fv->cell_coeff_name);CHKERRQ(ierr);
    ierr = PetscFree(fv->cell_coefficient);CHKERRQ(ierr);
    ierr = PetscFree(fv->cell_coeff_size);CHKERRQ(ierr);
  }
  if (fv->face_coeff_name) {
    for (c=0; c<fv->ncoeff_face; c++) {
      ierr = PetscFree(fv->face_coeff_name[c]);CHKERRQ(ierr);
      ierr = PetscFree(fv->face_coefficient[c]);CHKERRQ(ierr);
    }
    ierr = PetscFree(fv->face_coeff_name);CHKERRQ(ierr);
    ierr = PetscFree(fv->face_coefficient);CHKERRQ(ierr);
    ierr = PetscFree(fv->face_coeff_size);CHKERRQ(ierr);
  }
  
  ierr = PetscFree(fv->face_element_map);CHKERRQ(ierr);
  ierr = PetscFree(fv->face_fv_map);CHKERRQ(ierr);
  ierr = PetscFree(fv->face_normal);CHKERRQ(ierr);
  ierr = PetscFree(fv->face_centroid);CHKERRQ(ierr);
  ierr = PetscFree(fv->face_location);CHKERRQ(ierr);
  ierr = PetscFree(fv->face_type);CHKERRQ(ierr);
  ierr = PetscFree(fv->face_idx_interior);CHKERRQ(ierr);
  ierr = PetscFree(fv->face_idx_boundary);CHKERRQ(ierr);

  ierr = PetscFree(fv->boundary_flux);CHKERRQ(ierr);
  ierr = PetscFree(fv->boundary_value);CHKERRQ(ierr);
  
  if (fv->ctx_destroy) {
    ierr = fv->ctx_destroy(fv);CHKERRQ(ierr);
  }
  
  ierr = PetscFree(fv);CHKERRQ(ierr);
  *_fv = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDASetDimension(FVDA fv,PetscInt dim)
{
  PetscFunctionBegin;
  fv->dim = dim;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDASetSizes(FVDA fv,const PetscInt mi[],const PetscInt Mi[])
{
  PetscInt d;
  PetscFunctionBegin;
  if (mi) {
    for (d=0; d<fv->dim; d++) {
      if (fv->mi[d] != PETSC_DECIDE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot change local cell size (direction %D)",d);
      fv->mi[d] = mi[d];
    }
  }
  if (Mi) {
    for (d=0; d<fv->dim; d++) {
      if (fv->Mi[d] != PETSC_DECIDE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot change global cell size (direction %D)",d);
      fv->Mi[d] = Mi[d];
    }
  }
  PetscFunctionReturn(0);
}

/*
 steady state, advection             , upwind, upwind-hr
 steady state, diffusion             , none
 steady state, advection + diffusion , upwind
 time dep.,    advection             , upwind
 time dep.,    diffusion             , none
 time dep.,    advection + diffusion , upwind
 */

PetscErrorCode FVDASetProblemType(FVDA fv,PetscBool Qdot,FVDAPDEType equation_type,PetscInt numerical_flux,PetscInt reconstruction)
{
  PetscFunctionBegin;
  if (fv->setup) SETERRQ(fv->comm,PETSC_ERR_ORDER,"Must call FVDASetProblemType() before FVDASetUp()");
  fv->q_dot = Qdot;
  fv->equation_type = equation_type; /* 0: hyperbolic, 1: elliptic: 2: parabolic */
  fv->numerical_flux = numerical_flux;
  fv->reconstruction = reconstruction;
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDASetUp(FVDA fv)
{
  PetscErrorCode ierr;
  PetscBool      geometry_provided = PETSC_FALSE;
  PetscInt       mp[]={0,0,0};
  
  
  PetscFunctionBegin;
  if (fv->setup) PetscFunctionReturn(0);
  
  if (fv->dm_geometry) {
    PetscInt       nen;
    const PetscInt *e;
    
    geometry_provided = PETSC_TRUE;
    ierr = DMDAGetElements(fv->dm_geometry,&fv->ncells,&nen,&e);CHKERRQ(ierr);
  } else {
    PetscInt       mi[]={0,0,0},nen;
    const PetscInt *e;
    /* cannot build FV from only local cell sizes */

    if (fv->dim >= 1) {
      if (fv->Mi[0] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Cannot define geometry from only local size (i)");
    }
    if (fv->dim >= 2) {
      if (fv->Mi[1] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Cannot define geometry from only local size (j)");
    }
    if (fv->dim >=3) {
      if (fv->Mi[2] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Cannot define geometry from only local size (k)");
    }
    
    if (fv->dim == 2) {
      ierr = DMDACreate2d(fv->comm,
                          DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                          DMDA_STENCIL_BOX,
                          fv->Mi[0]+1,fv->Mi[1]+1,
                          PETSC_DECIDE,PETSC_DECIDE,
                          2,
                          1,
                          NULL,NULL,&fv->dm_geometry);CHKERRQ(ierr);
    } else if (fv->dim == 3) {
      ierr = DMDACreate3d(fv->comm,
                          DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                          DMDA_STENCIL_BOX,
                          fv->Mi[0]+1,fv->Mi[1]+1,fv->Mi[2]+1,
                          PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                          3,
                          1,
                          NULL,NULL,NULL,&fv->dm_geometry);CHKERRQ(ierr);
    }
    ierr = DMSetFromOptions(fv->dm_geometry);CHKERRQ(ierr);
    ierr = DMSetUp(fv->dm_geometry);CHKERRQ(ierr);

    ierr = DMDASetElementType(fv->dm_geometry,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
    ierr = DMDAGetElements(fv->dm_geometry,&fv->ncells,&nen,&e);CHKERRQ(ierr);
    ierr = DMDAGetElementsSizes(fv->dm_geometry,&mi[0],&mi[1],&mi[2]);CHKERRQ(ierr);
    
    ierr = FVDASetSizes(fv,(const PetscInt*)mi,NULL);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(fv->dm_geometry,&fv->vertex_coor_geometry);CHKERRQ(ierr);
  }

  if (fv->dim >= 1) {
    if (fv->mi[0] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Local size cannot be PETSC_DECIDE (i)");
    if (fv->Mi[0] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Global size cannot be PETSC_DECIDE (i)");
  }
  if (fv->dim >= 2) {
    if (fv->mi[1] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Local size cannot be PETSC_DECIDE (j)");
    if (fv->Mi[1] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Global size cannot be PETSC_DECIDE (j)");
  }
  if (fv->dim == 3) {
    if (fv->mi[2] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Local size cannot be PETSC_DECIDE (k)");
    if (fv->Mi[2] == PETSC_DECIDE) SETERRQ(fv->comm,PETSC_ERR_ARG_WRONG,"Global size cannot be PETSC_DECIDE (k)");
  }
  
  ierr = DMDAGetInfo(fv->dm_geometry,NULL,NULL,NULL,NULL,&mp[0],&mp[1],&mp[2],NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (fv->dim == 2) {
    ierr = DMDAGetElementOwnershipRanges2d(fv->dm_geometry,&fv->cell_ownership_i,&fv->cell_ownership_j);CHKERRQ(ierr);
    
#if 0
    ierr = DMDACreate2d(fv->comm,
                        DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX, /* [NOTE] BOX stencil */
                        fv->Mi[0],fv->Mi[1],
                        mp[0],mp[1],
                        1,
                        2, /* [NOTE] stencil width 2 */
                        fv->cell_ownership_i,fv->cell_ownership_j,&fv->dm_fv_2);CHKERRQ(ierr);
#endif
    
    ierr = DMDACreate2d(fv->comm,
                        DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX, /* [NOTE] BOX stencil */
                        fv->Mi[0],fv->Mi[1],
                        mp[0],mp[1],
                        1,
                        1, /* [NOTE] stencil width 1 */
                        fv->cell_ownership_i,fv->cell_ownership_j,&fv->dm_fv);CHKERRQ(ierr);
  }

  if (fv->dim == 3) {
    ierr = DMDAGetElementOwnershipRanges3d(fv->dm_geometry,&fv->cell_ownership_i,&fv->cell_ownership_j,&fv->cell_ownership_k);CHKERRQ(ierr);

#if 0
    ierr = DMDACreate3d(fv->comm,
                        DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX, /* [NOTE] BOX stencil */
                        fv->Mi[0],fv->Mi[1],fv->Mi[2],
                        mp[0],mp[1],mp[2],
                        1,
                        2, /* [NOTE] stencil width 2 */
                        fv->cell_ownership_i,fv->cell_ownership_j,fv->cell_ownership_k,&fv->dm_fv_2);CHKERRQ(ierr);
#endif
    
    ierr = DMDACreate3d(fv->comm,
                        DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX, /* [NOTE] BOX stencil */
                        fv->Mi[0],fv->Mi[1],fv->Mi[2],
                        mp[0],mp[1],mp[2],
                        1,
                        1, /* [NOTE] stencil width 1 */
                        fv->cell_ownership_i,fv->cell_ownership_j,fv->cell_ownership_k,&fv->dm_fv);CHKERRQ(ierr);
  }
  
  /*ierr = DMSetUp(fv->dm_fv_2);CHKERRQ(ierr);*/
  /*ierr = DMSetFromOptions(fv->dm_fv_2);CHKERRQ(ierr);*/
  
  //ierr = DMSetFromOptions(fv->dm_fv);CHKERRQ(ierr);
  ierr = DMSetUp(fv->dm_fv);CHKERRQ(ierr);
  
  /* force creation of global coordinate on fv DM */
  ierr = DMDASetUniformCoordinates(fv->dm_fv, 0.0,1.0, 0.0,1.0, 0.0,1.0);CHKERRQ(ierr);
  /*ierr = DMDASetUniformCoordinates(fv->dm_fv_2, 0.0,1.0, 0.0,1.0, 0.0,1.0);CHKERRQ(ierr);*/
  
  /* select dm_fv or dm_fv_2 based on reconstruction type */
  /*
  {
    DM dmfv = fv->dm_fv;
    DM cdm;
    ierr = DMGetCoordinateDM(dmfv,&cdm);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(cdm,&fv->cell_coor_geometry_local);CHKERRQ(ierr);
  }
  */
   
  ierr = private_FVDACreateFaceLabels(fv->dm_geometry,&fv->nfaces,&fv->face_element_map,&fv->face_type,&fv->face_location);CHKERRQ(ierr);
  ierr = PetscCalloc1((fv->dim)*fv->nfaces,&fv->face_normal);CHKERRQ(ierr);
  ierr = PetscCalloc1((fv->dim)*fv->nfaces,&fv->face_centroid);CHKERRQ(ierr);
  //printf("<mem> face_normal      %1.2e (MB)\n",sizeof(PetscReal)*fv->nfaces*3 * 1.0e-6);
  //printf("<mem> face_centroid    %1.2e (MB)\n",sizeof(PetscReal)*fv->nfaces*3 * 1.0e-6);
  
  /*ierr = private_FVDACreateElementFaceLabels3d(fv);CHKERRQ(ierr);*/
  
  ierr = private_FVDASetUpBoundaryLabels(fv);CHKERRQ(ierr);

  ierr = PetscCalloc1(fv->nfaces_boundary,&fv->boundary_flux);CHKERRQ(ierr);
  ierr = PetscCalloc1(fv->nfaces_boundary,&fv->boundary_value);CHKERRQ(ierr);
  //printf("<mem> boundary_flux    %1.2e (MB)\n",sizeof(FVFluxType)*fv->nfaces_boundary * 1.0e-6);
  //printf("<mem> boundary_value   %1.2e (MB)\n",sizeof(PetscReal)*fv->nfaces_boundary * 1.0e-6);
  
  /* select dm_fv or dm_fv_2 based on reconstruction type */
  switch (fv->dim) {
    case 2:
      ierr = private_FVDASetUpLocalFVIndices2d(fv,fv->dm_fv);CHKERRQ(ierr);
      break;
      
    case 3:
      ierr = private_FVDASetUpLocalFVIndices3d(fv,fv->dm_fv);CHKERRQ(ierr);
      break;
      
    default:
      break;
  }
  
  fv->setup = PETSC_TRUE;
  
  /* update geometry */
  if (geometry_provided) {
    ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FVDASetGeometryDM(FVDA fv,DM dm)
{
  PetscErrorCode ierr;
  PetscInt bs,dim;
  PetscFunctionBegin;
  if (fv->dim < 0) SETERRQ(fv->comm,PETSC_ERR_SUP,"Must call FVDASetDimension() first");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMGetBlockSize(dm,&bs);CHKERRQ(ierr);
  if (dim != fv->dim) SETERRQ(fv->comm,PETSC_ERR_SUP,"Dimension of geometry DM must match dimension of FVDA");
  if (bs != fv->dim) SETERRQ(fv->comm,PETSC_ERR_SUP,"Block size of geometry DM must match dimension of FVDA");
  if (fv->dm_geometry) {
    if (fv->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ORDER,"Cannot call FVDASetUp() before calling FVDASetGeometryDM()");
    ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
    ierr = DMDestroy(&fv->dm_geometry);CHKERRQ(ierr);
    fv->dm_geometry = dm;
  } else {
    ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
    fv->dm_geometry = dm;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetGeometryDM(FVDA fv,DM *dm)
{
  PetscFunctionBegin;
  if (dm) *dm = fv->dm_geometry;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetDM(FVDA fv,DM *dm)
{
  PetscFunctionBegin;
  if (dm) *dm = fv->dm_fv;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetGeometryCoordinates(FVDA fv,Vec *c)
{
  PetscFunctionBegin;
  if (c) *c = fv->vertex_coor_geometry;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAUpdateGeometry(FVDA fv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  switch (fv->dim) {
    case 2:
      SETERRQ(fv->comm,PETSC_ERR_SUP,"Not implemented in 2d");
      break;
    case 3:
      ierr = private_FVDAUpdateGeometry3d(fv);CHKERRQ(ierr);
      break;
      
    default:
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode private_FVDAUpdateGeometry3d(FVDA fv)
{
  const PetscInt  NSD = 3;
  PetscErrorCode  ierr;
  const PetscReal xi2d[] = { 0.0 , 0.0 };
  PetscReal       cell_coor[3 * DACELL3D_Q1_SIZE];
  Vec             coorl;
  const PetscReal *_coorl;
  PetscInt        c,f,dm_nel,dm_nen;
  const PetscInt  *dm_element;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(fv->comm,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  if (fv->dim != 3) SETERRQ(fv->comm,PETSC_ERR_SUP,"Only valid for 3d");
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  
  /* update face normals */
  /* update face centroids */
  for (f=0; f<fv->nfaces; f++) {
    PetscInt       e_minus,e_plus,eidx,d;
    const PetscInt *element;
    
    e_minus = fv->face_element_map[2*f+0];
    e_plus  = fv->face_element_map[2*f+1];
    eidx = e_minus;
    if (eidx == E_MINUS_OFF_RANK) {
      eidx = e_plus;
    }
    
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * eidx];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_coorl,cell_coor);CHKERRQ(ierr);
    
    ierr = _EvaluateFaceNormal3d(fv->face_type[f],cell_coor,xi2d,&fv->face_normal[NSD * f]);CHKERRQ(ierr);

    /* flip normal on a sub-domain boundary */
    if (e_minus == E_MINUS_OFF_RANK) {
      for (d=0; d<3; d++) {
        fv->face_normal[NSD * f + d] *= -1.0;
      }
    }
    
    ierr = _EvaluateFaceCoord3d(fv->face_type[f],cell_coor,xi2d,&fv->face_centroid[NSD * f]);CHKERRQ(ierr);
    /*
    printf("face %d: c %+1.4e %+1.4e %+1.4e : n %+1.4e %+1.4e %+1.4e\n",f,
           fv->face_centroid[NSD * f],fv->face_centroid[NSD * f+1],fv->face_centroid[NSD * f+2],
           fv->face_normal[NSD * f],fv->face_normal[NSD * f+1],fv->face_normal[NSD * f+2]);
    */
  }
  
  /* update cell centroids (default) */
  {
    Vec       fv_coor,fv_coorl;
    DM        cdm;
    PetscReal *_fv_coor;
    
    ierr = DMGetCoordinateDM(fv->dm_fv,&cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_fv,&fv_coor);CHKERRQ(ierr);
    ierr = VecGetArray(fv_coor,&_fv_coor);CHKERRQ(ierr);
    
#if 0
    for (f=0; f<fv->nfaces; f++) {
      PetscInt e_minus,e_plus,eidx,i,d;
      const PetscInt *element;
      
      e_minus = fv->face_element_map[2*f+0];
      e_plus  = fv->face_element_map[2*f+1];
      eidx = e_minus;
      if (eidx == E_MINUS_OFF_RANK) {
        eidx = e_plus;
      }
      
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * eidx];
      
      ierr = DACellGeometry3d_GetCoordinates(element,_coorl,cell_coor);CHKERRQ(ierr);
      
      /* average cell vertex coordinates */
      for (d=0; d<NSD; d++) {
        _fv_coor[NSD * eidx + d] = 0.0;
        for (i=0; i<DACELL3D_Q1_SIZE; i++) {
          _fv_coor[NSD * eidx + d] += cell_coor[NSD * i + d];
        }
        _fv_coor[NSD * eidx + d] /= (PetscReal)DACELL3D_Q1_SIZE;
      }
    }
#endif

    for (c=0; c<fv->ncells; c++) {
      PetscInt       eidx,i,d;
      const PetscInt *element;
      
      eidx = c;
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * eidx];
      ierr = DACellGeometry3d_GetCoordinates(element,_coorl,cell_coor);CHKERRQ(ierr);
      /* average cell vertex coordinates */
      for (d=0; d<NSD; d++) {
        _fv_coor[NSD * eidx + d] = 0.0;
        for (i=0; i<DACELL3D_Q1_SIZE; i++) {
          _fv_coor[NSD * eidx + d] += cell_coor[NSD * i + d];
        }
        _fv_coor[NSD * eidx + d] /= (PetscReal)DACELL3D_Q1_SIZE;
      }
    }

    ierr = VecRestoreArray(fv_coor,&_fv_coor);CHKERRQ(ierr);
    
    ierr = DMGetCoordinatesLocal(fv->dm_fv,&fv_coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(cdm,fv_coor,INSERT_VALUES,fv_coorl);CHKERRQ(ierr);
  }
  
  ierr = VecRestoreArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  
  /* update cell centroids (overlap 2) */
  /*
  {
    Vec fv_coor,fv2_coor,fv2_coorl;
    DM  cdm;
    
    ierr = DMGetCoordinates(fv->dm_fv,&fv_coor);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(fv->dm_fv_2,&cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_fv_2,&fv2_coor);CHKERRQ(ierr);
    ierr = VecCopy(fv_coor,fv2_coor);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(fv->dm_fv_2,&fv2_coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(cdm,fv2_coor,INSERT_VALUES,fv2_coorl);CHKERRQ(ierr);
  }
  */
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetBoundaryFaceIndicesRead(FVDA fv,DACellFace face,PetscInt *len,const PetscInt *indices[])
{
  PetscInt offset;
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  offset = fv->boundary_ranges[ (PetscInt)face ];

  if (len) *len = fv->boundary_ranges[ (PetscInt)face + 1] - fv->boundary_ranges[ (PetscInt)face ];
  if (indices) *indices = (const PetscInt*)&fv->face_idx_boundary[ offset ];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetBoundaryFaceIndicesOwnershipRange(FVDA fv,DACellFace face,PetscInt *start,PetscInt *end)
{
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  if (start) *start = fv->boundary_ranges[ (PetscInt)face ];
  if (end)   *end   = fv->boundary_ranges[ (PetscInt)face + 1];
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetFaceInfo(FVDA fv,PetscInt *nfaces,const DACellFaceLocation *l[],const DACellFace *f[],const PetscReal *n[],const PetscReal *c[])
{
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  if (nfaces) *nfaces = fv->nfaces;
  if (l) *l = (const DACellFaceLocation*)fv->face_location;
  if (f) *f = (const DACellFace*)fv->face_type;
  if (n) *n = (const PetscReal*)fv->face_normal;
  if (c) *c = (const PetscReal*)fv->face_centroid;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAFaceIterator(FVDA fv,DACellFace face,PetscBool require_normals,PetscReal time,
                                PetscErrorCode (*user_setter)(FVDA,
                                                              DACellFace,
                                                              PetscInt,
                                                              const PetscReal*,
                                                              const PetscReal*,
                                                              const PetscInt*,
                                                              PetscReal,FVFluxType*,PetscReal*,void*),
                                void *data)
{
  PetscInt       len,d,f,s,e;
  const PetscInt *indices;
  PetscReal      *face_coords;
  PetscReal      *face_normals = NULL;
  PetscInt       *cell;
  FVFluxType     *bc_type;
  PetscReal      *bc_value;
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  ierr = FVDAGetBoundaryFaceIndicesRead(fv,face,&len,&indices);CHKERRQ(ierr);
  ierr = FVDAGetBoundaryFaceIndicesOwnershipRange(fv,face,&s,&e);CHKERRQ(ierr);
 
  ierr = PetscCalloc1(fv->dim * len,&face_coords);CHKERRQ(ierr);
  for (f=0; f<len; f++) {
    PetscInt fid = indices[f];
    for (d=0; d<fv->dim; d++) {
      face_coords[fv->dim * f + d] = fv->face_centroid[fv->dim * fid + d];
    }
  }
  
  if (require_normals) {
    ierr = PetscCalloc1(fv->dim * len,&face_normals);CHKERRQ(ierr);
    for (f=0; f<len; f++) {
      PetscInt fid = indices[f];
      for (d=0; d<fv->dim; d++) {
        face_normals[fv->dim * f + d] = fv->face_normal[fv->dim * fid + d];
      }
    }
  }

  ierr = PetscCalloc1(len,&cell);CHKERRQ(ierr);
  for (f=0; f<len; f++) {
    PetscInt fid = indices[f];
    cell[f] = fv->face_element_map[2*fid + 0];
    if (cell[f] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Boundary facet [%D -> %D] is missing cell-. A boundary facet should always have a cell- living on the sub-domain",f,fid);
  }
  
  ierr = PetscCalloc1(len,&bc_type);CHKERRQ(ierr);
  for (f=0; f<len; f++) {
    fv->boundary_flux[s + f] = FVFLUX_UN_INITIALIZED;
    bc_type[f] = FVFLUX_UN_INITIALIZED;
  }
  
  ierr = PetscCalloc1(len,&bc_value);CHKERRQ(ierr);
  
  ierr = user_setter(fv, face, len, (const PetscReal*)face_coords, (const PetscReal*)face_normals, (const PetscInt*)cell, time, bc_type, bc_value, data);CHKERRQ(ierr);
  
  /* pack everything the user set */
  for (f=0; f<len; f++) {
    if (bc_type[f] != FVFLUX_DIRICHLET_CONSTRAINT && bc_type[f] != FVFLUX_NEUMANN_CONSTRAINT) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Boundary facet [%d] must have a value of FVFLUX_DIRICHLET_CONSTRAINT or FVFLUX_NEUMANN_CONSTRAINT. Found %D",f,bc_type[f]);
    }
    
    fv->boundary_flux[s + f]  = bc_type[f];
    fv->boundary_value[s + f] = bc_value[f];
  }
  
  ierr = PetscFree(face_coords);CHKERRQ(ierr);
  ierr = PetscFree(face_normals);CHKERRQ(ierr);
  ierr = PetscFree(cell);CHKERRQ(ierr);
  ierr = PetscFree(bc_type);CHKERRQ(ierr);
  ierr = PetscFree(bc_value);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAFaceIteratorPointwise(FVDA fv,DACellFace face,PetscBool require_normals,PetscReal time,
                                PetscErrorCode (*user_setter)(FVDA,
                                                              DACellFace,
                                                              const PetscReal*,
                                                              const PetscReal*,
                                                              PetscInt,
                                                              PetscReal,FVFluxType*,PetscReal*,void*),
                                void *data)
{
  PetscInt       len,d,f,s,e;
  const PetscInt *indices;
  PetscReal      face_coords[] = {0,0,0};
  PetscReal      face_normals[] = {0,0,0};
  PetscInt       cell;
  FVFluxType     bc_type;
  PetscReal      bc_value;
  PetscErrorCode ierr;

  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call FVDASetUp() first");
  ierr = FVDAGetBoundaryFaceIndicesRead(fv,face,&len,&indices);CHKERRQ(ierr);
  ierr = FVDAGetBoundaryFaceIndicesOwnershipRange(fv,face,&s,&e);CHKERRQ(ierr);

  for (f=0; f<len; f++) {
    PetscInt fid = indices[f];

    for (d=0; d<fv->dim; d++) {
      face_coords[d] = fv->face_centroid[fv->dim * fid + d];
    }
    if (require_normals) {
      for (d=0; d<fv->dim; d++) {
        face_normals[d] = fv->face_normal[fv->dim * fid + d];
      }
    }

    cell = fv->face_element_map[2*fid + 0];
    if (cell < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Boundary facet [%D -> %D] is missing cell-. A boundary facet should always have a cell- living on the sub-domain",f,fid);

    fv->boundary_flux[fid] = FVFLUX_UN_INITIALIZED;
    bc_type = FVFLUX_UN_INITIALIZED;
    bc_value = 0.0;
    ierr = user_setter(fv, face, (const PetscReal*)face_coords, (const PetscReal*)face_normals, cell, time, &bc_type, &bc_value, data);CHKERRQ(ierr);
    
    /* pack everything the user set */
    if (bc_type != FVFLUX_DIRICHLET_CONSTRAINT && bc_type != FVFLUX_NEUMANN_CONSTRAINT) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Boundary facet [%d] must have a value of FVFLUX_DIRICHLET_CONSTRAINT or FVFLUX_NEUMANN_CONSTRAINT. Found %D",f,bc_type);
    }
    
    fv->boundary_flux[s + f]  = bc_type;
    fv->boundary_value[s + f] = bc_value;
  }
  
  PetscFunctionReturn(0);
}

void EvaluateBasis_Q1_1D(const PetscReal xi[], PetscReal N[])
{
  N[0] = 0.5 * (1.0 - xi[0]);
  N[1] = 0.5 * (1.0 + xi[0]);
}

void EvaluateBasisDerivative_Q1_1D(const PetscReal xi[], PetscReal dN[][DACELL1D_Q1_SIZE])
{
  dN[0][0] = -0.5;
  dN[0][1] =  0.5;
}

void EvaluateBasis_Q1_2D(const PetscReal xi[], PetscReal N[])
{
  N[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]);
  N[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]);
  N[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]);
  N[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]);
}

void EvaluateBasisDerivative_Q1_2D(const PetscReal xi[], PetscReal dN[][DACELL2D_Q1_SIZE])
{
  dN[0][0] = -0.25 * (1.0 - xi[1]);
  dN[0][1] =  0.25 * (1.0 - xi[1]);
  dN[0][2] =  0.25 * (1.0 + xi[1]);
  dN[0][3] = -0.25 * (1.0 + xi[1]);
  
  dN[1][0] = -0.25 * (1.0 - xi[0]);
  dN[1][1] = -0.25 * (1.0 + xi[0]);
  dN[1][2] =  0.25 * (1.0 + xi[0]);
  dN[1][3] =  0.25 * (1.0 - xi[0]);
}

void EvaluateBasis_Q1_3D(const PetscReal xi[], PetscReal N[])
{
  N[0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]);
  N[1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]);
  N[2] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]);
  N[3] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]);
  
  N[4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]);
  N[5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]);
  N[6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]);
  N[7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]);
}

void EvaluateBasisDerivative_Q1_3D(const PetscReal xi[], PetscReal dN[][DACELL3D_Q1_SIZE])
{
  dN[0][0] = -0.125 * (1.0 - xi[1]) * (1.0 - xi[2]);
  dN[0][1] =  0.125 * (1.0 - xi[1]) * (1.0 - xi[2]);
  dN[0][2] =  0.125 * (1.0 + xi[1]) * (1.0 - xi[2]);
  dN[0][3] = -0.125 * (1.0 + xi[1]) * (1.0 - xi[2]);
  dN[0][4] = -0.125 * (1.0 - xi[1]) * (1.0 + xi[2]);
  dN[0][5] =  0.125 * (1.0 - xi[1]) * (1.0 + xi[2]);
  dN[0][6] =  0.125 * (1.0 + xi[1]) * (1.0 + xi[2]);
  dN[0][7] = -0.125 * (1.0 + xi[1]) * (1.0 + xi[2]);

  dN[1][0] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[2]);
  dN[1][1] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[2]);
  dN[1][2] =  0.125 * (1.0 + xi[0]) * (1.0 - xi[2]);
  dN[1][3] =  0.125 * (1.0 - xi[0]) * (1.0 - xi[2]);
  dN[1][4] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[2]);
  dN[1][5] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[2]);
  dN[1][6] =  0.125 * (1.0 + xi[0]) * (1.0 + xi[2]);
  dN[1][7] =  0.125 * (1.0 - xi[0]) * (1.0 + xi[2]);

  dN[2][0] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[1]);
  dN[2][1] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[1]);
  dN[2][2] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[1]);
  dN[2][3] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[1]);
  dN[2][4] =  0.125 * (1.0 - xi[0]) * (1.0 - xi[1]);
  dN[2][5] =  0.125 * (1.0 + xi[0]) * (1.0 - xi[1]);
  dN[2][6] =  0.125 * (1.0 + xi[0]) * (1.0 + xi[1]);
  dN[2][7] =  0.125 * (1.0 - xi[0]) * (1.0 + xi[1]);
}

/*
 [Notes] 
 - A consistent ordering (counter clock-wise) is chosen for each edge.
   This is essential to ensure that the computed edge normals are pointing outwards wrt to the cell.
 - The order of the faces in cell_face_labels[] must match the labels provided in typedef enum {} DACellFace, e.g.
   { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_N, DACELL_FACE_B, DACELL_FACE_F }
   For 2D we ignore the slots associated with DACELL_FACE_B and DACELL_FACE_F
 
 Ordering is determined from DMDAGetElements_2D() in $(PETSC_DIR_/src/dm/impls/da/dagetelem.c
 
 76:         cell[0] = (i-Xs  ) + (j-Ys  )*(Xe-Xs);
 77:         cell[1] = (i-Xs+1) + (j-Ys  )*(Xe-Xs);
 78:         cell[2] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs);
 79:         cell[3] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs);

       N
   3 ------ 2
   |        |
 W |        | E
   |        |
   0 ------ 1
       S
 
*/
PetscErrorCode DACellGeometry2d_GetFaceIndices(DM dm,DACellFace face,PetscInt fidx[])
{
  PetscErrorCode ierr;
  PetscInt       face_location;
  const PetscInt cell_face_labels[] = {
    3,0, /* west */
    1,2, /* east */
    0,1, /* south */
    2,3  /* north */
  };
  PetscFunctionBegin;
  face_location = (PetscInt)face;
  ierr = PetscMemcpy(fidx,&cell_face_labels[4*face_location],sizeof(PetscInt)*4);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 [Notes]
 - A counter clock-wise ordering is chosen for each face.
   This is essential to ensure that the computed face normals are pointing outwards wrt to the cell.
 - The order of the faces in cell_face_labels[] must match the labels provided in typedef enum {} DACellFace, e.g.
   { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_N, DACELL_FACE_B, DACELL_FACE_F }
 
 Ordering is determined from DMDAGetElements_3D() in $(PETSC_DIR_/src/dm/impls/da/dagetelem.c
 
 147:           cell[0] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
 148:           cell[1] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
 149:           cell[2] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
 150:           cell[3] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
 151:           cell[4] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
 152:           cell[5] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
 153:           cell[6] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
 154:           cell[7] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
 
                 N
             3 ------ 2
             |        |
           W |        | E (back face => k)
             |        |
             0 ------ 1
                 S
       N
   7 ------ 6
   |        |
 W |        | E (front face => k+1)
   |        |
   4 ------ 5
       S

*/
PetscErrorCode DACellGeometry3d_GetFaceIndices(DM dm,DACellFace face,PetscInt fidx[])
{
  PetscErrorCode ierr;
  PetscInt       face_location;
  const PetscInt cell_face_labels[] = {
    0,4,7,3, /* west */
    5,1,2,6, /* east */
    0,1,5,4, /* south */
    3,7,6,2, /* north */
    1,0,3,2, /* back */
    4,5,6,7  /* front */
  };
  PetscFunctionBegin;
  face_location = (PetscInt)face;
  ierr = PetscMemcpy(fidx,&cell_face_labels[4*face_location],sizeof(PetscInt)*4);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DACellGeometry2d_GetCoordinates(const PetscInt element[],const PetscReal mesh_coor[],PetscReal coor[])
{
  PetscInt i,d,vidx;
  PetscFunctionBegin;
  for (i=0; i<DACELL2D_Q1_SIZE; i++) {
    vidx = element[i];
    for (d=0; d<2; d++) {
      coor[2*i + d] = mesh_coor[2*vidx + d];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DACellGeometry3d_GetCoordinates(const PetscInt element[],const PetscReal mesh_coor[],PetscReal coor[])
{
  PetscInt i,d,vidx;
  PetscFunctionBegin;
  for (i=0; i<DACELL3D_Q1_SIZE; i++) {
    vidx = element[i];
    for (d=0; d<3; d++) {
      coor[3*i + d] = mesh_coor[3*vidx + d];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode _EvaluateFaceNormal3d(
                                     DACellFace face,
                                     const PetscReal coor[], /* should contain 8 points with dimension 3 (x,y,z) */
                                     const PetscReal xi0[],  /* should contain 1 point with dimension 2 (xi,eta) */
                                     PetscReal n0[])         /* n0[] contains 1 point with dimension 3 (x,y,z) */
{
  PetscReal      x_s,y_s,z_s;
  PetscReal      x_t,y_t,z_t;
  PetscReal      n[3],mag;
  PetscInt       i;
  PetscReal      GNi_st[2][DACELL3D_FACE_VERTS];
  PetscInt       fidx[DACELL3D_FACE_VERTS];
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  ierr = DACellGeometry3d_GetFaceIndices(NULL,face,fidx);CHKERRQ(ierr);
  
  /*
   x_s \cross x_t  = |  i   j   k  |
                     | x_s y_s z_s |
                     | x_t y_t z_t |
                     
                     n_i =  (y_s*z_t - z_s*y_t)
                     n_j = -(x_s*z_t - z_s*x_t)
                     n_k =  (x_s*y_t - y_s*x_t)
  */
  
  EvaluateBasisDerivative_Q1_2D(xi0,GNi_st);
  
  /* x_,s = dN_i/ds x_i */
  x_s = y_s = z_s = 0.0;
  x_t = y_t = z_t = 0.0;
  for (i=0; i<DACELL3D_FACE_VERTS; i++) {
    x_s += GNi_st[0][i] * coor[3*fidx[i]  ];
    y_s += GNi_st[0][i] * coor[3*fidx[i]+1];
    z_s += GNi_st[0][i] * coor[3*fidx[i]+2];
    
    x_t += GNi_st[1][i] * coor[3*fidx[i]  ];
    y_t += GNi_st[1][i] * coor[3*fidx[i]+1];
    z_t += GNi_st[1][i] * coor[3*fidx[i]+2];
  }
  n[0] =  (y_s*z_t - z_s*y_t);
  n[1] = -(x_s*z_t - z_s*x_t);
  n[2] =  (x_s*y_t - y_s*x_t);
  
  mag = PetscSqrtReal( n[0]*n[0] + n[1]*n[1] + n[2]*n[2] );
  
  n0[0] = n[0] / mag;
  n0[1] = n[1] / mag;
  n0[2] = n[2] / mag;
  PetscFunctionReturn(0);
}

PetscErrorCode _EvaluateFaceCoord3d(
                                    DACellFace face,
                                    const PetscReal coor[], /* should contain 8 points with dimension 3 (x,y,z) */
                                    const PetscReal xi0[],  /* should contain 1 point with dimension 2 (xi,eta) */
                                    PetscReal c0[])         /* c0[] contains 1 point with dimension 3 (x,y,z) */
{
  PetscInt       i,fidx[DACELL3D_FACE_VERTS];
  PetscReal      Ni_st[DACELL3D_FACE_VERTS];
  PetscErrorCode ierr;

  
  PetscFunctionBegin;
  ierr = DACellGeometry3d_GetFaceIndices(NULL,face,fidx);CHKERRQ(ierr);
  EvaluateBasis_Q1_2D(xi0,Ni_st);
  c0[0] = c0[1] = c0[2] = 0.0;
  for (i=0; i<DACELL3D_FACE_VERTS; i++) {
    c0[0] += Ni_st[i] * coor[3*fidx[i]  ];
    c0[1] += Ni_st[i] * coor[3*fidx[i]+1];
    c0[2] += Ni_st[i] * coor[3*fidx[i]+2];
  }
  PetscFunctionReturn(0);
}

void quadrature_gauss_legenre_2point_2d(PetscInt *nq,PetscReal q_w[],PetscReal q_xi[])
{
  const PetscReal s = 0.577350269189;
  /*const PetscReal w_1d[] = { 1.0, 1.0 };*/
  const PetscReal xi_1d[] = { -s, s };
  PetscInt        nI,nJ;
  
  *nq = 4;
  for (nI=0; nI<2; nI++) {
    for (nJ=0; nJ<2; nJ++) {
      PetscInt idx = nI + nJ*2;
      /*q_w[idx] = w_1d[nI] * w_1d[nJ];*/
      q_w[idx] = 1.0;
      q_xi[2*idx+0] = xi_1d[nI];
      q_xi[2*idx+1] = xi_1d[nJ];
    }
  }
}

void quadrature_gauss_legenre_2point_3d(PetscInt *nq,PetscReal q_w[],PetscReal q_xi[])
{
  const PetscReal s = 0.577350269189;
  /*const PetscReal w_1d[] = { 1.0, 1.0 };*/
  const PetscReal xi_1d[] = { -s, s };
  PetscInt        nI,nJ,nK;
  
  *nq = 8;
  for (nI=0; nI<2; nI++) {
    for (nJ=0; nJ<2; nJ++) {
      for (nK=0; nK<2; nK++) {
        PetscInt idx = nI + nJ*2 + nK*2*2;
        /*q_w[idx] = w_1d[nI] * w_1d[nJ] * w_1d[nK];*/
        q_w[idx] = 1.0;
        q_xi[3*idx+0] = xi_1d[nI];
        q_xi[3*idx+1] = xi_1d[nJ];
        q_xi[3*idx+2] = xi_1d[nK];
      }
    }
  }
}

void cell_geometry_3d(const PetscReal el_coords[3*DACELL3D_Q1_SIZE],
                                     PetscInt nqp,
                                     PetscReal GNI[][3][DACELL3D_Q1_SIZE],
                                     PetscReal detJ[])
{
  PetscInt  k,p;
  PetscReal J[3][3];
  
  for (p=0; p<nqp; p++) {
    J[0][0] = J[0][1] = J[0][2] = 0.0;
    J[1][0] = J[1][1] = J[1][2] = 0.0;
    J[2][0] = J[2][1] = J[2][2] = 0.0;

    for (k=0; k<DACELL3D_Q1_SIZE; k++) {
      PetscReal xc = el_coords[3*k+0];
      PetscReal yc = el_coords[3*k+1];
      PetscReal zc = el_coords[3*k+2];
      
      J[0][0] += GNI[p][0][k] * xc;
      J[0][1] += GNI[p][0][k] * yc;
      J[0][2] += GNI[p][0][k] * zc;
      
      J[1][0] += GNI[p][1][k] * xc;
      J[1][1] += GNI[p][1][k] * yc;
      J[1][2] += GNI[p][1][k] * zc;
      
      J[2][0] += GNI[p][2][k] * xc;
      J[2][1] += GNI[p][2][k] * yc;
      J[2][2] += GNI[p][2][k] * zc;
    }
    
    detJ[p] = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
            - J[0][1]*(J[1][0]*J[2][2] + J[1][2]*J[2][0])
            + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);
  }
}

void cell_geometry_pointwise_3d(const PetscReal el_coords[3*DACELL3D_Q1_SIZE],
                      PetscReal GNI[3][DACELL3D_Q1_SIZE],
                      PetscReal *detJ)
{
  PetscInt  k;
  PetscReal J[3][3];
  
  J[0][0] = J[0][1] = J[0][2] = 0.0;
  J[1][0] = J[1][1] = J[1][2] = 0.0;
  J[2][0] = J[2][1] = J[2][2] = 0.0;
  
  for (k=0; k<DACELL3D_Q1_SIZE; k++) {
    PetscReal xc = el_coords[3*k+0];
    PetscReal yc = el_coords[3*k+1];
    PetscReal zc = el_coords[3*k+2];
    
    J[0][0] += GNI[0][k] * xc;
    J[0][1] += GNI[0][k] * yc;
    J[0][2] += GNI[0][k] * zc;
    
    J[1][0] += GNI[1][k] * xc;
    J[1][1] += GNI[1][k] * yc;
    J[1][2] += GNI[1][k] * zc;
    
    J[2][0] += GNI[2][k] * xc;
    J[2][1] += GNI[2][k] * yc;
    J[2][2] += GNI[2][k] * zc;
  }
  
  *detJ = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
        - J[0][1]*(J[1][0]*J[2][2] + J[1][2]*J[2][0])
        + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);
}

PetscErrorCode _EvaluateFaceArea3d(DACellFace face,
                                   const PetscReal coor[], /* should contain 8 points with dimension 3 (x,y,z) */
                                   PetscReal *a)           /* returned facet area */
{
  PetscReal      x_s,y_s,z_s,x_t,y_t,z_t,n[3],q_w[4],q_xi[2*4],dJ,area = 0;
  PetscInt       i,q,nq;
  PetscReal      GNi_st[2][DACELL3D_FACE_VERTS];
  PetscInt       fidx[DACELL3D_FACE_VERTS];
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  ierr = DACellGeometry3d_GetFaceIndices(NULL,face,fidx);CHKERRQ(ierr);
  quadrature_gauss_legenre_2point_2d(&nq,q_w,q_xi);

  for (q=0; q<nq; q++) {
    EvaluateBasisDerivative_Q1_2D(&q_xi[2*q],GNi_st);
    
    /* x_,s = dN_i/ds x_i */
    x_s = y_s = z_s = 0.0;
    x_t = y_t = z_t = 0.0;
    for (i=0; i<DACELL3D_FACE_VERTS; i++) {
      x_s += GNi_st[0][i] * coor[3*fidx[i]  ];
      y_s += GNi_st[0][i] * coor[3*fidx[i]+1];
      z_s += GNi_st[0][i] * coor[3*fidx[i]+2];
      
      x_t += GNi_st[1][i] * coor[3*fidx[i]  ];
      y_t += GNi_st[1][i] * coor[3*fidx[i]+1];
      z_t += GNi_st[1][i] * coor[3*fidx[i]+2];
    }
    n[0] =  (y_s*z_t - z_s*y_t);
    n[1] = -(x_s*z_t - z_s*x_t);
    n[2] =  (x_s*y_t - y_s*x_t);
    
    dJ = PetscSqrtReal( n[0]*n[0] + n[1]*n[1] + n[2]*n[2] );
    
    area += q_w[q] * 1.0 * dJ;
  }
  *a = area;
  PetscFunctionReturn(0);
}

PetscErrorCode _EvaluateCellVolume3d(const PetscReal coor[], /* should contain 8 points with dimension 3 (x,y,z) */
                                     PetscReal *v)           /* returned cell volume */
{
  PetscReal q_w[8],q_xi[3*8],dN[3][DACELL3D_Q1_SIZE],dJ,vol = 0;
  PetscInt  q,nq;
  
  
  PetscFunctionBegin;
  quadrature_gauss_legenre_2point_3d(&nq,q_w,q_xi);
  for (q=0; q<nq; q++) {
    EvaluateBasisDerivative_Q1_3D(&q_xi[3*q],dN);
    cell_geometry_pointwise_3d(coor,dN,&dJ);
    vol += q_w[q] * 1.0 * dJ;
  }
  *v = vol;
  PetscFunctionReturn(0);
}

PetscErrorCode cart_convert_2d(PetscInt r,const PetscInt mp[],PetscInt rij[])
{
  PetscFunctionBegin;
  rij[1] = r/mp[0];
  rij[0] = r - rij[1] * mp[0];
#if defined(PETSC_USE_DEBUG)
  if (r != rij[0] + rij[1]*mp[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"cart_convert_2d() conversion failed");
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode cart_convert_3d(PetscInt r,const PetscInt mp[],PetscInt rijk[])
{
  PetscInt rij;
  PetscFunctionBegin;
  rijk[2] = r / (mp[0] * mp[1]);
  rij = r - rijk[2] * mp[0] * mp[1];
  rijk[1] = rij/mp[0];
  rijk[0] = rij - rijk[1] * mp[0];
#if defined(PETSC_USE_DEBUG)
  if (r != rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"cart_convert_3d() conversion failed");
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode _cart_convert_index_to_ijk(PetscInt r,const PetscInt mp[],PetscInt rijk[])
{
  PetscInt rij;
  PetscFunctionBegin;
  rijk[2] = r / (mp[0] * mp[1]);
  rij = r - rijk[2] * mp[0] * mp[1];
  rijk[1] = rij/mp[0];
  rijk[0] = rij - rijk[1] * mp[0];
#if defined(PETSC_USE_DEBUG)
  if (r != rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"_cart_convert_index_to_ijk() conversion failed");
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode _cart_convert_ijk_to_index(const PetscInt rijk[],const PetscInt mp[],PetscInt *r)
{
  PetscFunctionBegin;
  *r = rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1];
#if defined(PETSC_USE_DEBUG)
  if (*r != rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"_cart_convert_ijk_to_index() conversion failed");
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAGetElementOwnershipRanges2d(DM dm,PetscInt *_li[],PetscInt *_lj[])
{
  PetscErrorCode ierr;
  PetscInt       mp[2],mx[2],mxr[2];
  PetscInt       *li,*lj;
  PetscMPIInt    r,crank,csize,rij[2];
  MPI_Status     stat;
  MPI_Comm       comm;
  
  
  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)dm);
  ierr = MPI_Comm_size(comm,&csize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&crank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,NULL,NULL,NULL,&mp[0],&mp[1],NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(mp[0],&li);CHKERRQ(ierr);
  ierr = PetscCalloc1(mp[1],&lj);CHKERRQ(ierr);
  ierr = DMDAGetElementsSizes(dm,&mx[0],&mx[1],NULL);CHKERRQ(ierr);
  
  if (crank == 0) {
    li[0] = mx[0];
    lj[0] = mx[1];
    for (r=1; r<csize; r++) {
      ierr = MPI_Recv(mxr,2,MPIU_INT,r,r,comm,&stat);CHKERRQ(ierr);
      
      rij[1] = r/mp[0];
      rij[0] = r - rij[1] * mp[0];
      
      if (r != rij[0] + rij[1]*mp[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk conversion failed");
      
      li[ rij[0] ] = mxr[0];
      lj[ rij[1] ] = mxr[1];
    }
  } else {
    ierr = MPI_Send(mx,2,MPIU_INT,0,crank,comm);CHKERRQ(ierr);
  }
  
  ierr = MPI_Bcast(li,mp[0],MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(lj,mp[1],MPIU_INT,0,comm);CHKERRQ(ierr);
  
  if (_li) { *_li = li; } else { ierr = PetscFree(li);CHKERRQ(ierr); }
  if (_lj) { *_lj = lj; } else { ierr = PetscFree(lj);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAGetElementOwnershipRanges3d(DM dm,PetscInt *_li[],PetscInt *_lj[],PetscInt *_lk[])
{
  PetscErrorCode ierr;
  PetscInt       mp[3],mx[3],mxr[3];
  PetscInt       *li,*lj,*lk;
  PetscMPIInt    r,crank,csize,rij,rijk[3];
  MPI_Status     stat;
  MPI_Comm       comm;
  
  
  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)dm);
  ierr = MPI_Comm_size(comm,&csize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&crank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,NULL,NULL,NULL,&mp[0],&mp[1],&mp[2],NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(mp[0],&li);CHKERRQ(ierr);
  ierr = PetscCalloc1(mp[1],&lj);CHKERRQ(ierr);
  ierr = PetscCalloc1(mp[2],&lk);CHKERRQ(ierr);
  ierr = DMDAGetElementsSizes(dm,&mx[0],&mx[1],&mx[2]);CHKERRQ(ierr);
  
  if (crank == 0) {
    li[0] = mx[0];
    lj[0] = mx[1];
    lk[0] = mx[2];
    for (r=1; r<csize; r++) {
      ierr = MPI_Recv(mxr,3,MPIU_INT,r,r,comm,&stat);CHKERRQ(ierr);
      
      rijk[2] = r / (mp[0] * mp[1]);
      rij = r - rijk[2] * mp[0] * mp[1];
      rijk[1] = rij/mp[0];
      rijk[0] = rij - rijk[1] * mp[0];
      
      if (r != rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk conversion failed");
      
      if (rijk[0] > mp[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk[0] conversion failed");
      if (rijk[1] > mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk[1] conversion failed");
      if (rijk[2] > mp[2]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk[2] conversion failed");
      
      li[ rijk[0] ] = mxr[0];
      lj[ rijk[1] ] = mxr[1];
      lk[ rijk[2] ] = mxr[2];
    }
  } else {
    ierr = MPI_Send(mx,3,MPIU_INT,0,crank,comm);CHKERRQ(ierr);
  }
  
  ierr = MPI_Bcast(li,mp[0],MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(lj,mp[1],MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(lk,mp[2],MPIU_INT,0,comm);CHKERRQ(ierr);
  
  if (_li) { *_li = li; } else { ierr = PetscFree(li);CHKERRQ(ierr); }
  if (_lj) { *_lj = lj; } else { ierr = PetscFree(lj);CHKERRQ(ierr); }
  if (_lk) { *_lk = lk; } else { ierr = PetscFree(lk);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode private_FVDACreateFaceLabels2d(DM dm,PetscInt *ne,PetscInt *e2e[],DACellFace *ft[],DACellFaceLocation *fl[])
{
  PetscInt           *edge2element; /* [2 * nedges]: edge2element[0] -> element[low] */
  DACellFace         *face_type; /* [nedges]: stores cell face of element[low] */
  DACellFaceLocation *face_loc; /* [nedges] */
  PetscInt           mx,my,ei,ej,elidx;
  PetscInt           e,nedges;
  PetscInt           sv[2],ev[2],nv[2],Nv[2];
  PetscBool          has_west_face,has_south_face,has_east_face,has_north_face;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;
  
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dm,&sv[0],&sv[1],NULL,&nv[0],&nv[1],NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,&Nv[0],&Nv[1],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  has_west_face = PETSC_FALSE;
  has_south_face = PETSC_FALSE;
  has_east_face = PETSC_FALSE;
  has_north_face = PETSC_FALSE;
  ev[0] = sv[0] + nv[0];
  ev[1] = sv[1] + nv[1];
  if (sv[0] == 0) has_west_face = PETSC_TRUE;
  if (sv[1] == 0) has_south_face = PETSC_TRUE;
  
  if (ev[0] == Nv[0]) has_east_face = PETSC_TRUE;
  if (ev[1] == Nv[1]) has_north_face = PETSC_TRUE;
  
#ifdef FVDA_DEBUG
  PetscPrintf(PETSC_COMM_SELF,"[rank %d] Boundary faces on sub-domain: w=%d s=%d , e=%d n=%d\n",(int)rank,(int)has_west_face,(int)has_south_face,(int)has_east_face,(int)has_north_face);
#endif
  
  ierr = DMDAGetElementsSizes(dm,&mx,&my,NULL);CHKERRQ(ierr);
  /* number of edges */
  nedges  = (mx + 1) * my;
  nedges += mx * (my + 1);
  
  ierr = PetscCalloc1(nedges*2,&edge2element);CHKERRQ(ierr);
  ierr = PetscCalloc1(nedges,&face_type);CHKERRQ(ierr);
  ierr = PetscCalloc1(nedges,&face_loc);CHKERRQ(ierr);
  
  e = 0; /* edge counter initialize */
  
  /* x-normal face sweep */
  {
    for (ej=0; ej<my; ej++) {
      ei = 0;
      elidx = ei + ej * mx;
      if (has_west_face) {
        edge2element[2*e+0] = elidx;
        edge2element[2*e+1] = -1;
        face_type[e] = DACELL_FACE_W;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        edge2element[2*e+0] = E_MINUS_OFF_RANK;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_W;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
      
      for (ei=1; ei<mx; ei++) {
        elidx = (ei-1) + ej * mx;
        edge2element[2*e+0] = elidx;
        elidx = ei + ej * mx;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_E;
        face_loc[e] = DAFACE_INTERIOR;
        e++;
      }
      
      ei = mx - 1;
      elidx = ei + ej * mx;
      edge2element[2*e+0] = elidx;
      edge2element[2*e+1] = -1;
      if (has_east_face) {
        face_type[e] = DACELL_FACE_E;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        face_type[e] = DACELL_FACE_E;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
    }
  }
  
  
  /* y-normal face sweep */
  {
    for (ei=0; ei<mx; ei++) {
      ej = 0;
      elidx = ei + ej * mx;
      if (has_south_face) {
        edge2element[2*e+0] = elidx;
        edge2element[2*e+1] = -1;
        face_type[e] = DACELL_FACE_S;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        edge2element[2*e+0] = E_MINUS_OFF_RANK;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_S;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
      
      for (ej=1; ej<my; ej++) {
        elidx = ei + (ej-1) * mx;
        edge2element[2*e+0] = elidx;
        elidx = ei + ej * mx;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_N;
        face_loc[e] = DAFACE_INTERIOR;
        e++;
      }
      
      ej = my - 1;
      elidx = ei + ej * mx;
      edge2element[2*e+0] = elidx;
      edge2element[2*e+1] = -1;
      if (has_north_face) {
        face_type[e] = DACELL_FACE_N;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        face_type[e] = DACELL_FACE_N;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
    }
  }
  
  if (e != nedges) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Number of faces is incorrect");
  for (e=0; e<nedges; e++) {
    if (((PetscInt)face_type[e] >= DACELL2D_NFACES) || ((PetscInt)face_type[e] < 0)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Face %D contains out-of-range face type value %D",e,(PetscInt)face_type[e]);
  }
  
#ifdef FVDA_DEBUG
  for (e=0; e<nedges; e++) {
    printf("[rank %d] face %4d : type %d : (cell-,cell+) %+4d %+4d\n",(int)rank,(int)e,(int)face_type[e],(int)edge2element[2*e],(int)edge2element[2*e+1]);
  }
#endif
  
  *ne  = nedges;
  *e2e = edge2element;
  *ft  = face_type;
  *fl  = face_loc;
  
  PetscFunctionReturn(0);
}

static PetscErrorCode private_FVDACreateFaceLabels3d(DM dm,PetscInt *ne,PetscInt *e2e[],DACellFace *ft[],DACellFaceLocation *fl[])
{
  PetscInt           *edge2element; /* [2 * nedges]: edge2element[0] -> element[low] */
  DACellFace         *face_type; /* [nedges]: stores cell face of element[low] */
  DACellFaceLocation *face_loc; /* [nedges] */
  PetscInt           mx,my,mz,ei,ej,ek,elidx;
  PetscInt           e,nedges;
  PetscInt           sv[3],ev[3],nv[3],Nv[3];
  PetscBool          has_west_face,has_south_face,has_back_face,has_east_face,has_north_face,has_front_face;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;
  
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dm,&sv[0],&sv[1],&sv[2],&nv[0],&nv[1],&nv[2]);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,&Nv[0],&Nv[1],&Nv[2],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  has_west_face = PETSC_FALSE;
  has_south_face = PETSC_FALSE;
  has_back_face = PETSC_FALSE;
  has_east_face = PETSC_FALSE;
  has_north_face = PETSC_FALSE;
  has_front_face = PETSC_FALSE;
  ev[0] = sv[0] + nv[0];
  ev[1] = sv[1] + nv[1];
  ev[2] = sv[2] + nv[2];
  if (sv[0] == 0) has_west_face = PETSC_TRUE;
  if (sv[1] == 0) has_south_face = PETSC_TRUE;
  if (sv[2] == 0) has_back_face = PETSC_TRUE;
  
  if (ev[0] == Nv[0]) has_east_face = PETSC_TRUE;
  if (ev[1] == Nv[1]) has_north_face = PETSC_TRUE;
  if (ev[2] == Nv[2]) has_front_face = PETSC_TRUE;
  
#ifdef FVDA_DEBUG
  PetscPrintf(PETSC_COMM_SELF,"[rank %d] Boundary faces on sub-domain: w=%d s=%d b=%d , e=%d n=%d f=%d\n",(int)rank,(int)has_west_face,(int)has_south_face,(int)has_back_face,(int)has_east_face,(int)has_north_face,(int)has_front_face);
#endif
  
  ierr = DMDAGetElementsSizes(dm,&mx,&my,&mz);CHKERRQ(ierr);
  /* number of edges */
  nedges  = (mx + 1) * my * mz;
  nedges += mx * (my + 1) * mz;
  nedges += mx * my * (mz + 1);
  
  ierr = PetscCalloc1(nedges*2,&edge2element);CHKERRQ(ierr);
  ierr = PetscCalloc1(nedges,&face_type);CHKERRQ(ierr);
  ierr = PetscCalloc1(nedges,&face_loc);CHKERRQ(ierr);
  //printf("<mem> face_element_map %1.2e (MB)\n",sizeof(PetscInt)*nedges*2 * 1.0e-6);
  //printf("<mem> face_type        %1.2e (MB)\n",sizeof(DACellFace)*nedges * 1.0e-6);
  //printf("<mem> face_loc         %1.2e (MB)\n",sizeof(DACellFaceLocation)*nedges * 1.0e-6);

  e = 0; /* edge counter initialize */
  
  /* x-normal face sweep */
  for (ek=0; ek<mz; ek++) {
    for (ej=0; ej<my; ej++) {
      ei = 0;
      elidx = ei + ej * mx + ek * mx * my;
      if (has_west_face) {
        edge2element[2*e+0] = elidx;
        edge2element[2*e+1] = -1;
        face_type[e] = DACELL_FACE_W;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        edge2element[2*e+0] = E_MINUS_OFF_RANK;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_W;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
      
      for (ei=1; ei<mx; ei++) {
        elidx = (ei-1) + ej * mx + ek * mx * my;
        edge2element[2*e+0] = elidx;
        elidx = ei + ej * mx + ek * mx * my;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_E;
        face_loc[e] = DAFACE_INTERIOR;
        e++;
      }
      
      ei = mx - 1;
      elidx = ei + ej * mx + ek * mx * my;
      edge2element[2*e+0] = elidx;
      edge2element[2*e+1] = -1;
      if (has_east_face) {
        face_type[e] = DACELL_FACE_E;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        face_type[e] = DACELL_FACE_E;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
    }
  }
  
  
  /* y-normal face sweep */
  for (ek=0; ek<mz; ek++) {
    for (ei=0; ei<mx; ei++) {
      ej = 0;
      elidx = ei + ej * mx + ek * mx * my;
      if (has_south_face) {
        edge2element[2*e+0] = elidx;
        edge2element[2*e+1] = -1;
        face_type[e] = DACELL_FACE_S;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        edge2element[2*e+0] = E_MINUS_OFF_RANK;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_S;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
      
      for (ej=1; ej<my; ej++) {
        elidx = ei + (ej-1) * mx + ek * mx * my;
        edge2element[2*e+0] = elidx;
        elidx = ei + ej * mx + ek * mx * my;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_N;
        face_loc[e] = DAFACE_INTERIOR;
        e++;
      }
      
      ej = my - 1;
      elidx = ei + ej * mx + ek * mx * my;
      edge2element[2*e+0] = elidx;
      edge2element[2*e+1] = -1;
      if (has_north_face) {
        face_type[e] = DACELL_FACE_N;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        face_type[e] = DACELL_FACE_N;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
    }
  }
  
  
  /* z-normal face sweep */
  for (ej=0; ej<my; ej++) {
    for (ei=0; ei<mx; ei++) {
      ek = 0;
      elidx = ei + ej * mx + ek * mx * my;
      if (has_back_face) {
        edge2element[2*e+0] = elidx;
        edge2element[2*e+1] = -1;
        face_type[e] = DACELL_FACE_B;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        edge2element[2*e+0] = E_MINUS_OFF_RANK; /* off-rank */
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_B;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
      
      for (ek=1; ek<mz; ek++) {
        elidx = ei + ej * mx + (ek-1) * mx * my;
        edge2element[2*e+0] = elidx;
        elidx = ei + ej * mx + ek * mx * my;
        edge2element[2*e+1] = elidx;
        face_type[e] = DACELL_FACE_F;
        face_loc[e] = DAFACE_INTERIOR;
        e++;
      }
      
      ek = mz - 1;
      elidx = ei + ej * mx + ek * mx * my;
      edge2element[2*e+0] = elidx;
      edge2element[2*e+1] = -1;
      if (has_front_face) {
        face_type[e] = DACELL_FACE_F;
        face_loc[e]  = DAFACE_BOUNDARY;
      } else {
        face_type[e] = DACELL_FACE_F;
        face_loc[e]  = DAFACE_SUB_DOMAIN_BOUNDARY;
      }
      e++;
    }
  }
  
  if (e != nedges) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Number of faces is incorrect");
  for (e=0; e<nedges; e++) {
    if (((PetscInt)face_type[e] >= DACELL3D_NFACES) || ((PetscInt)face_type[e] < 0)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Face %D contains out-of-range face type value %D",e,(PetscInt)face_type[e]);
  }

#ifdef FVDA_DEBUG
  for (e=0; e<nedges; e++) {
    printf("[rank %d] face %4d : type %d : (cell-,cell+) %+4d %+4d\n",(int)rank,(int)e,(int)face_type[e],(int)edge2element[2*e],(int)edge2element[2*e+1]);
  }
#endif
  
  *ne  = nedges;
  *e2e = edge2element;
  *ft  = face_type;
  *fl  = face_loc;
  
  PetscFunctionReturn(0);
}

static PetscErrorCode private_FVDACreateFaceLabels(DM dm,PetscInt *ne,PetscInt *e2e[],DACellFace *ft[],DACellFaceLocation *fl[])
{
  PetscErrorCode ierr;
  PetscInt       dim;
  
  
  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 1:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"1D not supported");
      break;
    case 2:
      ierr = private_FVDACreateFaceLabels2d(dm,ne,e2e,ft,fl);CHKERRQ(ierr);
      break;
    case 3:
      ierr = private_FVDACreateFaceLabels3d(dm,ne,e2e,ft,fl);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unknown dimension");
      break;
  }
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode private_FVDACreateElementFaceLabels3d(FVDA fv)
{
  PetscInt *ncnt,*e2f;
  PetscErrorCode ierr;
  PetscInt f,c;
  
  ierr = PetscCalloc1(fv->ncells*DACELL3D_NFACES,&e2f);CHKERRQ(ierr);
  ierr = PetscCalloc1(fv->ncells,&ncnt);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    for (f=0; f<DACELL3D_NFACES; f++) {
      e2f[DACELL3D_NFACES*c + f] = -1;
    }
  }
  for (f=0; f<fv->nfaces; f++) {
    PetscInt eminus,eplus;
    
    eminus = fv->face_element_map[2*f+0];
    eplus  = fv->face_element_map[2*f+1];
    if (eminus != E_MINUS_OFF_RANK) {
      DACellFace faceid = fv->face_type[f];
      //e2f[DACELL3D_NFACES*eminus + ncnt[eminus]] = f;
      e2f[DACELL3D_NFACES*eminus + (PetscInt)faceid] = f;
      ncnt[eminus]++;
    }
    if (eplus != -1) {
      DACellFace faceid = fv->face_type[f];
      //e2f[DACELL3D_NFACES*eplus + ncnt[eplus]] = f;
      e2f[DACELL3D_NFACES*eplus + (PetscInt)faceid] = f;
      ncnt[eplus]++;
    }
    if (ncnt[eminus] > DACELL3D_NFACES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"eminus: face count > 6");
    if (ncnt[eplus] > DACELL3D_NFACES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"eplus: face count > 6");
  }
  
  for (c=0; c<fv->ncells; c++) {
    PetscBool throwerror = PETSC_FALSE;
    if (ncnt[c] != DACELL3D_NFACES) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"face count = %D for cell %D. Expected face count = 6",ncnt[c],c);
    for (f=0; f<DACELL3D_NFACES; f++) {
      if (e2f[DACELL3D_NFACES*c + f] == -1) throwerror = PETSC_TRUE;
      if (e2f[DACELL3D_NFACES*c + f] > fv->nfaces) throwerror = PETSC_TRUE;
    }
    if (throwerror) {
      PetscPrintf(PETSC_COMM_SELF,"Uninitialized or out-of-range-data detected. max cells %D , max faces %D\n",fv->ncells,fv->nfaces);
      for (f=0; f<DACELL3D_NFACES; f++) {
        if (e2f[DACELL3D_NFACES*c + f] == -1)    PetscPrintf(PETSC_COMM_SELF,"Uninitialized data detected. cell %D: face_index[%D]=%D\n",c,f,e2f[DACELL3D_NFACES*c + f]);
        if (e2f[DACELL3D_NFACES*c + f] > fv->nfaces) PetscPrintf(PETSC_COMM_SELF,"Out-of-range-data detected. cell %D: face_index[%D]=%D\n",c,f,e2f[DACELL3D_NFACES*c + f]);
      }
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"cell %D contains uninitialized or out-of-range-data face index",c);
    }
  }
  PetscFunctionReturn(0);
}
*/
 
static PetscErrorCode private_FVDASetUpLocalFVIndices2d(FVDA fv,DM dmfv)
{
  PetscMPIInt    rank;
  PetscInt       s[2],m[2],gs[2],gm[2],me[2],f,max_local_size;
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dmfv,&s[0],&s[1],NULL,&m[0],&m[1],NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dmfv,&gs[0],&gs[1],NULL,&gm[0],&gm[1],NULL);CHKERRQ(ierr);
  max_local_size = gm[0] * gm[1];
  ierr = DMDAGetElementsSizes(fv->dm_geometry,&me[0],&me[1],NULL);CHKERRQ(ierr);
  
  ierr = PetscCalloc1(2*fv->nfaces,&fv->face_fv_map);CHKERRQ(ierr);
  for (f=0; f<2*fv->nfaces; f++) {
    fv->face_fv_map[f] = -1;
  }
  
  /* Loop over fv->face_element_map */
  /* At least one of eminus, eplus must live on the current rank */
  /* Convert sub-domain rank local element index from fv->face_element_map[] to the global space */
  /* Define the neighbour based on the face_type[] value */
  for (f=0; f<fv->nfaces; f++) {
    PetscInt eminus=-1,eplus=-1,eijk[2],sub_domain_rank_local[2],local[2],lid_m,lid_p;
    
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    eminus = fv->face_element_map[2*f+0];
    eplus  = fv->face_element_map[2*f+1];
    if (eminus >= 0) {
      PetscInt shift[] = {0,0};
      
      ierr = cart_convert_2d(eminus,me,eijk);CHKERRQ(ierr);
      
      sub_domain_rank_local[0] = eijk[0] + s[0];
      sub_domain_rank_local[1] = eijk[1] + s[1];
      
      local[0] = sub_domain_rank_local[0] - gs[0];
      local[1] = sub_domain_rank_local[1] - gs[1];
      
      lid_m = (local[0]) + (local[1])*gm[0];
      
      switch (fv->face_type[f]) {
        case DACELL_FACE_E:
          shift[0] =  1;
          break;
        case DACELL_FACE_W:
          shift[0] = -1;
          break;
          
        case DACELL_FACE_N:
          shift[0] =  1;
          break;
        case DACELL_FACE_S:
          shift[0] = -1;
          break;
          
        case DACELL_FACE_F:
          break;
        case DACELL_FACE_B:
          break;
        default:
          break;
      }
      
      lid_p = (local[0] + shift[0]) + (local[1] + shift[1])*gm[0];
      
      fv->face_fv_map[2*f+0] = lid_m;
      fv->face_fv_map[2*f+1] = lid_p;
    } else if (eplus >= 0) {
      PetscInt shift[] = {0,0};
      
      ierr = cart_convert_2d(eplus,me,eijk);CHKERRQ(ierr);
      
      sub_domain_rank_local[0] = eijk[0] + s[0];
      sub_domain_rank_local[1] = eijk[1] + s[1];
      
      local[0] = sub_domain_rank_local[0] - gs[0];
      local[1] = sub_domain_rank_local[1] - gs[1];
      
      lid_p = (local[0]) + (local[1])*gm[0];
      
      switch (fv->face_type[f]) {
        case DACELL_FACE_E:
          shift[0] =  1;
          break;
        case DACELL_FACE_W:
          shift[0] = -1;
          break;
          
        case DACELL_FACE_N:
          shift[0] =  1;
          break;
        case DACELL_FACE_S:
          shift[0] = -1;
          break;
          
        case DACELL_FACE_F:
          break;
        case DACELL_FACE_B:
          break;
        default:
          break;
      }
      
      lid_m = (local[0] + shift[0]) + (local[1] + shift[1])*gm[0];
      
      fv->face_fv_map[2*f+0] = lid_m;
      fv->face_fv_map[2*f+1] = lid_p;
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Both cell- and cell+ are negative. Failed to convert element index to local fv cell index");
    }
  }
  
  for (f=0; f<fv->nfaces; f++) {
    PetscInt eminus=-1,eijk[2],sub_domain_rank_local[2],local[2],lid_m,lid_p;
    
    if (fv->face_location[f] != DAFACE_BOUNDARY) continue;
    
    eminus = fv->face_element_map[2*f+0];
    if (eminus >= 0) {
      PetscInt shift[] = {0,0};
      
      ierr = cart_convert_2d(eminus,me,eijk);CHKERRQ(ierr);
      
      sub_domain_rank_local[0] = eijk[0] + s[0];
      sub_domain_rank_local[1] = eijk[1] + s[1];
      
      local[0] = sub_domain_rank_local[0] - gs[0];
      local[1] = sub_domain_rank_local[1] - gs[1];
      
      lid_m = (local[0]) + (local[1])*gm[0];
      
      switch (fv->face_type[f]) {
        case DACELL_FACE_E:
          shift[0] =  1;
          break;
        case DACELL_FACE_W:
          shift[0] = -1;
          break;
          
        case DACELL_FACE_N:
          shift[1] =  1;
          break;
        case DACELL_FACE_S:
          shift[1] = -1;
          break;
          
        case DACELL_FACE_F:
          break;
        case DACELL_FACE_B:
          break;
        default:
          break;
      }
      
      lid_p = (local[0] + shift[0]) + (local[1] + shift[1])*gm[0];
      
      fv->face_fv_map[2*f+0] = lid_m;
      fv->face_fv_map[2*f+1] = -1;
    }
  }
  
  /* check all entries of face_fv_map[] have been initialized */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] != DAFACE_BOUNDARY) {
      if ((fv->face_fv_map[2*f+0] <= -1) || (fv->face_fv_map[2*f+1] <= -1)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"interior face[%D]->fv_local_space[%D,%D] invalid (un-initialized)",f,fv->face_fv_map[2*f+0],fv->face_fv_map[2*f+1]);
    } else {
      if (fv->face_fv_map[2*f+0] <= -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"boundary face[%D]->fv_local_space[%D,***] invalid (un-initialized)",f,fv->face_fv_map[2*f+0]);
    }
  }
  
  /* check all entries of face_fv_map[] are valid */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] != DAFACE_BOUNDARY) {
      if ((fv->face_fv_map[2*f+0] >= max_local_size) || (fv->face_fv_map[2*f+1] >= max_local_size)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"interior face[%D]->fv_local_space[%D,%D] invalid (out-of-bounds)",f,fv->face_fv_map[2*f+0],fv->face_fv_map[2*f+1]);
    } else {
      if (fv->face_fv_map[2*f+0] <= -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"boundary face[%D]->fv_local_space[%D,***] invalid (un-initialized)",f,fv->face_fv_map[2*f+0]);
    }
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode private_FVDASetUpLocalFVIndices3d(FVDA fv,DM dmfv)
{
  PetscMPIInt    rank;
  PetscInt       s[3],m[3],gs[3],gm[3],me[3],f,max_local_size;
  PetscErrorCode ierr;
  
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dmfv,&s[0],&s[1],&s[2],&m[0],&m[1],&m[2]);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dmfv,&gs[0],&gs[1],&gs[2],&gm[0],&gm[1],&gm[2]);CHKERRQ(ierr);
  max_local_size = gm[0] * gm[1] * gm[2];
  ierr = DMDAGetElementsSizes(fv->dm_geometry,&me[0],&me[1],&me[2]);CHKERRQ(ierr);

  ierr = PetscCalloc1(2*fv->nfaces,&fv->face_fv_map);CHKERRQ(ierr);
  //printf("<mem> face_fv_map      %1.2e (MB)\n",sizeof(PetscInt)*2*fv->nfaces * 1.0e-6);
  for (f=0; f<2*fv->nfaces; f++) {
    fv->face_fv_map[f] = -1;
  }
  
  /* Loop over fv->face_element_map */
  /* At least one of eminus, eplus must live on the current rank */
  /* Convert sub-domain rank local element index from fv->face_element_map[] to the global space */
  /* Define the neighbour based on the face_type[] value */
  for (f=0; f<fv->nfaces; f++) {
    PetscInt eminus=-1,eplus=-1,eijk[3],sub_domain_rank_local[3],local[3],lid_m,lid_p;
    
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    eminus = fv->face_element_map[2*f+0];
    eplus  = fv->face_element_map[2*f+1];
    if (eminus >= 0) {
      PetscInt shift[] = {0,0,0};
      
      ierr = cart_convert_3d(eminus,me,eijk);CHKERRQ(ierr);
      
      sub_domain_rank_local[0] = eijk[0] + s[0];
      sub_domain_rank_local[1] = eijk[1] + s[1];
      sub_domain_rank_local[2] = eijk[2] + s[2];
      
      local[0] = sub_domain_rank_local[0] - gs[0];
      local[1] = sub_domain_rank_local[1] - gs[1];
      local[2] = sub_domain_rank_local[2] - gs[2];
      
      lid_m = (local[0]) + (local[1])*gm[0] + (local[2])*gm[0]*gm[1];
      
      switch (fv->face_type[f]) {
        case DACELL_FACE_E:
          shift[0] =  1;
          break;
        case DACELL_FACE_W:
          shift[0] = -1;
          break;
          
        case DACELL_FACE_N:
          shift[1] =  1;
          break;
        case DACELL_FACE_S:
          shift[1] = -1;
          break;
          
        case DACELL_FACE_F:
          shift[2] =  1;
          break;
        case DACELL_FACE_B:
          shift[2] = -1;
          break;
          
        default:
          break;
      }
      
      lid_p = (local[0] + shift[0]) + (local[1] + shift[1])*gm[0] + (local[2] + shift[2])*gm[0]*gm[1];
      
      fv->face_fv_map[2*f+0] = lid_m;
      fv->face_fv_map[2*f+1] = lid_p;
    } else if (eplus >= 0) {
      PetscInt shift[] = {0,0,0};
      
      ierr = cart_convert_3d(eplus,me,eijk);CHKERRQ(ierr);
      
      sub_domain_rank_local[0] = eijk[0] + s[0];
      sub_domain_rank_local[1] = eijk[1] + s[1];
      sub_domain_rank_local[2] = eijk[2] + s[2];
      
      local[0] = sub_domain_rank_local[0] - gs[0];
      local[1] = sub_domain_rank_local[1] - gs[1];
      local[2] = sub_domain_rank_local[2] - gs[2];
      
      lid_p = (local[0]) + (local[1])*gm[0] + (local[2])*gm[0]*gm[1];
      
      switch (fv->face_type[f]) {
        case DACELL_FACE_E:
          shift[0] =  1;
          break;
        case DACELL_FACE_W:
          shift[0] = -1;
          break;
          
        case DACELL_FACE_N:
          shift[1] =  1;
          break;
        case DACELL_FACE_S:
          shift[1] = -1;
          break;
          
        case DACELL_FACE_F:
          shift[2] =  1;
          break;
        case DACELL_FACE_B:
          shift[2] = -1;
          break;
          
        default:
          break;
      }

      lid_m = (local[0] + shift[0]) + (local[1] + shift[1])*gm[0] + (local[2] + shift[2])*gm[0]*gm[1];
      
      fv->face_fv_map[2*f+0] = lid_m;
      fv->face_fv_map[2*f+1] = lid_p;
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Both cell- and cell+ are negative. Failed to convert element index to local fv cell index");
    }
  }

  for (f=0; f<fv->nfaces; f++) {
    PetscInt eminus=-1,eijk[3],sub_domain_rank_local[3],local[3],lid_m,lid_p;
    
    if (fv->face_location[f] != DAFACE_BOUNDARY) continue;
    
    eminus = fv->face_element_map[2*f+0];
    if (eminus >= 0) {
      PetscInt shift[] = {0,0,0};
      
      ierr = cart_convert_3d(eminus,me,eijk);CHKERRQ(ierr);
      
      sub_domain_rank_local[0] = eijk[0] + s[0];
      sub_domain_rank_local[1] = eijk[1] + s[1];
      sub_domain_rank_local[2] = eijk[2] + s[2];
      
      local[0] = sub_domain_rank_local[0] - gs[0];
      local[1] = sub_domain_rank_local[1] - gs[1];
      local[2] = sub_domain_rank_local[2] - gs[2];
      
      lid_m = (local[0]) + (local[1])*gm[0] + (local[2])*gm[0]*gm[1];
      
      switch (fv->face_type[f]) {
        case DACELL_FACE_E:
          shift[0] =  1;
          break;
        case DACELL_FACE_W:
          shift[0] = -1;
          break;
          
        case DACELL_FACE_N:
          shift[1] =  1;
          break;
        case DACELL_FACE_S:
          shift[1] = -1;
          break;
          
        case DACELL_FACE_F:
          shift[2] =  1;
          break;
        case DACELL_FACE_B:
          shift[2] = -1;
          break;
          
        default:
          break;
      }
      
      lid_p = (local[0] + shift[0]) + (local[1] + shift[1])*gm[0] + (local[2] + shift[2])*gm[0]*gm[1];
      
      fv->face_fv_map[2*f+0] = lid_m;
      fv->face_fv_map[2*f+1] = -1;
    }
  }

  /* check all entries of face_fv_map[] have been initialized */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] != DAFACE_BOUNDARY) {
      if ((fv->face_fv_map[2*f+0] <= -1) || (fv->face_fv_map[2*f+1] <= -1)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"interior face[%D]->fv_local_space[%D,%D] invalid (un-initialized)",f,fv->face_fv_map[2*f+0],fv->face_fv_map[2*f+1]);
    } else {
      if (fv->face_fv_map[2*f+0] <= -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"boundary face[%D]->fv_local_space[%D,***] invalid (un-initialized)",f,fv->face_fv_map[2*f+0]);
    }
  }
  
  /* check all entries of face_fv_map[] are valid */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] != DAFACE_BOUNDARY) {
      if ((fv->face_fv_map[2*f+0] >= max_local_size) || (fv->face_fv_map[2*f+1] >= max_local_size)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"interior face[%D]->fv_local_space[%D,%D] invalid (out-of-bounds)",f,fv->face_fv_map[2*f+0],fv->face_fv_map[2*f+1]);
    } else {
      if (fv->face_fv_map[2*f+0] <= -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"boundary face[%D]->fv_local_space[%D,***] invalid (un-initialized)",f,fv->face_fv_map[2*f+0]);
    }
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode private_FVDASetUpBoundaryLabels(FVDA fv)
{
  PetscInt       f,nfaces_interior,nfaces_interior_subdomain_boundary,cnt[]={0,0};
  PetscErrorCode ierr;

  
  PetscFunctionBegin;
  fv->nfaces_interior = 0;
  fv->nfaces_boundary = 0;
  nfaces_interior = 0;
  nfaces_interior_subdomain_boundary = 0;
  for (f=0; f<fv->nfaces; f++) {
    switch (fv->face_location[f]) {
      case DAFACE_INTERIOR:
        nfaces_interior++;
        break;
        
      case DAFACE_SUB_DOMAIN_BOUNDARY:
        nfaces_interior_subdomain_boundary++;
        break;
        
      case DAFACE_BOUNDARY:
        fv->nfaces_boundary++;
        break;
    }
  }
  if (fv->nfaces_boundary + nfaces_interior + nfaces_interior_subdomain_boundary != fv->nfaces) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sum of faces is inconsistent");
  fv->nfaces_interior = nfaces_interior_subdomain_boundary + nfaces_interior;
  
  ierr = PetscCalloc1(fv->nfaces_interior,&fv->face_idx_interior);CHKERRQ(ierr);
  ierr = PetscCalloc1(fv->nfaces_boundary,&fv->face_idx_boundary);CHKERRQ(ierr);
  //printf("<mem> face_id_inter.   %1.2e (MB)\n",sizeof(PetscInt)*fv->nfaces_interior * 1.0e-6);
  //printf("<mem> face_id_bound.   %1.2e (MB)\n",sizeof(PetscInt)*fv->nfaces_boundary * 1.0e-6);

  for (f=0; f<fv->nfaces; f++) {
    switch (fv->face_location[f]) {
      case DAFACE_INTERIOR:
        fv->face_idx_interior[ cnt[0] ] = f;
        cnt[0]++;
        break;
        
      case DAFACE_SUB_DOMAIN_BOUNDARY:
        fv->face_idx_interior[ cnt[0] ] = f;
        cnt[0]++;
        break;
        
      case DAFACE_BOUNDARY:
        break;
    }
  }

  /* order the exterior boundary facets in a special manner (domain boundary wise) */
  cnt[1] = 0;
  if (fv->dim == 2) {
    const DACellFace flist[] = { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_N };
    PetscInt         l;
    
    fv->boundary_ranges[0] = 0;
    
    for (l=0; l<sizeof(flist)/sizeof(DACellFace); l++) {
      for (f=0; f<fv->nfaces; f++) {
        if (fv->face_location[f] != DAFACE_BOUNDARY) continue;
        
        if (fv->face_type[f] == flist[l]) {
          fv->face_idx_boundary[ cnt[1] ] = f;
          cnt[1]++;
        }
      }
      
      fv->boundary_ranges[l+1] = cnt[1];
    }
  }
  
  if (fv->dim == 3) {
    //typedef enum { DACELL_FACE_W=0, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_N, DACELL_FACE_B, DACELL_FACE_F } DACellFace;
    const DACellFace flist[] = { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_S, DACELL_FACE_N, DACELL_FACE_B, DACELL_FACE_F };
    PetscInt         l;
    
    fv->boundary_ranges[0] = 0;
    
    for (l=0; l<sizeof(flist)/sizeof(DACellFace); l++) {
      for (f=0; f<fv->nfaces; f++) {
        if (fv->face_location[f] != DAFACE_BOUNDARY) continue;

        if (fv->face_type[f] == flist[l]) {
          fv->face_idx_boundary[ cnt[1] ] = f;
          cnt[1]++;
        }
      }

      fv->boundary_ranges[l+1] = cnt[1];
    }
  }
  
#ifdef FVDA_DEBUG
  {
    PetscInt    l,ncell_faces;
    PetscMPIInt rank;
    
    ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
    if (fv->dim == 2) ncell_faces = DACELL2D_NFACES;
    if (fv->dim == 3) ncell_faces = DACELL3D_NFACES;

    for (l=0; l<ncell_faces; l++) {
      PetscPrintf(PETSC_COMM_WORLD,"[rank %d] boundary face %d: range[ %4d , %4d )\n",(int)rank,(int)l,(int)fv->boundary_ranges[l],(int)fv->boundary_ranges[l+1]);
    }
  }
#endif

  
  for (f=0; f<fv->nfaces_boundary; f++) {
    PetscInt fid = fv->face_idx_boundary[f];
    PetscInt cell_minus = fv->face_element_map[2*fid + 0];
    if (cell_minus < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Boundary facet [%D -> %D] is missing cell-. A boundary facet should always have a cell- living on the sub-domain",f,fid);
  }

#ifdef FVDA_DEBUG
  {
    PetscInt    l,ncell_faces;
    PetscMPIInt rank;
    
    ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
    if (fv->dim == 2) ncell_faces = DACELL2D_NFACES;
    if (fv->dim == 3) ncell_faces = DACELL3D_NFACES;
    /*
    for (f=0; f<fv->nfaces_boundary; f++) {
      printf("boundary face %4d -> fidx %4d\n",f,fv->face_idx_boundary[f]);
    }
    */
    for (l=0; l<ncell_faces; l++) {
      PetscInt s,e;
      s = fv->boundary_ranges[l];
      e = fv->boundary_ranges[l+1];
      printf("[rank %d] ----- boundary %.2d ----- \n",(int)rank,(int)l);
      for (f=s; f<e; f++) {
        printf("[rank %d]  boundary face %4d -> fidx %4d --> cell- %4d \n",(int)rank,(int)f,(int)fv->face_idx_boundary[f],(int)fv->face_element_map[2*fv->face_idx_boundary[f]+0]);
      }
    }
  }
#endif
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetValidElement(FVDA fv,PetscInt faceid,PetscInt *cellid)
{
  PetscInt eminus;
  
  PetscFunctionBegin;
  eminus = fv->face_element_map[2*faceid+0];
  switch (eminus) {
    case E_MINUS_OFF_RANK:
      *cellid = fv->face_element_map[2*faceid+1];
      break;
    default:
      *cellid = eminus;
      break;
  }
  if (*cellid < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Negative cell index returned");
  PetscFunctionReturn(0);
}

/*
 Evaluates div(F), where F = \vec u X
 
 Uses divergence theorem, hence we evaluate
 \int_V \div(\vec u X) = \int_S (\vec u X) \dot n
                       ~ (\vec u X) \dot n dS
*/
PetscErrorCode eval_F_upwind_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb;
  const PetscReal *vdotn;
  PetscReal       v_n,X_m,X_p,flux;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];

  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);

  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"v.n",&vdotn);CHKERRQ(ierr);

  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];

#ifdef FVDA_DEBUG
    printf("interior f %d: c-/+ %d %d\n",f,c_m,c_p);
#endif
#ifdef FVDA_DEBUG
    printf("interior f %d: n %+1.4e %+1.4e %+1.4e\n",f,fv->face_normal[3*f+0],fv->face_normal[3*f+1],fv->face_normal[3*f+2]);
    printf("interior f %d: v.n %+1.4e\n",f,v_n);
#endif


    /*
     if (v_n > 0.0) { // out
     F[c_m] += (v_n * X_m) * dS; // cell[-]
     F[c_p] -= (v_n * X_m) * dS; // cell[+]
     } else {
     F[c_m] += (v_n * X_p) * dS; // cell[-]
     F[c_p] -= (v_n * X_p) * dS; // cell[+]
     }
     */

    flux = 0.5 * ( v_n * (X_p + X_m) - PetscAbsReal(v_n) * (X_p - X_m ) );

    F[c_m] += flux * dS; // cell[-]
    F[c_p] -= flux * dS; // cell[+]
    
  }

  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    
    if (fv->face_location[f] != DAFACE_BOUNDARY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"YOU SHOULD NEVER BE IN LOOP IF YOU ARE NOT A BOUNDARY");
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    v_n = vdotn[f];

    c_m = fv->face_fv_map[2*f+0];
    if (c_m < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"negative c- value detected");
    
    X_m = X[c_m];

    flux = v_n * X_m;

    //printf("face %d (<= fb %d) bc v.n %+1.4e\n",f,fb,v_n);
    
    if (v_n > 0.0) { /* outflow */
      F[c_m] += flux * dS; // cell[-]
    } else { /* inflow */
      
      switch (bctype) {
        
        case FVFLUX_DIRICHLET_CONSTRAINT:
        {
          PetscReal g_D = bcvalue;
          X_p = 2.0 * g_D - X_m;
        }
          break;
        
        case FVFLUX_NEUMANN_CONSTRAINT:
        {
          /* What to do with non-zero flux?? */
          PetscReal g_N = bcvalue;
          X_p = (0.0) * g_N + X_m;
        }
          break;
          
        default:
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
          break;
      }
      
      flux = v_n * X_p;
      F[c_m] += flux * dS;
      
    }
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_central_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb;
  const PetscReal *vdotn;
  PetscReal       v_n,X_m,X_p,flux;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"v.n",&vdotn);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    flux = 0.5 * v_n * ( X_p + X_m );
    
    F[c_m] += flux * dS; // cell[-]
    F[c_p] -= flux * dS; // cell[+]
  }
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    
    if (fv->face_location[f] != DAFACE_BOUNDARY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"YOU SHOULD NEVER BE IN LOOP IF YOU ARE NOT A BOUNDARY");
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    if (c_m < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"negative c- value detected");
    
    X_m = X[c_m];
    
    flux = v_n * X_m;
    
    if (v_n > 0.0) { /* outflow */
      /* 
         Notes: 
         - Currently I am using an upwind flux. Does this makes sense for a central flux??
         - Maybe for the central flux, the "do nothing" condition might simply be to enforce grad(Q).n = 0.
           That is, set Q+ (outside the domain) = Q-.
           Then the central flux would be
           Q* = 1/2(Q+ + Q-) = 0
      */
      F[c_m] += flux * dS; // cell[-]
    } else { /* inflow */
      PetscReal X_avg;
      
      switch (bctype) {
          
        case FVFLUX_DIRICHLET_CONSTRAINT:
        { /* Weak imposition of Dirichlet */
          PetscReal g_D = bcvalue;
          
          X_p = 2.0 * g_D - X_m;
        }
          break;
          
        case FVFLUX_NEUMANN_CONSTRAINT:
        { /* Weak imposition of Neumann */
          /* What to do with non-zero flux?? */
          PetscReal g_N = bcvalue;
          X_p = (0.0) * g_N + X_m;
        }
          break;
          
        default:
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
          break;
      }
      
      X_avg = 0.5 * (X_p + X_m);
      flux  = v_n * X_avg;
      F[c_m] += flux * dS;
      
    }
  }
  PetscFunctionReturn(0);
}

/*
 
 -div(-k grad(X))
 
 \int_V 1. -div( -k grad(X))
 \int_S ( k grad(X) . n )
 ~ k grad(X) . n dS
 
*/
PetscErrorCode eval_F_diffusion_7point_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb,d;
  const PetscReal *k;
  PetscReal       k_face,X_m,X_p;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  
  
  PetscFunctionBegin;
  dm = fv->dm_fv;

  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);

  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"k",&k);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    PetscReal dl[]={0,0,0};
    PetscReal dsn=0,flux;
    
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);

    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    for (d=0; d<3; d++) {
      dl[d] = fv_coor[3*c_p + d] - fv_coor[3*c_m + d];
      dsn += dl[d] * dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    flux = k_face * (X_p - X_m) / dsn;
    
    F[c_m] += flux * dS; // cell[-]
    F[c_p] -= flux * dS; // cell[+]
  }
  
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    PetscReal  dl[]={0,0,0};
    PetscReal  dsn=0,flux;

    if (fv->face_location[f] != DAFACE_BOUNDARY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"YOU SHOULD NEVER BE IN LOOP IF YOU ARE NOT A BOUNDARY");
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);

    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    k_face = k[f];

    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    for (d=0; d<3; d++) {
      dl[d] = 2.0 * (fv->face_centroid[3*f + d] - fv_coor[3*c_m + d]);
      dsn += dl[d]*dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    
    switch (bctype) {
      case FVFLUX_DIRICHLET_CONSTRAINT:
      { /* Weak imposition of Dirichlet */
        PetscReal g_D = bcvalue;
        
        X_p = 2.0 * g_D - X_m;
      }
        break;
        
      case FVFLUX_NEUMANN_CONSTRAINT:
      { /* Weak imposition of Neumann */
        PetscReal g_N = bcvalue;
        X_p = (dsn/k_face) * g_N + X_m;
      }
        break;

      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
        break;
    }
    flux = k_face * (X_p - X_m) / dsn;
    F[c_m] += flux * dS;
  }
  
  
  
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F(SNES snes,Vec X,Vec F,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;

  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dm = fv->dm_fv;

  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(Fl);CHKERRQ(ierr);
  ierr = VecGetArray(Fl,&_F);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_upwind_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
      {
        PetscInt k,m;
        ierr = VecGetSize(Fl,&m);CHKERRQ(ierr);
        for (k=0; k<m; k++) {
          _F[k] *= -1.0;
        }
      }
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
    }
  }

  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fl,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);

  ierr = DMLocalToGlobal(dm,Fl,INSERT_VALUES,F);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  
#if 0
  {
    PetscInt fb;
    
    ierr = VecGetArrayRead(X,&_X);CHKERRQ(ierr);
    ierr = VecGetArray(F,&_F);CHKERRQ(ierr);
    
    
    for (fb=0; fb<fv->nfaces_boundary; fb++) {
      PetscInt   f = fv->face_idx_boundary[fb];
      FVFluxType bctype;
      PetscReal  bcvalue;
      PetscInt   c_m;
      
      if (fv->face_location[f] != DAFACE_BOUNDARY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"YOU SHOULD NEVER BE IN LOOP IF YOU ARE NOT A BOUNDARY");
      
      bctype = fv->boundary_flux[fb];
      bcvalue = fv->boundary_value[fb];
      
      c_m = fv->face_fv_map[2*f+0];
      
      switch (bctype) {
        /* Strong imposition of Dirichlet */
        case FVFLUX_DIRICHLET_CONSTRAINT:
          _F[c_m] = _X[c_m] - bcvalue;
          break;
        default:
          break;
      }
    }
    ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X,&_X);CHKERRQ(ierr);
  }
#endif
  
  PetscFunctionReturn(0);
}


PetscErrorCode eval_J_upwind_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal X[],Mat J)
{
  const PetscReal scalefactor = -1.0;
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb;
  const PetscReal *vdotn;
  PetscReal       v_n;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"v.n",&vdotn);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    if (fv->face_element_map[2*f+0] == E_MINUS_OFF_RANK) continue;

    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    c_p = fv->face_fv_map[2*f+1];
    //X_m = X[c_m];
    //X_p = X[c_p];
    
    //flux = 0.5 * ( v_n * X_p + v_n * X_m - PetscAbsReal(v_n) * (X_p - X_m ) );
    //F[c_m] += flux * dS; // cell[-]
    //F[c_p] -= flux * dS; // cell[+]
    /*
    if (v_n > 0.0) { // out
      F[c_m] += (v_n * X_m) * dS; // cell[-]
      F[c_p] -= (v_n * X_m) * dS; // cell[+]
    } else {
      F[c_m] += (v_n * X_p) * dS; // cell[-]
      F[c_p] -= (v_n * X_p) * dS; // cell[+]
    }
    */
    if (v_n > 0.0) {
      ierr = MatSetValueLocal(J, c_m, c_m,  v_n * dS * scalefactor, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValueLocal(J, c_p, c_m, -v_n * dS * scalefactor, ADD_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatSetValueLocal(J, c_m, c_p,  v_n * dS * scalefactor, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValueLocal(J, c_p, c_p, -v_n * dS * scalefactor, ADD_VALUES);CHKERRQ(ierr);
    }

  }

  /* process outflow */
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt f = fv->face_idx_boundary[fb];
    
    v_n = vdotn[f];
    if (v_n <= 0.0) { continue; }
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    c_m = fv->face_fv_map[2*f+0];
    //X_m = X[c_m];
    //flux = v_n * X_m;
    //F[c_m] += flux * dS; // cell[-]
    ierr = MatSetValueLocal(J, c_m, c_m, v_n * dS * scalefactor, ADD_VALUES);CHKERRQ(ierr);
  }
  
  /* process inflow */
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    
    v_n = vdotn[f];
    if (v_n > 0.0) { continue; }

    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype  = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    c_m = fv->face_fv_map[2*f+0];
    //X_m = X[c_m];
    //flux = v_n * X_m;
    
    switch (bctype) {
      /* Weak imposition of Dirichlet */
      case FVFLUX_DIRICHLET_CONSTRAINT:
        //X_p   = 2.0 * g_D - X_m;
        //flux  = v_n * X_p = v_n ( 2 g_D - X_m );
        ierr = MatSetValueLocal(J, c_m, c_m, -1.0 * v_n * dS * scalefactor, ADD_VALUES);CHKERRQ(ierr);
        break;
      case FVFLUX_NEUMANN_CONSTRAINT:
        //X_p   = (dL/kface) * g_N + X_m;
        //flux  = v_n * X_p = v_n ( (dL/kface) * g_N + X_m );
        ierr = MatSetValue(J, c_m, c_m, 1.0 * v_n * dS * scalefactor, ADD_VALUES);CHKERRQ(ierr);
        break;
      default:
        break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode eval_J_diffusion_7point_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],Mat J)
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb,d;
  const PetscReal *k;
  PetscReal       k_face;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  /* test */
  /*
  {
    PetscInt        faceid=0;
    PetscReal       _dV,_dS;
    
    ierr = FVDAGetValidElement(fv,faceid,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,cell_geom_coor,cell_coor);CHKERRQ(ierr);
    
    _EvaluateCellVolume3d(cell_coor,&dV);
    _EvaluateFaceArea3d(fv->face_type[faceid],cell_coor,&_dS);
    printf("cell volume %+1.4e : face area %+1.4e\n",_dV,_dS);
  }
  */
   
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"k",&k);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    PetscReal dl[]={0,0,0};
    PetscReal dsn=0;
    
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    if (fv->face_element_map[2*f+0] == E_MINUS_OFF_RANK) continue;

    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);

    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    c_p = fv->face_fv_map[2*f+1];
    //X_m = X[c_m];
    //X_p = X[c_p];
    
    for (d=0; d<3; d++) {
      dl[d] = fv_coor[3*c_p + d] - fv_coor[3*c_m + d];
      dsn += dl[d] * dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    //flux = k_face * (X_p - X_m) / dsn;
    
    ierr = MatSetValueLocal(J, c_m, c_m, -k_face * dS/dsn, ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValueLocal(J, c_m, c_p,  k_face * dS/dsn, ADD_VALUES);CHKERRQ(ierr);
    
    ierr = MatSetValueLocal(J, c_p, c_m,  k_face * dS/dsn, ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValueLocal(J, c_p, c_p, -k_face * dS/dsn, ADD_VALUES);CHKERRQ(ierr);
  }
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    PetscReal  dl[]={0,0,0};
    PetscReal  dsn=0;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);

    bctype  = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    //X_m = X[c_m];
    
    for (d=0; d<3; d++) {
      dl[d] = 2.0 * (fv->face_centroid[3*f + d] - fv_coor[3*c_m + d]);
      dsn += dl[d]*dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    
    switch (bctype) {
      /* Weak imposition of Dirichlet */
      case FVFLUX_DIRICHLET_CONSTRAINT:
        //X_p   = 2.0 * g_D - X_m;
        //flux = k_face * (X_p - X_m) / dsn = (k_face/dsn) * (2 g_D - X_m - X_m)
        ierr = MatSetValueLocal(J, c_m, c_m, -2.0 * k_face * dS/dsn, ADD_VALUES);CHKERRQ(ierr);
        break;
      case FVFLUX_NEUMANN_CONSTRAINT:
        //X_p   = (dsn/k_face) * g_N + X_m;
        //flux = (k_face/dsn) * (X_p - X_m) = (k_face/dsn) * ((dsn/k_face) * g_N + X_m - X_m) = 0
        break;
      default:
        break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode eval_J(SNES snes,Vec X,Mat Ja,Mat Jb,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,coorl,geometry_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor;
  DM                dm;
  FVDA              fv = NULL;
  
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dm = fv->dm_fv;
  
  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = MatZeroEntries(Jb);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_J_upwind_local(fv,_geom_coor,_X,Jb);CHKERRQ(ierr);
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_J_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,Jb);CHKERRQ(ierr);
    }
  }
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Ja != Jb) {
    ierr = MatAssemblyBegin(Ja,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ja,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
PetscErrorCode bcset_natural(FVDA fv,
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
#endif

PetscReal vant_L(PetscReal a, PetscReal b, PetscReal w)
{
  PetscReal t1,t2;
  t1 = a*a + 2.0*a*b + w;
  t2 = a*a + 2.0*b*b + a*b + w;
  return(t1/t2);
}

PetscReal vant_L_v2(PetscReal y, PetscReal w)
{
  PetscReal t1,t2;
  t1 = y*y + 2.0*y;
  t2 = y*y + y + 2.0;
  return(t1/t2);
}

PetscErrorCode eval_F_upwind_hr_local_2reconstructions(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb;
  const PetscReal *vdotn;
  PetscReal       v_n,X_m,X_p,flux,Xhr_m,Xhr_p,flux_hr;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  PetscInt        n_neigh,neigh[27];
  PetscReal       coeff[3];
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"v.n",&vdotn);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];

    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];

    flux = 0.5 * ( v_n * (X_p + X_m) - PetscAbsReal(v_n) * (X_p - X_m ) );

    ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
    ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
    ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr_m);CHKERRQ(ierr);

    /*
    min = PETSC_MAX_REAL;
    max = PETSC_MIN_REAL;
    for (PetscInt kk=0; kk<n_neigh; kk++) {
      min = PetscMin(min,X[neigh[kk]]);
      max = PetscMax(max,X[neigh[kk]]);
    }

    {
      const PetscInt cid = c_m;
      PetscReal Xhr      = Xhr_m;
      
      PetscReal delta,phi_ij,phi = PETSC_MAX_REAL;
      const PetscReal *r = &fv_coor[3*cid];
      PetscReal rij[3];// = &fv->face_centroid[3*f];
      PetscReal xi2d[]={0,0};
      PetscInt kk;
      
      
      for (kk=0; kk<6; kk++) {
        ierr = _EvaluateFaceCoord3d(kk,cell_coor,xi2d,rij);CHKERRQ(ierr);

        delta =  coeff[0] * (rij[0] - r[0]);
        delta += coeff[1] * (rij[1] - r[1]);
        delta += coeff[2] * (rij[2] - r[2]);

        if (delta > 0) {
          phi_ij = PetscMin(1.0, (max - Xhr)/delta );
          phi_ij = vant_L(max - Xhr,delta,0);
          phi_ij = vant_L_v2((max - Xhr)/delta,0);
        } else if (delta < 0) {
          phi_ij = PetscMin(1.0, (min - Xhr)/delta );
          phi_ij = vant_L(min - Xhr,delta,0);
          phi_ij = vant_L_v2((min - Xhr)/delta,0);
        } else {
          phi_ij = 1.0;
        }
        phi = PetscMin(phi,phi_ij);
      }
      Xhr_m =   coeff[0] * (fv->face_centroid[3*f+0] - r[0])
              + coeff[1] * (fv->face_centroid[3*f+1] - r[1])
              + coeff[2] * (fv->face_centroid[3*f+2] - r[2])
              + X[cid];
    }
    */
    
    ierr = FVDAGetReconstructionStencil_AtCell(fv,c_p,&n_neigh,neigh);CHKERRQ(ierr);
    ierr = setup_coeff(fv,c_p,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
    ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_p,(const PetscReal*)&fv_coor[3*c_p],X,coeff,&Xhr_p);CHKERRQ(ierr);

    /*
    min = PETSC_MAX_REAL;
    max = PETSC_MIN_REAL;
    for (PetscInt kk=0; kk<n_neigh; kk++) {
      min = PetscMin(min,X[neigh[kk]]);
      max = PetscMax(max,X[neigh[kk]]);
    }

    {
      const PetscInt cid = c_p;
      PetscReal Xhr      = Xhr_p;
      
      PetscReal delta,phi_ij,phi = PETSC_MAX_REAL;
      const PetscReal *r = &fv_coor[3*cid];
      PetscReal rij[3];// = &fv->face_centroid[3*f];
      PetscReal xi2d[]={0,0};
      PetscInt kk;
      
      
      for (kk=0; kk<6; kk++) {
        ierr = _EvaluateFaceCoord3d(kk,cell_coor,xi2d,rij);CHKERRQ(ierr);
        
        delta =  coeff[0] * (rij[0] - r[0]);
        delta += coeff[1] * (rij[1] - r[1]);
        delta += coeff[2] * (rij[2] - r[2]);
        
        if (delta > 0) {
          phi_ij = PetscMin(1.0, (max - Xhr)/delta );
          phi_ij = vant_L(max - Xhr,delta,0);
          phi_ij = vant_L_v2((max - Xhr)/delta,0);
        } else if (delta < 0) {
          phi_ij = PetscMin(1.0, (min - Xhr)/delta );
          phi_ij = vant_L(min - Xhr,delta,0);
          phi_ij = vant_L_v2((min - Xhr)/delta,0);
        } else {
          phi_ij = 1.0;
        }
        phi = PetscMin(phi,phi_ij);
      }
  
      Xhr_p =   coeff[0] * (fv->face_centroid[3*f+0] - r[0])
              + coeff[1] * (fv->face_centroid[3*f+1] - r[1])
              + coeff[2] * (fv->face_centroid[3*f+2] - r[2])
              + X[cid];
    }
    */

    
    //printf("Q_m %+1.4e %+1.4e [hr]  Q_p %+1.4e %+1.4e [hr]\n",X_m,Xhr_m,X_p,Xhr_p);

    flux_hr = 0.5 * ( v_n * (Xhr_p + Xhr_m) - PetscAbsReal(v_n) * (Xhr_p - Xhr_m ) );
    

    F[c_m] += flux_hr * dS; // cell[-]
    F[c_p] -= flux_hr * dS; // cell[+]
  }
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    v_n = vdotn[f];
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    flux = v_n * X_m;
    
    
    if (v_n > 0.0) { /* outflow */
      ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
      ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr_m);CHKERRQ(ierr);

      flux_hr = v_n * Xhr_m;

      F[c_m] += flux_hr * dS; // cell[-]
    } else { /* inflow */
      
      switch (bctype) {
          
        case FVFLUX_DIRICHLET_CONSTRAINT:
        {
          PetscReal g_D = bcvalue;
          X_p = 2.0 * g_D - X_m;
        }
          break;
          
        case FVFLUX_NEUMANN_CONSTRAINT:
        {
          /* What to do with non-zero flux?? */
          PetscReal g_N = bcvalue;
          X_p = (0.0) * g_N + X_m;
        }
          break;
          
        default:
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
          break;
      }
      flux = v_n * X_p;
      F[c_m] += flux * dS;
    }
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_upwind_hr_local_SEQ(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb;
  const PetscReal *vdotn;
  PetscReal       v_n,X_m,X_p,flux,Xhr,flux_hr;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  PetscInt        n_neigh,neigh[27];
  PetscReal       coeff[3];
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"v.n",&vdotn);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    if (v_n > 0.0) { // out flow
      
      flux = v_n * X_m;
    
      ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
      ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
    
      flux_hr = v_n * Xhr;
    } else {
      
      flux = v_n * X_m;
      
      ierr = FVDAGetReconstructionStencil_AtCell(fv,c_p,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c_p,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
      ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_p,(const PetscReal*)&fv_coor[3*c_p],X,coeff,&Xhr);CHKERRQ(ierr);

      flux_hr = v_n * Xhr;
    }
    
    F[c_m] += flux_hr * dS; // cell[-]
    F[c_p] -= flux_hr * dS; // cell[+]
  }
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    v_n = vdotn[f];
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    flux = v_n * X_m;
    
    
    if (v_n > 0.0) { /* outflow */
      ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
      ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
      
      flux_hr = v_n * Xhr;
      
      F[c_m] += flux_hr * dS; // cell[-]
    } else { /* inflow */
      switch (bctype) {
        case FVFLUX_DIRICHLET_CONSTRAINT:
        {
          PetscReal g_D = bcvalue;
          X_p = 2.0 * g_D - X_m;
        }
          break;
          
        case FVFLUX_NEUMANN_CONSTRAINT:
        {
          /* What to do with non-zero flux?? */
          PetscReal g_N = bcvalue;
          X_p = (0.0) * g_N + X_m;
        }
          break;
          
        default:
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
          break;
      }
      flux = v_n * X_p;
      F[c_m] += flux * dS;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_upwind_hr_local_MPI(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb;
  const PetscReal *vdotn;
  PetscReal       v_n,X_m,X_p,flux,Xhr,flux_hr;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  PetscInt        n_neigh,neigh[27];
  PetscReal       coeff[3];
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"v.n",&vdotn);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    flux_hr = 0;
    
    if (fv->face_location[f] == DAFACE_SUB_DOMAIN_BOUNDARY) {

      if (v_n > 0.0 ) { /* outflow: use -ve side to compute flux */
        
        if (fv->face_element_map[2*f+0] >= 0) {
          
          ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
          ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
          ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
          
          flux_hr = v_n * Xhr;
          
          F[c_m] += flux_hr * dS; // cell[-]
          F[c_p] -= flux_hr * dS; // cell[+]
        
        } else  if (fv->face_element_map[2*f+1] >= 0) {
          
          /* cannot process the high order flux for the + side - neigbour rank will take care of that */
          
        } else {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"v.n > 0 sub-domain boundary not valid");
        }
        
      } else { /* inflow: use +ve side to compute flux */
        if (fv->face_element_map[2*f+0] >= 0) {

          /* cannot process the high order flux for the - side - neigbour rank will take care of that */

        } else  if (fv->face_element_map[2*f+1] >= 0) {

          ierr = FVDAGetReconstructionStencil_AtCell(fv,c_p,&n_neigh,neigh);CHKERRQ(ierr);
          ierr = setup_coeff(fv,c_p,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
          ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_p,(const PetscReal*)&fv_coor[3*c_p],X,coeff,&Xhr);CHKERRQ(ierr);
          
          flux_hr = v_n * Xhr;
          
          F[c_m] += flux_hr * dS; // cell[-]
          F[c_p] -= flux_hr * dS; // cell[+]

        } else {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"v.n <= 0 sub-domain boundary not valid");
        }
        
      }
      
    } else { /* interior */

      if (v_n > 0.0) { /* outflow */
        ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
        ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
        ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
      } else {
        ierr = FVDAGetReconstructionStencil_AtCell(fv,c_p,&n_neigh,neigh);CHKERRQ(ierr);
        ierr = setup_coeff(fv,c_p,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
        ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_p,(const PetscReal*)&fv_coor[3*c_p],X,coeff,&Xhr);CHKERRQ(ierr);
      }
      
      flux_hr = v_n * Xhr;
      
      F[c_m] += flux_hr * dS; // cell[-]
      F[c_p] -= flux_hr * dS; // cell[+]
    }
  }
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    v_n = vdotn[f];
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    flux = v_n * X_m;
    
    
    if (v_n > 0.0) { /* outflow */
      ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
      ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
      
      flux_hr = v_n * Xhr;
      
      F[c_m] += flux_hr * dS; // cell[-]
    } else { /* inflow */
      switch (bctype) {
          
        case FVFLUX_DIRICHLET_CONSTRAINT:
        {
          PetscReal g_D = bcvalue;
          X_p = 2.0 * g_D - X_m;
        }
          break;
          
        case FVFLUX_NEUMANN_CONSTRAINT:
        {
          /* What to do with non-zero flux?? */
          PetscReal g_N = bcvalue;
          X_p = (0.0) * g_N + X_m;
        }
          break;
          
        default:
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
          break;
      }
      flux = v_n * X_p;
      F[c_m] += flux * dS;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_upwind_hr_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscMPIInt     commsize;
  ierr = MPI_Comm_size(fv->comm,&commsize);CHKERRQ(ierr);
  if (commsize == 1) {
    ierr = eval_F_upwind_hr_local_SEQ(fv,domain_geom_coor,fv_coor,X,F);CHKERRQ(ierr);
  } else {
    ierr = eval_F_upwind_hr_local_MPI(fv,domain_geom_coor,fv_coor,X,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


//
// Based on eval_F_upwind_hr_local_MPI()
//
PetscErrorCode eval_F_upwind_hr_bound_local(FVDA fv,const PetscReal range[],const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb;
  const PetscReal *vdotn;
  PetscReal       v_n,X_m,X_p,flux,Xhr,flux_hr;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  PetscInt        n_neigh,neigh[27];
  PetscReal       coeff[3];
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"v.n",&vdotn);CHKERRQ(ierr);

#if 0
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    flux_hr = 0;
    if (v_n > 0.0 && fv->face_element_map[2*f+0] >= 0) { /* outflow */
      
      flux = v_n * X_m;
      
      ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
      ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
      
      flux_hr = v_n * Xhr;
      if (Xhr < range[0] || Xhr > range[1]) {
        flux_hr = flux;
      }
      
      F[c_m] += flux_hr * dS; // cell[-]
      F[c_p] -= flux_hr * dS; // cell[+]
    } else {
      // do nothing
    }
  }
#endif
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    v_n = vdotn[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    flux = 0;
    flux_hr = 0;
    
    if (fv->face_location[f] == DAFACE_SUB_DOMAIN_BOUNDARY) {
      
      if (v_n > 0.0 ) { /* outflow: use -ve side to compute flux */
        
        if (fv->face_element_map[2*f+0] >= 0) {
          
          ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
          ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
          ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
          
          flux = v_n * X_m;
          flux_hr = v_n * Xhr;
          if (Xhr < range[0] || Xhr > range[1]) {
            flux_hr = flux;
          }
          
          F[c_m] += flux_hr * dS; // cell[-]
          F[c_p] -= flux_hr * dS; // cell[+]
          
        } else  if (fv->face_element_map[2*f+1] >= 0) {
          
          /* cannot process the high order flux for the + side - neigbour rank will take care of that */
          
        } else {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"v.n > 0 sub-domain boundary not valid");
        }
        
      } else { /* inflow: use +ve side to compute flux */
        if (fv->face_element_map[2*f+0] >= 0) {
          
          /* cannot process the high order flux for the - side - neigbour rank will take care of that */
          
        } else  if (fv->face_element_map[2*f+1] >= 0) {
          
          ierr = FVDAGetReconstructionStencil_AtCell(fv,c_p,&n_neigh,neigh);CHKERRQ(ierr);
          ierr = setup_coeff(fv,c_p,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
          ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_p,(const PetscReal*)&fv_coor[3*c_p],X,coeff,&Xhr);CHKERRQ(ierr);
          
          flux = v_n * X_p;
          flux_hr = v_n * Xhr;
          if (Xhr < range[0] || Xhr > range[1]) {
            flux_hr = flux;
          }
          
          F[c_m] += flux_hr * dS; // cell[-]
          F[c_p] -= flux_hr * dS; // cell[+]
          
        } else {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"v.n <= 0 sub-domain boundary not valid");
        }
        
      }
      
    } else { /* interior */
      
      if (v_n > 0.0) { /* outflow */
        ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
        ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
        ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
        flux = v_n * X_m;
      } else {
        ierr = FVDAGetReconstructionStencil_AtCell(fv,c_p,&n_neigh,neigh);CHKERRQ(ierr);
        ierr = setup_coeff(fv,c_p,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
        ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_p,(const PetscReal*)&fv_coor[3*c_p],X,coeff,&Xhr);CHKERRQ(ierr);
        flux = v_n * X_p;
      }
      
      flux_hr = v_n * Xhr;
      if (Xhr < range[0] || Xhr > range[1]) {
        flux_hr = flux;
      }
      
      F[c_m] += flux_hr * dS; // cell[-]
      F[c_p] -= flux_hr * dS; // cell[+]
    }
  }

  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    v_n = vdotn[f];
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    flux = v_n * X_m;
    
    
    if (v_n > 0.0) { /* outflow */
      ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
      ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr);CHKERRQ(ierr);
      
      flux_hr = v_n * Xhr;
      if (Xhr < range[0] || Xhr > range[1]) {
        flux_hr = flux;
      }
      
      F[c_m] += flux_hr * dS; // cell[-]
    } else { /* inflow */
      switch (bctype) {
        case FVFLUX_DIRICHLET_CONSTRAINT:
        {
          PetscReal g_D = bcvalue;
          X_p = 2.0 * g_D - X_m;
        }
          break;
          
        case FVFLUX_NEUMANN_CONSTRAINT:
        {
          /* What to do with non-zero flux?? */
          PetscReal g_N = bcvalue;
          X_p = (0.0) * g_N + X_m;
        }
          break;
          
        default:
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
          break;
      }
      flux = v_n * X_p;
      F[c_m] += flux * dS;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_diffusion_7point_hr_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb,d;
  const PetscReal *k;
  PetscReal       k_face,X_m,X_p;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  PetscInt        n_neigh,neigh[27];
  PetscReal       coeff[3],Xhr_m,Xhr_p,grad_m[3],grad_p[3],flux_m,flux_p,s[3];
  
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"k",&k);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    PetscReal dl[]={0,0,0};
    PetscReal dsn=0,flux,flux_hr;
    
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    for (d=0; d<3; d++) {
      dl[d] = fv_coor[3*c_p + d] - fv_coor[3*c_m + d];
      dsn += dl[d] * dl[d];
      s[d] = dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    for (d=0; d<3; d++) {
      s[d] = s[d] / dsn;
    }
    flux = k_face * (X_p - X_m) / dsn;
    
    //printf("face %d: normal %+1.4e %+1.4e %+1.4e\n",f,fv->face_normal[3*f+0],fv->face_normal[3*f+1],fv->face_normal[3*f+2]);
    //printf("cell %d- %d+: dsn %+1.4e: flux %+1.4e\n",c_m,c_p,dsn,flux);
    //printf("Q- %+1.4e, Q+ %+1.4e\n",X_m,X_p);
    
    ierr = FVDAGetReconstructionStencil_AtCell(fv,c_m,&n_neigh,neigh);CHKERRQ(ierr);
    ierr = setup_coeff(fv,c_m,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
    ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,(const PetscReal*)&fv_coor[3*c_m],X,coeff,&Xhr_m);CHKERRQ(ierr);
    grad_m[0] = coeff[0];
    grad_m[1] = coeff[1];
    grad_m[2] = coeff[2];
    //printf("  grad[-] %+1.4e %+1.4e %+1.4e\n",grad_m[0],grad_m[1],grad_m[2]);

    ierr = FVDAGetReconstructionStencil_AtCell(fv,c_p,&n_neigh,neigh);CHKERRQ(ierr);
    ierr = setup_coeff(fv,c_p,n_neigh,(const PetscInt*)neigh,fv_coor,X,coeff);CHKERRQ(ierr);
    ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_p,(const PetscReal*)&fv_coor[3*c_p],X,coeff,&Xhr_p);CHKERRQ(ierr);
    grad_p[0] = coeff[0];
    grad_p[1] = coeff[1];
    grad_p[2] = coeff[2];
    //printf("  grad[+] %+1.4e %+1.4e %+1.4e\n",grad_p[0],grad_p[1],grad_p[2]);

    //printf("[HR] Q- %+1.4e, Q+ %+1.4e\n",Xhr_m,Xhr_p);

    flux_m =   grad_m[0] * dl[0] + grad_m[1] * dl[1] + grad_m[2] * dl[2];
    flux_p = -(grad_p[0] * dl[0] + grad_p[1] * dl[1] + grad_p[2] * dl[2]);

#if 0
    flux_hr  = 0.5 * ( grad_m[0] + grad_p[0] ) * fv->face_normal[3*f+0];
    flux_hr += 0.5 * ( grad_m[1] + grad_p[1] ) * fv->face_normal[3*f+1];
    flux_hr += 0.5 * ( grad_m[2] + grad_p[2] ) * fv->face_normal[3*f+2];
    
    flux_hr = flux_hr * k_face;
    //printf("  flux[hr] %+1.4e\n",flux_hr);
#endif


    {
      PetscReal grad_e[3];
      
      grad_e[0] = 0.5 * ( grad_m[0] + grad_p[0] );
      grad_e[1] = 0.5 * ( grad_m[1] + grad_p[1] );
      grad_e[2] = 0.5 * ( grad_m[2] + grad_p[2] );
      
      flux_hr  = grad_e[0] * (fv->face_normal[3*f+0] - s[0]);
      flux_hr += grad_e[1] * (fv->face_normal[3*f+1] - s[1]);
      flux_hr += grad_e[2] * (fv->face_normal[3*f+2] - s[2]);
      
      flux_hr = flux_hr * k_face;
    }
    
/*
    if (fabs(flux_hr - flux) > 1.0e-10) {
      printf("flux seems to be wrong \n");
      exit(0);
    }
*/
    F[c_m] += (flux + flux_hr) * dS; // cell[-]
    F[c_p] -= (flux + flux_hr) * dS; // cell[+]
  }
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    PetscReal  dl[]={0,0,0};
    PetscReal  dsn=0,flux;
    
    if (fv->face_location[f] != DAFACE_BOUNDARY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"YOU SHOULD NEVER BE IN LOOP IF YOU ARE NOT A BOUNDARY");
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    for (d=0; d<3; d++) {
      dl[d] = 2.0 * (fv->face_centroid[3*f + d] - fv_coor[3*c_m + d]);
      dsn += dl[d]*dl[d];
    }
    dsn = PetscSqrtReal(dsn);

    switch (bctype) {
      case FVFLUX_DIRICHLET_CONSTRAINT:
      { /* Weak imposition of Dirichlet */
        PetscReal g_D = bcvalue;
        
        X_p = 2.0 * g_D - X_m;
      }
        break;
        
      case FVFLUX_NEUMANN_CONSTRAINT:
      { /* Weak imposition of Neumann */
        PetscReal g_N = bcvalue;
        X_p = (dsn/k_face) * g_N + X_m;
      }
        break;
        
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
        break;
    }
    flux = k_face * (X_p - X_m) / dsn;
    F[c_m] += flux * dS;

  }
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_diffusion_7point_hr_local_store(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb,d,c;
  const PetscReal *k;
  PetscReal       k_face,X_m,X_p;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  PetscInt        n_neigh,neigh[27];
  PetscReal       *coeff,_coeff[3],grad_m[3],grad_p[3],s[3];
  
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = PetscMalloc1(fv->ncells*3,&coeff);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    ierr = FVDAGetReconstructionStencil_AtCell(fv,c,&n_neigh,neigh);CHKERRQ(ierr);
    ierr = setup_coeff(fv,c,n_neigh,(const PetscInt*)neigh,fv_coor,X,_coeff);CHKERRQ(ierr);
    coeff[3*c+0] = _coeff[0];
    coeff[3*c+1] = _coeff[1];
    coeff[3*c+2] = _coeff[2];
  }
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"k",&k);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    PetscReal dl[]={0,0,0};
    PetscReal dsn=0,flux,flux_hr;
    
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    for (d=0; d<3; d++) {
      dl[d] = fv_coor[3*c_p + d] - fv_coor[3*c_m + d];
      dsn += dl[d] * dl[d];
      s[d] = dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    for (d=0; d<3; d++) {
      s[d] = s[d] / dsn;
    }
    flux = k_face * (X_p - X_m) / dsn;
    
    grad_m[0] = coeff[3*c_m+0];
    grad_m[1] = coeff[3*c_m+1];
    grad_m[2] = coeff[3*c_m+2];
    
    grad_p[0] = coeff[3*c_p+0];
    grad_p[1] = coeff[3*c_p+1];
    grad_p[2] = coeff[3*c_p+2];
    
    {
      PetscReal grad_e[3];
      
      grad_e[0] = 0.5 * ( grad_m[0] + grad_p[0] );
      grad_e[1] = 0.5 * ( grad_m[1] + grad_p[1] );
      grad_e[2] = 0.5 * ( grad_m[2] + grad_p[2] );
      
      flux_hr  = grad_e[0] * (fv->face_normal[3*f+0] - s[0]);
      flux_hr += grad_e[1] * (fv->face_normal[3*f+1] - s[1]);
      flux_hr += grad_e[2] * (fv->face_normal[3*f+2] - s[2]);
      
      flux_hr = flux_hr * k_face;
    }
    
    F[c_m] += (flux + flux_hr) * dS; // cell[-]
    F[c_p] -= (flux + flux_hr) * dS; // cell[+]
  }
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    PetscReal  dl[]={0,0,0};
    PetscReal  dsn=0,flux;
    
    if (fv->face_location[f] != DAFACE_BOUNDARY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"YOU SHOULD NEVER BE IN LOOP IF YOU ARE NOT A BOUNDARY");
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    for (d=0; d<3; d++) {
      dl[d] = 2.0 * (fv->face_centroid[3*f + d] - fv_coor[3*c_m + d]);
      dsn += dl[d]*dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    
    switch (bctype) {
      case FVFLUX_DIRICHLET_CONSTRAINT:
      { /* Weak imposition of Dirichlet */
        PetscReal g_D = bcvalue;
        
        X_p = 2.0 * g_D - X_m;
      }
        break;
        
      case FVFLUX_NEUMANN_CONSTRAINT:
      { /* Weak imposition of Neumann */
        PetscReal g_N = bcvalue;
        X_p = (dsn/k_face) * g_N + X_m;
      }
        break;
        
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
        break;
    }
    flux = k_face * (X_p - X_m) / dsn;
    F[c_m] += flux * dS;
    
  }
  ierr = PetscFree(coeff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_diffusion_7point_hr_local_store_MPI(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[])
{
  PetscErrorCode  ierr;
  PetscInt        f,c_m,c_p,fb,d,c;
  const PetscReal *k;
  PetscReal       k_face,X_m,X_p;
  DM              dm;
  PetscReal       dS;
  PetscInt        dm_nel,dm_nen,cellid;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  PetscInt        n_neigh,neigh[27];
  PetscReal       *coeff,_coeff[3],grad_m[3],grad_p[3],s[3];
  
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = PetscCalloc1(fv->ncells*3,&coeff);CHKERRQ(ierr);
  {
    PetscInt e,fv_start[3],fv_range[3],fv_start_local[3],fv_ghost_offset[3],fv_ghost_range[3];
    
    ierr = DMDAGetCorners(fv->dm_fv,&fv_start[0],&fv_start[1],&fv_start[2],&fv_range[0],&fv_range[1],&fv_range[2]);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(fv->dm_fv,&fv_start_local[0],&fv_start_local[1],&fv_start_local[2],&fv_ghost_range[0],&fv_ghost_range[1],&fv_ghost_range[2]);CHKERRQ(ierr);
    fv_ghost_offset[0] = fv_start[0] - fv_start_local[0];
    fv_ghost_offset[1] = fv_start[1] - fv_start_local[1];
    fv_ghost_offset[2] = fv_start[2] - fv_start_local[2];

    for (e=0; e<fv->ncells; e++) {
      PetscInt cijk[3];
      
      ierr = _cart_convert_index_to_ijk(e,(const PetscInt*)fv_range,cijk);CHKERRQ(ierr);
      cijk[0] += fv_ghost_offset[0];
      cijk[1] += fv_ghost_offset[1];
      cijk[2] += fv_ghost_offset[2];
      
      ierr = _cart_convert_ijk_to_index((const PetscInt*)cijk,(const PetscInt*)fv_ghost_range,&c);CHKERRQ(ierr);

      ierr = FVDAGetReconstructionStencil_AtCell(fv,c,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c,n_neigh,(const PetscInt*)neigh,fv_coor,X,_coeff);CHKERRQ(ierr);
      coeff[3*e+0] = _coeff[0];
      coeff[3*e+1] = _coeff[1];
      coeff[3*e+2] = _coeff[2];
    }
  }
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,"k",&k);CHKERRQ(ierr);
  
  /* interior face loop */
  for (f=0; f<fv->nfaces; f++) {
    PetscReal dl[]={0,0,0};
    PetscReal dsn=0,flux,flux_hr;
    
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    X_p = X[c_p];
    
    for (d=0; d<3; d++) {
      dl[d] = fv_coor[3*c_p + d] - fv_coor[3*c_m + d];
      dsn += dl[d] * dl[d];
      s[d] = dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    for (d=0; d<3; d++) {
      s[d] = s[d] / dsn;
    }
    flux = k_face * (X_p - X_m) / dsn;
    
    if (fv->face_element_map[2*f+0] >= 0) {
      F[c_m] += 0.5*flux * dS; // cell[-]
      F[c_p] -= 0.5*flux * dS; // cell[+]
    }
    if (fv->face_element_map[2*f+1] >= 0) {
      F[c_m] += 0.5*flux * dS; // cell[-]
      F[c_p] -= 0.5*flux * dS; // cell[+]
    }
    
#if 0
    grad_m[0] = coeff[3*c_m+0];
    grad_m[1] = coeff[3*c_m+1];
    grad_m[2] = coeff[3*c_m+2];
    
    grad_p[0] = coeff[3*c_p+0];
    grad_p[1] = coeff[3*c_p+1];
    grad_p[2] = coeff[3*c_p+2];
    
    {
      PetscReal grad_e[3];
      
      grad_e[0] = 0.5 * ( grad_m[0] + grad_p[0] );
      grad_e[1] = 0.5 * ( grad_m[1] + grad_p[1] );
      grad_e[2] = 0.5 * ( grad_m[2] + grad_p[2] );
      
      flux_hr  = grad_e[0] * (fv->face_normal[3*f+0] - s[0]);
      flux_hr += grad_e[1] * (fv->face_normal[3*f+1] - s[1]);
      flux_hr += grad_e[2] * (fv->face_normal[3*f+2] - s[2]);
      
      flux_hr = flux_hr * k_face;
    }
    
    F[c_m] += (flux + flux_hr) * dS; // cell[-]
    F[c_p] -= (flux + flux_hr) * dS; // cell[+]
#endif
    
    if (fv->face_element_map[2*f+0] >= 0) {
      PetscReal grad_e[3];
      PetscInt cl = fv->face_element_map[2*f+0];
      
      grad_m[0] = coeff[3*cl+0];
      grad_m[1] = coeff[3*cl+1];
      grad_m[2] = coeff[3*cl+2];
      
      grad_e[0] = 0.5 * ( grad_m[0] );
      grad_e[1] = 0.5 * ( grad_m[1] );
      grad_e[2] = 0.5 * ( grad_m[2] );
      
      flux_hr  = grad_e[0] * (fv->face_normal[3*f+0] - s[0]);
      flux_hr += grad_e[1] * (fv->face_normal[3*f+1] - s[1]);
      flux_hr += grad_e[2] * (fv->face_normal[3*f+2] - s[2]);
      
      flux_hr = flux_hr * k_face;

      F[c_m] += (flux_hr) * dS; // cell[-]
      F[c_p] -= (flux_hr) * dS; // cell[+]
    }

    if (fv->face_element_map[2*f+1] >= 0) {
      PetscReal grad_e[3];
      PetscInt cl = fv->face_element_map[2*f+1];
      
      grad_p[0] = coeff[3*cl+0];
      grad_p[1] = coeff[3*cl+1];
      grad_p[2] = coeff[3*cl+2];
      
      grad_e[0] = 0.5 * ( grad_p[0] );
      grad_e[1] = 0.5 * ( grad_p[1] );
      grad_e[2] = 0.5 * ( grad_p[2] );
      
      flux_hr  = grad_e[0] * (fv->face_normal[3*f+0] - s[0]);
      flux_hr += grad_e[1] * (fv->face_normal[3*f+1] - s[1]);
      flux_hr += grad_e[2] * (fv->face_normal[3*f+2] - s[2]);
      
      flux_hr = flux_hr * k_face;
      
      F[c_m] += (flux_hr) * dS; // cell[-]
      F[c_p] -= (flux_hr) * dS; // cell[+]
    }

    
  }
  
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    PetscReal  dl[]={0,0,0};
    PetscReal  dsn=0,flux;
    
    if (fv->face_location[f] != DAFACE_BOUNDARY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"YOU SHOULD NEVER BE IN LOOP IF YOU ARE NOT A BOUNDARY");
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,domain_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];
    
    k_face = k[f];
    
    c_m = fv->face_fv_map[2*f+0];
    X_m = X[c_m];
    
    for (d=0; d<3; d++) {
      dl[d] = 2.0 * (fv->face_centroid[3*f + d] - fv_coor[3*c_m + d]);
      dsn += dl[d]*dl[d];
    }
    dsn = PetscSqrtReal(dsn);
    
    switch (bctype) {
      case FVFLUX_DIRICHLET_CONSTRAINT:
      { /* Weak imposition of Dirichlet */
        PetscReal g_D = bcvalue;
        
        X_p = 2.0 * g_D - X_m;
      }
        break;
        
      case FVFLUX_NEUMANN_CONSTRAINT:
      { /* Weak imposition of Neumann */
        PetscReal g_N = bcvalue;
        X_p = (dsn/k_face) * g_N + X_m;
      }
        break;
        
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must set one of Dirichlet or Neumann");
        break;
    }
    flux = k_face * (X_p - X_m) / dsn;
    F[c_m] += flux * dS;
    
  }
  ierr = PetscFree(coeff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_hr(SNES snes,Vec X,Vec F,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;
  
  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dm = fv->dm_fv;
  
  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(Fl);CHKERRQ(ierr);
  ierr = VecGetArray(Fl,&_F);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_upwind_hr_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
      {
        PetscInt k,m;
        ierr = VecGetSize(Fl,&m);CHKERRQ(ierr);
        for (k=0; k<m; k++) {
          _F[k] *= -1.0;
        }
      }
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      //ierr = eval_F_diffusion_7point_hr_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
      //ierr = eval_F_diffusion_7point_hr_local_store(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
      ierr = eval_F_diffusion_7point_hr_local_store_MPI(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
    }
  }
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fl,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dm,Fl,ADD_VALUES,F);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAVecTraverse(FVDA fv,Vec X,PetscReal time,PetscInt dof,
                               PetscBool user_setter(PetscScalar*,PetscScalar*,void*),
                              void *data)
{
  
  Vec               cellcoor;
  const PetscScalar *_cellcoor;
  PetscScalar       *_X;
  PetscInt          c,bs,b;
  PetscScalar       vals[10];
  PetscBool         impose;
  PetscErrorCode    ierr;
  
  ierr = DMDAGetInfo(fv->dm_fv,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&bs,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (bs > 10) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_SUP,"Need to increase static allocation of block-size");
  ierr = DMGetCoordinates(fv->dm_fv,&cellcoor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellcoor,&_cellcoor);CHKERRQ(ierr);
  ierr = VecGetArray(X,&_X);CHKERRQ(ierr);
  if (dof >= 0) {
    for (c=0; c<fv->ncells; c++) {
      impose = user_setter((PetscScalar*)&_cellcoor[fv->dim*c],vals,data);CHKERRQ(ierr);
      if (impose) {
        _X[bs*c + dof] = vals[0];
      }
    }
  } else {
    for (c=0; c<fv->ncells; c++) {
      impose = user_setter((PetscScalar*)&_cellcoor[fv->dim*c],vals,data);
      if (impose) {
        for (b=0; b<bs; b++) {
          _X[bs*c + b] = vals[b];
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(cellcoor,&_cellcoor);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&_X);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode FVDACreateMatrix(FVDA fv,DMDAStencilType type,Mat *A)
{
  //DM            dm;
  DM_DA           *da = (DM_DA*)fv->dm_fv->data;
  DMDAStencilType stype = da->stencil_type;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  da->stencil_type = type;
  //ierr = DMClone(fv->dm_fv,&dm);CHKERRQ(ierr);
  //ierr = DMCreateMatrix(dm,A);CHKERRQ(ierr);
  ierr = DMCreateMatrix(fv->dm_fv,A);CHKERRQ(ierr);
  da->stencil_type = stype;
  //ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 If snes->ksp->pc is type PCMG, configure the levels such
 that they have an appropriate DM. The appropriate DM is one
 consistent with fv->dm_fv without coordinates.
 It is important to not have coordinates attached as PETSc
 does not implement injection of coordinates for P0 interpolation,
 or for the case when the number of cells is even.
*/
PetscErrorCode SNESFVDAConfigureGalerkinMG(SNES snes,FVDA fv)
{
  KSP       ksp;
  PC        pc;
  PetscBool ismg;
  DM        dm,*dml;
  Mat       interp;
  PetscInt  nlevels,k;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {
    ierr = DMClone(fv->dm_fv,&dm);CHKERRQ(ierr);
    ierr = DMDASetInterpolationType(dm,DMDA_Q0);CHKERRQ(ierr);
    ierr = PCMGGetLevels(pc,&nlevels);CHKERRQ(ierr);
    ierr = PetscCalloc1(nlevels,&dml);CHKERRQ(ierr);
    dml[0] = dm;
    for (k=1; k<nlevels; k++) {
      ierr = DMCoarsen(dml[k-1],fv->comm,&dml[k]);CHKERRQ(ierr);
      ierr = DMDASetInterpolationType(dml[k],DMDA_Q0);CHKERRQ(ierr);
    }
    for (k=1; k<nlevels; k++) {
      ierr = DMCreateInterpolation(dml[k],dml[k-1],&interp,NULL);CHKERRQ(ierr);
      ierr = PCMGSetInterpolation(pc,nlevels-k,interp);CHKERRQ(ierr);
      ierr = MatDestroy(&interp);CHKERRQ(ierr);
    }
    for (k=0; k<1; k++) {
      KSP smth;
      ierr = PCMGGetSmoother(pc,k,&smth);CHKERRQ(ierr);
      ierr = KSPSetDM(smth,dml[nlevels-1-k]);CHKERRQ(ierr);
      ierr = KSPSetDMActive(smth,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH);CHKERRQ(ierr);
    for (k=0; k<nlevels; k++) {
      ierr = DMDestroy(&dml[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dml);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
