#include "ptatin3d.h"
#include "ptatin3d_energy.h"
#include "private/ptatin_impl.h"
#include "ptatin_init.h"

#include "data_bucket.h"
#include "material_point_utils.h"
#include "material_point_std_utils.h"
#include "ptatin_log.h"
#include "ptatin_models.h"
#include "ptatin_utils.h"
#include "stokes_form_function.h"
#include "stokes_operators.h"
#include "sub_comm.h"
#include "dmda_redundant.h"
#include "stokes_output.h"
#include "MPntStd_def.h"

#include "parse.h"
#include "point_in_tetra.h"

typedef struct
{
  DM        da;
  char      region_file[PETSC_MAX_PATH_LEN],mesh_file[PETSC_MAX_PATH_LEN];
  PetscReal O[3],L[3];
  int       method;
} GMSHCtx;

static PetscErrorCode CreateGMSHCtx(GMSHCtx **data)
{
  GMSHCtx        *ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(GMSHCtx),&ctx);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx,sizeof(GMSHCtx));CHKERRQ(ierr);
  *data = ctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyGMSHCtx(GMSHCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!data) { PetscFunctionReturn(0); }
  ierr = DMDestroy(&data->da);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetRegionIndexFromGMSH(pTatinCtx ptatin, GMSHCtx *data)
{
  Mesh           mesh;
  DataBucket     db;
  DataField      PField_std;
  long int       *region_idx = NULL;
  int            p,n_mp_points;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatinGetMaterialPoints(ptatin,&db,NULL);CHKERRQ(ierr);
  DataBucketGetSizes(db,&n_mp_points,0,0);
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);

  /* get user mesh from file */
  parse_mesh(data->mesh_file,&mesh);
  if (!mesh) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"mesh = NULL. Aborting.\n"); }
  /* get region index from file */
  parse_field(mesh,data->region_file,'c',(void**)&region_idx);
  
  for (p=0; p<n_mp_points; p++) {
    MPntStd  *marker_std;
    long int np = 1,found;
    long int ep[] = {-1};
    double   xip[] = {0.0,0.0,0.0};

    DataFieldAccessPoint(PField_std,p,(void**)&marker_std);

    /* locate point */
    switch (data->method) {
    case 0:
      PointLocation_BruteForce(mesh,np,(const double*)marker_std->coor,ep,xip,&found);
      break;

    case 1:
      PointLocation_PartitionedBoundingBox(mesh,np,(const double*)marker_std->coor,ep,xip,&found);
      break;
    }
    /* assign marker phase */
    marker_std->phase = region_idx[ep[0]];
  }
  DataFieldRestoreAccess(PField_std);
  free(region_idx);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAInitialMeshGeometry(pTatinCtx ptatin, GMSHCtx *data)
{
  PetscInt n[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  n[0] = 2*ptatin->mx + 1;
  n[1] = 2*ptatin->my + 1;
  n[2] = 2*ptatin->mz + 1;

  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,n[0],n[1],n[2],PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 3,2, 0,0,0,&data->da);CHKERRQ(ierr);
  ierr = DMSetUp(data->da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(data->da,data->O[0],data->L[0],data->O[1],data->L[1],data->O[2],data->L[2]);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelInitialize(pTatinCtx ptatin, GMSHCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ptatin->mx = ptatin->my = ptatin->mz = 16;

  data->O[0] = data->O[1] = data->O[2] = 0.0;
  data->L[0] = 1000.0e3;
  data->L[1] = 200.0e3;
  data->L[2] = 1000.0e3;

  ierr = PetscSNPrintf(data->mesh_file,PETSC_MAX_PATH_LEN-1,"md.bin");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-mesh_file",data->mesh_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(data->region_file,PETSC_MAX_PATH_LEN-1,"region_cell.bin");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-region_file",data->region_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);

  data->method = 0;
  ierr = PetscOptionsGetInt(NULL,NULL,"-method",&data->method,NULL);CHKERRQ(ierr);

  ierr = DMDAInitialMeshGeometry(ptatin,data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode OutputMarkers(pTatinCtx ptatin,const char prefix[])
{
  DataBucket               materialpoint_db;
  int                      nf;
  const MaterialPointField mp_prop_list[] = { MPField_Std };//, MPField_Stokes, MPField_StokesPl, MPField_Energy };
  char                     mp_file_prefix[256];
  PetscErrorCode           ierr;

  PetscFunctionBegin;

  nf = sizeof(mp_prop_list)/sizeof(mp_prop_list[0]);

  ierr = pTatinGetMaterialPoints(ptatin,&materialpoint_db,NULL);CHKERRQ(ierr);
  sprintf(mp_file_prefix,"%s_mpoints",prefix);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,ptatin->outputpath,mp_file_prefix);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode pTatin3d_ICFromGMSH(int argc,char **argv)
{
  pTatinCtx       ptatin;
  GMSHCtx         *data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  
  ierr = pTatin3dCreateContext(&ptatin);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(ptatin);CHKERRQ(ierr);

  ierr = CreateGMSHCtx(&data);CHKERRQ(ierr);
  ierr = ModelInitialize(ptatin,data);CHKERRQ(ierr);
  ierr = pTatin3dCreateMaterialPoints(ptatin,data->da);CHKERRQ(ierr);
  /* interpolate material point coordinates (needed if mesh was modified) */
  ierr = MaterialPointCoordinateSetUp(ptatin,data->da);CHKERRQ(ierr);
  /* use dave's library to assign material points region index */
  ierr = SetRegionIndexFromGMSH(ptatin,data);CHKERRQ(ierr);
  /* output */
  ierr = OutputMarkers(ptatin,"MPStd");CHKERRQ(ierr);

  ierr = DestroyGMSHCtx(data);CHKERRQ(ierr);
  ierr = pTatin3dDestroyContext(&ptatin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = pTatinInitialize(&argc,&argv,0,NULL);CHKERRQ(ierr);

  ierr = pTatin3d_ICFromGMSH(argc,argv);CHKERRQ(ierr);

  ierr = pTatinFinalize();CHKERRQ(ierr);
  return 0;
}