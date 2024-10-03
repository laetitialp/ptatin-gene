#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>

#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin_init.h"
#include "ptatin_utils.h"
#include "dmda_update_coords.h"
#include "dmda_remesh.h"
#include "mesh_update.h"
#include "dmda_view_petscvtk.h"
#include "dmdae.h"
#include "dmda_element_q2p1.h"
#include "element_utils_q2.h"
#include "element_utils_q1.h"
#include "dmda_element_q1.h"

#include "model_utils.h"

struct _p_ModelData {
  DM        da;
  PetscInt  n[3],nsteps; 
  PetscReal O[3],L[3];
  char      outputpath[PETSC_MAX_PATH_LEN];
};

typedef struct _p_ModelData *ModelData;

static PetscErrorCode ModelDestroy(ModelData *data)
{
  ModelData      _data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!data) { PetscFunctionReturn(0); }

  _data = *data;
  if (_data->da) { ierr = DMDestroy(&_data->da);CHKERRQ(ierr); }
  ierr = PetscFree(_data);CHKERRQ(ierr);
  *data = NULL;

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetUp(ModelData *_data)
{
  ModelData      data;
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(struct _p_ModelData),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(struct _p_ModelData));CHKERRQ(ierr);
  /* mesh size */
  data->n[0] = data->n[1] = data->n[2] = 21;
  /* domain size */
  data->O[0] = data->O[1] = data->O[2] = -10.0;
  data->L[0] = data->L[1] = data->L[2] = 10.0;
  /* number of steps */
  data->nsteps = 10;

  /* options */
  nn = 3;
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-n",data->n,&nn,&found);CHKERRQ(ierr);
  if (found) { if (nn != 3) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -O. Found %d",nn); } }

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-O",data->O,&nn,&found);CHKERRQ(ierr);
  if (found) { if (nn != 3) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -O. Found %d",nn); } }

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-L",data->L,&nn,&found);CHKERRQ(ierr);
  if (found) { if (nn != 3) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -L. Found %d",nn); } }

  ierr = PetscSNPrintf(data->outputpath,PETSC_MAX_PATH_LEN-1,"output");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-output_path",data->outputpath,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-nsteps",& data->nsteps,NULL);

  *_data = data;

  PetscFunctionReturn(0);
}

static PetscErrorCode WritePVD(char fname[], PetscInt nsteps)
{
  FILE* fp = NULL;
  PetscInt n;

  PetscFunctionBegin;
  if ((fp = fopen ( fname, "w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",fname );
  }

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  fprintf(fp, "<Collection>\n");
  for (n=0; n<nsteps; n++) {
    fprintf(fp, "  <DataSet timestep=\"%f\" file=\"step%d.vts\"/>\n",(float)n,n);
  }
  fprintf(fp, "</Collection>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose( fp );
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutput(ModelData data, PetscInt step)
{
  Vec            x;
  PetscViewer    viewer;
  char           fname[PETSC_MAX_PATH_LEN];
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = pTatinTestDirectory(data->outputpath,'w',&found);CHKERRQ(ierr);
  if (!found) {
    ierr = pTatinCreateDirectory(data->outputpath);CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/step%d.vts",data->outputpath,step);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Writing file %s\n",fname);

  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
  ierr = DMView(data->da,viewer);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(data->da,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName( (PetscObject)x, "empty_field" );CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode DefineInitialMeshGeometry(ModelData data)
{
  DM             cda;
  Vec            coord;
  PetscInt       M,N,P,si,sj,sk,nx,ny,nz,i,j,k;
  PetscScalar    *LA_coords;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,data->n[0],data->n[1],data->n[2],PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 3,2, 0,0,0,&data->da);CHKERRQ(ierr);
  ierr = DMSetUp(data->da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(data->da,data->O[0],data->L[0],data->O[1],data->L[1],data->O[2],data->L[2]);CHKERRQ(ierr);

  ierr = DMDAGetInfo(data->da,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(data->da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);
  /* Get DM coords */
  ierr = DMGetCoordinateDM(data->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(data->da,&coord);CHKERRQ(ierr);
  ierr = VecGetArray(coord,&LA_coords);CHKERRQ(ierr);

  /* Surface of the mesh */
  if (sj+ny == N) {
    j = ny-1;//N-1;
    /* Modify coordinates */
    for (k=0;k<nz;k++){
      for (i=0;i<nx;i++){
        PetscInt      nidx;
        PetscReal     x,z;

        nidx = i + j*nx + k*nx*ny;
        x = LA_coords[3*nidx + 0];
        z = LA_coords[3*nidx + 2];

        LA_coords[3*nidx + 1] += PetscCosReal(x) * PetscSinReal(z);

      }
    }
  }
  ierr = VecRestoreArray(coord,&LA_coords);CHKERRQ(ierr);
  ierr = DMDAUpdateGhostedCoordinates(data->da);CHKERRQ(ierr);
  /* billinearize Q2 mesh */
  ierr = DMDABilinearizeQ2Elements(data->da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RunModel()
{
  ModelData      data;
  PetscReal      dt,diffusivity,baselevel,diffusivities[2];
  PetscInt       step;
  char           pvd_name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = ModelSetUp(&data);CHKERRQ(ierr);
  ierr = DefineInitialMeshGeometry(data);CHKERRQ(ierr);

  dt = 0.5;
  diffusivity = 0.1;
  diffusivities[0] = 0.001;
  diffusivities[1] = 0.1;
  baselevel = 0.0;

  /* Output */
  ierr = ModelOutput(data,0);CHKERRQ(ierr);
  for (step=1; step<data->nsteps; step++) {
    /* Apply surface diffusion */
    //ierr = UpdateMeshGeometry_ApplyDiffusionJMAX(data->da,diffusivity,dt,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
    ierr = UpdateMeshGeometry_ApplyDiffusionJMAX_BaseLevel(data->da,diffusivities,baselevel,dt,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
    /* Output */
    ierr = ModelOutput(data,step);CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(pvd_name,PETSC_MAX_PATH_LEN-1,"%s/domain.pvd",data->outputpath);CHKERRQ(ierr);  
  ierr = WritePVD(pvd_name,data->nsteps);CHKERRQ(ierr);
  ierr = ModelDestroy(&data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main( int argc,char **argv )
{
  PetscErrorCode ierr;
  ierr = pTatinInitialize(&argc,&argv,(char *)0,NULL);CHKERRQ(ierr);
  ierr = RunModel();CHKERRQ(ierr);
  ierr = pTatinFinalize();CHKERRQ(ierr);
  return 0;
}