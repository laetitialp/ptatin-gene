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
#include "mesh_entity.h"
#include "surface_constraint.h"
#include "surfbclist.h"

#include "parse.h"
#include "point_in_tetra.h"

typedef struct
{
  DM        da;
  char      region_file[PETSC_MAX_PATH_LEN],mesh_file[PETSC_MAX_PATH_LEN];
  PetscReal O[3],L[3];
  int       method;
  PetscInt  n_bcfaces;
  PetscInt  *tag_table;
  SurfaceConstraint *sc;
  long int  *f2c,*tag_facets;
} GMSHCtx;

static PetscErrorCode CreateGMSHCtx(GMSHCtx **data)
{
  GMSHCtx        *ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = PetscMalloc(sizeof(GMSHCtx),&ctx);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx,sizeof(GMSHCtx));CHKERRQ(ierr);
  *data = ctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyGMSHCtx(GMSHCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  if (!data) { PetscFunctionReturn(0); }
  ierr = PetscFree(data->tag_table);CHKERRQ(ierr);
  ierr = PetscFree(data->sc);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetRegionIndexFromMesh(pTatinCtx ptatin, Mesh mesh, long int *region_idx, int method)
{
  DataBucket     db;
  DataField      PField_std;
  int            p,n_mp_points;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinGetMaterialPoints(ptatin,&db,NULL);CHKERRQ(ierr);
  DataBucketGetSizes(db,&n_mp_points,0,0);
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);

  for (p=0; p<n_mp_points; p++) {
    MPntStd  *marker_std;
    long int np = 1,found;
    long int ep[] = {-1};
    double   xip[] = {0.0,0.0,0.0};

    DataFieldAccessPoint(PField_std,p,(void**)&marker_std);

    /* locate point */
    switch (method) {
      case 0:
        PointLocation_BruteForce(mesh,np,(const double*)marker_std->coor,ep,xip,&found);
        break;
      case 1:
        PointLocation_PartitionedBoundingBox(mesh,np,(const double*)marker_std->coor,ep,xip,&found);
        break;
      default:
        PointLocation_PartitionedBoundingBox(mesh,np,(const double*)marker_std->coor,ep,xip,&found);
        break;
    }
    /* assign marker phase */
    marker_std->phase = region_idx[ep[0]];
  }
  DataFieldRestoreAccess(PField_std);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetRegionIndexFromGMSH(pTatinCtx ptatin, GMSHCtx *data)
{
  Mesh           mesh;
  long int       *region_idx = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* get user mesh from file */
  parse_mesh(data->mesh_file,&mesh);
  if (!mesh) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"mesh = NULL. Aborting.\n"); }
  /* get region index from file */
  parse_field(mesh,data->region_file,'c',(void**)&region_idx,NULL);
  
  /* use dave's library to assign material points region index */
  ierr = SetRegionIndexFromMesh(ptatin,mesh,region_idx,data->method);CHKERRQ(ierr);

  free(region_idx);
  MeshDestroy(&mesh);
  PetscFunctionReturn(0);
}

static PetscErrorCode MarkBoundaryFacetFromMesh(
  MeshEntity e, 
  MeshFacetInfo fi,
  Mesh mesh,
  PetscInt method)
{
  PetscErrorCode ierr;
  PetscInt *facet_to_keep,nmarked=0,f;
  Facet cell_facet;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  
  if (e->type != MESH_ENTITY_FACET) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid for MESH_ENTITY_FACET");

  if (e->dm != fi->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid if DMs refer to the same object");

  ierr = MeshFacetInfoGetCoords(fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);

  ierr = PetscMalloc1(fi->n_facets,&facet_to_keep);CHKERRQ(ierr);
  for (f=0; f<fi->n_facets; f++) {
    long int np = 1,found;
    long int ep[] = {-1};
    double   xip[] = {0.0,0.0,0.0};
    
    /* pack data */
    ierr = FacetPack(cell_facet, f, fi);CHKERRQ(ierr);
    
    switch (method) {
      case 0:
        PointLocation_BruteForce(mesh,np,(const double*)cell_facet->centroid,ep,xip,&found);
        break;
      case 1:
        PointLocation_PartitionedBoundingBox(mesh,np,(const double*)cell_facet->centroid,ep,xip,&found);
        break;
      default:
        PointLocation_PartitionedBoundingBox(mesh,np,(const double*)cell_facet->centroid,ep,xip,&found);
        break;
    }

    /*
    if (found == 0) {
      PetscPrintf(PETSC_COMM_WORLD,"Point[%d]: ( %1.4e, %1.4e, %1.4e ) not found!\n",f,cell_facet->centroid[0],cell_facet->centroid[1],cell_facet->centroid[2]);
    }
    */
    if (found == 0) continue;
    /* select if the point is found */
    facet_to_keep[nmarked] = f;
    nmarked++;
  }
  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = MeshFacetInfoRestoreCoords(fi);CHKERRQ(ierr);
  
  ierr = MeshEntitySetValues(e,nmarked,(const PetscInt*)facet_to_keep);CHKERRQ(ierr);
  
  ierr = PetscFree(facet_to_keep);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode MarkFacetsFromGMSH(GMSHCtx *data)
{
  PetscInt       sf;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  for (sf=0; sf<data->n_bcfaces; sf++) {
    Mesh       mesh;
    MeshEntity mesh_entity;
    char       meshfile[PETSC_MAX_PATH_LEN],opt_name[PETSC_MAX_PATH_LEN];

    PetscPrintf(PETSC_COMM_WORLD,"Processing sc[%d]: %s, tag = [%d]\n",sf,data->sc[sf]->name,data->tag_table[sf]);

    /* read the facets mesh corresponding to tag */
    ierr = PetscSNPrintf(meshfile,PETSC_MAX_PATH_LEN-1,"facet_%d_mesh.bin",data->tag_table[sf]);CHKERRQ(ierr);
    ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-facet_mesh_file_%d",data->tag_table[sf]);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,NULL,opt_name,meshfile,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);

    /* get facet mesh */
    parse_mesh(meshfile,&mesh);

    ierr = SurfaceConstraintGetFacets(data->sc[sf],&mesh_entity);CHKERRQ(ierr);
    ierr = MarkBoundaryFacetFromMesh(mesh_entity,data->sc[sf]->fi,mesh,1);CHKERRQ(ierr);

    MeshDestroy(&mesh);;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode SurfaceConstaintCreateFromOptions(pTatinCtx ptatin, PetscBool insert_if_not_found, GMSHCtx *data)
{
  PhysCompStokes stokes;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);

  for (f=0; f<data->n_bcfaces; f++) {
    PetscInt  tag,sc_type;
    char      opt_name[PETSC_MAX_PATH_LEN],sc_name[PETSC_MAX_PATH_LEN];
    PetscBool found;

    tag = data->tag_table[f];

    ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-sc_name_%d",tag);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,NULL,opt_name,sc_name,PETSC_MAX_PATH_LEN-1,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Providing a name to -sc_name_%d is mandatory!",data->tag_table[f]); }

    ierr = PetscSNPrintf(opt_name,PETSC_MAX_PATH_LEN-1,"-sc_type_%d",tag);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,opt_name,&sc_type,&found);CHKERRQ(ierr);
    if (!found) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Providing a type to -sc_type_%d is mandatory!",data->tag_table[f]); }

    ierr = SurfBCListGetConstraint(stokes->surf_bclist,sc_name,&data->sc[f]);CHKERRQ(ierr);
    if (!data->sc[f]) {
      if (insert_if_not_found) {
        ierr = SurfBCListAddConstraint(stokes->surf_bclist,sc_name,&data->sc[f]);CHKERRQ(ierr);
        ierr = SurfaceConstraintSetType(data->sc[f],sc_type);CHKERRQ(ierr);
      } else { 
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found"); 
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelInitialize(pTatinCtx ptatin, GMSHCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ptatin->mx = ptatin->my = ptatin->mz = 16;

  data->O[0] = data->O[1] = data->O[2] = 0.0;
  data->L[0] = 600.0e3;
  data->L[1] = -250.0e3;
  data->L[2] = 300.0e3;

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-O",data->O,&nn,&found);CHKERRQ(ierr);
  if (found) { if (nn != 3) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -O. Found %d",nn); } }

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-L",data->L,&nn,&found);CHKERRQ(ierr);
  if (found) { if (nn != 3) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -L. Found %d",nn); } }

  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&ptatin->mx,NULL);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&ptatin->my,NULL);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mz",&ptatin->mz,NULL);

  ierr = PetscSNPrintf(data->mesh_file,PETSC_MAX_PATH_LEN-1,"md.bin");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-mesh_file",data->mesh_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(data->region_file,PETSC_MAX_PATH_LEN-1,"region_cell.bin");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-region_file",data->region_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);

  /* Create boundaries data */
  ierr = PetscOptionsGetInt(NULL,NULL,"-n_bc_subfaces",&data->n_bcfaces,&found);CHKERRQ(ierr);
  if (!found) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Option -n_bc_subfaces not found!\n"); }
  ierr = PetscCalloc1(data->n_bcfaces,&data->tag_table);CHKERRQ(ierr);
  ierr = PetscCalloc1(data->n_bcfaces,&data->sc);

  /* get the number of subfaces and their tag correspondance */
  nn = data->n_bcfaces;
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-bc_tag_list",data->tag_table,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != data->n_bcfaces) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"n_bc_subfaces (%d) and the number of entries in bc_tag_list (%d) mismatch!\n",data->n_bcfaces,nn);
    }
  }

  data->method = 0;
  ierr = PetscOptionsGetInt(NULL,NULL,"-method",&data->method,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode OutputSurfaceConstraint(pTatinCtx ptatin, GMSHCtx *data)
{
  PhysCompStokes    stokes;
  PetscInt          nsc;
  char              root[PETSC_MAX_PATH_LEN];
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s/",ptatin->outputpath,ptatin->step);CHKERRQ(ierr);

  for (nsc=0; nsc<data->n_bcfaces; nsc++) {
    ierr = SurfaceConstraintViewParaview(data->sc[nsc],root,data->sc[nsc]->name);CHKERRQ(ierr);
  }

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
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

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
  DM              multipys_pack,dav;
  Vec             X,F;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  
  ierr = pTatin3dCreateContext(&ptatin);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(ptatin);CHKERRQ(ierr);

  ierr = CreateGMSHCtx(&data);CHKERRQ(ierr);
  ierr = ModelInitialize(ptatin,data);CHKERRQ(ierr);

  /* Generate physics modules */
  ierr = pTatin3d_PhysCompStokesCreate(ptatin);CHKERRQ(ierr);
  /* Pack all physics together */
  /* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
  ierr = PetscObjectReference((PetscObject)ptatin->stokes_ctx->stokes_pack);CHKERRQ(ierr);
  ptatin->pack = ptatin->stokes_ctx->stokes_pack;

  /* fetch some local variables */
  multipys_pack = ptatin->pack;
  dav           = ptatin->stokes_ctx->dav;

  ierr = DMDASetUniformCoordinates(dav,data->O[0],data->L[0],data->O[1],data->L[1],data->O[2],data->L[2]);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = pTatinPhysCompAttachData_Stokes(ptatin,X);CHKERRQ(ierr);
  
  ierr = pTatin3dCreateMaterialPoints(ptatin,dav);CHKERRQ(ierr);
  /* interpolate material point coordinates (needed if mesh was modified) */
  ierr = MaterialPointCoordinateSetUp(ptatin,dav);CHKERRQ(ierr);

  // ASSIGN PHASE FROM MESH
  ierr = SetRegionIndexFromGMSH(ptatin,data);CHKERRQ(ierr);

  ierr = PhysCompStokesUpdateSurfaceQuadratureGeometry(ptatin->stokes_ctx);CHKERRQ(ierr);
  ierr = SurfaceQuadratureViewParaview_Stokes(ptatin->stokes_ctx,ptatin->outputpath,"def");CHKERRQ(ierr);

  ierr = SurfaceConstaintCreateFromOptions(ptatin,PETSC_TRUE,data);CHKERRQ(ierr);

  /* 
  Would call BC function
  Check if required to call Mark each time BC function is called, maybe once per step, maybe once for all ?
  Lagrangian models will fail if called at each time step, likely that it will be called only once for all 
  and it would also be better for memory management to only do the process once and clean up everything related 
  to external data;
  */
  ierr = MarkFacetsFromGMSH(data);CHKERRQ(ierr);

  /* output */
  ierr = OutputMarkers(ptatin,"MPStd");CHKERRQ(ierr);
  ierr = OutputSurfaceConstraint(ptatin,data);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
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