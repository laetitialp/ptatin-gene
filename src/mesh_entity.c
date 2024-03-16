
#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>
#include <element_type_Q2.h>
#include <dmda_element_q2p1.h>
#include <mesh_entity.h>

//typedef enum { MESH_ENTITY_CELL=0, MESH_ENTITY_FACET, MESH_ENTITY_VERTEX } MeshEntityType;
const char *MeshEntityTypeNames[] = { "cell", "facet", "vertex", 0 };

PetscErrorCode MeshEntityView(MeshEntity e)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)e->dm);
  if (e->name) { PetscPrintf(comm,"MeshEntity: %s\n",e->name); }
  else { PetscPrintf(comm,"MeshEntity:\n"); }
  PetscPrintf(comm,"  type: %s\n",MeshEntityTypeNames[(PetscInt)e->type]);
  PetscPrintf(comm,"  n_entities: %D (global)\n",e->n_entities_global);
  PetscPrintf(comm,"  range: [ %D , %D )\n",e->range_index[0],e->range_index[1]);
  PetscPrintf(comm,"  empty?: %D\n",(PetscInt)e->empty);
  PetscPrintf(comm,"  set_values_called?: %D\n",(PetscInt)e->set_values_called);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntityViewer(MeshEntity e,PetscViewer v)
{
  if (e->name) { PetscViewerASCIIPrintf(v,"MeshEntity: %s\n",e->name); }
  else { PetscViewerASCIIPrintf(v,"MeshEntity:\n"); }
  PetscViewerASCIIPushTab(v);
  PetscViewerASCIIPrintf(v,"type: %s\n",MeshEntityTypeNames[(PetscInt)e->type]);
  PetscViewerASCIIPrintf(v,"n_entities: %D (global)\n",e->n_entities_global);
  PetscViewerASCIIPrintf(v,"range: [ %D , %D )\n",e->range_index[0],e->range_index[1]);
  PetscViewerASCIIPrintf(v,"empty?: %D\n",(PetscInt)e->empty);
  PetscViewerASCIIPrintf(v,"set_values_called?: %D\n",(PetscInt)e->set_values_called);
  PetscViewerASCIIPopTab(v);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntityCreate(MeshEntity *_e)
{
  MeshEntity     e;
  PetscErrorCode ierr;
  
  ierr = PetscMalloc(sizeof(struct _p_MeshEntity),&e);CHKERRQ(ierr);
  ierr = PetscMemzero(e,sizeof(struct _p_MeshEntity));CHKERRQ(ierr);
  e->n_entities = 0;
  e->empty = PETSC_TRUE;
  e->set_values_called = PETSC_FALSE;
  e->ref_cnt = 1;
  *_e = e;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntityDestroy(MeshEntity *_e)
{
  MeshEntity     e;
  PetscErrorCode ierr;
  
  if (!_e) PetscFunctionReturn(0);
  e = *_e;
  if (!e) PetscFunctionReturn(0);
  e->ref_cnt--;
  if (e->ref_cnt > 0) PetscFunctionReturn(0);
  
  ierr = PetscFree(e->local_index);CHKERRQ(ierr);
  ierr = PetscFree(e->name);CHKERRQ(ierr);
  ierr = PetscFree(e);CHKERRQ(ierr);
  *_e = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntityIncrementRef(MeshEntity e)
{
  e->ref_cnt++;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntitySetName(MeshEntity e, const char name[])
{
  PetscErrorCode ierr;
  ierr = PetscFree(e->name);CHKERRQ(ierr);
  e->name = NULL;
  if (name) {
    ierr = PetscStrallocpy(name,&e->name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntityReset(MeshEntity e)
{
  PetscErrorCode ierr;
  ierr = PetscFree(e->local_index);CHKERRQ(ierr);
  e->local_index = NULL;
  e->n_entities = 0;
  e->empty = PETSC_TRUE;
  e->set_values_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntitySetRange(MeshEntity e,const PetscInt range[])
{
  e->range_index[0] = range[0];
  e->range_index[1] = range[1];
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntitySetDM(MeshEntity e,DM dm)
{
  e->dm = dm;
  PetscFunctionReturn(0);
}

/* Collective on e->dm */
PetscErrorCode MeshEntitySetValues(MeshEntity e,PetscInt len, const PetscInt vals[])
{
  InsertMode mode = INSERT_VALUES;
  PetscInt i,nnew;
  PetscErrorCode ierr;
  
  /* count */
  nnew = 0;
  for (i=0; i<len; i++) {
    if ((vals[i] >= e->range_index[0]) && (vals[i] < e->range_index[1])) { nnew++; }
  }
  
  if (mode == INSERT_VALUES) {
    ierr = MeshEntityReset(e);CHKERRQ(ierr);
    ierr = PetscMalloc1(nnew,&e->local_index);CHKERRQ(ierr);
    nnew = 0;
    for (i=0; i<len; i++) {
      if ((vals[i] >= e->range_index[0]) && (vals[i] < e->range_index[1])) {
        e->local_index[nnew] = vals[i];
        nnew++;
      }
    }
    e->n_entities = nnew;
  }
  
  e->empty = PETSC_FALSE;
  if (e->n_entities == 0) {
    e->empty = PETSC_TRUE;
  }
  e->n_entities_global = e->n_entities;
  ierr = MPI_Allreduce(MPI_IN_PLACE,&e->n_entities_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)e->dm));CHKERRQ(ierr);
  e->set_values_called = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Facet support */
PetscErrorCode FacetCreate(Facet *_f)
{
  Facet  f;
  PetscErrorCode ierr;
  //ierr = PetscMalloc1(1,&f);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(struct _p_Facet),&f);CHKERRQ(ierr);
  ierr = PetscMemzero(f,sizeof(struct _p_Facet));CHKERRQ(ierr);

  *_f = f;
  PetscFunctionReturn(0);
}

PetscErrorCode FacetDestroy(Facet *_f)
{
  Facet  f;
  PetscErrorCode ierr;
  
  if (!_f) PetscFunctionReturn(0);
  f = *_f;
  ierr = PetscFree(f);CHKERRQ(ierr);
  *_f = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshFacetInfoIncrementRef(MeshFacetInfo e)
{
  e->ref_cnt++;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshFacetInfoCreate(MeshFacetInfo *_e)
{
  MeshFacetInfo  e;
  PetscErrorCode ierr;
  
  //ierr = PetscMalloc1(1,&e);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(struct _p_MeshFacetInfo),&e);CHKERRQ(ierr);
  ierr = PetscMemzero(e,sizeof(struct _p_MeshFacetInfo));CHKERRQ(ierr);

  ElementTypeCreate_Q2(&e->element,3);
  
  e->ref_cnt = 1;
  *_e = e;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshFacetInfoDestroy(MeshFacetInfo *_e)
{
  MeshFacetInfo  e;
  PetscErrorCode ierr;
  
  if (!_e) PetscFunctionReturn(0);
  e = *_e;
  if (!e) PetscFunctionReturn(0);
  e->ref_cnt--;
  if (e->ref_cnt > 0) PetscFunctionReturn(0);

  // dm left hanging
  ierr = PetscFree(e->facet_cell_index);CHKERRQ(ierr);
  ierr = PetscFree(e->facet_label);CHKERRQ(ierr);
  ierr = PetscFree(e->facet_label_offset);CHKERRQ(ierr);
  
  ElementTypeDestroy_Q2(&e->element);
  
  ierr = PetscFree(e);CHKERRQ(ierr);
  *_e = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode _MeshFacetInfoSetUp(MeshFacetInfo e)
{
  PetscErrorCode ierr;
  PetscInt eli,elj,elk;
  PetscInt si,sj,sk,ni,nj,nk,M,N,P,lmx,lmy,lmz;
  PetscInt cnt,elidx;
  
  PetscFunctionBegin;
  
  ierr = DMDAGetInfo(e->dm,NULL,&M,&N,&P, NULL,NULL,NULL,NULL, NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(e->dm,&si,&sj,&sk,&ni,&nj,&nk);CHKERRQ(ierr);
  ierr = DMDAGetLocalSizeElementQ2(e->dm,&lmx,&lmy,&lmz);CHKERRQ(ierr);

  cnt = 0;

  //case HEX_FACE_Pxi:
  if (si+ni == M) {
  eli = lmx - 1;
  for (elk=0; elk<lmz; elk++) {
    for (elj=0; elj<lmy; elj++) {
      elidx = eli + elj * lmx + elk * lmx*lmy;
      e->facet_cell_index[ cnt ] = elidx;
      e->facet_label[cnt] = (PetscInt)HEX_FACE_Pxi;
      cnt++;
    }
  }}

  
  //case HEX_FACE_Nxi:
  if (si == 0) {
  eli = 0;
  for (elk=0; elk<lmz; elk++) {
    for (elj=0; elj<lmy; elj++) {
      elidx = eli + elj * lmx + elk * lmx*lmy;
      e->facet_cell_index[ cnt ] = elidx;
      e->facet_label[cnt] = (PetscInt)HEX_FACE_Nxi;
      cnt++;
    }
  }}

  
  //case HEX_FACE_Peta:
  if (sj+nj == N) {
  elj = lmy - 1;
  for (elk=0; elk<lmz; elk++) {
    for (eli=0; eli<lmx; eli++) {
      elidx = eli + elj * lmx + elk * lmx*lmy;
      e->facet_cell_index[ cnt ] = elidx;
      e->facet_label[cnt] = (PetscInt)HEX_FACE_Peta;
      cnt++;
    }
  }}
  
  
  //case HEX_FACE_Neta:
  if (sj == 0) {
  elj = 0;
  for (elk=0; elk<lmz; elk++) {
    for (eli=0; eli<lmx; eli++) {
      elidx = eli + elj * lmx + elk * lmx*lmy;
      e->facet_cell_index[ cnt ] = elidx;
      e->facet_label[cnt] = (PetscInt)HEX_FACE_Neta;
      cnt++;
    }
  }}

  
  //case HEX_FACE_Pzeta:
  if (sk+nk == P) {
  elk = lmz - 1;
  for (elj=0; elj<lmy; elj++) {
    for (eli=0; eli<lmx; eli++) {
      elidx = eli + elj * lmx + elk * lmx*lmy;
      e->facet_cell_index[ cnt ] = elidx;
      e->facet_label[cnt] = (PetscInt)HEX_FACE_Pzeta;
      cnt++;
    }
  }}
  
  
  //case HEX_FACE_Nzeta:
  if (sk == 0) {
  elk = 0;
  for (elj=0; elj<lmy; elj++) {
    for (eli=0; eli<lmx; eli++) {
      elidx = eli + elj * lmx + elk * lmx*lmy;
      e->facet_cell_index[ cnt ] = elidx;
      e->facet_label[cnt] = (PetscInt)HEX_FACE_Nzeta;
      cnt++;
    }
  }}
  
  PetscFunctionReturn(0);
}

PetscErrorCode MeshFacetInfoSetUp(MeshFacetInfo e, DM dm)
{
  PetscErrorCode ierr;
  PetscInt nface_list[HEX_EDGES];
  PetscInt lmx,lmy,lmz,M,N,P,si,sj,sk,ni,nj,nk,k;
  
  if (e->setup) PetscFunctionReturn(0);
  e->dm = dm;

  ierr = DMDAGetInfo(e->dm,NULL,&M,&N,&P, NULL,NULL,NULL,NULL, NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(e->dm,&si,&sj,&sk,&ni,&nj,&nk);CHKERRQ(ierr);
  ierr = DMDAGetLocalSizeElementQ2(e->dm,&lmx,&lmy,&lmz);CHKERRQ(ierr);
  
  nface_list[0] = nface_list[1] = 0;
  nface_list[2] = nface_list[3] = 0;
  nface_list[4] = nface_list[5] = 0;
  if (si+ni == M) { nface_list[HEX_FACE_Pxi]   = lmy*lmz; }
  if (si == 0)    { nface_list[HEX_FACE_Nxi]   = lmy*lmz; }
  if (sj+nj == N) { nface_list[HEX_FACE_Peta]  = lmx*lmz; }
  if (sj == 0)    { nface_list[HEX_FACE_Neta]  = lmx*lmz; }
  if (sk+nk == P) { nface_list[HEX_FACE_Pzeta] = lmx*lmy; }
  if (sk == 0)    { nface_list[HEX_FACE_Nzeta] = lmx*lmy; }
  e->n_facets = 0;
  for (k=0; k<6; k++) {
    e->n_facets += nface_list[k];
  }

  e->n_facet_labels = 6; /* hex has six faces */
  ierr = PetscMalloc1(6+1,&e->facet_label_offset);CHKERRQ(ierr); /* six faces of hex */
  e->facet_label_offset[0] = 0;
  for (k=0; k<6; k++) {
    e->facet_label_offset[k+1] = e->facet_label_offset[k] + nface_list[k];
  }
  /*
  for (k=0; k<6; k++) {
    printf("f-s %d | f-e %d\n",e->facet_label_offset[k],e->facet_label_offset[k+1]);
  }
  */
  ierr = PetscMalloc1(e->n_facets,&e->facet_cell_index);CHKERRQ(ierr);
  ierr = PetscMalloc1(e->n_facets,&e->facet_label);CHKERRQ(ierr);
  
  ierr = _MeshFacetInfoSetUp(e);CHKERRQ(ierr);
  
  e->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshFacetInfoCreate2(DM dm,MeshFacetInfo *_e)
{
  PetscErrorCode ierr;
  MeshFacetInfo e;
  ierr = MeshFacetInfoCreate(&e);CHKERRQ(ierr);
  ierr = MeshFacetInfoSetUp(e,dm);CHKERRQ(ierr);
  *_e = e;
  PetscFunctionReturn(0);
}

/* Collective on e->dm */
PetscErrorCode MeshFacetInfoGetCoords(MeshFacetInfo e)
{
  PetscErrorCode ierr;
  if (!e->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MeshFacetInfoSetUp() first");
  
  ierr = DMGetCoordinatesLocal(e->dm,&e->coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(e->coor,&e->_mesh_coor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* Collective on e->dm */
PetscErrorCode MeshFacetInfoRestoreCoords(MeshFacetInfo e)
{
  PetscErrorCode ierr;
  if (!e->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MeshFacetInfoSetUp() first");
  if (!e->_mesh_coor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MeshFacetInfoGetCoords() first");
  
  ierr = VecRestoreArrayRead(e->coor,&e->_mesh_coor);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MeshFacetCreate(const char name[], DM dm, MeshEntity *e)
{
  PetscErrorCode ierr;
  
  ierr = MeshEntityCreate(e);CHKERRQ(ierr);
  (*e)->type = MESH_ENTITY_FACET;
  ierr = MeshEntitySetName(*e,name);CHKERRQ(ierr);
  
  ierr = MeshEntitySetDM(*e,dm);CHKERRQ(ierr);
  {
    PetscInt range[2],nfacets;
    
    ierr = DMDAGetLocalSizeFacetQ2(dm,&nfacets);CHKERRQ(ierr);
    range[0] = 0;
    range[1] = nfacets;
    ierr = MeshEntitySetRange(*e,(const PetscInt*)range);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode MeshFacetDestroy(MeshEntity *e)
{
  PetscErrorCode ierr;
  ierr = MeshEntityDestroy(e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode FacetPack(Facet f, PetscInt index, MeshFacetInfo fi)
{
  PetscInt k,d;
  const PetscInt *elnidx;
  PetscInt       nel,nen;
  double         elcoords[3*Q2_NODES_PER_EL_3D];
  QPoint2d       qp2;
  PetscErrorCode ierr;

  PetscInt facets[6][4] = {
    { 2,8, 20,26 }, // int fid_px[] =
    { 6,0, 24,18 }, // int fid_mx[] =
    { 8,6, 26,24 }, // int fid_pe[] =
    { 0,2, 18,20 }, // int fid_me[] =
    { 18,20, 24,26 }, // int fid_pz[] =
    { 2,0, 8,6 } }; // int fid_mz[] =

  ConformingElementFamily element = fi->element;

  f->index = index; /* facet local index */
  f->label = fi->facet_label[index]; /* side label */
  f->cell_index = fi->facet_cell_index[index];
  
  ierr = DMDAGetElements_pTatinQ2P1(fi->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);
  ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*f->cell_index],(PetscReal*)fi->_mesh_coor);CHKERRQ(ierr);

  //cell_vertices[4 * 3];
  for (k=0; k<4; k++) {
    PetscInt vid = facets[f->label][k];

    for (d=0; d<3; d++) {
      f->cell_vertices[3*k + d] = elcoords[3*vid + d];
    }
  }

  ////cell_coords[9 * 3];
  
  //centroid[3]
  {
    for (d=0; d<3; d++) {
      f->centroid[d] = 0.0;
      for (k=0; k<4; k++) {
        f->centroid[d] += f->cell_vertices[3*k + d];
      }
      f->centroid[d] *= 0.25; /* 4 verts / face */
    }
  }
  
  //centroid_normal[3], centroid_tangent1[3], centroid_tangent2[3];
  qp2.xi = 0.0;
  qp2.eta = 0.0;
  
  element->compute_surface_normal_3D(
                                     element,
                                     elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                     f->label,   // edge index 0,1,2,3,4,5,6,7 //
                                     &qp2, // should contain 1 point with dimension 2 (xi,eta)   //
                                     f->centroid_normal ); // normal[] contains 1 point with dimension 3 (x,y,z) //
  element->compute_surface_tangents_3D(
                                       element,
                                       elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                       f->label,
                                       &qp2, // should contain 1 point with dimension 2 (xi,eta)   //
                                       f->centroid_tangent1,f->centroid_tangent2 ); // t1[],t2[] contains 1 point with dimension 3 (x,y,z) //

  PetscFunctionReturn(0);
}

/* Collective on e->dm */
PetscErrorCode MeshFacetMark(MeshEntity e, MeshFacetInfo fi, PetscBool (*mark)(Facet,void*), void *data)
{
  PetscErrorCode ierr;
  PetscInt *facet_to_keep,nmarked=0,f;
  PetscBool selected;
  Facet cell_facet;
  
  if (e->type != MESH_ENTITY_FACET) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid for MESH_ENTITY_FACET");

  if (e->dm != fi->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid if DMs refer to the same object");

  ierr = MeshFacetInfoGetCoords(fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(fi->n_facets,&facet_to_keep);CHKERRQ(ierr);
  for (f=0; f<fi->n_facets; f++) {
    selected = PETSC_FALSE;
    
    /* pack data */
    ierr = FacetPack(cell_facet, f, fi);CHKERRQ(ierr);
    
    /* user select */
    selected = mark(cell_facet, data);
    if (selected) {
      facet_to_keep[nmarked] = f;
      nmarked++;
    }
  }
  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = MeshFacetInfoRestoreCoords(fi);CHKERRQ(ierr);
  
  ierr = MeshEntitySetValues(e,nmarked,(const PetscInt*)facet_to_keep);CHKERRQ(ierr);
  
  ierr = PetscFree(facet_to_keep);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* Collective on e->dm */
PetscErrorCode MeshFacetMarkDomainFaces(
                      MeshEntity e, MeshFacetInfo fi,
                      PetscInt nsides, HexElementFace sides[])
{
  PetscErrorCode ierr;
  PetscInt *facet_to_keep,nmarked=0,f,k,f_start,f_end;
  PetscInt side_mark[HEX_EDGES];
  
  if (e->type != MESH_ENTITY_FACET) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid for MESH_ENTITY_FACET");
  
  if (e->dm != fi->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid if DMs refer to the same object");

  /* check side list does not contain duplicates */
  for (k=0; k<HEX_EDGES; k++) {
    side_mark[k] = 0;
  }
  for (k=0; k<nsides; k++) {
    side_mark[ sides[k] ]++;
  }
  for (k=0; k<HEX_EDGES; k++) {
    if (side_mark[k] > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"sides[] must not contain duplicate entries");
  }
  
  ierr = PetscMalloc1(fi->n_facets,&facet_to_keep);CHKERRQ(ierr);
  for (k=0; k<nsides; k++) {
    f_start = fi->facet_label_offset[ sides[k] ];
    f_end   = fi->facet_label_offset[ sides[k]+1 ];
    for (f=f_start; f<f_end; f++) {
      facet_to_keep[nmarked] = f;
      nmarked++;
    }
  }

  ierr = MeshEntitySetValues(e,nmarked,(const PetscInt*)facet_to_keep);CHKERRQ(ierr);
  
  ierr = PetscFree(facet_to_keep);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* Collective on e->dm */
PetscErrorCode MeshFacetMarkDomainFaceSubset(
                                        MeshEntity e, MeshFacetInfo fi,
                                        PetscInt nsides, HexElementFace sides[],
                                        PetscBool (*mark)(Facet,void*), void *data)
{
  PetscErrorCode ierr;
  PetscInt *facet_to_keep,nmarked=0,f,k,f_start,f_end;
  PetscInt side_mark[HEX_EDGES];
  Facet cell_facet;
  
  if (e->type != MESH_ENTITY_FACET) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid for MESH_ENTITY_FACET");
  
  if (e->dm != fi->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid if DMs refer to the same object");
  
  /* check side list does not contain duplicates */
  for (k=0; k<HEX_EDGES; k++) {
    side_mark[k] = 0;
  }
  for (k=0; k<nsides; k++) {
    side_mark[ sides[k] ]++;
  }
  for (k=0; k<HEX_EDGES; k++) {
    if (side_mark[k] > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"sides[] must not contain duplicate entries");
  }
  
  ierr = MeshFacetInfoGetCoords(fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(fi->n_facets,&facet_to_keep);CHKERRQ(ierr);
  nmarked = 0;
  for (k=0; k<nsides; k++) {
    f_start = fi->facet_label_offset[ sides[k] ];
    f_end   = fi->facet_label_offset[ sides[k]+1 ];
    
    for (f=f_start; f<f_end; f++) {
      PetscBool selected = PETSC_FALSE;
      
      /* pack data */
      ierr = FacetPack(cell_facet, f, fi);CHKERRQ(ierr);
      
      /* user select */
      selected = mark(cell_facet, data);
      if (selected) {
        facet_to_keep[nmarked] = f;
        nmarked++;
      }
    }
  }
  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = MeshFacetInfoRestoreCoords(fi);CHKERRQ(ierr);

  ierr = MeshEntitySetValues(e,nmarked,(const PetscInt*)facet_to_keep);CHKERRQ(ierr);
  
  ierr = PetscFree(facet_to_keep);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 
  MeshFacetMarkByBoundary() provides the same API as the general setter MeshFacetMark().
 
  Usage:
 
  MarkDomainFaceContext ctx;
  MarkDomainFaceContextInit(&ctx);
  n_domain_faces = 3;
  ctx.domain_face[0] = HEX_FACE_Pxi;
  ctx.domain_face[1] = HEX_FACE_Nxi;
  ctx.domain_face[2] = HEX_FACE_Peta;
 
  ctx.mark = user_code_to_select_facet; // This will result in MeshFacetMarkDomainFaceSubset() being called.
  ctx.user_data = (void*)pointer_to_any_user_data; // only used if ctx.mark != NULL
  ctx.mark = NULL; // This will result in MeshFacetMarkDomainFaces() being called.
 
*/
PetscErrorCode MarkDomainFaceContextInit(MarkDomainFaceContext *ctx)
{
  PetscErrorCode ierr;
  ctx->n_domain_faces = 0;
  ierr = PetscMemzero(ctx->domain_face,30*sizeof(HexElementFace));
  ctx->mark = NULL;
  ctx->user_data = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MeshFacetMarkByBoundary(MeshEntity e, MeshFacetInfo fi, PetscBool (*mark)(Facet,void*), void *data)
{
  PetscErrorCode ierr;
  MarkDomainFaceContext *ctx = NULL;
  
  if (!data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Require non-NULL data. Expected pointer to MarkDomainFaceContext");
  ctx = (MarkDomainFaceContext*)data;

  if (mark) {
    PetscPrintf(PETSC_COMM_SELF,"[Warning] MeshFacetMarkBoundary: over-riding method ctx->mark() with provided method mark()");
    ctx->mark = mark;
  }

  if (ctx->mark) {
    ierr = MeshFacetMarkDomainFaceSubset(e,fi,ctx->n_domain_faces,ctx->domain_face,ctx->mark,ctx->user_data);CHKERRQ(ierr);
  } else {
    ierr = MeshFacetMarkDomainFaces(e,fi,ctx->n_domain_faces,ctx->domain_face);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Basic MeshEntity viewer */
PetscErrorCode _MeshEntityView_Facet(MeshEntity me, PetscInt tag, PetscInt fileindex)
{
  PetscErrorCode ierr;
  MeshFacetInfo fi;
  int facets[6][4] = {
    { 2,8, 20,26 }, // int fid_px[] =
    { 6,0, 24,18 }, // int fid_mx[] =
    { 8,6, 26,24 }, // int fid_pe[] =
    { 0,2, 18,20 }, // int fid_me[] =
    { 18,20, 24,26 }, // int fid_pz[] =
    { 2,0, 8,6 } }; // int fid_mz[] =
  char filename[PETSC_MAX_PATH_LEN];
  FILE* fp = NULL;
  PetscInt npoints,ncells,f,k,c;

  DM             cda;
  Vec            gcoords;
  PetscScalar    *LA_gcoords;
  double         elcoords[3*Q2_NODES_PER_EL_3D];
  const PetscInt *elnidx;
  PetscInt       nel,nen;
  PetscMPIInt    rank;
  
  ierr = MeshFacetInfoCreate(&fi);CHKERRQ(ierr);
  ierr = MeshFacetInfoSetUp(fi,me->dm);CHKERRQ(ierr);

  ncells = me->n_entities;
  npoints = 4 * ncells;
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(me->dm,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(me->dm,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(me->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)me->dm),&rank);CHKERRQ(ierr);
  if (!me->name) {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"facets-index%d-r%1.5d.vtu",fileindex,rank);
  } else {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"facets-%s-r%1.5d.vtu",me->name,rank);
  }
  
  if ((fp = fopen(filename,"w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open new VTU file %s",filename);
  }

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  fprintf(fp, "  <UnstructuredGrid>\n");
  fprintf(fp, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\" >\n",npoints,ncells);

  fprintf(fp, "    <Points>\n");
  fprintf(fp, "      <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (f=0; f<me->n_entities; f++) {
    PetscInt element_index;
    PetscInt facet_label;
    PetscInt vtk_facet[4];
    
    element_index = fi->facet_cell_index[ me->local_index[f] ];
    
    facet_label = fi->facet_label[ me->local_index[f] ];
    
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*element_index],LA_gcoords);CHKERRQ(ierr);

    /* vtk wants the vertices numbered like [0, 1, 3, 2] --> counter clockwise */
    vtk_facet[0] = facets[facet_label][0];
    vtk_facet[1] = facets[facet_label][1];
    vtk_facet[2] = facets[facet_label][3];
    vtk_facet[3] = facets[facet_label][2];
    for (k=0; k<4; k++) {
      PetscInt vid = vtk_facet[k];
      fprintf(fp, "      %1.4e %1.4e %1.4e \n",elcoords[3*vid+0],elcoords[3*vid+1],elcoords[3*vid+2]);
    }
  }
  fprintf(fp, "      </DataArray>\n");
  fprintf(fp, "    </Points>\n");
  
  /* POINT-DATA HEADER - OPEN */
  fprintf(fp, "    <CellData>\n");
  fprintf(fp, "      <DataArray type=\"Int32\" Name=\"tag\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (f=0; f<me->n_entities; f++) {
    fprintf(fp, "      %d \n",(int)tag);
  }
  fprintf(fp, "      </DataArray>\n");
  /* POINT-DATA HEADER - CLOSE */
  fprintf(fp, "    </CellData>\n");

  /* UNSTRUCTURED GRID DATA */
  fprintf(fp, "    <Cells>\n");
  
  // connectivity //
  fprintf(fp, "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<ncells; c++) {
    fprintf(fp,"%d %d %d %d ", 4*c,4*c+1,4*c+2,4*c+3);
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");
  
  // offsets //
  fprintf(fp, "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<ncells; c++) {
    fprintf(fp,"%d ", 4*c + 4);
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");
  
  // types //
  fprintf(fp, "      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<ncells; c++) {
    fprintf(fp,"%d ", 9); // VTK_QUAD (=9)
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");
  
  fprintf(fp, "    </Cells>\n");

  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </UnstructuredGrid>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose(fp);
  
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = MeshFacetInfoDestroy(&fi);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode MeshEntityViewPV(PetscInt n,MeshEntity m[])
{
  PetscErrorCode ierr;
  PetscInt k,tag;
  MeshEntityType type;

  type = m[0]->type;
  for (k=1; k<n; k++) {
    if (type != m[k]->type) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"All types must be the same");
  }

  switch (type) {
    case MESH_ENTITY_FACET:
      for (k=0; k<n; k++) {
        tag = k;
        ierr = _MeshEntityView_Facet(m[k], tag, k);CHKERRQ(ierr);
      }
      break;

    case MESH_ENTITY_CELL:
      break;
      
    case MESH_ENTITY_VERTEX:
      break;
      
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of MESH_ENTITY_FACET, MESH_ENTITY_CELL, MESH_ENTITY_VERTEX");
      break;
  }
  
  PetscFunctionReturn(0);
}

PetscBool MarkFacetsFromPoint(Facet facets, void *ctx)
{
  MarkFromPointCtx *data = (MarkFromPointCtx*)ctx;
  PetscBool impose = PETSC_FALSE;
  PetscFunctionBegin;
  
  /* Select the entire cell based on its centroid coordinate */
  if (data->greater) {
    if (facets->centroid[ data->dim ] >= data->x) { impose = PETSC_TRUE; }
  } else {
    if (facets->centroid[ data->dim ] <= data->x) { impose = PETSC_TRUE; }
  }

  PetscFunctionReturn(impose);
}

PetscErrorCode MeshFacetMarkFromMesh(MeshEntity e, MeshFacetInfo fi, Mesh mesh, PetscInt method, PetscReal length_scale)
{
  PetscInt       f,nmarked=0;
  PetscInt       *facet_to_keep;
  Facet          cell_facet;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  if (e->type != MESH_ENTITY_FACET) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid for MESH_ENTITY_FACET");
  if (e->dm != fi->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid if DMs refer to the same object");

  ierr = MeshFacetInfoGetCoords(fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);

  ierr = PetscMalloc1(fi->n_facets,&facet_to_keep);CHKERRQ(ierr);
  for (f=0; f<fi->n_facets; f++) {
    int      d;
    long int np = 1,found;
    long int ep[] = {-1};
    double   xip[] = {0.0,0.0};
    double   cell_centroid[3];
    
    /* pack data */
    ierr = FacetPack(cell_facet, f, fi);CHKERRQ(ierr);

    /* scale for user mesh length scale */
    for (d=0; d<3; d++) {
      cell_centroid[d] = cell_facet->centroid[d] * length_scale;
    }

    switch (method) {
      case 0:
        PointLocation_BruteForce_Triangles(mesh,np,(const double*)cell_centroid,ep,xip,&found);
        break;
      case 1:
        PointLocation_PartitionedBoundingBox_Triangles(mesh,np,(const double*)cell_centroid,ep,xip,&found);
        break;
      default:
        PointLocation_PartitionedBoundingBox_Triangles(mesh,np,(const double*)cell_centroid,ep,xip,&found);
        break;
    }
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