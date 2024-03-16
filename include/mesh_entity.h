
#ifndef __ptatin3d_mesh_entity_h__
#define __ptatin3d_mesh_entity_h__

#include <petsc.h>
#include <petscdm.h>
#include <element_type_Q2.h>
#include <parse.h>
#include <point_in_tetra.h>

typedef enum { MESH_ENTITY_CELL=0, MESH_ENTITY_FACET, MESH_ENTITY_VERTEX } MeshEntityType;

typedef struct _p_MeshEntity *MeshEntity;

struct _p_MeshEntity {
  PetscInt       range_index[2];
  PetscInt       n_entities,n_entities_global;
  PetscInt       *local_index;
  char           *name;
  MeshEntityType type;
  PetscInt       ref_cnt;
  PetscBool      empty,set_values_called;
  DM             dm;
};


PetscErrorCode MeshEntityView(MeshEntity e);
PetscErrorCode MeshEntityViewer(MeshEntity e,PetscViewer v);
PetscErrorCode MeshEntityCreate(MeshEntity *_e);
PetscErrorCode MeshEntityDestroy(MeshEntity *_e);
PetscErrorCode MeshEntityIncrementRef(MeshEntity e);
PetscErrorCode MeshEntitySetName(MeshEntity e, const char name[]);
PetscErrorCode MeshEntityReset(MeshEntity e);
PetscErrorCode MeshEntitySetRange(MeshEntity e,const PetscInt range[]);
PetscErrorCode MeshEntitySetDM(MeshEntity e,DM dm);
PetscErrorCode MeshEntitySetValues(MeshEntity e,PetscInt len, const PetscInt vals[]);


typedef struct _p_MeshFacetInfo *MeshFacetInfo;

struct _p_MeshFacetInfo {
  DM        dm;
  Vec       coor;
  const PetscReal *_mesh_coor;
  PetscInt  n_facets;
  PetscInt  *facet_cell_index;
  PetscInt  n_facet_labels;
  PetscInt  *facet_label;
  PetscInt  *facet_label_offset;
  PetscInt  ref_cnt;
  PetscBool setup;
  ConformingElementFamily element;
};

typedef struct _p_Facet *Facet;

struct _p_Facet {
  PetscInt index;
  PetscInt label;
  PetscInt cell_index;
  PetscReal cell_vertices[4 * 3];
  //PetscReal cell_coords[9 * 3];
  PetscReal centroid[3],centroid_normal[3],centroid_tangent1[3],centroid_tangent2[3];
};

PetscErrorCode MeshFacetInfoIncrementRef(MeshFacetInfo e);
PetscErrorCode MeshFacetInfoCreate(MeshFacetInfo *_e);
PetscErrorCode MeshFacetInfoDestroy(MeshFacetInfo *_e);
PetscErrorCode MeshFacetInfoSetUp(MeshFacetInfo e, DM dm);
PetscErrorCode MeshFacetInfoCreate2(DM dm,MeshFacetInfo *_e);
PetscErrorCode MeshFacetInfoGetCoords(MeshFacetInfo e);
PetscErrorCode MeshFacetInfoRestoreCoords(MeshFacetInfo e);

PetscErrorCode FacetCreate(Facet *_f);
PetscErrorCode FacetPack(Facet f, PetscInt index, MeshFacetInfo fi);
PetscErrorCode FacetDestroy(Facet *_f);

PetscErrorCode MeshFacetCreate(const char name[], DM dm, MeshEntity *e);
PetscErrorCode MeshFacetDestroy(MeshEntity *e);

typedef struct {
  PetscInt n_domain_faces;
  HexElementFace domain_face[30];
  PetscBool (*mark)(Facet,void*);
  void *user_data;
} MarkDomainFaceContext;

PetscErrorCode MeshFacetMark(MeshEntity e, MeshFacetInfo fi, PetscBool (*mark)(Facet,void*), void *data);
PetscErrorCode MeshFacetMarkDomainFaces(MeshEntity e, MeshFacetInfo fi,PetscInt nsides, HexElementFace sides[]);
PetscErrorCode MeshFacetMarkDomainFaceSubset(
                                             MeshEntity e, MeshFacetInfo fi,
                                             PetscInt nsides, HexElementFace sides[],
                                             PetscBool (*mark)(Facet,void*), void *data);

PetscErrorCode MarkDomainFaceContextInit(MarkDomainFaceContext *ctx);
PetscErrorCode MeshFacetMarkByBoundary(MeshEntity e, MeshFacetInfo fi, PetscBool (*mark)(Facet,void*), void *data);


PetscErrorCode MeshEntityViewPV(PetscInt n,MeshEntity m[]);

typedef struct {
  PetscBool greater;
  PetscInt  dim;
  PetscReal x;
} MarkFromPointCtx;

PetscBool MarkFacetsFromPoint(Facet facets, void *ctx);
PetscErrorCode MeshFacetMarkFromMesh(MeshEntity e, MeshFacetInfo fi, Mesh mesh, PetscInt method);
#endif
