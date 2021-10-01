
#ifndef __fvda_h__
#define __fvda_h__

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

//#define FVDA_DEBUG

#define DACELL1D_Q1_SIZE    2

#define DACELL2D_Q1_SIZE    4
#define DACELL2D_VERTS      4
#define DACELL2D_FACE_VERTS 2
#define DACELL2D_NFACES     4

#define DACELL3D_Q1_SIZE    8
#define DACELL3D_VERTS      8
#define DACELL3D_FACE_VERTS 4
#define DACELL3D_NFACES     6

#define E_MINUS_OFF_RANK -2
#define CELL_GHOST       -1
#define CELL_OFF_RANK    -2

/*
 Do not ever change the order of the entries in this enum.
 The result returned from
   DACellGeometry2d_GetFaceIndices()
 and
   DACellGeometry2d_GetFaceIndices()
 implicitly assume the order in the enum.
*/
typedef enum {
  DACELL_FACE_W=0,
  DACELL_FACE_E,
  DACELL_FACE_S,
  DACELL_FACE_N,
  DACELL_FACE_B,
  DACELL_FACE_F
} DACellFace;

typedef enum {
  DAFACE_BOUNDARY=0,
  DAFACE_INTERIOR,
  DAFACE_SUB_DOMAIN_BOUNDARY
} DACellFaceLocation;

typedef enum {
  FVFLUX_UN_INITIALIZED=0,
  FVFLUX_IN_FLUX,
  FVFLUX_OUT_FLUX,
  FVFLUX_DIRICHLET_CONSTRAINT,
  FVFLUX_NEUMANN_CONSTRAINT,
  FVFLUX_NATIVE
} FVFluxType;

typedef enum {
  FVDA_HYPERBOLIC=0,
  FVDA_ELLIPTIC,
  FVDA_PARABOLIC
} FVDAPDEType;

typedef enum {
  FVPRIMITIVE_CELL=0,
  FVPRIMITIVE_FACE,
  FVPRIMITIVE_VERTEX
} FVPrimitiveType;

typedef struct _p_FVDA *FVDA;
typedef struct _p_FVALE *FVALE;
typedef struct _p_FVTD *FVTD;


PetscErrorCode FVDACreate(MPI_Comm comm,FVDA *_fv);
PetscErrorCode FVDASetDimension(FVDA fv,PetscInt dim);
PetscErrorCode FVDASetSizes(FVDA fv,const PetscInt mi[],const PetscInt Mi[]);
PetscErrorCode FVDASetUp(FVDA fv);
PetscErrorCode FVDACreate2d(MPI_Comm comm,PetscInt Mi[],FVDA *_fv);
PetscErrorCode FVDACreate3d(MPI_Comm comm,PetscInt Mi[],FVDA *_fv);
PetscErrorCode FVDACreateFromDMDA(DM vertex_layout,FVDA *_fv);
PetscErrorCode FVDASetGeometryDM(FVDA fv,DM dm);
PetscErrorCode FVDAGetGeometryDM(FVDA fv,DM *dm);
PetscErrorCode FVDAGetDM(FVDA fv,DM *dm);
PetscErrorCode FVDAGetGeometryCoordinates(FVDA fv,Vec *c);
PetscErrorCode FVDAUpdateGeometry(FVDA fv);
PetscErrorCode FVDASetProblemType(FVDA fv,PetscBool Qdot,FVDAPDEType equation_type,PetscInt numerical_flux,PetscInt reconstruction);
PetscErrorCode FVDADestroy(FVDA *_fv);

PetscErrorCode FVDAGetBoundaryFaceIndicesRead(FVDA fv,DACellFace face,PetscInt *len,const PetscInt *indices[]);
PetscErrorCode FVDAGetBoundaryFaceIndicesOwnershipRange(FVDA fv,DACellFace face,PetscInt *start,PetscInt *end);

PetscErrorCode FVDAGetFaceInfo(FVDA fv,PetscInt *nfaces,const DACellFaceLocation *l[],const DACellFace *f[],const PetscReal *n[],const PetscReal *c[]);

PetscErrorCode DACellGeometry2d_GetCoordinates(const PetscInt element[],const PetscReal mesh_coor[],PetscReal coor[]);
PetscErrorCode DACellGeometry3d_GetCoordinates(const PetscInt element[],const PetscReal mesh_coor[],PetscReal coor[]);

PetscErrorCode _EvaluateFaceNormal3d(DACellFace face,const PetscReal coor[],const PetscReal xi0[],PetscReal n0[]);
PetscErrorCode _EvaluateFaceCoord3d(DACellFace face,const PetscReal coor[],const PetscReal xi0[],PetscReal c0[]);

PetscErrorCode FVDAFaceIterator(FVDA fv,DACellFace face,PetscBool require_normals,PetscReal time,
                                PetscErrorCode (*user_setter)(FVDA,
                                                              DACellFace,
                                                              PetscInt,
                                                              const PetscReal*,
                                                              const PetscReal*,
                                                              const PetscInt*,
                                                              PetscReal,FVFluxType*,PetscReal*,void*),
                                void *data);

PetscErrorCode eval_F(SNES snes,Vec X,Vec F,void *data);
PetscErrorCode eval_J(SNES snes,Vec X,Mat Ja,Mat Jb,void *data);

PetscErrorCode eval_F_hr(SNES snes,Vec X,Vec F,void *data);

PetscErrorCode FVDAGetValidElement(FVDA fv,PetscInt faceid,PetscInt *cellid);

PetscErrorCode _EvaluateCellVolume3d(const PetscReal coor[],PetscReal *v);
PetscErrorCode _EvaluateFaceArea3d(DACellFace face,const PetscReal coor[],PetscReal *a);

PetscErrorCode FVDAVecTraverse(FVDA fv,Vec X,PetscReal time,PetscInt dof,
                               PetscBool user_setter(PetscScalar*,PetscScalar*,void*),
                               void *data);

PetscErrorCode _cart_convert_index_to_ijk(PetscInt r,const PetscInt mp[],PetscInt rijk[]);
PetscErrorCode _cart_convert_ijk_to_index(const PetscInt rijk[],const PetscInt mp[],PetscInt *r);

PetscErrorCode FVDACreateMatrix(FVDA fv,DMDAStencilType type,Mat *A);
PetscErrorCode SNESFVDAConfigureGalerkinMG(SNES snes,FVDA fv);

PetscErrorCode eval_F_diffusion_7point_hr_local_store_MPI(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[]);
PetscErrorCode eval_J_diffusion_7point_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],Mat J);

#endif
