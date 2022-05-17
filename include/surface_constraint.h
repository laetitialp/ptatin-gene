
#ifndef __ptatin3d_surface_constraint_h__
#define __ptatin3d_surface_constraint_h__

#include <petsc.h>
#include <petscdm.h>
#include <data_bucket.h>
#include <quadrature.h>
#include <mesh_entity.h>

typedef enum {
  SC_NONE = 0,
  SC_TRACTION,
  SC_FSSA,
  SC_NITSCHE_DIRICHLET,
  SC_NITSCHE_NAVIER_SLIP,
  SC_NITSCHE_A_NAVIER_SLIP
} SurfaceConstraintType;


typedef struct _p_SurfaceConstraint *SurfaceConstraint;

struct _SurfaceConstraintOps {
  PetscErrorCode (*setup)(SurfaceConstraint);
  PetscErrorCode (*destroy)(SurfaceConstraint);
  
  PetscErrorCode (*residual_F)(SurfaceConstraint,DM,const PetscScalar*,DM,const PetscScalar*,PetscScalar*); /* not sure we actually need this - possibly remove */
  PetscErrorCode (*residual_Fu)(SurfaceConstraint,DM,const PetscScalar*,DM,const PetscScalar*,PetscScalar*);
  PetscErrorCode (*residual_Fp)(SurfaceConstraint,DM,const PetscScalar*,DM,const PetscScalar*,PetscScalar*);
  
  PetscErrorCode (*action_A)(void);
  PetscErrorCode (*asmb_A)(void);
  PetscErrorCode (*diag_A)(void);
  
  PetscErrorCode (*action_Auu)(void);
  PetscErrorCode (*asmb_Auu)(void);
  PetscErrorCode (*diag_Auu)(void);
  
  PetscErrorCode (*action_Aup)(void);
  PetscErrorCode (*asmb_Aup)(void);
  
  PetscErrorCode (*action_Apu)(void);
  PetscErrorCode (*asmb_Apu)(void);
};

struct _p_SurfaceConstraint {
  SurfaceConstraintType type;
  MeshEntity            facets;
  PetscBool             linear;
  MeshFacetInfo         fi;
  DM                    dm;
  PetscInt              nqp_facet;
  DataBucket            properties_db;
  PetscBool             setup;
  struct _SurfaceConstraintOps ops;
  void                         *data; /* for implementations */
  SurfaceQuadrature     quadrature;
  PetscErrorCode        (*user_set_values)(void*);
  void                  *user_data_set_values;
  PetscErrorCode        (*user_mark_facets)(void*);
  void                  *user_data_mark_facets;
  char                  *name;
};


PetscErrorCode SurfaceConstraintCreate(SurfaceConstraint *_sc);
PetscErrorCode SurfaceConstraintDestroy(SurfaceConstraint *_sc);
PetscErrorCode SurfaceConstraintSetDM(SurfaceConstraint sc, DM dm);
PetscErrorCode SurfaceConstraintCreateWithFacetInfo(MeshFacetInfo mfi,SurfaceConstraint *_sc);
PetscErrorCode SurfaceConstraintViewer(SurfaceConstraint sc,PetscViewer v);
PetscErrorCode SurfaceConstraintReset(SurfaceConstraint sc);
PetscErrorCode SurfaceConstraintSetName(SurfaceConstraint sc, const char name[]);
PetscErrorCode SurfaceConstraintSetType(SurfaceConstraint sc, SurfaceConstraintType type);
PetscErrorCode SurfaceConstraintSetQuadrature(SurfaceConstraint sc, SurfaceQuadrature q);
PetscErrorCode SurfaceConstraintGetFacets(SurfaceConstraint sc, MeshEntity *f);

PetscErrorCode SurfaceConstraintDuplicate(SurfaceConstraint sc, MeshFacetInfo mfi, SurfaceQuadrature surfQ, SurfaceConstraint *_dup);
PetscErrorCode SurfaceConstraintDuplicateOperatorA11(SurfaceConstraint sc, MeshFacetInfo mfi, SurfaceQuadrature surfQ, SurfaceConstraint *_dup);

PetscErrorCode SurfaceConstraintSetResidualOnly(SurfaceConstraint sc);
PetscErrorCode SurfaceConstraintSetOperatorOnly(SurfaceConstraint sc);

typedef PetscErrorCode (*SurfConstraintSetter)(void*);
typedef PetscErrorCode (*SurfConstraintSetTraction)(Facet,const PetscReal*,PetscReal*,void*);


PetscErrorCode SurfaceConstraintSetValues_TRACTION(SurfaceConstraint sc,
                                                   PetscErrorCode (*set)(Facet,const PetscReal*,PetscReal*,void*),
                                                   void *data);


PetscErrorCode SurfaceConstraintSetValues(SurfaceConstraint sc,
                                          SurfConstraintSetter set,
                                          void *data);

PetscErrorCode _SurfaceConstraintViewParaviewVTU(SurfaceConstraint sc,const char name[]);
PetscErrorCode SurfaceConstraintViewParaview(SurfaceConstraint sc, const char path[], const char prefix[]);

#endif
