
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
  SC_NITSCHE_CUSTOM_SLIP
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

/* function pointer typedefs for implementation specific setters */
typedef PetscErrorCode (*SurfCSetValuesGeneric)(void*);
typedef PetscErrorCode (*SurfCSetValuesTraction)(Facet,const PetscReal*,PetscReal*,void*); /* <in> coor[3] : <out> traction[3] */
typedef PetscErrorCode (*SurfCSetValuesNitscheDirichlet)(Facet,const PetscReal*,PetscReal*,void*); /* <in> coor[3] : <out> velocity[3] */
typedef PetscErrorCode (*SurfCSetValuesNitscheNavierSlip)(Facet,const PetscReal*,PetscReal*,void*); /* <in> coor[3] : <out> velocity[1] */
typedef PetscErrorCode (*SurfCSetValuesNitscheCustomSlip)(Facet,const PetscReal*,PetscReal*,PetscReal*,void*); /* <in> coor[3] : <out> velocity[1], normal_hat[3] */


/* Test type by assignment and compile time errors */
#define SURFC_CHKSETVALS_SC_NONE(setter)
#define SURFC_CHKSETVALS_SC_TRACTION(setter)            { SurfCSetValuesTraction _v1_ = (setter); _v1_ = NULL; }
#define SURFC_CHKSETVALS_SC_FSSA(setter)
#define SURFC_CHKSETVALS_SC_NITSCHE_DIRICHLET(setter)   { SurfCSetValuesNitscheDirichlet _v1_ = (setter); _v1_ = NULL; }
#define SURFC_CHKSETVALS_SC_NITSCHE_NAVIER_SLIP(setter) { SurfCSetValuesNitscheNavierSlip _v1_ = (setter); _v1_ = NULL; }
#define SURFC_CHKSETVALS_SC_NITSCHE_CUSTOM_SLIP(setter) { SurfCSetValuesNitscheCustomSlip _v1_ = (setter); _v1_ = NULL; }

#define SURFC_CHKSETVALS(type, setter) SURFC_CHKSETVALS_##type((setter))


PetscErrorCode SurfaceConstraintSetValues_TRACTION           (SurfaceConstraint sc,SurfCSetValuesTraction set,         void *data);
PetscErrorCode SurfaceConstraintSetValues_NITSCHE_DIRICHLET  (SurfaceConstraint sc,SurfCSetValuesNitscheDirichlet set, void *data);
PetscErrorCode SurfaceConstraintSetValues_NITSCHE_NAVIER_SLIP(SurfaceConstraint sc,SurfCSetValuesNitscheNavierSlip set,void *data);
PetscErrorCode SurfaceConstraintSetValues_NITSCHE_CUSTOM_SLIP(SurfaceConstraint sc,SurfCSetValuesNitscheCustomSlip set,void *data);


PetscErrorCode SurfaceConstraintSetValues(SurfaceConstraint sc,
                                          SurfCSetValuesGeneric set,
                                          void *data);

PetscErrorCode _SurfaceConstraintViewParaviewVTU(SurfaceConstraint sc,const char name[]);
PetscErrorCode SurfaceConstraintViewParaview(SurfaceConstraint sc, const char path[], const char prefix[]);

#endif
