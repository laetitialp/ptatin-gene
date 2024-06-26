
#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>
#include <element_type_Q2.h>
#include <dmda_element_q2p1.h>
#include <mesh_entity.h>
#include <ptatin3d_defs.h>
#include <ptatin3d.h>
#include <private/ptatin_impl.h>
#include <element_utils_q2.h>
#include <quadrature.h>
#include <private/quadrature_impl.h>
#include <ptatin3d_stokes.h>

#include <surface_constraint.h>
#include <sc_generic.h>

/*
typedef enum {
  SC_NONE = 1,
  SC_TRACTION,
  SC_FSSA,
  SC_NITSCHE_DIRICHLET,
  SC_NITSCHE_NAVIER_SLIP,
  SC_NITSCHE_A_NAVIER_SLIP
} SurfaceConstraintType;
*/
const char *SurfaceConstraintTypeNames[] = { "none", "traction", "demo", "fssa", "nitsche_dirichlet", "nitsche_navier_slip", "nitsche_generalized_navier_slip", 0 };

static PetscErrorCode _ops_residual_only(SurfaceConstraint sc);
static PetscErrorCode _ops_operator_only(SurfaceConstraint sc);

static PetscErrorCode _SetType_NONE(SurfaceConstraint sc);
static PetscErrorCode _SetType_TRACTION(SurfaceConstraint sc);
static PetscErrorCode _SetType_FSSA(SurfaceConstraint sc);
PetscErrorCode _SetType_DEMO(SurfaceConstraint sc);

PetscErrorCode _SetType_NITSCHE_DIRICHLET(SurfaceConstraint sc);
PetscErrorCode SurfaceConstraintSetValues_NITSCHE_DIRICHLET(SurfaceConstraint sc,
                                                            SurfCSetValuesNitscheDirichlet set,
                                                            void *data);

PetscErrorCode _SetType_NITSCHE_NAVIER_SLIP(SurfaceConstraint sc);
PetscErrorCode SurfaceConstraintSetValues_NITSCHE_NAVIER_SLIP(SurfaceConstraint sc,
                                                              SurfCSetValuesNitscheNavierSlip set,
                                                              void *data);

PetscErrorCode _SetType_NITSCHE_GENERAL_SLIP(SurfaceConstraint sc);
PetscErrorCode SurfaceConstraintSetValues_NITSCHE_GENERAL_SLIP(SurfaceConstraint sc,
                                                              SurfCSetValuesNitscheGeneralSlip set,
                                                              void *data);

PetscErrorCode SurfaceConstraintCreate(SurfaceConstraint *_sc)
{
  SurfaceConstraint sc;
  PetscErrorCode    ierr;
  
  ierr = PetscMalloc(sizeof(struct _p_SurfaceConstraint),&sc);CHKERRQ(ierr);
  ierr = PetscMemzero(sc,sizeof(struct _p_SurfaceConstraint));CHKERRQ(ierr);
  
  DataBucketCreate(&sc->properties_db);

  ierr = _SetType_NONE(sc);CHKERRQ(ierr);
  ierr = SurfaceConstraintSetName(sc,"default");CHKERRQ(ierr);

  *_sc = sc;

  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintDestroy(SurfaceConstraint *_sc)
{
  SurfaceConstraint sc;
  PetscErrorCode    ierr;
  
  if (!_sc) PetscFunctionReturn(0);
  sc = *_sc;
  if (!sc) PetscFunctionReturn(0);

  if (sc->ops.destroy) { ierr = sc->ops.destroy(sc);CHKERRQ(ierr); }
  sc->data = NULL;
  ierr = MeshFacetInfoDestroy(&sc->fi);CHKERRQ(ierr);
  ierr = MeshFacetDestroy(&sc->facets);CHKERRQ(ierr);
  DataBucketDestroy(&sc->properties_db);
  ierr = PetscFree(sc->name);CHKERRQ(ierr);
  
  ierr = PetscFree(sc);CHKERRQ(ierr);
  *_sc = NULL;
  
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintCreateWithFacetInfo(MeshFacetInfo mfi,SurfaceConstraint *_sc)
{
  PetscErrorCode    ierr;
  SurfaceConstraint sc;
  
  ierr = SurfaceConstraintCreate(&sc);CHKERRQ(ierr);
  ierr = MeshFacetInfoIncrementRef(mfi);CHKERRQ(ierr);
  sc->fi = mfi;
  sc->dm = mfi->dm;
  
  // build facet info for the domain
  //ierr = MeshFacetInfoCreate2(sc->dm,&sc->fi);CHKERRQ(ierr);
  ierr = MeshFacetInfoIncrementRef(mfi);CHKERRQ(ierr);
  if (sc->fi) { ierr = MeshFacetInfoDestroy(&sc->fi);CHKERRQ(ierr); }
  sc->fi = mfi;
  
  // create empty facet list named "default"
  ierr = MeshFacetCreate("default",sc->dm,&sc->facets);CHKERRQ(ierr);
  
  sc->nqp_facet = 9; /* Really should choose this a better way. Query fi->element ?? */
  *_sc = sc;
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetDM(SurfaceConstraint sc, DM dm)
{
  PetscErrorCode ierr;
  
  if (sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"DM is already set");
  if (sc->fi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"MeshFacetInfo is already set");
  sc->dm = dm;
  
  // build facet info for the domain
  ierr = MeshFacetInfoCreate(&sc->fi);CHKERRQ(ierr);
  ierr = MeshFacetInfoSetUp(sc->fi,dm);CHKERRQ(ierr);

  // create empty facet list named "default"
  ierr = MeshFacetCreate("default",dm,&sc->facets);CHKERRQ(ierr);

  sc->nqp_facet = 9; /* Really should choose this a better way. Query fi->element ?? */
  
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetName(SurfaceConstraint sc, const char name[])
{
  PetscErrorCode ierr;
  ierr = PetscFree(sc->name);CHKERRQ(ierr);
  sc->name = NULL;
  if (name) {
    ierr = PetscStrallocpy(name,&sc->name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintReset(SurfaceConstraint sc)
{
  PetscErrorCode ierr;
  ierr = MeshEntityReset(sc->facets);CHKERRQ(ierr);
  DataBucketSetSizes(sc->properties_db,0,-1);
  sc->user_set_values = NULL;
  sc->user_data_set_values = NULL;
  sc->user_mark_facets = NULL;
  sc->user_data_mark_facets = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintViewer(SurfaceConstraint sc,PetscViewer v)
{
  PetscErrorCode ierr;
  if (sc->name) { PetscViewerASCIIPrintf(v,"SurfaceConstraint: %s\n",sc->name); }
  else { PetscViewerASCIIPrintf(v,"SurfaceConstraint:\n"); }
  PetscViewerASCIIPushTab(v);
  PetscViewerASCIIPrintf(v,"type: %s\n",SurfaceConstraintTypeNames[(PetscInt)sc->type]);
  ierr = MeshEntityViewer(sc->facets,v);CHKERRQ(ierr);
  PetscViewerASCIIPopTab(v);
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetOperatorOnly(SurfaceConstraint sc)
{
  PetscErrorCode ierr;
  switch (sc->type) {
    case SC_NONE:
      PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetOperatorOnly: operator_only = true has no effect with type SC_NONE as it does not support operators");
      break;
    case SC_TRACTION:
      PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetOperatorOnly: operator_only = true has no effect with type SC_TRACTION as it does not support operators");
      break;
    default:
      break;
  }
  ierr = _ops_operator_only(sc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetResidualOnly(SurfaceConstraint sc)
{
  PetscErrorCode ierr;
  switch (sc->type) {
    case SC_NONE:
      PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetResidualOnly: residual_only = true has no effect with type SC_NONE as it does not support operators");
      break;
    case SC_TRACTION:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"residual_only = true not valid with SC_TRACTION as it does not support operators");
      break;
    default:
      break;
  }
  ierr = _ops_residual_only(sc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetType(SurfaceConstraint sc, SurfaceConstraintType type)
{
  PetscErrorCode ierr;

  if (sc->type != SC_NONE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"SurfaceConstraint type already set");
  /*
  if (type == sc->type) {
    ierr = SurfaceConstraintReset(sc);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (type != sc->type && sc->type != SC_NONE) {
    ierr = SurfaceConstraintReset(sc);CHKERRQ(ierr);
    if (sc->ops.destroy) { ierr = sc->ops.destroy(sc);CHKERRQ(ierr); }
    sc->data = NULL;
    DataBucketDestroy(&sc->properties_db);
    DataBucketCreate(&sc->properties_db);
  }
  */
  sc->type = type;

  switch (type) {
    case SC_NONE:
      ierr = _SetType_NONE(sc);CHKERRQ(ierr);
      break;

    case SC_TRACTION:
      ierr = _SetType_TRACTION(sc);CHKERRQ(ierr);
      break;

    case SC_DEMO:
      ierr = _SetType_DEMO(sc);CHKERRQ(ierr);
      break;

    case SC_FSSA:
      ierr = _SetType_FSSA(sc);CHKERRQ(ierr);
      break;

    case SC_NITSCHE_DIRICHLET:
      ierr = _SetType_NITSCHE_DIRICHLET(sc);CHKERRQ(ierr);
      break;

    case SC_NITSCHE_NAVIER_SLIP:
      ierr = _SetType_NITSCHE_NAVIER_SLIP(sc);CHKERRQ(ierr);
      break;

    case SC_NITSCHE_GENERAL_SLIP:
      ierr = _SetType_NITSCHE_GENERAL_SLIP(sc);CHKERRQ(ierr);
      break;

    case SC_DIRICHLET:
      ierr = _SetType_NONE(sc);CHKERRQ(ierr);
      sc->type = SC_DIRICHLET;
      DataBucketFinalize(sc->properties_db);
      break;

    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfaceConstraint type %d not implemented",(int)type);
      break;
  }
  
  DataBucketSetInitialSizes(sc->properties_db,1,1);
  DataBucketSetSizes(sc->properties_db,0,-1);

  PetscFunctionReturn(0);
}


PetscErrorCode SurfaceConstraintSetQuadrature(SurfaceConstraint sc, SurfaceQuadrature q)
{
  sc->quadrature = q;
  PetscFunctionReturn(0);
}

/*
PetscErrorCode SurfaceConstraintSetFacets(SurfaceConstraint sc, MeshEntity facets)
{
  // increment self
  MeshEntityIncrementRef(facets);
  // destroy self
  MeshEntityDestroy(&sc->facets);
  // copy
  sc->facets = facets;
}
*/

PetscErrorCode SurfaceConstraintGetFacets(SurfaceConstraint sc, MeshEntity *f)
{
  PetscFunctionBegin;
  if (f) {
    if (sc) {
      *f = sc->facets;
    } else {
      *f = NULL;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintGetMeshFacetInfo(SurfaceConstraint sc, MeshFacetInfo *fi)
{
  PetscFunctionBegin;
  if (fi) {
    if (sc) {
      *fi = sc->fi;
    } else {
      *fi = NULL;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintDuplicate(SurfaceConstraint sc, MeshFacetInfo mfi, SurfaceQuadrature surfQ, SurfaceConstraint *_dup)
{
  PetscErrorCode ierr;
  SurfaceConstraint dup;
  *_dup = NULL;
  
  ierr = SurfaceConstraintCreateWithFacetInfo(mfi,&dup);CHKERRQ(ierr);
  ierr = SurfaceConstraintSetQuadrature(dup,surfQ);CHKERRQ(ierr);
  if (sc->name) {
    ierr = SurfaceConstraintSetName(dup,sc->name);CHKERRQ(ierr);
  }
  ierr = SurfaceConstraintSetType(dup,sc->type);CHKERRQ(ierr);
  dup->linear = sc->linear;
  *_dup = dup;
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintDuplicateOperatorA11(SurfaceConstraint sc, MeshFacetInfo mfi, SurfaceQuadrature surfQ, SurfaceConstraint *_dup)
{
  PetscErrorCode ierr;
  SurfaceConstraint dup;
  *_dup = NULL;

  ierr = SurfaceConstraintDuplicate(sc,mfi,surfQ,&dup);CHKERRQ(ierr);
  if (sc->ops.action_Auu || sc->ops.asmb_Auu) {
    ierr = _ops_operator_only(dup);CHKERRQ(ierr);
    dup->ops.action_A = NULL;
    
    dup->ops.action_Aup = NULL;
    dup->ops.asmb_Aup   = NULL;
    
    dup->ops.action_Apu = NULL;
    dup->ops.asmb_Apu   = NULL;
    *_dup = dup;
  } else {
    ierr = SurfaceConstraintDestroy(&dup);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


static PetscErrorCode _ops_residual_only(SurfaceConstraint sc)
{
  sc->ops.action_A = NULL;

  sc->ops.action_Auu = NULL;
  sc->ops.asmb_Auu   = NULL;
  sc->ops.diag_Auu   = NULL;

  sc->ops.action_Aup = NULL;
  sc->ops.asmb_Aup   = NULL;

  sc->ops.action_Apu = NULL;
  sc->ops.asmb_Apu   = NULL;
  
  PetscFunctionReturn(0);
}

static PetscErrorCode _ops_operator_only(SurfaceConstraint sc)
{
  sc->ops.residual_F  = NULL;
  sc->ops.residual_Fu = NULL;
  sc->ops.residual_Fp = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode _resize_facet_quadrature_data(SurfaceConstraint sc)
{
  PetscInt nfacets;
  if (!sc->facets->set_values_called) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
  }
  nfacets = sc->facets->n_entities;
  /*
  DataBucketGetSizes(sc->properties_db,&len,NULL,NULL);
  if (len != sc->nqp_facet*nfacets) {
    DataBucketSetSizes(sc->properties_db,sc->nqp_facet*nfacets,-1);
  }
  */
  // db is smart enough not to resize the object if the length did not change
  //DataBucketSetSizes(sc->properties_db,sc->nqp_facet*nfacets,-1);
  // call set initial sizes will memset(0) all data which will help debug issues when set_values has not been called.
  DataBucketSetInitialSizes(sc->properties_db,sc->nqp_facet*nfacets,-1);
  PetscFunctionReturn(0);
}

PetscErrorCode _sc_check_sizes(SurfaceConstraint sc)
{
  PetscErrorCode ierr;
  PetscInt nfacets,nfields,nquadpoints;
  nfacets = sc->facets->n_entities;
  DataBucketGetDataFields(sc->properties_db,&nfields,NULL);
  if (nfields) {
    DataBucketGetSizes(sc->properties_db,&nquadpoints,NULL,NULL);
    if ( (PetscInt)(nquadpoints / sc->nqp_facet) != nfacets) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Must call SurfaceConstraintSetValues(). Found %D facets and %D quadrature points.",nfacets,nquadpoints);
  }
  PetscFunctionReturn(0);
}

/* residual */
PetscErrorCode SurfaceConstraintOps_EvaluateF(SurfaceConstraint sc,
                                              DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],PetscScalar Ru[],
                                              PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.residual_F) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.residual_F(sc,dau,ufield,dap,pfield,Ru);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_EvaluateF[name %s]: residual_F = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintOps_EvaluateFu(SurfaceConstraint sc,
                                              DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],PetscScalar Ru[],
                                              PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.residual_Fu) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.residual_Fu(sc,dau,ufield,dap,pfield,Ru);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_EvaluateF[name %s]: residual_Fu = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintOps_EvaluateFp(SurfaceConstraint sc,
                                              DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],PetscScalar Ru[],
                                              PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.residual_Fp) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.residual_Fp(sc,dau,ufield,dap,pfield,Ru);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_EvaluateF[name %s]: residual_Fp = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

/* action */
PetscErrorCode SurfaceConstraintOps_ActionA(SurfaceConstraint sc,
                                              DM dau,const PetscScalar ufield[],
                                              DM dap,const PetscScalar pfield[],
                                              PetscScalar Yu[],PetscScalar Yp[],
                                              PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.action_A) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.action_A(sc,dau,ufield,dap,pfield,Yu,Yp);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_ActionA[name %s]: action_A = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintOps_ActionA11(SurfaceConstraint sc,
                                            DM dau,const PetscScalar ufield[],
                                            PetscScalar Yu[],
                                            PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.action_Auu) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.action_Auu(sc,dau,ufield,Yu);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_ActionAuu[name %s]: action_Auu = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintOps_ActionA12(SurfaceConstraint sc,
                                            DM dau,
                                            DM dap,const PetscScalar pfield[],
                                            PetscScalar Yu[],
                                            PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.action_Aup) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.action_Aup(sc,dau,dap,pfield,Yu);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_ActionAup[name %s]: action_Aup = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintOps_ActionA21(SurfaceConstraint sc,
                                            DM dau,const PetscScalar ufield[],
                                            DM dap,
                                            PetscScalar Yp[],
                                            PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.action_Apu) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.action_Apu(sc,dau,ufield,dap,Yp);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_ActionA21[name %s]: action_A21 = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

/* assemble */
PetscErrorCode SurfaceConstraintOps_AssembleA11(SurfaceConstraint sc,
                                              DM dau,Mat A,
                                              PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.asmb_Auu) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.asmb_Auu(sc,dau,A);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_AssembleAuu[name %s]: asmb_Auu = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintOps_AssembleA12(SurfaceConstraint sc,
                                              DM dau,
                                              DM dap,
                                              Mat A,
                                              PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.asmb_Aup) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.asmb_Aup(sc,dau,dap,A);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_AssembleAup[name %s]: asmb_Aup = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintOps_AssembleA21(SurfaceConstraint sc,
                                              DM dau,
                                              DM dap,
                                              Mat A,
                                              PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.asmb_Apu) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.asmb_Apu(sc,dau,dap,A);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_AseembleA21[name %s]: asmb_Apu = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}

/* assemble diag */
PetscErrorCode SurfaceConstraintOps_AssembleDiagA11(SurfaceConstraint sc,
                                                DM dau,Vec diagA,
                                                PetscBool error_if_null)
{
  PetscErrorCode ierr;
  if (sc->ops.diag_Auu) {
    ierr = _sc_check_sizes(sc);CHKERRQ(ierr);
    ierr = sc->ops.diag_Auu(sc,dau,diagA);CHKERRQ(ierr);
  } else {
    if (error_if_null) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"SurfaceConstraintOps_DiagAuu[name %s]: diag_Auu = NULL",sc->name);
  }
  PetscFunctionReturn(0);
}




/* NONE */
static PetscErrorCode _SetType_NONE(SurfaceConstraint sc)
{
  /* set methods */
  sc->ops.setup = NULL;
  sc->ops.destroy = NULL;
  /* allocate implementation data */
  sc->data = NULL;
  sc->type = SC_NONE;
  PetscFunctionReturn(0);
}

PetscErrorCode user_traction_set_constant(Facet F,
                                          const PetscReal qp_coor[],
                                          PetscReal traction[],
                                          void *data)
{
  PetscReal *input;
  input = (PetscReal*)data;
  traction[0] = input[0];
  traction[1] = input[1];
  traction[2] = input[2];
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetValues_TRACTION(SurfaceConstraint sc,
                                         SurfCSetValuesTraction set,
                                         void *data)
{
  PetscInt e,facet_index,cell_side,cell_index,q,qp_offset;
  Facet cell_facet;
  PetscReal qp_coor[3],traction[3];
  PetscErrorCode ierr;
  PetscReal *traction_qp;
  double Ni[27];
  const PetscInt *elnidx;
  PetscInt       nel,nen;
  double         elcoords[3*Q2_NODES_PER_EL_3D];


  if (sc->type != SC_TRACTION) {
    PetscPrintf(PetscObjectComm((PetscObject)sc->dm),"[ignoring] SurfaceConstraintSetValues_TRACTION() called with different type on object with name \"%s\"\n",sc->name);
    PetscFunctionReturn(0);
  }
  
  if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
  if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
  if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");

  /* resize traction qp data */
  ierr = _resize_facet_quadrature_data(sc);CHKERRQ(ierr);
  
  DataBucketGetEntriesdByName(sc->properties_db,"traction",(void**)&traction_qp);
  
  ierr = MeshFacetInfoGetCoords(sc->fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(sc->fi->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);
  
  for (e=0; e<sc->facets->n_entities; e++) {
    facet_index = sc->facets->local_index[e]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    //printf("cell_side %d\n",cell_side);
    
    ierr = FacetPack(cell_facet, facet_index, sc->fi);CHKERRQ(ierr);

    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*cell_index],(PetscReal*)sc->fi->_mesh_coor);CHKERRQ(ierr);

    //qp_offset = sc->nqp_facet * facet_index; /* offset into entire domain qp list */
    qp_offset = sc->nqp_facet * e; /* offset into facet qp list */
    for (q=0; q<sc->nqp_facet; q++) {

      {
        PetscInt d,k;
        
        for (d=0; d<3; d++) { qp_coor[d] = 0.0; }
        sc->fi->element->basis_NI_3D(&sc->quadrature->gp3[cell_side][q],Ni);
        for (k=0; k<sc->fi->element->n_nodes_3D; k++) {
          for (d=0; d<3; d++) {
            qp_coor[d] += Ni[k] * elcoords[3*k+d];
          }
        }
      }

      ierr = set(cell_facet, qp_coor, traction, data);CHKERRQ(ierr);
      
      //printf("local fe %d q %d index %d %d %d\n",e,q,3*(qp_offset+q)+0,3*(qp_offset+q)+1,3*(qp_offset+q)+2);
      ierr = PetscMemcpy(&traction_qp[3*(qp_offset+q)],traction,sizeof(PetscReal)*3);CHKERRQ(ierr);
    }
  }
  
  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = MeshFacetInfoRestoreCoords(sc->fi);CHKERRQ(ierr);

  DataBucketRestoreEntriesdByName(sc->properties_db,"traction",(void**)&traction_qp);

  PetscFunctionReturn(0);
}

static PetscErrorCode _FormFunctionLocal_Fu_TRACTION(SurfaceConstraint sc,DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],PetscScalar Ru[])
{
  PetscErrorCode ierr;
  DM cda;
  Vec gcoords;
  const PetscReal *LA_gcoords;
  PetscInt nel=0,nen_u=0,nen_p=0,k,fe,q,nqp=0;
  const PetscInt *elnidx_u;
  const PetscInt *elnidx_p;
  PetscReal elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal elu[3*Q2_NODES_PER_EL_3D],elp[P_BASIS_FUNCTIONS];
  PetscReal Fe[3*Q2_NODES_PER_EL_3D],Be[3*Q2_NODES_PER_EL_2D];
  PetscInt vel_el_lidx[3*U_BASIS_FUNCTIONS];
  PetscReal NIu_surf[NQP][Q2_NODES_PER_EL_2D];
  const PetscReal *domain_traction_qp;
  ConformingElementFamily element = NULL;
  PetscLogDouble t0,t1;
  
  PetscFunctionBegin;
  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);
  
  element = sc->fi->element;

  DataBucketGetEntriesdByName(sc->properties_db,"traction",(void**)&domain_traction_qp);
  
  PetscTime(&t0);
  for (fe=0; fe<sc->facets->n_entities; fe++) {
    PetscInt          facet_index,cell_side,cell_index;
    QPoint2d          *qp2 = NULL;
    const PetscReal   *cell_traction_qp;
    int               *face_local_indices = NULL;
    
    facet_index = sc->facets->local_index[fe]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];

    nqp = sc->quadrature->npoints;
    qp2 = sc->quadrature->gp2[cell_side];

    /* evaluate the quadrature points using the 1D basis for this edge */
    for (q=0; q<nqp; q++) {
      element->basis_NI_2D(&qp2[q],NIu_surf[q]);
    }

    face_local_indices = element->face_node_list[cell_side];

    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*cell_index]);CHKERRQ(ierr);
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*cell_index],(PetscReal*)LA_gcoords);CHKERRQ(ierr);
    ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*cell_index],(PetscReal*)ufield);CHKERRQ(ierr);
    ierr = DMDAGetScalarElementField(elp,nen_p,(PetscInt*)&elnidx_p[nen_p*cell_index],(PetscReal*)pfield);CHKERRQ(ierr);

    /* initialise element stiffness matrix */
    ierr = PetscMemzero(Fe,sizeof(PetscScalar)*Q2_NODES_PER_EL_3D*3);CHKERRQ(ierr);
    ierr = PetscMemzero(Be,sizeof(PetscScalar)*Q2_NODES_PER_EL_2D*3);CHKERRQ(ierr);
    
    cell_traction_qp = &domain_traction_qp[fe * 3 * nqp];
    
    for (q=0; q<nqp; q++) {
      PetscScalar fac,surfJ_q;
      const PetscReal *traction_qp;
      
      element->compute_surface_geometry_3D(
                                           element,
                                           elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                           cell_side,   // edge index 0,...,7 //
                                           &qp2[q], // should contain 1 point with dimension 2 (xi,eta)   //
                                           NULL,NULL,&surfJ_q); // n0[],t0 contains 1 point with dimension 3 (x,y,z) //
      fac = qp2[q].w * surfJ_q;

      traction_qp = &cell_traction_qp[3 * q];

      for (k=0; k<Q2_NODES_PER_EL_2D; k++) {
        Be[3*k  ] += - fac * NIu_surf[q][k] * traction_qp[0];
        Be[3*k+1] += - fac * NIu_surf[q][k] * traction_qp[1];
        Be[3*k+2] += - fac * NIu_surf[q][k] * traction_qp[2];
      }
    }
    
    /* combine body force with A.x */
    for (k=0; k<Q2_NODES_PER_EL_2D; k++) {
      int nidx3d;
      
      /* map 1D index over element edge to 2D element space */
      nidx3d = face_local_indices[k];
      Fe[3*nidx3d  ] = Be[3*k  ];
      Fe[3*nidx3d+1] = Be[3*k+1];
      Fe[3*nidx3d+2] = Be[3*k+2];
    }
    
    ierr = DMDASetValuesLocalStencil_AddValues_Stokes_Velocity(Ru,vel_el_lidx,Fe);CHKERRQ(ierr);
  }
  PetscTime(&t1);
  PetscPrintf(PetscObjectComm((PetscObject)dau),"Assembled int_S N traction dS, = %1.4e (sec)\n",t1-t0);
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"traction",(void**)&domain_traction_qp);
  ierr = VecRestoreArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode _SetType_TRACTION(SurfaceConstraint sc)
{
  /* set methods */
  //sc->ops.setup = _SetUp_TRACTION;
  sc->ops.residual_F  = NULL;
  sc->ops.residual_Fu = _FormFunctionLocal_Fu_TRACTION;
  sc->ops.residual_Fp = NULL;
  /* allocate implementation data */
  /* insert properties into quadrature bucket */
  DataBucketRegisterField(sc->properties_db,"traction",sizeof(PetscReal)*3,NULL);
  DataBucketFinalize(sc->properties_db);
  sc->type = SC_TRACTION;
  PetscFunctionReturn(0);
}

static PetscErrorCode _FormFunctionLocal_Fu_FSSA_diag(SurfaceConstraint sc,DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],PetscScalar Ru[]);
static PetscErrorCode _FormFunctionLocal_Fu_FSSA(SurfaceConstraint sc,DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],PetscScalar Ru[]);
static PetscErrorCode _FormJacobianCell_A11_FSSA_diag(SurfaceConstraint sc,DM dau,Mat A);
static PetscErrorCode _FormJacobianCell_A11_FSSA(SurfaceConstraint sc,DM dau,Mat A);

typedef struct {
  PetscBool use_diag_approximation;
  PetscReal theta;
  PetscReal dt;
} FSSA_Context;

static PetscErrorCode _Destroy_FSSA(SurfaceConstraint sc)
{
  FSSA_Context *ctx;
  PetscErrorCode ierr;
  if (sc->data) {
    ctx = (FSSA_Context*)sc->data;
    ierr = PetscFree(ctx);CHKERRQ(ierr);
    sc->data = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _SetType_FSSA(SurfaceConstraint sc)
{
  FSSA_Context *ctx;
  PetscErrorCode ierr;
  
  /* set methods */
  sc->ops.setup = NULL;
  sc->ops.destroy = _Destroy_FSSA;
  sc->ops.residual_F  = NULL;
  sc->ops.residual_Fu = _FormFunctionLocal_Fu_FSSA_diag;
  sc->ops.asmb_Auu    = _FormJacobianCell_A11_FSSA_diag;
  sc->ops.residual_Fp = NULL;
  /* allocate implementation data */
  ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
  ctx->use_diag_approximation = PETSC_TRUE;
  ctx->theta = 0.5;
  ctx->dt = 0.0;
  sc->data = (void*)ctx;
  /* insert properties into quadrature bucket */
  DataBucketFinalize(sc->properties_db);
  sc->type = SC_FSSA;
 PetscFunctionReturn(0);
}

static PetscErrorCode _FormFunctionLocal_Fu_FSSA_diag(SurfaceConstraint sc,DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],PetscScalar Ru[])
{
  PetscErrorCode ierr;
  PetscFunctionReturn(0);
}

static PetscErrorCode _FormFunctionLocal_Fu_FSSA(SurfaceConstraint sc,DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],PetscScalar Ru[])
{
  PetscErrorCode ierr;
  PetscFunctionReturn(0);
}

static PetscErrorCode _FormJacobianCell_A11_FSSA_diag(SurfaceConstraint sc,DM dau,Mat A)
{
  PetscErrorCode ierr;
  PetscFunctionReturn(0);
}

static PetscErrorCode _FormJacobianCell_A11_FSSA(SurfaceConstraint sc,DM dau,Mat A)
{
  PetscErrorCode ierr;
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintFSSA_SetParams(SurfaceConstraint sc,PetscBool use_diag,PetscReal theta)
{
  FSSA_Context *ctx;
  if (sc->type != SC_FSSA) PetscFunctionReturn(0);
  ctx = (FSSA_Context*)sc->data;
  ctx->use_diag_approximation = use_diag;
  if (ctx->use_diag_approximation) {
    sc->ops.residual_Fu = _FormFunctionLocal_Fu_FSSA_diag;
    sc->ops.asmb_Auu    = _FormJacobianCell_A11_FSSA_diag;
  } else {
    sc->ops.residual_Fu = _FormFunctionLocal_Fu_FSSA;
    sc->ops.asmb_Auu    = _FormJacobianCell_A11_FSSA;
  }
  if (theta > 0) {
    ctx->theta = theta;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintFSSA_SetTimestep(SurfaceConstraint sc,PetscReal dt)
{
  FSSA_Context *ctx;
  if (sc->type != SC_FSSA) PetscFunctionReturn(0);
  ctx = (FSSA_Context*)sc->data;
  ctx->dt = dt;
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetValues(SurfaceConstraint sc,SurfCSetValuesGeneric set,void *data)

{
  PetscErrorCode ierr;
  
  if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
  if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
  if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");

  switch (sc->type) {
    case SC_NONE:
      PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetValues: type NONE does not have a setter");
      break;

    case SC_TRACTION:
      ierr = SurfaceConstraintSetValues_TRACTION(sc, (SurfCSetValuesTraction)set, data);CHKERRQ(ierr);
      break;

    case SC_DEMO:
      PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetValues: type DEMO does not have a setter");
      break;

    case SC_FSSA:
      PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetValues: type FSSA does not have a setter");
      break;

    case SC_NITSCHE_DIRICHLET:
      ierr = SurfaceConstraintSetValues_NITSCHE_DIRICHLET(sc, (SurfCSetValuesNitscheDirichlet)set, data);CHKERRQ(ierr);
      break;

    case SC_NITSCHE_NAVIER_SLIP:
      ierr = SurfaceConstraintSetValues_NITSCHE_NAVIER_SLIP(sc, (SurfCSetValuesNitscheNavierSlip)set, data);CHKERRQ(ierr);
      break;

    case SC_NITSCHE_GENERAL_SLIP:
      ierr = SurfaceConstraintSetValues_NITSCHE_GENERAL_SLIP(sc, (SurfCSetValuesNitscheGeneralSlip)set, data);CHKERRQ(ierr);
      break;

    case SC_DIRICHLET:
      PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetValues: type DIRICHLET does not have a setter");
      break;

    default:
      break;
  }
  PetscFunctionReturn(0);
}





#include <output_paraview.h>


PetscErrorCode _SurfaceConstraintViewParaviewVTU(SurfaceConstraint sc,const char name[])
{
  PetscErrorCode ierr;
  PetscInt fe,n,e,k,ngp,npoints;
  QPntSurfCoefStokes *all_qpoint;
  QPntSurfCoefStokes *cell_qpoint;
  FILE* fp = NULL;
  double xp,yp,zp;
  QPntSurfCoefStokes *qpoint;
  DM             cda;
  Vec            gcoords;
  PetscScalar    *LA_gcoords;
  double         elcoords[3*Q2_NODES_PER_EL_3D];
  double         Ni[27];
  const PetscInt *elnidx;
  PetscInt       nel,nen,nfaces;
  ConformingElementFamily element;
  int            c,npoints32;
  DM da;
  SurfaceQuadrature surfQ;
  MeshFacetInfo mfi;
  
  PetscFunctionBegin;
  if ((fp = fopen(name,"w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name);
  }
  
  surfQ = sc->quadrature;
  mfi = sc->fi;
  da = mfi->dm;
  element = mfi->element;
  
  ngp = surfQ->npoints;
  nfaces = sc->facets->n_entities;
  npoints = nfaces * surfQ->npoints;
  PetscMPIIntCast(npoints,&npoints32);
  
  /* setup for quadrature point properties */
  ierr = SurfaceQuadratureGetAllCellData_Stokes(surfQ,&all_qpoint);CHKERRQ(ierr);
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = DMDAGetElements_pTatinQ2P1(da,&nel,&nen,&elnidx);CHKERRQ(ierr);
  
  /* VTU HEADER - OPEN */
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  fprintf(fp,"  <UnstructuredGrid>\n");
  fprintf(fp,"    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\" >\n",npoints32,npoints32);
  
  /* POINT COORDS */
  fprintf(fp,"    <Points>\n");
  fprintf(fp,"      <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (fe=0; fe<nfaces; fe++) {
    PetscInt face_id,facet_index;

    facet_index = sc->facets->local_index[fe];
    e = mfi->facet_cell_index[facet_index];
    face_id = mfi->facet_label[facet_index];

    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,facet_index,&cell_qpoint);CHKERRQ(ierr);
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*e],LA_gcoords);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);
    
    for (n=0; n<ngp; n++) {
      qpoint = &cell_qpoint[n];
      
      /* interpolate global coords */
      element->basis_NI_3D(&surfQ->gp3[face_id][n],Ni);
      xp = yp = zp = 0.0;
      for (k=0; k<element->n_nodes_3D; k++) {
        xp += Ni[k] * elcoords[3*k  ];
        yp += Ni[k] * elcoords[3*k+1];
        zp += Ni[k] * elcoords[3*k+2];
      }
      
      fprintf(fp,"      %1.4e %1.4e %1.4e \n",xp,yp,zp);
    }
  }
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Points>\n");
  
  /* POINT-DATA HEADER - OPEN */
  fprintf(fp,"    <PointData>\n");
  
  /* POINT-DATA FIELDS */
  
  /* eta/rho */
  fprintf(fp,"      <DataArray type=\"Float32\" Name=\"eta\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (fe=0; fe<nfaces; fe++) {
    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      double field;
      qpoint = &cell_qpoint[n];
      
      QPntSurfCoefStokesGetField_viscosity(qpoint,&field);
      fprintf(fp,"      %1.4e \n",field);
    }
  }
  fprintf(fp,"      </DataArray>\n");
  
  fprintf(fp,"      <DataArray type=\"Float32\" Name=\"rho\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (fe=0; fe<nfaces; fe++) {
    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      double field;
      qpoint = &cell_qpoint[n];
      
      QPntSurfCoefStokesGetField_density(qpoint,&field);
      fprintf(fp,"      %1.4e \n",field );
    }
  }
  fprintf(fp,"      </DataArray>\n");
  
  /* user fields - pretty hack - pretty unsafe */
  {
    int df,nfields;
    DataField *dfields;
    const double *_data;
    
    DataBucketGetDataFields(sc->properties_db,&nfields,&dfields);
    
    for (df=0; df<nfields; df++) {
      int bs,blocksize = dfields[df]->atomic_size / sizeof(double);
      
      for (bs=0; bs<blocksize; bs++) {
        if (blocksize == 1) {
          fprintf(fp,"      <DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"1\" format=\"ascii\">\n",dfields[df]->name);
        } else {
          fprintf(fp,"      <DataArray type=\"Float32\" Name=\"%s_%d\" NumberOfComponents=\"1\" format=\"ascii\">\n",dfields[df]->name,bs);
        }
        
        _data = (const double*)dfields[df]->data;
        for (fe=0; fe<nfaces; fe++) {
          for (n=0; n<ngp; n++) {
            int pidx;
            double field;

            pidx = ngp * fe + n;
            field = _data[blocksize * pidx + bs];
            fprintf(fp,"      %1.4e \n",field);
          }
        }
        fprintf(fp,"      </DataArray>\n");
      }
    }
  }
  
  /* POINT-DATA HEADER - CLOSE */
  fprintf(fp,"    </PointData>\n");
  
  /* UNSTRUCTURED GRID DATA */
  fprintf(fp,"    <Cells>\n");
  
  // connectivity //
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ",c);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  // offsets //
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ",(c+1));
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  // types //
  fprintf(fp,"      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ",1);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  fprintf(fp,"    </Cells>\n");
  
  /* VTU HEADER - CLOSE */
  fprintf(fp,"    </Piece>\n");
  fprintf(fp,"  </UnstructuredGrid>\n");
  fprintf(fp,"</VTKFile>\n");
  
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  fclose(fp);
  PetscFunctionReturn(0);
}


static PetscErrorCode _SurfaceConstraintViewParaviewPVTU(SurfaceConstraint sc,const char prefix[],const char name[])
{
  PetscErrorCode ierr;
  FILE* fp = NULL;
  PetscMPIInt nproc;
  PetscInt i;
  char *sourcename;
  
  PetscFunctionBegin;
  if ((fp = fopen(name,"w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name);
  }
  
  /* PVTU HEADER - OPEN */
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
  fprintf(fp,"<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  /* define size of the nodal mesh based on the cell DM */
  fprintf(fp,"  <PUnstructuredGrid GhostLevel=\"0\">\n" ); /* note overlap = 0 */
  
  /* POINT COORDS */
  fprintf(fp,"    <PPoints>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\"/>\n");
  fprintf(fp,"    </PPoints>\n");
  
  /* CELL-DATA HEADER - OPEN */
  fprintf(fp,"    <PCellData>\n");
  /* CELL-DATA HEADER - CLOSE */
  fprintf(fp,"    </PCellData>\n");
  
  /* POINT-DATA HEADER - OPEN */
  fprintf(fp,"    <PPointData>\n");
  /* POINT-DATA FIELDS */
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"eta\" NumberOfComponents=\"1\"/>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"rho\" NumberOfComponents=\"1\"/>\n");
  
  {
    int df,nfields;
    DataField *dfields;
    
    DataBucketGetDataFields(sc->properties_db,&nfields,&dfields);
    
    for (df=0; df<nfields; df++) {
      int bs,blocksize = dfields[df]->atomic_size / sizeof(double);
      
      for (bs=0; bs<blocksize; bs++) {
        if (blocksize == 1) {
          fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"1\"/>\n",dfields[df]->name);
        } else {
          fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"%s_%d\" NumberOfComponents=\"1\"/>\n",dfields[df]->name,bs);
        }
      }
    }
  }
  
  /* POINT-DATA HEADER - CLOSE */
  fprintf(fp,"    </PPointData>\n");
  
  /* PVTU write sources */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRQ(ierr);
  for (i=0; i<nproc; i++) {
    int i32;
    
    PetscMPIIntCast(i,&i32);
  
    if (asprintf(&sourcename,"%s_sc-subdomain%1.5d.vtu",prefix,i32) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    fprintf(fp,"    <Piece Source=\"%s\"/>\n",sourcename);
    free(sourcename);
  }
  
  /* PVTU HEADER - CLOSE */
  fprintf(fp,"  </PUnstructuredGrid>\n");
  fprintf(fp,"</VTKFile>\n");
  fclose(fp);
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintViewParaview(SurfaceConstraint sc, const char path[], const char prefix[])
{
  char *vtkfilename,*filename;
  PetscMPIInt rank;
  char *appended;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  if (asprintf(&appended,"%s_sc",prefix) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = pTatinGenerateParallelVTKName(appended,"vtu",&vtkfilename);CHKERRQ(ierr);
  if (path) {
    if (asprintf(&filename,"%s/%s",path,vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  } else {
    if (asprintf(&filename,"./%s",vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  }
  
  ierr = _SurfaceConstraintViewParaviewVTU(sc,filename);CHKERRQ(ierr);
  
  free(filename);
  free(vtkfilename);
  free(appended);
  
  if (asprintf(&appended,"%s_sc",prefix) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = pTatinGenerateVTKName(appended,"pvtu",&vtkfilename);CHKERRQ(ierr);
  if (path) {
    if (asprintf(&filename,"%s/%s",path,vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  } else {
    if (asprintf(&filename,"./%s",vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  }
  if (rank == 0) { /* not we are a bit tricky about which name we pass in here to define the edge data sets */
    ierr = _SurfaceConstraintViewParaviewPVTU(sc,prefix,filename);CHKERRQ(ierr);
  }
  free(filename);
  free(vtkfilename);
  free(appended);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _sc_get_hF(HexElementFace side,PetscReal elcoor[],PetscReal *hf)
{
  PetscInt k,d;
  const PetscInt center = 4;
  PetscReal dx,dx2,maxh = 0.0;
  
  switch (side) {
    case HEX_FACE_Pxi:
    {
      const PetscInt fid[] = { 2,5,8, 11,14,17, 20,23,26 };
      for (k=0; k<9; k++) {
        dx2 = 0.0;
        for (d=0; d<3; d++) {
          dx = elcoor[3*fid[k]+d] - elcoor[3*fid[center]+d];
          dx2 += dx*dx;
        }
        maxh = PetscMax(maxh,dx2);
      }
    }
      break;

    case HEX_FACE_Nxi:
    {
      const PetscInt fid[] = { 6,3,0, 15,12,9,  24,21,18 };
      for (k=0; k<9; k++) {
        dx2 = 0.0;
        for (d=0; d<3; d++) {
          dx = elcoor[3*fid[k]+d] - elcoor[3*fid[center]+d];
          dx2 += dx*dx;
        }
        maxh = PetscMax(maxh,dx2);
      }
    }
      break;
      
    case HEX_FACE_Peta:
    {
      const PetscInt fid[] = { 8,7,6, 17,16,15, 26,25,24 };
      for (k=0; k<9; k++) {
        dx2 = 0.0;
        for (d=0; d<3; d++) {
          dx = elcoor[3*fid[k]+d] - elcoor[3*fid[center]+d];
          dx2 += dx*dx;
        }
        maxh = PetscMax(maxh,dx2);
      }
    }
      break;
      
    case HEX_FACE_Neta:
    {
      const PetscInt fid[] = { 0,1,2, 9,10,11, 18,19,20 };
      for (k=0; k<9; k++) {
        dx2 = 0.0;
        for (d=0; d<3; d++) {
          dx = elcoor[3*fid[k]+d] - elcoor[3*fid[center]+d];
          dx2 += dx*dx;
        }
        maxh = PetscMax(maxh,dx2);
      }
    }
      break;
      
    case HEX_FACE_Pzeta:
    {
      const PetscInt fid[] = { 18,19,20, 21,22,23, 24,25,26 };
      for (k=0; k<9; k++) {
        dx2 = 0.0;
        for (d=0; d<3; d++) {
          dx = elcoor[3*fid[k]+d] - elcoor[3*fid[center]+d];
          dx2 += dx*dx;
        }
        maxh = PetscMax(maxh,dx2);
      }
    }
      break;
      
    case HEX_FACE_Nzeta:
    {
      const PetscInt fid[] = { 2,1,0, 5,4,3, 8,7,6 };
      for (k=0; k<9; k++) {
        dx2 = 0.0;
        for (d=0; d<3; d++) {
          dx = elcoor[3*fid[k]+d] - elcoor[3*fid[center]+d];
          dx2 += dx*dx;
        }
        maxh = PetscMax(maxh,dx2);
      }
    }
      break;
      
    default:
      break;
  }
  
  *hf = PetscSqrtReal(maxh) * 2.0;
  
  PetscFunctionReturn(0);
}


PetscErrorCode compute_surface_volume_ratio(SurfaceConstraint sc,PetscInt fe,PetscReal elcoords[],PetscReal *A_on_V)
{
  PetscInt facet_index,cell_side,cell_index;
  PetscInt q,nqp_v,nqp_s;
  QPoint3d qp3[27];
  QPoint2d *qp2 = NULL;
  PetscReal Ae = 0.0,Ve = 0.0,volJ_q,surfJ_q;
  ConformingElementFamily element = NULL;
  double __GNi[3 * 27];
  const double *GNi[] = { &__GNi[0], &__GNi[27], &__GNi[2*27] };
  
  element = sc->fi->element;
  facet_index = sc->facets->local_index[fe]; /* facet local index */
  cell_side  = sc->fi->facet_label[facet_index]; /* side label */
  cell_index = sc->fi->facet_cell_index[facet_index];
  
  
  /* compute the volume */
  element->generate_volume_quadrature_3D(&nqp_v,qp3);
  Ve = 0.0;
  for (q=0; q<nqp_v; q++) {
    element->basis_GNI_3D(&qp3[q],(double**)GNi);
    element->compute_volume_geometry_3D(elcoords,GNi,NULL,&volJ_q);
    Ve += qp3[q].w * volJ_q;
  }
  
  /* compute the area */
  nqp_s = sc->quadrature->npoints;
  qp2 = sc->quadrature->gp2[cell_side];
  
  Ae = 0.0;
  for (q=0; q<nqp_s; q++) {
    element->compute_surface_geometry_3D(
                                         element,
                                         elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                         cell_side,   // edge index 0,...,7 //
                                         &qp2[q],     // should contain 1 point with dimension 2 (xi,eta)   //
                                         NULL,NULL,&surfJ_q); // n0[],t0 contains 1 point with dimension 3 (x,y,z) //
    Ae += qp2[q].w * surfJ_q;
  }
  *A_on_V = Ae / Ve;
  PetscFunctionReturn(0);
}

PetscErrorCode compute_penalty_nitsche_warburton(SurfaceConstraint sc,PetscInt fe,PetscReal elcoords[],PetscReal *penalty)
{
  PetscReal A_on_V;
  PetscReal p = 2.0; // poly degree
  PetscReal d = 3.0; // spatial dimension
  PetscErrorCode ierr;
  ierr = compute_surface_volume_ratio(sc,fe,elcoords,&A_on_V);CHKERRQ(ierr);
  *penalty = ((p + 1.0) * (p + d) / d) * A_on_V;
  PetscFunctionReturn(0);
}

/*
 Koen Hillewaert
 Development of the discontinuous Galerkin method for high-resolution, large scale CFD and acoustics in industrial geometries. 
 Presses univ. de Louvain, 2013.

 Table 3.1
*/
PetscErrorCode compute_penalty_nitsche_hillewaert(SurfaceConstraint sc,PetscInt fe,PetscReal elcoords[],PetscReal *penalty)
{
  PetscReal A_on_V;
  PetscReal p = 2.0; // poly degree
  PetscErrorCode ierr;
  ierr = compute_surface_volume_ratio(sc,fe,elcoords,&A_on_V);CHKERRQ(ierr);
  *penalty = (p + 1.0) * (p + 1.0) * A_on_V;
  PetscFunctionReturn(0);
}

PetscErrorCode compute_global_penalty_nitsche(SurfaceConstraint sc,PetscInt type,PetscReal *penalty)
{
  PetscErrorCode ierr;
  PetscReal      penalty_k,penalty_local = 0.0;
  PetscInt       fe,facet_index,cell_index;
  const PetscInt *elnidx;
  PetscInt       nel,nen;
  double         elcoords[3*Q2_NODES_PER_EL_3D];
  
  ierr = MeshFacetInfoGetCoords(sc->fi);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(sc->fi->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);

  switch (type) {
    case 0:
      for (fe=0; fe<sc->facets->n_entities; fe++) {
        facet_index = sc->facets->local_index[fe]; /* facet local index */
        cell_index = sc->fi->facet_cell_index[facet_index];
        
        ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*cell_index],(PetscReal*)sc->fi->_mesh_coor);CHKERRQ(ierr);
        
        ierr = compute_penalty_nitsche_warburton(sc,fe,elcoords,&penalty_k);CHKERRQ(ierr);
        penalty_local = PetscMax(penalty_local,penalty_k);
      }
      break;

    case 1:
      for (fe=0; fe<sc->facets->n_entities; fe++) {
        facet_index = sc->facets->local_index[fe]; /* facet local index */
        cell_index = sc->fi->facet_cell_index[facet_index];
        
        ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*cell_index],(PetscReal*)sc->fi->_mesh_coor);CHKERRQ(ierr);
        
        ierr = compute_penalty_nitsche_hillewaert(sc,fe,elcoords,&penalty_k);CHKERRQ(ierr);
        penalty_local = PetscMax(penalty_local,penalty_k);
      }
      break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only type {0, 1} supported");
      break;
  }
  
  ierr = MeshFacetInfoRestoreCoords(sc->fi);CHKERRQ(ierr);
  
  ierr = MPI_Allreduce(&penalty_local,penalty,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)sc->fi->dm));CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode DMDABCListTraverseFacets3d(BCList list,DM da,SurfaceConstraint sc,PetscInt dof_idx,PetscBool (*eval)(PetscScalar*,PetscScalar*,void*),void *ctx)
{
  ConformingElementFamily element = NULL;
  Facet                   cell_facet;
  PetscInt                d,e,k,L,M,N,P,ndof,facet_index,cell_side,cell_index,nel,nen;
  PetscInt                *idx;
  const PetscInt          *elnidx;
  PetscScalar             *vals;
  PetscScalar             bc_val,elcoords[3*Q2_NODES_PER_EL_3D];
  PetscBool               impose_dirichlet;
  PetscErrorCode          ierr;

  MPI_Comm       comm;
  PetscMPIInt    rank;

  PetscFunctionBegin;

  comm = PetscObjectComm((PetscObject)da);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (sc->type != SC_DIRICHLET) {
    PetscPrintf(PetscObjectComm((PetscObject)sc->dm),"[ignoring] SurfaceConstraintSetValues_TRACTION() called with different type on object with name \"%s\"\n",sc->name);
    PetscFunctionReturn(0);
  }

  if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
  if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
  if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");

  element = sc->fi->element;

  ierr = DMDAGetInfo(sc->fi->dm,NULL, &M,&N,&P, NULL,NULL,NULL, &ndof,NULL, NULL,NULL,NULL, NULL);CHKERRQ(ierr);
  /* Get Q2 elements connectivity table */
  ierr = MeshFacetInfoGetCoords(sc->fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(sc->fi->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);

  ierr = BCListGetLocalIndices(list,&L,&idx);
  ierr = BCListGetLocalValues(list,&L,&vals);

  for (e=0; e<sc->facets->n_entities; e++) {
    int      *face_local_indices = NULL;

    facet_index = sc->facets->local_index[e];            /* facet local index */
    cell_side   = sc->fi->facet_label[facet_index];      /* side label */
    cell_index  = sc->fi->facet_cell_index[facet_index];

    ierr = FacetPack(cell_facet, facet_index, sc->fi);CHKERRQ(ierr);

    face_local_indices = element->face_node_list[cell_side];

    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*cell_index],(PetscReal*)sc->fi->_mesh_coor);CHKERRQ(ierr);

    for (k=0; k<Q2_NODES_PER_EL_2D; k++) {
      PetscInt    blockloc,loc,nidx3d;   
      PetscScalar pos[3];

      /* maps nodes 0-8 (surface) -> 0-27 (volume) */
      nidx3d   = face_local_indices[k];
      /* get node idx */
      blockloc = elnidx[nen*cell_index + nidx3d]; 
      /* get dof idx */
      loc      = blockloc*ndof+dof_idx; 
      for (d=0; d<3; d++) {
        pos[d] = elcoords[3*nidx3d + d];
      }
      /* get user defined function */
      impose_dirichlet = eval(pos,&bc_val,ctx);

      idx[loc]  = BCList_DIRICHLET;
      vals[loc] = bc_val;
    }
  }

  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = BCListLocalToGlobal(list);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
