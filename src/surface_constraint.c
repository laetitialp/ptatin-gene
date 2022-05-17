
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
const char *SurfaceConstraintTypeNames[] = { "none", "traction", "fssa", "nitsche_dirichlet", "nitsche_navier_slip", "nitsche_custom_navier_slip", 0 };

static PetscErrorCode _ops_residual_only(SurfaceConstraint sc);
static PetscErrorCode _ops_operator_only(SurfaceConstraint sc);

static PetscErrorCode _SetType_NONE(SurfaceConstraint sc);
static PetscErrorCode _SetType_TRACTION(SurfaceConstraint sc);

PetscErrorCode SurfaceConstraintCreate(SurfaceConstraint *_sc)
{
  SurfaceConstraint sc;
  PetscErrorCode    ierr;
  
  ierr = PetscMalloc(sizeof(struct _p_SurfaceConstraint),&sc);CHKERRQ(ierr);
  ierr = PetscMemzero(sc,sizeof(struct _p_SurfaceConstraint));CHKERRQ(ierr);
  
  DataBucketCreate(&sc->properties_db);

  ierr = _SetType_NONE(sc);CHKERRQ(ierr);

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
  
  if (sc->fi) SETERRQ(PetscObjectComm((PetscObject)sc->dm),PETSC_ERR_USER,"sc->fi already set");
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

PetscErrorCode SurfaceConstraintSetType(SurfaceConstraint sc, SurfaceConstraintType type, PetscBool residual_only, PetscBool operator_only)
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
      if (residual_only) {
        PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetType: residual_only = true has no effect with type SC_NONE");
      }
      if (operator_only) {
        PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetType: operator_only = true has no effect with type SC_NONE");
      }
      break;

    case SC_TRACTION:
      ierr = _SetType_TRACTION(sc);CHKERRQ(ierr);
      if (operator_only) {
        PetscPrintf(PETSC_COMM_SELF,"[Warning] SurfaceConstraintSetType: operator_only = true has no effect with type SC_TRACTION");
      }
      break;

    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfaceConstraint type %d not implemented",(int)type);
      break;
  }
  
  DataBucketSetInitialSizes(sc->properties_db,1,1);
  DataBucketSetSizes(sc->properties_db,0,-1);

  if (residual_only) {
    ierr = _ops_residual_only(sc);CHKERRQ(ierr);
  }
  if (operator_only) {
    ierr = _ops_operator_only(sc);CHKERRQ(ierr);
  }
  
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
  if (f) {
    *f = sc->facets;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintDuplicate(SurfaceConstraint sc, SurfaceConstraint *dup)
{
  PetscFunctionReturn(0);
}


static PetscErrorCode _ops_residual_only(SurfaceConstraint sc)
{
  sc->ops.action_A = NULL;
  sc->ops.asmb_A   = NULL;
  sc->ops.diag_A   = NULL;

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

static PetscErrorCode _resize_facet_quadrature_data(SurfaceConstraint sc)
{
  PetscInt nfacets;
  if (!sc->facets->set_values_called) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
  }
  nfacets = sc->facets->n_entities;
  DataBucketSetSizes(sc->properties_db,sc->nqp_facet*nfacets,-1);
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

/* TRACTION */
#if 0
static PetscErrorCode _SetUp_TRACTION(SurfaceConstraint sc)
{
  PetscErrorCode ierr;
  PetscInt nfaces;
  
  if (!sc->facets->set_values_called) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
  }
  nfaces = sc->facets->n_entities;
  if (nfaces != 0) {
    DataBucketSetSizes(sc->properties_db,sc->nqp_facet*nfaces,-1);
  } else {
    /* Done in SetType() */
    //DataBucketSetInitialSizes(sc->properties_db,1,1);
    //DataBucketSetSizes(sc->properties_db,0,-1);
  }
  
  PetscFunctionReturn(0);
}
#endif

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
                                         PetscErrorCode (*set)(Facet,const PetscReal*,PetscReal*,void*),
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

  
  if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
  if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
  if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");

  if (sc->type != SC_TRACTION) {
    PetscPrintf(PetscObjectComm((PetscObject)sc->dm),"[ignoring] SurfaceConstraintSetValues_TRACTION() called with different type\n");
    PetscFunctionReturn(0);
  }

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
    qp_offset = sc->nqp_facet * e; /* offset into entire domain qp list */
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
      
      ierr = PetscMemcpy(&traction_qp[3*qp_offset],traction,sizeof(PetscReal)*3);CHKERRQ(ierr);
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
  
  PetscFunctionReturn(0);
}
