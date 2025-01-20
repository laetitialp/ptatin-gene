#include <petsc.h>
#include <ptatin3d_defs.h>
#include <ptatin3d.h>
#include <private/ptatin_impl.h>
#include <quadrature.h>
#include <private/quadrature_impl.h>
#include <element_type_Q2.h>
#include <dmda_element_q2p1.h>
#include <element_utils_q2.h>
#include <ptatin3d_stokes.h>
#include <mesh_entity.h>
#include <surface_constraint.h>
#include <sc_generic.h>


/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
// fe_form_compiler.py version: ba26980b8db4ac4412a8ded988cf48aa986fbc80
// sympy version: 1.6.1
// using common substring elimination: True
// form file: nitsche-custom-h_IJ.py version: cd6c585d0922009ee6b85bfb39c0efd122a5046c

#include "neumann_deviatoric_nts_q2_3d_forms.h"
/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */

typedef enum { V_X1=0, V_X2 } StokesSubVec;

typedef enum { M_A11=0, M_A12, M_A21, M_A22 } StokesSubMat;

typedef struct {
  QPntSurfCoefStokes *boundary_qp;
  /* user fields */
  double             *traction_qp;  // "traction"
} FormContextDemo;

PetscErrorCode _resize_facet_quadrature_data(SurfaceConstraint sc);

/* surface constraint implementation specific */
static PetscErrorCode _form_access_demo(StokesForm *form)
{
  PetscErrorCode    ierr;
  SurfaceConstraint sc;
  SurfaceQuadrature boundary_q;
  FormContextDemo   *formdata = NULL;
  PetscFunctionBegin;
  
  sc         = form->sc;
  formdata   = (FormContextDemo*)form->data;
  boundary_q = sc->quadrature;
  ierr = SurfaceQuadratureGetAllCellData_Stokes(boundary_q,&formdata->boundary_qp);CHKERRQ(ierr);
  
  DataBucketGetEntriesdByName(sc->properties_db,"traction",(void**)&formdata->traction_qp);

  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode _form_restore_demo(StokesForm *form)
{
  SurfaceConstraint sc;
  FormContextDemo   *formdata = NULL;
  PetscFunctionBegin;

  formdata = (FormContextDemo*)form->data;
  sc       = form->sc;
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"traction",(void**)&formdata->traction_qp);
  formdata->boundary_qp = NULL;
  
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode StokesFormSetupContext_Demo(StokesForm *F, FormContextDemo *formdata)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* data */
  ierr    = PetscMemzero(formdata,sizeof(FormContextDemo));CHKERRQ(ierr);
  F->data = (void*)formdata;
  
  /* methods */
  F->access  = _form_access_demo;
  F->restore = _form_restore_demo;
  F->apply   = NULL;
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode StokesFormSetup_Demo(StokesForm *form, SurfaceConstraint sc, FormContextDemo *formdata)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StokesFormInit(form,FORM_UNINIT,sc);CHKERRQ(ierr);
  ierr = StokeFormSetFunctionSpace_Q2P1(form);CHKERRQ(ierr);
  ierr = StokesFormSetupContext_Demo(form,formdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* action (residual) */
/* point-wise kernels */
static PetscErrorCode _form_residual_F1(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  FormContextDemo *formdata = NULL;
  QPntSurfCoefStokes *qp_data = NULL;
  PetscInt        sq_index,sc_index,qp_offset;
  double          *normal,*tangent1,*tangent2,*traction;
  PetscFunctionBegin;
  
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  qp_data  = &formdata->boundary_qp[ sq_index ];
  normal   = (PetscReal*)qp_data->normal;
  tangent1 = (PetscReal*)qp_data->tangent1;
  tangent2 = (PetscReal*)qp_data->tangent2;

  qp_offset = 3*sc_index;
  traction = &formdata->traction_qp[qp_offset];
  
  neumann_deviatoric_nts_q2_3d_residual_w(
    form->test->W, // velocity test function
    form->X[1]->W, // pressure trial function
    form->p_elfield_0, // pressure field
    traction,  // parameter
    normal,  // parameter
    tangent1,  // parameter
    tangent2,  // parameter
    ds[0], F
  );

  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_Residual(StokesForm *form,StokesSubVec op)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StokesFormSetType(form,FORM_RESIDUAL);CHKERRQ(ierr);
  switch (op) {
    case V_X1:
      form->apply = _form_residual_F1;
      break;
    case V_X2:
      form->apply = NULL;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of X1, X2");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_residual_F1(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp,const PetscScalar pfield[], PetscScalar R[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  PetscFunctionBegin;
  
  //printf("_Residual_F1\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A12(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  FormContextDemo    *formdata = NULL;
  QPntSurfCoefStokes *qp_data = NULL;
  PetscInt           sq_index;
  PetscReal          *normal;
  PetscFunctionBegin;
  
  formdata = (FormContextDemo*)form->data;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  qp_data  = &formdata->boundary_qp[ sq_index ];
  
  normal   = (PetscReal*)qp_data->normal;
  neumann_deviatoric_nts_q2_3d_spmv_wp(form->test->W, form->trial->W, form->p_elfield_0, normal, ds[0], F);

  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StokesFormConfigureAction_SpMV(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StokesFormSetType(form,FORM_SPMV);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = NULL;
      break;
    case M_A12:
      form->apply = _form_spmv_A12;
      break;
    case M_A21:
      form->apply = NULL;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of A11, A12, A21");
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A12(
  SurfaceConstraint sc, DM dmu, DM dmp,const PetscScalar pfield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  PetscFunctionBegin;

  //printf("_SpMV_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StokesFormConfigureAction_SpMV(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,NULL, dmp,pfield, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A12(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  FormContextDemo    *formdata = NULL;
  QPntSurfCoefStokes *qp_data = NULL;
  PetscInt           sq_index;
  double             *normal;
  PetscFunctionBegin;
  
  formdata = (FormContextDemo*)form->data;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  qp_data  = &formdata->boundary_qp[ sq_index ];
  
  normal = (PetscReal*)qp_data->normal;

  neumann_deviatoric_nts_q2_3d_asmb_wp(form->test->W,form->trial->W,normal,ds[0],A);

  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_Assemble(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StokesFormSetType(form,FORM_ASSEMBLE);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = NULL;
      break;
    case M_A12:
      form->apply = _form_asmb_A12;
      break;
    case M_A21:
      form->apply = NULL;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of A11, A12, A21");
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_asmb_A12(SurfaceConstraint sc, DM dmu, DM dmp, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  PetscFunctionBegin;
  
  //printf("_Assemble_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.u,&F.p, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_wA(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  FormContextDemo    *formdata = NULL;
  QPntSurfCoefStokes *qp_data = NULL;
  PetscInt           sq_index;
  double             *normal;
  PetscFunctionBegin;
  
  formdata = (FormContextDemo*)form->data;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  qp_data = &formdata->boundary_qp[ sq_index ];

  normal = (PetscReal*)qp_data->normal;

  neumann_deviatoric_nts_q2_3d_spmv_w_up(form->test->W,form->X[1]->W,form->p_elfield_0,normal,ds[0], F);

  PetscFunctionReturn(0);
}

static PetscErrorCode StoksFormConfigureAction_AuResidual(StokesForm *form,StokesSubVec op)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StokesFormSetType(form,FORM_RESIDUAL);CHKERRQ(ierr);
  switch (op) {
    case V_X1:
      form->apply = _form_spmv_wA;
      break;
    case V_X2:
      form->apply = NULL;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of X1, X2");
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A(SurfaceConstraint sc,
                                DM dmu,const PetscScalar ufield[],
                                DM dmp,const PetscScalar pfield[],
                                PetscScalar Yu[], PetscScalar Yp[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  PetscFunctionBegin;
  //printf("_SpMV_A\n");
  
  //printf("_Residual_A11X1_A12X2\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AuResidual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, Yu);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _SetType_DEVIATORIC_TRACTION(SurfaceConstraint sc)
{
  PetscFunctionBegin;
  /* set methods */
  sc->ops.setup   = NULL; /* always null */
  sc->ops.destroy = NULL;
  
  sc->ops.residual_F  = NULL; /* always null */
  sc->ops.residual_Fu = sc_residual_F1;
  sc->ops.residual_Fp = NULL;
  
  sc->ops.action_A   = sc_spmv_A;
  sc->ops.action_Auu = NULL;
  sc->ops.action_Aup = sc_spmv_A12;
  sc->ops.action_Apu = NULL;
  
  sc->ops.asmb_Auu = NULL;
  sc->ops.asmb_Aup = sc_asmb_A12;
  sc->ops.asmb_Apu = NULL;
  
  sc->ops.diag_Auu = NULL;
  
  /* insert properties into quadrature bucket */
  DataBucketRegister_double(sc->properties_db,"traction" ,3);
  
  DataBucketFinalize(sc->properties_db);
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetValues_Stress_DEVIATORIC_TRACTION(
  SurfaceConstraint sc,
  SurfCSetValuesTraction set,
  void *data
)
{
  PetscInt       e,facet_index,cell_side,cell_index,q,qp_offset;
  Facet          cell_facet;
  PetscReal      qp_coor[3],traction[3];
  PetscReal      *traction_qp;
  double         Ni[27];
  const PetscInt *elnidx;
  PetscInt       nel,nen;
  double         elcoords[3*Q2_NODES_PER_EL_3D];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  if (sc->type != SC_DEVIATORIC_TRACTION) {
    PetscPrintf(PetscObjectComm((PetscObject)sc->dm),"[ignoring] SurfaceConstraintSetValues_Stress_DEVIATORIC_TRACTION() called with different type on object with name \"%s\"\n",sc->name);
    PetscFunctionReturn(0);
  }
  
  if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
  if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
  if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
  
  /* resize qp data */
  ierr = _resize_facet_quadrature_data(sc);CHKERRQ(ierr);
  
  DataBucketGetEntriesdByName(sc->properties_db,"traction", (void**)&traction_qp);
  
  ierr = MeshFacetInfoGetCoords(sc->fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(sc->fi->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);
  
  for (e=0; e<sc->facets->n_entities; e++) {
    facet_index = sc->facets->local_index[e]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    
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
      
      ierr = PetscMemzero(traction, sizeof(double)*3);CHKERRQ(ierr);
      
      ierr = set(cell_facet, qp_coor, traction, data);CHKERRQ(ierr);
      
      ierr = PetscMemcpy(&traction_qp[3*(qp_offset+q)], traction, sizeof(PetscReal)*3);CHKERRQ(ierr);
    }
  }
  
  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = MeshFacetInfoRestoreCoords(sc->fi);CHKERRQ(ierr);
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"traction", (void**)&traction_qp);
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintSetValues_StrainRate_DEVIATORIC_TRACTION(
  SurfaceConstraint sc,
  SurfCSetValuesTraction set,
  void *data)
{
  PetscInt           e,facet_index,cell_side,cell_index,q,qp_offset;
  Facet              cell_facet;
  PetscReal          qp_coor[3],traction[3];
  PetscReal          *traction_qp;
  double             Ni[27];
  const PetscInt     *elnidx;
  PetscInt           nel,nen;
  double             elcoords[3*Q2_NODES_PER_EL_3D];
  QPntSurfCoefStokes *all_surf_gausspoints,*cell_surf_gausspoints;
  PetscErrorCode     ierr;
  PetscFunctionBegin;
  
  if (sc->type != SC_DEVIATORIC_TRACTION) {
    PetscPrintf(PetscObjectComm((PetscObject)sc->dm),"[ignoring] SurfaceConstraintSetValues_StrainRate_DEVIATORIC_TRACTION() called with different type on object with name \"%s\"\n",sc->name);
    PetscFunctionReturn(0);
  }
  
  if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
  if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
  if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
  
  /* resize qp data */
  ierr = _resize_facet_quadrature_data(sc);CHKERRQ(ierr);
  
  DataBucketGetEntriesdByName(sc->properties_db,"traction", (void**)&traction_qp);
  
  ierr = MeshFacetInfoGetCoords(sc->fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(sc->fi->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);
  
  ierr = SurfaceQuadratureGetAllCellData_Stokes(sc->quadrature,&all_surf_gausspoints);CHKERRQ(ierr);

  for (e=0; e<sc->facets->n_entities; e++) {
    facet_index = sc->facets->local_index[e]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    
    ierr = FacetPack(cell_facet, facet_index, sc->fi);CHKERRQ(ierr);
    
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*cell_index],(PetscReal*)sc->fi->_mesh_coor);CHKERRQ(ierr);
    ierr = SurfaceQuadratureGetCellData_Stokes(sc->quadrature,all_surf_gausspoints,facet_index,&cell_surf_gausspoints);CHKERRQ(ierr);

    //qp_offset = sc->nqp_facet * facet_index; /* offset into entire domain qp list */
    qp_offset = sc->nqp_facet * e; /* offset into facet qp list */
    
    for (q=0; q<sc->nqp_facet; q++) {
      QPntSurfCoefStokes *qp_data = &cell_surf_gausspoints[q];
      double             eta = qp_data->eta;
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
      
      ierr = PetscMemzero(traction, sizeof(double)*3);CHKERRQ(ierr);
      
      ierr = set(cell_facet, qp_coor, traction, data);CHKERRQ(ierr);
      {
        int k;
        /* Use surface qp viscosity to compute stress */
        //for (k=0; k<3; k++) { traction[k] = 2.0 * cell_surf_gausspoints[q].eta * traction[k]; }
        for (k=0; k<3; k++) { traction[k] = 2.0 * eta * traction[k]; }
      }
      ierr = PetscMemcpy(&traction_qp[3*(qp_offset+q)], traction, sizeof(PetscReal)*3);CHKERRQ(ierr);
    }
  }
  
  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = MeshFacetInfoRestoreCoords(sc->fi);CHKERRQ(ierr);
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"traction", (void**)&traction_qp);
  PetscFunctionReturn(0);
}
