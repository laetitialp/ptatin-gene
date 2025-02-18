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
} DeviatoricTractionContext;

PetscErrorCode _resize_facet_quadrature_data(SurfaceConstraint sc);

/* surface constraint implementation specific */
static PetscErrorCode _form_access_demo(StokesForm *form)
{
  PetscErrorCode    ierr;
  SurfaceConstraint sc;
  SurfaceQuadrature boundary_q;
  DeviatoricTractionContext   *formdata = NULL;
  PetscFunctionBegin;
  
  sc         = form->sc;
  formdata   = (DeviatoricTractionContext*)form->data;
  boundary_q = sc->quadrature;
  ierr = SurfaceQuadratureGetAllCellData_Stokes(boundary_q,&formdata->boundary_qp);CHKERRQ(ierr);
  
  DataBucketGetEntriesdByName(sc->properties_db,"traction",(void**)&formdata->traction_qp);

  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode _form_restore_demo(StokesForm *form)
{
  SurfaceConstraint sc;
  DeviatoricTractionContext   *formdata = NULL;
  PetscFunctionBegin;

  formdata = (DeviatoricTractionContext*)form->data;
  sc       = form->sc;
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"traction",(void**)&formdata->traction_qp);
  formdata->boundary_qp = NULL;
  
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode StokesFormSetupContext_Demo(StokesForm *F, DeviatoricTractionContext *formdata)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* data */
  ierr    = PetscMemzero(formdata,sizeof(DeviatoricTractionContext));CHKERRQ(ierr);
  F->data = (void*)formdata;
  
  /* methods */
  F->access  = _form_access_demo;
  F->restore = _form_restore_demo;
  F->apply   = NULL;
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode StokesFormSetup_Demo(StokesForm *form, SurfaceConstraint sc, DeviatoricTractionContext *formdata)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StokesFormInit(form,FORM_UNINIT,sc);CHKERRQ(ierr);
  ierr = StokeFormSetFunctionSpace_Q2P1(form);CHKERRQ(ierr);
  ierr = StokesFormSetupContext_Demo(form,formdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#if 0
/* action (residual) */
/* point-wise kernels */
static PetscErrorCode _form_residual_F1(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  DeviatoricTractionContext *formdata = NULL;
  QPntSurfCoefStokes *qp_data = NULL;
  PetscInt        sq_index,sc_index,qp_offset;
  double          *normal,*tangent1,*tangent2,*traction;
  PetscFunctionBegin;
  
  formdata = (DeviatoricTractionContext*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;

  //PetscPrintf(PETSC_COMM_WORLD,"[facet_sc_i, facet_i, point_i, sq_index, sc_index] = [%d, %d, %d, %d, %d]\n",form->facet_sc_i,form->facet_i,form->point_i,sq_index,sc_index);
  
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
  DeviatoricTractionContext formdata;
  PetscFunctionBegin;
  
  printf("_Residual_F1\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A12(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  DeviatoricTractionContext    *formdata = NULL;
  QPntSurfCoefStokes *qp_data = NULL;
  PetscInt           sq_index;
  PetscReal          *normal;
  PetscFunctionBegin;
  
  formdata = (DeviatoricTractionContext*)form->data;
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
  DeviatoricTractionContext formdata;
  PetscFunctionBegin;

  printf("_SpMV_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StokesFormConfigureAction_SpMV(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,NULL, dmp,pfield, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A12(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  DeviatoricTractionContext    *formdata = NULL;
  QPntSurfCoefStokes *qp_data = NULL;
  PetscInt           sq_index;
  double             *normal;
  PetscFunctionBegin;
  
  formdata = (DeviatoricTractionContext*)form->data;
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
  DeviatoricTractionContext formdata;
  PetscFunctionBegin;
  
  printf("_Assemble_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.u,&F.p, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_wA(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  DeviatoricTractionContext    *formdata = NULL;
  QPntSurfCoefStokes *qp_data = NULL;
  PetscInt           sq_index;
  double             *normal;
  PetscFunctionBegin;
  
  formdata = (DeviatoricTractionContext*)form->data;
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
  DeviatoricTractionContext formdata;
  PetscFunctionBegin;
  
  printf("_SpMV_A\n");
  printf("_Residual_A11X1_A12X2\n");
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
#endif

#if 0
/* action (residual) */
/* point-wise kernels */
static PetscErrorCode _form_residual_F1(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  DeviatoricTractionContext    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset;
  double             *normal,*tangent1,*tangent2,*traction;
  PetscErrorCode     ierr;
  PetscFunctionBegin;

  formdata = (DeviatoricTractionContext*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  normal   = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  tangent1 = (PetscReal*)formdata->boundary_qp[ sq_index ].tangent1;
  tangent2 = (PetscReal*)formdata->boundary_qp[ sq_index ].tangent2;
  
  qp_offset = 3*sc_index;
  traction  = &formdata->traction_qp[qp_offset];

  neumann_deviatoric_nts_q2_3d_residual_w(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
    form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
    form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
    form->p_elfield_0,
    traction,  // parameter
    normal,  // parameter
    tangent1,  // parameter
    tangent2,  // parameter
    ds[0], F
  );
  
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_residual_F2(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  PetscFunctionBegin;
  neumann_deviatoric_nts_q2_3d_residual_q(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
    form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
    form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
    form->p_elfield_0,
    ds[0], F
  );
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_Residual(StokesForm *form,StokesSubVec op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_RESIDUAL);CHKERRQ(ierr);
  switch (op) {
    case V_X1:
      form->apply = _form_residual_F1;
      break;
    case V_X2:
      form->apply = _form_residual_F2;
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
  DeviatoricTractionContext formdata;
  
  //printf("_Residual_F1\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_residual_F2(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp,const PetscScalar pfield[], PetscScalar R[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  DeviatoricTractionContext formdata;
  
  //printf("_Residual_F2\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X2);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.p, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* action (spmv) */
/* point-wise kernels */
static PetscErrorCode _form_spmv_A11(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  PetscFunctionBegin;
  neumann_deviatoric_nts_q2_3d_spmv_wu(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
    form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
    ds[0], F
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A12(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  DeviatoricTractionContext    *formdata = NULL;
  PetscInt           sq_index,sc_index;
  double             *normal;
  PetscFunctionBegin;
  formdata = (DeviatoricTractionContext*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  neumann_deviatoric_nts_q2_3d_spmv_wp(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
    form->p_elfield_0,
    normal,  // parameter
    ds[0], F
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A21(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  PetscFunctionBegin; 
  neumann_deviatoric_nts_q2_3d_spmv_qu(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
    form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
    ds[0], F
  );
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_SpMV(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_SPMV);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_spmv_A11;
      break;
    case M_A12:
      form->apply = _form_spmv_A12;
      break;
    case M_A21:
      form->apply = _form_spmv_A21;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of A11, A12, A21");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_spmv_A11(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  DeviatoricTractionContext formdata;
  
  //printf("_SpMV_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A11);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, NULL,NULL, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A12(
  SurfaceConstraint sc, DM dmu, DM dmp,const PetscScalar pfield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  DeviatoricTractionContext formdata;
  
  printf("_SpMV_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,NULL, dmp,pfield, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A21(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp, PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  DeviatoricTractionContext formdata;
  
  //printf("_SpMV_A21\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A21);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.p, dmu, dmu,ufield, dmp,NULL, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* assemble */
/* point-wise kernels */
static PetscErrorCode _form_asmb_A11(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  PetscFunctionBegin;
  neumann_deviatoric_nts_q2_3d_asmb_wu(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
    ds[0], A
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A12(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  DeviatoricTractionContext    *formdata = NULL;
  PetscInt           sq_index,sc_index;
  double             *normal;
  PetscFunctionBegin;

  formdata = (DeviatoricTractionContext*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  neumann_deviatoric_nts_q2_3d_asmb_wp(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
    normal,  // parameter
    ds[0], A
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A21(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  PetscFunctionBegin;
  neumann_deviatoric_nts_q2_3d_asmb_qu(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
    ds[0], A
  );
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_Assemble(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_ASSEMBLE);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_asmb_A11;
      break;
    case M_A12:
      form->apply = _form_asmb_A12;
      break;
    case M_A21:
      form->apply = _form_asmb_A21;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of A11, A12, A21");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_asmb_A11(SurfaceConstraint sc, DM dmu, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  DeviatoricTractionContext formdata;
  
  //printf("_Assemble_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A11);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.u,&F.u, dmu, dmu, NULL, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_asmb_A12(SurfaceConstraint sc, DM dmu, DM dmp, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  DeviatoricTractionContext formdata;
  PetscFunctionBegin;
  printf("_Assemble_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.u,&F.p, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_asmb_A21(SurfaceConstraint sc, DM dmu, DM dmp, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  DeviatoricTractionContext formdata;
  
  //printf("_Assemble_A21\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A21);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.p,&F.u, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* assemble diagonal */
/* point-wise kernels */
static PetscErrorCode _form_asmbdiag_A11(StokesForm *form,PetscReal ds[],PetscReal diagA[])
{
  PetscFunctionBegin;
  neumann_deviatoric_nts_q2_3d_asmbdiag_wu(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
    ds[0], diagA
  );
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_AssembleDiagonal(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_ASSEMBLE_DIAG);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_asmbdiag_A11;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only be A11");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_asmbdiag_A11(SurfaceConstraint sc, DM dmu, Vec diagA)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  DeviatoricTractionContext formdata;
  
  //printf("_AssembleDiagonal_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AssembleDiagonal(&F,M_A11);CHKERRQ(ierr);
  ierr = generic_facet_assemble_diagonal(&F, &F.u,dmu,  dmu, diagA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_wA(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  DeviatoricTractionContext    *formdata = NULL;
  PetscInt           sq_index,sc_index;
  double             *normal;
  PetscFunctionBegin;
  
  formdata = (DeviatoricTractionContext*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  neumann_deviatoric_nts_q2_3d_spmv_w_up(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
    form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
    form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
    form->p_elfield_0,
    normal,  // parameter
    ds[0], F
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_qA(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  PetscFunctionBegin;
  neumann_deviatoric_nts_q2_3d_spmv_q_up(
    form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
    form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
    form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
    form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
    form->p_elfield_0,
    ds[0], F
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode StoksFormConfigureAction_AuResidual(StokesForm *form,StokesSubVec op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_RESIDUAL);CHKERRQ(ierr);
  switch (op) {
    case V_X1:
      form->apply = _form_spmv_wA;
      break;
    case V_X2:
      form->apply = _form_spmv_qA;
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
  DeviatoricTractionContext formdata;
  
  //printf("_SpMV_A\n");
  
  //printf("_Residual_A11X1_A12X2\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AuResidual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, Yu);CHKERRQ(ierr);
  
  //printf("_Residual_A21X1\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AuResidual(&F,V_X2);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.p, dmu, dmu,ufield, dmp,pfield, Yp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _SetType_DEVIATORIC_TRACTION(SurfaceConstraint sc)
{
  PetscErrorCode ierr;
  
  /* set methods */
  sc->ops.setup   = NULL; /* always null */
  sc->ops.destroy = NULL;
  
  sc->ops.residual_F  = NULL; /* always null */
  sc->ops.residual_Fu = sc_residual_F1;
  sc->ops.residual_Fp = sc_residual_F2;
  
  sc->ops.action_A   = sc_spmv_A;
  sc->ops.action_Auu = sc_spmv_A11;
  sc->ops.action_Aup = sc_spmv_A12;
  sc->ops.action_Apu = sc_spmv_A21;
  
  sc->ops.asmb_Auu = sc_asmb_A11;
  sc->ops.asmb_Aup = sc_asmb_A12;
  sc->ops.asmb_Apu = sc_asmb_A21;
  
  sc->ops.diag_Auu = sc_asmbdiag_A11;
  
  /* allocate implementation data */
  /* insert properties into quadrature bucket */
  DataBucketRegister_double(sc->properties_db,"traction" ,3);
  DataBucketFinalize(sc->properties_db);
  PetscFunctionReturn(0);
}
#endif

#if 0
typedef enum { RESIDUAL_F1, SPMV_A12 } TypeOfForm;

static inline PetscErrorCode DeviatoricNeumann_Residual_F1_Form(
  PetscReal Ni[],
  PetscReal Nip[],
  PetscReal el_pfield[],
  PetscReal normal[],
  PetscReal tangent1[],
  PetscReal tangent2[],
  const PetscReal traction[],
  PetscReal fac, 
  PetscScalar Fe[]
)
{
  PetscInt  q,k,d;
  PetscReal pressure;
  PetscFunctionBegin;

  /* evaluate pressure at quadrature point */
  pressure = 0.0;
  for (q=0; q<P_BASIS_FUNCTIONS; q++) {
    pressure += Nip[q] * el_pfield[q]; 
  }
  /* int (w.n)*p - (w.n)T0 - (w.t1)T1 - (w.t2)T2 * ds */
  for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
    for (d=0; d<NSD; d++) {
      PetscReal nt  = normal[d]   * traction[0];
      PetscReal t1t = tangent1[d] * traction[1];
      PetscReal t2t = tangent2[d] * traction[2]; 
      Fe[3*k + d] += fac * Ni[k] * (normal[d]*pressure - (nt + t1t + t2t));
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DeviatoricNeumann_SpMV_Form(
  PetscReal Ni[],
  PetscReal Nip[],
  PetscReal el_pfield[],
  PetscReal normal[],
  PetscReal fac, 
  PetscScalar Fe[]
)
{
  PetscInt  q,k,d;
  PetscReal pressure;
  PetscFunctionBegin;

  /* evaluate pressure at quadrature point */
  pressure = 0.0;
  for (q=0; q<P_BASIS_FUNCTIONS; q++) {
    pressure += Nip[q]*el_pfield[q];
  }
  /* int (w.n)*p * ds */
  for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
    for (d=0; d<NSD; d++) {
      Fe[3*k + d] += fac*Ni[k]*normal[d]*pressure;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _FormFunctionLocal(
  SurfaceConstraint sc, DM dau, DM dap, const PetscScalar pfield[], PetscScalar Ru[], TypeOfForm type)
{
  DM              cda;
  Vec             gcoords;
  const PetscReal *LA_gcoords;
  PetscInt        fe,nel,nen_u,nen_p;
  const PetscInt  *elnidx_u,*elnidx_p;
  const PetscReal *domain_traction_qp;
  PetscScalar     Fe[3*Q2_NODES_PER_EL_3D];
  ConformingElementFamily element = NULL;
  QPntSurfCoefStokes *all_surf_gausspoints,*cell_surf_gausspoints;
  PetscLogDouble  t0,t1;
  PetscErrorCode  ierr;
  PetscFunctionBegin;

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);
  ierr = SurfaceQuadratureGetAllCellData_Stokes(sc->quadrature,&all_surf_gausspoints);CHKERRQ(ierr);
  
  element = sc->fi->element;

  DataBucketGetEntriesdByName(sc->properties_db,"traction",(void**)&domain_traction_qp);
  PetscTime(&t0);
  for (fe=0; fe<sc->facets->n_entities; fe++) {
    PetscInt        q,nqp,facet_index,cell_side,cell_index;
    PetscInt        vel_el_lidx[3*U_BASIS_FUNCTIONS];
    QPoint3d        *qp3 = NULL;
    QPoint2d        *qp2 = NULL;
    const PetscReal *cell_traction_qp;
    PetscReal       elcoords[3*Q2_NODES_PER_EL_3D];
    PetscReal       elp[P_BASIS_FUNCTIONS];
    
    facet_index = sc->facets->local_index[fe]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];

    nqp = sc->quadrature->npoints;
    qp2 = sc->quadrature->gp2[cell_side];
    qp3 = sc->quadrature->gp3[cell_side];

    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*cell_index]);CHKERRQ(ierr);
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*cell_index],(PetscReal*)LA_gcoords);CHKERRQ(ierr);
    ierr = DMDAGetScalarElementField(elp,nen_p,(PetscInt*)&elnidx_p[nen_p*cell_index],(PetscReal*)pfield);CHKERRQ(ierr);

    ierr = SurfaceQuadratureGetCellData_Stokes(sc->quadrature,all_surf_gausspoints,facet_index,&cell_surf_gausspoints);CHKERRQ(ierr);

    /* initialise element stiffness matrix */
    ierr = PetscMemzero(Fe,sizeof(PetscScalar)*Q2_NODES_PER_EL_3D*3);CHKERRQ(ierr);
    
    cell_traction_qp = &domain_traction_qp[fe * 3 * nqp];
    
    for (q=0; q<nqp; q++) {
      QPntSurfCoefStokes *qp_data = &cell_surf_gausspoints[q];
      PetscScalar        xip[] = { qp3[q].xi, qp3[q].eta, qp3[q].zeta };
      PetscScalar        fac,surfJ_q;
      const PetscReal    *traction_qp;
      PetscReal          Ni[Q2_NODES_PER_EL_3D],Nip[P_BASIS_FUNCTIONS];
      double             *normal,*tangent1,*tangent2;
      
      element->compute_surface_geometry_3D(
                                           element,
                                           elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                           cell_side,   // edge index 0,...,7 //
                                           &qp2[q], // should contain 1 point with dimension 2 (xi,eta)   //
                                           NULL,NULL,&surfJ_q); // n0[],t0 contains 1 point with dimension 3 (x,y,z) //
      fac = qp2[q].w * surfJ_q;

      traction_qp = &cell_traction_qp[3 * q];
      P3D_ConstructNi_Q2_3D(xip,Ni);
      ConstructNi_pressure(xip,elcoords,Nip);

      normal   = qp_data->normal;
      tangent1 = qp_data->tangent1;
      tangent2 = qp_data->tangent2;

      switch (type) {
        case RESIDUAL_F1:
          ierr = DeviatoricNeumann_Residual_F1_Form(Ni,Nip,elp,normal,tangent1,tangent2,traction_qp,fac,Fe);
          break;
        
        case SPMV_A12:
          ierr = DeviatoricNeumann_SpMV_Form(Ni,Nip,elp,normal,fac,Fe);
          break;
      }
    }
    ierr = DMDASetValuesLocalStencil_AddValues_Stokes_Velocity(Ru,vel_el_lidx,Fe);CHKERRQ(ierr);
  }
  PetscTime(&t1);
  PetscPrintf(PetscObjectComm((PetscObject)dau),"Assembled int_S N traction dS, = %1.4e (sec)\n",t1-t0);
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"traction",(void**)&domain_traction_qp);
  ierr = VecRestoreArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_residual_F1(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp,const PetscScalar pfield[], PetscScalar R[])
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = _FormFunctionLocal(sc,dmu,dmp,pfield,R,RESIDUAL_F1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A12(
  SurfaceConstraint sc, DM dmu, DM dmp,const PetscScalar pfield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = _FormFunctionLocal(sc,dmu,dmp,pfield,Y,SPMV_A12);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A(SurfaceConstraint sc,
                                DM dmu,const PetscScalar ufield[],
                                DM dmp,const PetscScalar pfield[],
                                PetscScalar Yu[], PetscScalar Yp[])
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = _FormFunctionLocal(sc,dmu,dmp,pfield,Yu,SPMV_A12);CHKERRQ(ierr);
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
  sc->ops.asmb_Aup = NULL;//sc_asmb_A12;
  sc->ops.asmb_Apu = NULL;
  
  sc->ops.diag_Auu = NULL;
  
  /* insert properties into quadrature bucket */
  DataBucketRegister_double(sc->properties_db,"traction" ,3);
  
  DataBucketFinalize(sc->properties_db);
  PetscFunctionReturn(0);
}
#endif

typedef enum { RESIDUAL_F1, SPMV_A12 } TypeOfForm;

static inline PetscErrorCode DeviatoricNeumann_Residual_F1_FormSurface(
  PetscReal Ni[],
  PetscReal Nip[],
  PetscReal el_pfield[],
  PetscReal normal[],
  PetscReal tangent1[],
  PetscReal tangent2[],
  const PetscReal traction[],
  PetscReal fac, 
  PetscScalar Fe[]
)
{
  PetscInt  q,k,d;
  PetscReal pressure;
  PetscFunctionBegin;

  /* evaluate pressure at quadrature point */
  pressure = 0.0;
  for (q=0; q<P_BASIS_FUNCTIONS; q++) {
    pressure += Nip[q] * el_pfield[q]; 
  }
  /* int (w.n)*p - (w.n)T0 - (w.t1)T1 - (w.t2)T2 * ds */
  for (k=0; k<Q2_NODES_PER_EL_2D; k++) {
    for (d=0; d<NSD; d++) {
      PetscReal nt  = normal[d]   * traction[0];
      PetscReal t1t = tangent1[d] * traction[1];
      PetscReal t2t = tangent2[d] * traction[2]; 
      Fe[3*k + d] += fac * Ni[k] * (normal[d]*pressure - (nt + t1t + t2t));
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DeviatoricNeumann_SpMV_FormSurface(
  PetscReal Ni[],
  PetscReal Nip[],
  PetscReal el_pfield[],
  PetscReal normal[],
  PetscReal fac, 
  PetscScalar Fe[]
)
{
  PetscInt  q,k,d;
  PetscReal pressure;
  PetscFunctionBegin;

  /* evaluate pressure at quadrature point */
  pressure = 0.0;
  for (q=0; q<P_BASIS_FUNCTIONS; q++) {
    pressure += Nip[q]*el_pfield[q];
  }
  /* int (w.n)*p * ds */
  for (k=0; k<Q2_NODES_PER_EL_2D; k++) {
    for (d=0; d<NSD; d++) {
      Fe[3*k + d] += fac*Ni[k]*normal[d]*pressure;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _FormFunctionLocal(
  SurfaceConstraint sc, DM dau, DM dap, const PetscScalar pfield[], PetscScalar Ru[], TypeOfForm type)
{
  DM              cda;
  Vec             gcoords;
  const PetscReal *LA_gcoords;
  PetscInt        fe,nel,nen_u,nen_p;
  const PetscInt  *elnidx_u,*elnidx_p;
  const PetscReal *domain_traction_qp;
  PetscScalar     Fe[3*Q2_NODES_PER_EL_3D],Be[3*Q2_NODES_PER_EL_2D];
  ConformingElementFamily element = NULL;
  QPntSurfCoefStokes *all_surf_gausspoints,*cell_surf_gausspoints;
  PetscLogDouble  t0,t1;
  PetscErrorCode  ierr;
  PetscFunctionBegin;

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);
  ierr = SurfaceQuadratureGetAllCellData_Stokes(sc->quadrature,&all_surf_gausspoints);CHKERRQ(ierr);
  
  element = sc->fi->element;

  DataBucketGetEntriesdByName(sc->properties_db,"traction",(void**)&domain_traction_qp);
  PetscTime(&t0);
  for (fe=0; fe<sc->facets->n_entities; fe++) {
    PetscInt        k,q,nqp,facet_index,cell_side,cell_index;
    PetscInt        vel_el_lidx[3*U_BASIS_FUNCTIONS];
    QPoint3d        *qp3 = NULL;
    QPoint2d        *qp2 = NULL;
    const PetscReal *cell_traction_qp;
    PetscReal       elcoords[3*Q2_NODES_PER_EL_3D];
    PetscReal       elp[P_BASIS_FUNCTIONS];
    int             *face_local_indices = NULL;
    
    facet_index = sc->facets->local_index[fe]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    
    face_local_indices = element->face_node_list[cell_side];

    nqp = sc->quadrature->npoints;
    qp2 = sc->quadrature->gp2[cell_side];
    qp3 = sc->quadrature->gp3[cell_side];

    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*cell_index]);CHKERRQ(ierr);
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*cell_index],(PetscReal*)LA_gcoords);CHKERRQ(ierr);
    ierr = DMDAGetScalarElementField(elp,nen_p,(PetscInt*)&elnidx_p[nen_p*cell_index],(PetscReal*)pfield);CHKERRQ(ierr);

    ierr = SurfaceQuadratureGetCellData_Stokes(sc->quadrature,all_surf_gausspoints,facet_index,&cell_surf_gausspoints);CHKERRQ(ierr);

    /* initialise element stiffness matrix */
    ierr = PetscMemzero(Fe,sizeof(PetscScalar)*Q2_NODES_PER_EL_3D*3);CHKERRQ(ierr);
    ierr = PetscMemzero(Be,sizeof(PetscScalar)*Q2_NODES_PER_EL_2D*3);CHKERRQ(ierr);
    
    cell_traction_qp = &domain_traction_qp[fe * 3 * nqp];
    
    for (q=0; q<nqp; q++) {
      QPntSurfCoefStokes *qp_data = &cell_surf_gausspoints[q];
      PetscScalar        xip[] = { qp3[q].xi, qp3[q].eta, qp3[q].zeta };
      PetscScalar        fac,surfJ_q;
      const PetscReal    *traction_qp;
      PetscReal          Ni_surf[Q2_NODES_PER_EL_2D],Nip[P_BASIS_FUNCTIONS];
      double             *normal,*tangent1,*tangent2;
      
      element->compute_surface_geometry_3D(
                                           element,
                                           elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                           cell_side,   // edge index 0,...,7 //
                                           &qp2[q], // should contain 1 point with dimension 2 (xi,eta)   //
                                           NULL,NULL,&surfJ_q); // n0[],t0 contains 1 point with dimension 3 (x,y,z) //
      fac = qp2[q].w * surfJ_q;

      traction_qp = &cell_traction_qp[3 * q];
      /* surface basis function */
      element->basis_NI_2D(&qp2[q],Ni_surf);
      ConstructNi_pressure(xip,elcoords,Nip);

      normal   = qp_data->normal;
      tangent1 = qp_data->tangent1;
      tangent2 = qp_data->tangent2;

      switch (type) {
        case RESIDUAL_F1:
          ierr = DeviatoricNeumann_Residual_F1_FormSurface(Ni_surf,Nip,elp,normal,tangent1,tangent2,traction_qp,fac,Be);
          break;
        
        case SPMV_A12:
          ierr = DeviatoricNeumann_SpMV_FormSurface(Ni_surf,Nip,elp,normal,fac,Be);
          break;
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

static PetscErrorCode sc_residual_F1(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp,const PetscScalar pfield[], PetscScalar R[])
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = _FormFunctionLocal(sc,dmu,dmp,pfield,R,RESIDUAL_F1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A12(
  SurfaceConstraint sc, DM dmu, DM dmp,const PetscScalar pfield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = _FormFunctionLocal(sc,dmu,dmp,pfield,Y,SPMV_A12);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A(SurfaceConstraint sc,
                                DM dmu,const PetscScalar ufield[],
                                DM dmp,const PetscScalar pfield[],
                                PetscScalar Yu[], PetscScalar Yp[])
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = _FormFunctionLocal(sc,dmu,dmp,pfield,Yu,SPMV_A12);CHKERRQ(ierr);
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
  sc->ops.asmb_Aup = NULL;//sc_asmb_A12;
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
