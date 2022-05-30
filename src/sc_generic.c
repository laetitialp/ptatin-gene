
#include <petsc.h>

#include <ptatin3d_defs.h>
#include <ptatin3d.h>
#include <private/ptatin_impl.h>
#include <element_type_Q2.h>
#include <dmda_element_q2p1.h>
#include <element_utils_q2.h>
#include <surface_constraint.h>
#include <sc_generic.h>


PetscErrorCode FunctionSpaceSet_VelocityQ1(FunctionSpace *s)
{
  PetscErrorCode ierr;
  ierr = PetscMemzero(s,sizeof(FunctionSpace));CHKERRQ(ierr);
  s->basis  = BASIS_Q1_3;
  s->nbasis = Q1_NODES_PER_EL_3D;
  s->dim    = 3;
  PetscFunctionReturn(0);
}

PetscErrorCode FunctionSpaceSet_VelocityQ2(FunctionSpace *s)
{
  PetscErrorCode ierr;
  ierr = PetscMemzero(s,sizeof(FunctionSpace));CHKERRQ(ierr);
  s->basis  = BASIS_Q2_3;
  s->nbasis = Q2_NODES_PER_EL_3D;
  s->dim    = 3;
  PetscFunctionReturn(0);
}

PetscErrorCode FunctionSpaceSet_PressureP0(FunctionSpace *s)
{
  PetscErrorCode ierr;
  ierr = PetscMemzero(s,sizeof(FunctionSpace));CHKERRQ(ierr);
  s->basis  = BASIS_P0_1;
  s->nbasis = 1;
  s->dim    = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode FunctionSpaceSet_PressureP1(FunctionSpace *s)
{
  PetscErrorCode ierr;
  ierr = PetscMemzero(s,sizeof(FunctionSpace));CHKERRQ(ierr);
  s->basis  = BASIS_P1_1;
  s->nbasis = P_BASIS_FUNCTIONS;
  s->dim    = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode FunctionSpaceSet_Q2P1(FunctionSpace *u,FunctionSpace *p)
{
  PetscErrorCode ierr;
  ierr = FunctionSpaceSet_VelocityQ2(u);CHKERRQ(ierr);
  ierr = FunctionSpaceSet_PressureP1(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokeFormSetFunctionSpace_Q2P1(StokesForm *f)
{
  PetscErrorCode ierr;
  ierr = FunctionSpaceSet_VelocityQ2(&f->u);CHKERRQ(ierr);
  ierr = FunctionSpaceSet_PressureP1(&f->p);CHKERRQ(ierr);
  f->X[0] = &f->u;
  f->X[1] = &f->p;
  PetscFunctionReturn(0);
}

PetscErrorCode StokesFormSetType(StokesForm *f,FormType type)
{
  f->type = type;
  PetscFunctionReturn(0);
}

PetscErrorCode StokesFormInit(StokesForm *f,FormType type,SurfaceConstraint sc)
{
  PetscErrorCode ierr;
  ierr = PetscMemzero(f,sizeof(StokesForm));CHKERRQ(ierr);
  f->type = type;
  if (sc) { f->sc = sc; }
  f->nqp = sc->nqp_facet;
  f->cell_i = -1;
  f->facet_sc_i = -1;
  f->facet_i = -1;
  f->point_i = -1;
  PetscFunctionReturn(0);
}

PetscErrorCode StokesFormFlush(StokesForm *f)
{
  f->u.W = NULL;
  f->u.Wx = NULL;
  f->u.Wy = NULL;
  f->u.Wz = NULL;
  f->p.W = NULL;
  f->p.Wx = NULL;
  f->p.Wy = NULL;
  f->p.Wz = NULL;
  f->x_elfield = NULL;
  f->u_elfield_0 = NULL;
  f->u_elfield_1 = NULL;
  f->u_elfield_2 = NULL;
  f->p_elfield_0 = NULL;
  f->cell_i = -1;
  f->facet_sc_i = -1;
  f->facet_i = -1;
  f->point_i = -1;
  f->test = NULL;
  f->trial = NULL;
  PetscFunctionReturn(0);
}

/*
 Generic "action" for surface integral terms.
 
 This variant calls the form->apply() method point-wise on each quadrature point.
 It will be simple to replace this with a variant which calls form->apply() one per cell.
 The trick to enable this (without changing the definition of Form is
 
 (i) set form->point_i = -1 (flag to indicate quad point index makes no sense)
 (ii) Allocate packed data for basis.
 Change
   double N[27];
 to
   double N[27 * nqp]
 then in form->apply() access via 
   Ni = &form->u->W[27*q]
 (iii) Tabulate quadrature points for velocity on all faces. qp[HEX_FACES][NQP][27]
 (iv)
*/
PetscErrorCode generic_facet_action(StokesForm *form,
                                           FunctionSpace *test_function,
                                           DM dm, /* always be q2 mesh */
                                           DM dau,const PetscScalar ufield[],
                                           DM dap,const PetscScalar pfield[],
                                           PetscScalar Y[])
{
  PetscErrorCode ierr;
  PetscReal Feu[3*Q2_NODES_PER_EL_3D];
  PetscReal elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal _elu[3*Q2_NODES_PER_EL_3D];
  PetscReal elu[3][Q2_NODES_PER_EL_3D];
  PetscReal N[Q2_NODES_PER_EL_3D];
  PetscReal Nxi[3][Q2_NODES_PER_EL_3D],Nx[3][Q2_NODES_PER_EL_3D];
  PetscReal Fep[P_BASIS_FUNCTIONS];
  PetscReal elp[P_BASIS_FUNCTIONS];
  PetscReal M[P_BASIS_FUNCTIONS];
  //PetscReal Mxi[3][P_BASIS_FUNCTIONS],Mx[3][P_BASIS_FUNCTIONS];
  PetscInt size = 0;
  PetscReal *Fe = NULL;
  PetscBool require_U = PETSC_FALSE;
  PetscBool require_P = PETSC_FALSE;
  
  PetscInt fe,i,q;
  PetscInt nel,nen_u,nen_p,nqp;
  DM cda;
  Vec gcoords;
  const PetscReal *LA_gcoords = NULL;
  const PetscInt *elnidx_u = NULL,*elnidx_p = NULL;
  PetscInt velocity_el_lidx[3*Q2_NODES_PER_EL_3D],pressure_el_lidx[P_BASIS_FUNCTIONS];
  ConformingElementFamily element;
  SurfaceConstraint sc;
  PetscLogDouble t0,t1;
  MPI_Comm comm;
  
  /* error checking */
  comm = PetscObjectComm((PetscObject)dm);
  if (!form->apply) SETERRQ(comm,PETSC_ERR_USER,"Form cannot be applied. form->apply() is NULL");
  
  if (form->type != FORM_SPMV && form->type != FORM_RESIDUAL) SETERRQ(comm,PETSC_ERR_USER,"Form type must be FORM_SPMV or FORM_RESIDUAL");
  
  if (ufield && !dau) SETERRQ(comm,PETSC_ERR_USER,"ufield[] required non-NULL dau");
  if (pfield && !dap) SETERRQ(comm,PETSC_ERR_USER,"pfield[] required non-NULL dap");

  if (form->type == FORM_RESIDUAL) {
    if (!dau) SETERRQ(comm,PETSC_ERR_USER,"FORM_RESIDUAL requires dau be non-NULL");
    if (!dap) SETERRQ(comm,PETSC_ERR_USER,"FORM_RESIDUAL requires dap be non-NULL");
    if (!ufield) SETERRQ(comm,PETSC_ERR_USER,"FORM_RESIDUAL requires ufield[] be non-NULL");
    if (!pfield) SETERRQ(comm,PETSC_ERR_USER,"FORM_RESIDUAL requires pfield[] be non-NULL");
  }
  
  if (test_function == &form->u) {
    if (!dau) SETERRQ(comm,PETSC_ERR_USER,"Test function u requires dau be non-NULL");
    size = form->u.nbasis * form->u.dim;
    Fe = Feu;
    require_U = PETSC_TRUE;
    
    if (ufield) {
      if (pfield) {
        printf("action will define residual F1\n");
        if (form->type != FORM_RESIDUAL) SETERRQ(comm,PETSC_ERR_USER,"Both u,p fields provided but form->type is not FORM_RESIDUAL");
        require_P = PETSC_TRUE;
      } else {
        printf("action will define SpMV A11 X1\n");
        if (form->type != FORM_SPMV) SETERRQ(comm,PETSC_ERR_USER,"form->type is not FORM_SPMV");
        form->trial = &form->u;
      }
    } else if (pfield) {
      if (ufield) { // done above - maps to residual F1
      } else {
        printf("action will define SpMV A12 X2\n");
        if (form->type != FORM_SPMV) SETERRQ(comm,PETSC_ERR_USER,"form->type is not FORM_SPMV");
        if (!dap) SETERRQ(comm,PETSC_ERR_USER,"SpMV A12 requires dap be non-NULL");
        form->trial = &form->p;
        require_P = PETSC_TRUE;
      }
    } else {
      SETERRQ(comm,PETSC_ERR_USER,"Input suggests you want to assemble an operator (A11 or A12)");
    }
  } else if (test_function == &form->p) {
    if (!dap) SETERRQ(comm,PETSC_ERR_USER,"Test function p requires dap be non-NULL");
    size = form->p.nbasis * form->p.dim;
    Fe = Fep;
    require_P = PETSC_TRUE;
    
    if (ufield) {
      if (pfield) {
        printf("action will define residual F2\n");
        if (form->type != FORM_RESIDUAL) SETERRQ(comm,PETSC_ERR_USER,"Both u,p fields provided but form->type is not FORM_RESIDUAL");
        require_U = PETSC_TRUE;
      } else {
        printf("action will define SpMV A21 X1\n");
        if (form->type != FORM_SPMV) SETERRQ(comm,PETSC_ERR_USER,"form->type is not FORM_SPMV");
        if (!dau) SETERRQ(comm,PETSC_ERR_USER,"SpMV A21 requires dau be non-NULL");
        form->trial = &form->u;
        require_U = PETSC_TRUE;
      }
    } else if (pfield) {
      if (ufield) { // done above - maps to residual F2
      } else {
        printf("action will define SpMV A22 X2\n");
        if (form->type != FORM_SPMV) SETERRQ(comm,PETSC_ERR_USER,"form->type is not FORM_SPMV");
        form->trial = &form->p;
      }
    } else {
      SETERRQ(comm,PETSC_ERR_USER,"Input suggests you want to assemble an operator (A21 or A22)");
    }
    
  } else {
    SETERRQ(comm,PETSC_ERR_USER,"Test function must be one of form->u or form->p");
  }

  /* Only set pointers for basis if you require them - this should make chasing bugs easier with gdb/valgrind */
  if (require_U) {
    switch (form->u.basis) {
      case BASIS_Q2_3:
        form->u.W = N;
        form->u.Wx = Nx[0]; form->u.Wy = Nx[1]; form->u.Wz = Nx[2];
        break;
        
      default:
        SETERRQ(comm,PETSC_ERR_SUP,"FunctionSpace(u): Only BASIS_Q2_3 are supported");
        break;
    }
  }
  
  if (require_P) {
    switch (form->p.basis) {
      case BASIS_P1_1:
        form->p.W = M;
        form->p.Wx = NULL; form->p.Wy = NULL; form->p.Wz = NULL;
        break;
        
      default:
        SETERRQ(comm,PETSC_ERR_SUP,"FunctionSpace(p): Only BASIS_P1_1 are supported");
        break;
    }
  }
  
  sc = form->sc;
  element = sc->fi->element;

  form->x_elfield = elcoords; /* pack form */
  form->u_elfield_0 = elu[0]; /* pack form */
  form->u_elfield_1 = elu[1]; /* pack form */
  form->u_elfield_2 = elu[2]; /* pack form */
  form->p_elfield_0 = elp;    /* pack form */
  form->test = test_function; /* pack form */
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(dm,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = DMDAGetElements_pTatinQ2P1(dm,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr); /* coordinates */
  if (dau) {
    ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  }
  if (dap) {
    ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);
  }
  
  /* Get access to any data required by the form */
  if (form->access) {
    ierr = form->access(form);CHKERRQ(ierr);
  }
  
  PetscTime(&t0);
  for (fe=0; fe<sc->facets->n_entities; fe++) {
    PetscInt          facet_index,cell_side,cell_index;
    QPoint2d          *qp2 = NULL;
    QPoint3d          *qp3 = NULL;
    
    facet_index = sc->facets->local_index[fe]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    
    nqp = sc->quadrature->npoints;
    qp2 = sc->quadrature->gp2[cell_side];
    qp3 = sc->quadrature->gp3[cell_side];
    
    /* pack form */
    form->nqp = nqp;

    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*cell_index],(PetscReal*)LA_gcoords);CHKERRQ(ierr);
    
    if (dau) {
      ierr = StokesVelocity_GetElementLocalIndices(velocity_el_lidx,(PetscInt*)&elnidx_u[nen_u*cell_index]);CHKERRQ(ierr);
    }
    if (ufield) {
      ierr = DMDAGetVectorElementFieldQ2_3D(_elu,(PetscInt*)&elnidx_u[nen_u*cell_index],(PetscReal*)ufield);CHKERRQ(ierr);
      for (i=0; i<nen_u; i++) {
        elu[0][i] = _elu[3*i+0];
        elu[1][i] = _elu[3*i+1];
        elu[2][i] = _elu[3*i+2];
      }
    }
    
    if (dap) {
      ierr = StokesPressure_GetElementLocalIndices(pressure_el_lidx,(PetscInt*)&elnidx_p[nen_p*cell_index]);CHKERRQ(ierr);
    }
    if (pfield) {
      ierr = DMDAGetScalarElementField(elp,nen_p,(PetscInt*)&elnidx_p[nen_p*cell_index],(PetscReal*)pfield);CHKERRQ(ierr);
    }
    
    /* initialise element stiffness matrix */
    ierr = PetscMemzero(Fe,sizeof(PetscScalar)*size);CHKERRQ(ierr);
    
    for (q=0; q<nqp; q++) {
      PetscScalar fac,J_q,surfJ_q;
      PetscScalar xip[] = { qp3[q].xi, qp3[q].eta, qp3[q].zeta };
      
      element->compute_surface_geometry_3D(element,
                                           elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                           cell_side,   // edge index 0,...,7 //
                                           &qp2[q], // should contain 1 point with dimension 2 (xi,eta)   //
                                           NULL,NULL,&surfJ_q); // n0[],t0 contains 1 point with dimension 3 (x,y,z) //
      fac = qp2[q].w * surfJ_q;
      
      if (require_U) {
        /* evaluate the basis for u, gradu */
        //element->basis_NI_3D(&qp3[q],N);
        P3D_ConstructNi_Q2_3D(xip,N);
        //element->basis_GNI_3D(&qp3[q],Nxi);
        P3D_ConstructGNi_Q2_3D(xip,Nxi);
        P3D_evaluate_geometry_elementQ2(1,elcoords,&Nxi,&J_q,&Nx[0],&Nx[1],&Nx[2]);
      }
      if (require_P) {
        /* evaluate the basis for p */
        ConstructNi_pressure(xip,elcoords,M);
      }
      
      /* form->apply() */
      /* pack form */
      form->cell_i     = cell_index;
      form->facet_sc_i = fe; /* required to access qp data associated with SurfacConstraint */
      form->facet_i    = facet_index; /* required to access qp data assocuated with SurfaceQuadrature */
      form->point_i    = q; /* Required for point-wise iterator */
      
      ierr = form->apply(form,&fac,Fe);CHKERRQ(ierr);
    }
    
    if (test_function == &form->u) {
      ierr = DMDASetValuesLocalStencil_AddValues_Stokes_Velocity(Y,velocity_el_lidx,Fe);CHKERRQ(ierr);
    }
    if (test_function == &form->p) {
      ierr = DMDASetValuesLocalStencil_AddValues_Stokes_Pressure(Y,pressure_el_lidx,Fe);CHKERRQ(ierr);
    }
  }
  PetscTime(&t1);
  {
    double time = (double)(t1 - t0);
    ierr = MPI_Allreduce(MPI_IN_PLACE,&time,1,MPI_REAL,MPI_MAX,comm);CHKERRQ(ierr);
    PetscPrintf(comm,"generic_facet_action(): Assembled form in %1.4e (sec) [max-collective]\n",time);
  }

  /* Restore access to any data required by the form */
  if (form->restore) {
    ierr = form->restore(form);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = StokesFormFlush(form);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PetscErrorCode generic_facet_assemble(StokesForm *form,
                                    FunctionSpace *test_function,
                                    FunctionSpace *trial_function,
                                    DM dm, /* always be q2 mesh */
                                    DM dau,
                                    DM dap,
                                    Mat A)
{
  PetscErrorCode ierr;
  PetscReal *Ae_ij = NULL;
  PetscReal elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal N[Q2_NODES_PER_EL_3D];
  PetscReal Nxi[3][Q2_NODES_PER_EL_3D],Nx[3][Q2_NODES_PER_EL_3D];
  PetscReal M[P_BASIS_FUNCTIONS];
  //PetscReal Mxi[3][P_BASIS_FUNCTIONS],Mx[3][P_BASIS_FUNCTIONS];
  PetscInt size = 0;
  PetscBool require_U = PETSC_FALSE;
  PetscBool require_P = PETSC_FALSE;
  PetscBool use_set_values_local = PETSC_FALSE;
  ISLocalToGlobalMapping ltog_u = NULL,ltog_p = NULL;
  const PetscInt *GINDICES_p;
  const PetscInt *GINDICES_u;
  PetscInt       NUM_GINDICES_u,ge_eqnums_u[3*Q2_NODES_PER_EL_3D];
  PetscInt       NUM_GINDICES_p,ge_eqnums_p[P_BASIS_FUNCTIONS];
  PetscInt fe,i,q;
  PetscInt nel,nen_u,nen_p,nqp;
  DM cda;
  Vec gcoords;
  const PetscReal *LA_gcoords = NULL;
  const PetscInt *elnidx_u = NULL,*elnidx_p = NULL;
  PetscInt velocity_el_lidx[3*Q2_NODES_PER_EL_3D],pressure_el_lidx[P_BASIS_FUNCTIONS];
  PetscInt *test_lidx = NULL,*trial_lidx = NULL;
  PetscInt *test_gidx = NULL,*trial_gidx = NULL,test_size,trial_size;
  ConformingElementFamily element;
  SurfaceConstraint sc;
  PetscLogDouble t0,t1;
  MPI_Comm comm;
  
  
  /* error checking */
  comm = PetscObjectComm((PetscObject)dm);
  if (!form->apply) SETERRQ(comm,PETSC_ERR_USER,"Form cannot be applied. form->apply() is NULL");
  
  if (form->type != FORM_ASSEMBLE) SETERRQ(comm,PETSC_ERR_USER,"Form type must be FORM_ASSEMBLE");

  if (test_function == &form->u) {
    if (!dau) SETERRQ(comm,PETSC_ERR_USER,"TestFunction[u]: FORM_ASSEMBLE requires dau be non-NULL");
  }
  if (test_function == &form->p) {
    if (!dap) SETERRQ(comm,PETSC_ERR_USER,"TestFunction[p]: FORM_ASSEMBLE requires dap be non-NULL");
  }

  if (trial_function == &form->u) {
    if (!dau) SETERRQ(comm,PETSC_ERR_USER,"TrialFunction[u]: FORM_ASSEMBLE requires dau be non-NULL");
    if ((test_function == trial_function) && dap) {
      SETERRQ(comm,PETSC_ERR_USER,"TestFunction[u] = TrialFunction[u]: dap must be NULL");
    }
  }
  if (trial_function == &form->p) {
    if (!dap) SETERRQ(comm,PETSC_ERR_USER,"TrialFunction[p]: FORM_ASSEMBLE requires dap be non-NULL");
    if ((test_function == trial_function) && dau) {
      SETERRQ(comm,PETSC_ERR_USER,"TestFunction[p] = TrialFunction[p]: dau must be NULL");
    }
  }

  form->test = test_function;   /* pack form */
  form->trial = trial_function; /* pack form */

  test_size  = form->test->nbasis * form->test->dim;
  trial_size = form->trial->nbasis * form->trial->dim;
  
  size = test_size * trial_size;
  ierr = PetscMalloc1(size,&Ae_ij);CHKERRQ(ierr);

  if (form->test == &form->u || form->trial == &form->u) {
    require_U = PETSC_TRUE;
  }
  if (form->test == &form->p || form->trial == &form->p) {
    require_P = PETSC_TRUE;
  }

  if (form->test == &form->u) {
    test_lidx = velocity_el_lidx;
    test_gidx = ge_eqnums_u;
  } else if (form->test == &form->p) {
    test_lidx = pressure_el_lidx;
    test_gidx = ge_eqnums_p;
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Cannot evaluate form - test function space is not in form->X[]");
  }

  if (form->trial == &form->u) {
    trial_lidx = velocity_el_lidx;
    trial_gidx = ge_eqnums_u;
  } else if (form->trial == &form->p) {
    trial_lidx = pressure_el_lidx;
    trial_gidx = ge_eqnums_p;
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Cannot evaluate form - trial function space is not in form->X[]");
  }

  if ((form->test == form->trial) && (form->test == &form->u)) {
    use_set_values_local = PETSC_TRUE;
  }
  
  /* Only set pointers for basis if you require them - this should make chasing bugs easier with gdb/valgrind */
  if (require_U) {
    switch (form->u.basis) {
      case BASIS_Q2_3:
        form->u.W = N;
        form->u.Wx = Nx[0]; form->u.Wy = Nx[1]; form->u.Wz = Nx[2];
        break;
        
      default:
        SETERRQ(comm,PETSC_ERR_SUP,"FunctionSpace(u): Only BASIS_Q2_3 are supported");
        break;
    }
  }
  
  if (require_P) {
    switch (form->p.basis) {
      case BASIS_P1_1:
        form->p.W = M;
        form->p.Wx = NULL; form->p.Wy = NULL; form->p.Wz = NULL;
        break;
        
      default:
        SETERRQ(comm,PETSC_ERR_SUP,"FunctionSpace(p): Only BASIS_P1_1 are supported");
        break;
    }
  }
  
  sc = form->sc;
  element = sc->fi->element;
  
  form->x_elfield = elcoords; /* pack form */

  /* setup for coords */
  ierr = DMGetCoordinateDM(dm,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = DMDAGetElements_pTatinQ2P1(dm,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr); /* coordinates */
  if (dau) {
    ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
    
    ierr = DMGetLocalToGlobalMapping(dau, &ltog_u);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(ltog_u, &NUM_GINDICES_u);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltog_u, &GINDICES_u);CHKERRQ(ierr);
  }
  if (dap) {
    ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);
    
    ierr = DMGetLocalToGlobalMapping(dap, &ltog_p);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(ltog_p, &NUM_GINDICES_p);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltog_p, &GINDICES_p);CHKERRQ(ierr);
  }
  
  /* Get access to any data required by the form */
  if (form->access) {
    ierr = form->access(form);CHKERRQ(ierr);
  }
  
  PetscTime(&t0);
  for (fe=0; fe<sc->facets->n_entities; fe++) {
    PetscInt          facet_index,cell_side,cell_index;
    QPoint2d          *qp2 = NULL;
    QPoint3d          *qp3 = NULL;
    
    facet_index = sc->facets->local_index[fe]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    
    nqp = sc->quadrature->npoints;
    qp2 = sc->quadrature->gp2[cell_side];
    qp3 = sc->quadrature->gp3[cell_side];
    
    /* pack form */
    form->nqp = nqp;
    
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*cell_index],(PetscReal*)LA_gcoords);CHKERRQ(ierr);
    
    if (dau) {
      ierr = StokesVelocity_GetElementLocalIndices(velocity_el_lidx,(PetscInt*)&elnidx_u[nen_u*cell_index]);CHKERRQ(ierr);
      
      /* get global indices */ // U
      for (i=0; i<nen_u; i++) {
        const int nid = elnidx_u[nen_u * cell_index + i];
        ge_eqnums_u[3*i  ] = GINDICES_u[ 3*nid   ];
        ge_eqnums_u[3*i+1] = GINDICES_u[ 3*nid+1 ];
        ge_eqnums_u[3*i+2] = GINDICES_u[ 3*nid+2 ];
      }
    }
    
    if (dap) {
      ierr = StokesPressure_GetElementLocalIndices(pressure_el_lidx,(PetscInt*)&elnidx_p[nen_p*cell_index]);CHKERRQ(ierr);
      
      // P
      for (i=0; i<nen_p; i++) {
        const int nid = elnidx_p[nen_p * cell_index + i];
        ge_eqnums_p[i] = GINDICES_p[ nid ];
      }
    }
    
    /* initialise element stiffness matrix */
    ierr = PetscMemzero(Ae_ij,sizeof(PetscScalar)*size);CHKERRQ(ierr);
    
    for (q=0; q<nqp; q++) {
      PetscScalar fac,J_q,surfJ_q;
      PetscScalar xip[] = { qp3[q].xi, qp3[q].eta, qp3[q].zeta };
      
      element->compute_surface_geometry_3D(element,
                                           elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                           cell_side,   // edge index 0,...,7 //
                                           &qp2[q], // should contain 1 point with dimension 2 (xi,eta)   //
                                           NULL,NULL,&surfJ_q); // n0[],t0 contains 1 point with dimension 3 (x,y,z) //
      fac = qp2[q].w * surfJ_q;
      
      if (require_U) { /* evaluate the basis for u, gradu */
        P3D_ConstructNi_Q2_3D(xip,N);
        P3D_ConstructGNi_Q2_3D(xip,Nxi);
        P3D_evaluate_geometry_elementQ2(1,elcoords,&Nxi,&J_q,&Nx[0],&Nx[1],&Nx[2]);
      }
      if (require_P) { /* evaluate the basis for p */
        ConstructNi_pressure(xip,elcoords,M);
      }
      
      /* form->apply() */ /* pack form */
      form->cell_i     = cell_index;
      form->facet_sc_i = fe; /* required to access qp data associated with SurfacConstraint */
      form->facet_i    = facet_index; /* required to access qp data assocuated with SurfaceQuadrature */
      form->point_i    = q; /* Required for point-wise iterator */
      
      ierr = form->apply(form,&fac,Ae_ij);CHKERRQ(ierr);
    }

    if (use_set_values_local) {
      ierr = MatSetValuesLocal(A, test_size,test_lidx, trial_size,trial_gidx, Ae_ij,ADD_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatSetValues(A, test_size,test_gidx, trial_size,trial_gidx, Ae_ij,ADD_VALUES);CHKERRQ(ierr);
    }
    
  }
  ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  
  PetscTime(&t1);
  {
    double time = (double)(t1 - t0);
    ierr = MPI_Allreduce(MPI_IN_PLACE,&time,1,MPI_REAL,MPI_MAX,comm);CHKERRQ(ierr);
    PetscPrintf(comm,"generic_facet_assemble(): Assembled form in %1.4e (sec) [max-collective]\n",time);
  }
  
  /* Restore access to any data required by the form */
  if (form->restore) {
    ierr = form->restore(form);CHKERRQ(ierr);
  }
  
  if (dau) {
    ierr = ISLocalToGlobalMappingRestoreIndices(ltog_u, &GINDICES_u);CHKERRQ(ierr);
  }
  if (dap) {
    ierr = ISLocalToGlobalMappingRestoreIndices(ltog_p, &GINDICES_p);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = StokesFormFlush(form);CHKERRQ(ierr);
  ierr = PetscFree(Ae_ij);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PetscErrorCode generic_facet_assemble_diagonal(StokesForm *form,
                                      FunctionSpace *test_function,
                                      DM dm, /* always be q2 mesh */
                                      DM dmX,
                                      Vec diagA)
{
  PetscErrorCode ierr;
  PetscReal *Ae_ij = NULL;
  PetscReal elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal N[Q2_NODES_PER_EL_3D];
  PetscReal Nxi[3][Q2_NODES_PER_EL_3D],Nx[3][Q2_NODES_PER_EL_3D];
  PetscReal M[P_BASIS_FUNCTIONS];
  //PetscReal Mxi[3][P_BASIS_FUNCTIONS],Mx[3][P_BASIS_FUNCTIONS];
  PetscInt size = 0;
  PetscBool require_U = PETSC_FALSE;
  PetscBool require_P = PETSC_FALSE;
  PetscBool use_set_values_local = PETSC_FALSE;
  ISLocalToGlobalMapping ltog = NULL;
  const PetscInt *GINDICES;
  PetscInt       NUM_GINDICES;
  PetscInt ge_eqnums_u[3*Q2_NODES_PER_EL_3D],ge_eqnums_p[P_BASIS_FUNCTIONS];
  PetscInt fe,i,q;
  PetscInt nel,nen,nen_c,nqp;
  DM cda,dau = NULL,dap = NULL;
  Vec gcoords;
  const PetscReal *LA_gcoords = NULL;
  const PetscInt *elnidx = NULL;
  const PetscInt *elnidx_c = NULL;
  PetscInt velocity_el_lidx[3*Q2_NODES_PER_EL_3D],pressure_el_lidx[P_BASIS_FUNCTIONS];
  PetscInt *test_lidx = NULL;
  PetscInt *test_gidx = NULL,test_size;
  ConformingElementFamily element;
  SurfaceConstraint sc;
  PetscLogDouble t0,t1;
  MPI_Comm comm;
  
  
  /* error checking */
  comm = PetscObjectComm((PetscObject)dm);
  if (!form->apply) SETERRQ(comm,PETSC_ERR_USER,"Form cannot be applied. form->apply() is NULL");
  
  if (form->type != FORM_ASSEMBLE) SETERRQ(comm,PETSC_ERR_USER,"Form type must be FORM_ASSEMBLE");
  if (!dm) SETERRQ(comm,PETSC_ERR_USER,"Form ASSEMBLE_DIAG requires non-NULL dm");
  if (!dmX) SETERRQ(comm,PETSC_ERR_USER,"Form ASSEMBLE_DIAG requires non-NULL dmX");

  if (test_function == &form->u) {
    dau = dmX;
  }
  if (test_function == &form->p) {
    dau = dmX;
  }
  
  form->test = test_function;   /* pack form */
  form->trial = test_function; /* pack form */
  
  test_size = form->test->nbasis * form->test->dim;
  
  size = test_size;
  ierr = PetscMalloc1(size,&Ae_ij);CHKERRQ(ierr);
  
  if (form->test == &form->u || form->trial == &form->u) {
    require_U = PETSC_TRUE;
  }
  if (form->test == &form->p || form->trial == &form->p) {
    require_P = PETSC_TRUE;
  }
  
  if (form->test == &form->u) {
    test_lidx = velocity_el_lidx;
    test_gidx = ge_eqnums_u;
  } else if (form->test == &form->p) {
    test_lidx = pressure_el_lidx;
    test_gidx = ge_eqnums_p;
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Cannot evaluate form - test function space is not in form->X[]");
  }
  
  if ((form->test == form->trial) && (form->test == &form->u)) {
    use_set_values_local = PETSC_TRUE;
  }
  
  /* Only set pointers for basis if you require them - this should make chasing bugs easier with gdb/valgrind */
  if (require_U) {
    switch (form->u.basis) {
      case BASIS_Q2_3:
        form->u.W = N;
        form->u.Wx = Nx[0]; form->u.Wy = Nx[1]; form->u.Wz = Nx[2];
        break;
        
      default:
        SETERRQ(comm,PETSC_ERR_SUP,"FunctionSpace(u): Only BASIS_Q2_3 are supported");
        break;
    }
  }
  
  if (require_P) {
    switch (form->p.basis) {
      case BASIS_P1_1:
        form->p.W = M;
        form->p.Wx = NULL; form->p.Wy = NULL; form->p.Wz = NULL;
        break;
        
      default:
        SETERRQ(comm,PETSC_ERR_SUP,"FunctionSpace(p): Only BASIS_P1_1 are supported");
        break;
    }
  }
  
  sc = form->sc;
  element = sc->fi->element;
  
  form->x_elfield = elcoords; /* pack form */
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(dm,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = DMDAGetElements_pTatinQ2P1(dm,&nel,&nen_c,&elnidx_c);CHKERRQ(ierr); /* coordinates */
  if (dau) {
    ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen,&elnidx);CHKERRQ(ierr);
    
    ierr = DMGetLocalToGlobalMapping(dau, &ltog);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES);CHKERRQ(ierr);
  }
  if (dap) {
    ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen,&elnidx);CHKERRQ(ierr);
    
    ierr = DMGetLocalToGlobalMapping(dap, &ltog);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(ltog, &NUM_GINDICES);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltog, &GINDICES);CHKERRQ(ierr);
  }
  
  /* Get access to any data required by the form */
  if (form->access) {
    ierr = form->access(form);CHKERRQ(ierr);
  }
  
  PetscTime(&t0);
  for (fe=0; fe<sc->facets->n_entities; fe++) {
    PetscInt          facet_index,cell_side,cell_index;
    QPoint2d          *qp2 = NULL;
    QPoint3d          *qp3 = NULL;
    
    facet_index = sc->facets->local_index[fe]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    
    nqp = sc->quadrature->npoints;
    qp2 = sc->quadrature->gp2[cell_side];
    qp3 = sc->quadrature->gp3[cell_side];
    
    /* pack form */
    form->nqp = nqp;
    
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_c[nen_c*cell_index],(PetscReal*)LA_gcoords);CHKERRQ(ierr);
    
    if (dau) {
      ierr = StokesVelocity_GetElementLocalIndices(velocity_el_lidx,(PetscInt*)&elnidx[nen*cell_index]);CHKERRQ(ierr);
      
      /* get global indices */ // U
      for (i=0; i<nen; i++) {
        const int nid = elnidx[nen * cell_index + i];
        ge_eqnums_u[3*i  ] = GINDICES[ 3*nid   ];
        ge_eqnums_u[3*i+1] = GINDICES[ 3*nid+1 ];
        ge_eqnums_u[3*i+2] = GINDICES[ 3*nid+2 ];
      }
    }
    
    if (dap) {
      ierr = StokesPressure_GetElementLocalIndices(pressure_el_lidx,(PetscInt*)&elnidx[nen*cell_index]);CHKERRQ(ierr);
      
      // P
      for (i=0; i<nen; i++) {
        const int nid = elnidx[nen * cell_index + i];
        ge_eqnums_p[i] = GINDICES[ nid ];
      }
    }
    
    /* initialise element stiffness matrix */
    ierr = PetscMemzero(Ae_ij,sizeof(PetscScalar)*size);CHKERRQ(ierr);
    
    for (q=0; q<nqp; q++) {
      PetscScalar fac,J_q,surfJ_q;
      PetscScalar xip[] = { qp3[q].xi, qp3[q].eta, qp3[q].zeta };
      
      element->compute_surface_geometry_3D(element,
                                           elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                           cell_side,   // edge index 0,...,7 //
                                           &qp2[q], // should contain 1 point with dimension 2 (xi,eta)   //
                                           NULL,NULL,&surfJ_q); // n0[],t0 contains 1 point with dimension 3 (x,y,z) //
      fac = qp2[q].w * surfJ_q;
      
      if (require_U) { /* evaluate the basis for u, gradu */
        P3D_ConstructNi_Q2_3D(xip,N);
        P3D_ConstructGNi_Q2_3D(xip,Nxi);
        P3D_evaluate_geometry_elementQ2(1,elcoords,&Nxi,&J_q,&Nx[0],&Nx[1],&Nx[2]);
      }
      if (require_P) { /* evaluate the basis for p */
        ConstructNi_pressure(xip,elcoords,M);
      }
      
      /* form->apply() */ /* pack form */
      form->cell_i     = cell_index;
      form->facet_sc_i = fe; /* required to access qp data associated with SurfacConstraint */
      form->facet_i    = facet_index; /* required to access qp data assocuated with SurfaceQuadrature */
      form->point_i    = q; /* Required for point-wise iterator */
      
      ierr = form->apply(form,&fac,Ae_ij);CHKERRQ(ierr);
    }
    
    if (use_set_values_local) {
      ierr = VecSetValuesLocal(diagA, test_size,test_lidx, Ae_ij,ADD_VALUES);CHKERRQ(ierr);
    } else {
      ierr = VecSetValues(diagA, test_size,test_gidx, Ae_ij,ADD_VALUES);CHKERRQ(ierr);
    }
    
  }
  ierr = VecAssemblyBegin(diagA);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(diagA);CHKERRQ(ierr);
  
  PetscTime(&t1);
  {
    double time = (double)(t1 - t0);
    ierr = MPI_Allreduce(MPI_IN_PLACE,&time,1,MPI_REAL,MPI_MAX,comm);CHKERRQ(ierr);
    PetscPrintf(comm,"generic_facet_assemble_diagonal(): Assembled form in %1.4e (sec) [max-collective]\n",time);
  }
  
  /* Restore access to any data required by the form */
  if (form->restore) {
    ierr = form->restore(form);CHKERRQ(ierr);
  }
  
  if (ltog) {
    ierr = ISLocalToGlobalMappingRestoreIndices(ltog, &GINDICES);CHKERRQ(ierr);
  }
  
  ierr = VecRestoreArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  ierr = StokesFormFlush(form);CHKERRQ(ierr);
  ierr = PetscFree(Ae_ij);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}



