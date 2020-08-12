/*
 Tests the following functionality
 
 - Methods associated with type(FVReconstructionCell)
     * FVReconstructionP1Create();
     * FVReconstructionP1Interpolate();

 - Methods associated with type(DIMap)
     * DIMapCreate_FVDACell_RankLocalToLocal();
     * DIMapCreate_FVDACell_LocalToRankLocal();
     * DIMapApply()
*/

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <ptatin3d_defs.h>
#include <ptatin3d.h>
#include <quadrature.h>
#include <element_utils_q1.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_private.h>
#include <fvda_utils.h>

PetscInt test_function_type = 1;

PetscErrorCode func_1(PetscScalar x[],PetscScalar val[])
{
  val[0] = x[0] * x[0] * 2.2 + x[1] + x[1] * x[2] * 0.2;
  return 0;
}

PetscErrorCode grad_func_1(PetscScalar x[],PetscScalar val[])
{
  val[0] = x[0] * 4.4;
  val[1] = 1.0 + x[2] * 0.2;
  val[2] = x[1] * 0.2;
  return 0;
}

PetscErrorCode func_2(PetscScalar x[],PetscScalar val[])
{
  val[0] = x[0] * PetscSinReal(PETSC_PI * 3.3 * x[0]) * PetscCosReal(PETSC_PI * 1.2 * x[1])
         + x[1] * PetscSinReal(PETSC_PI * 3.3 * x[1]) * PetscCosReal(PETSC_PI * 1.2 * x[2])
         + x[2] * PetscSinReal(PETSC_PI * 3.3 * x[0]) * PetscCosReal(PETSC_PI * 1.2 * x[2]);
  return 0;
}

PetscErrorCode grad_func_2(PetscScalar x[],PetscScalar val[])
{
  val[0] = PetscSinReal(PETSC_PI * 3.3 * x[0]) * PetscCosReal(PETSC_PI * 1.2 * x[1])
         + x[0] * PETSC_PI * 3.3 * PetscCosReal(PETSC_PI * 3.3 * x[0]) * PetscCosReal(PETSC_PI * 1.2 * x[1])
         + x[2] * PETSC_PI * 3.3 * PetscCosReal(PETSC_PI * 3.3 * x[0]) * PetscCosReal(PETSC_PI * 1.2 * x[2]);
  
  val[1] = -x[0] * PetscSinReal(PETSC_PI * 3.3 * x[0]) * PetscSinReal(PETSC_PI * 1.2 * x[1]) * PETSC_PI * 1.2
         + PetscSinReal(PETSC_PI * 3.3 * x[1]) * PetscCosReal(PETSC_PI * 1.2 * x[2])
         + x[1] * PETSC_PI * 3.3 * PetscCosReal(PETSC_PI * 3.3 * x[1]) * PetscCosReal(PETSC_PI * 1.2 * x[2]);
  
  val[2] = -x[1] * PetscSinReal(PETSC_PI * 3.3 * x[1]) * PetscSinReal(PETSC_PI * 1.2 * x[2]) * PETSC_PI * 1.2
         + PetscSinReal(PETSC_PI * 3.3 * x[0]) * PetscCosReal(PETSC_PI * 1.2 * x[2])
         - x[2] * PetscSinReal(PETSC_PI * 3.3 * x[0]) * PetscSinReal(PETSC_PI * 1.2 * x[2]) * PETSC_PI * 1.2;
  return 0;
}

PetscErrorCode func(PetscScalar x[],PetscScalar val[])
{
  PetscErrorCode ierr;
  switch (test_function_type) {
    case 1:
      ierr = func_1(x,val);CHKERRQ(ierr);
      break;
    case 2:
      ierr = func_2(x,val);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown test function selected");
      break;
  }
  return 0;
}

PetscErrorCode grad_func(PetscScalar x[],PetscScalar val[])
{
  PetscErrorCode ierr;
  switch (test_function_type) {
    case 1:
      ierr = grad_func_1(x,val);CHKERRQ(ierr);
      break;
    case 2:
      ierr = grad_func_2(x,val);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown gradient test function selected");
      break;
  }
  return 0;
}



PetscBool func_eval(PetscScalar x[],PetscScalar val[],void *ctx)
{
  PetscErrorCode ierr;
  ierr = func(x,val);CHKERRABORT(PETSC_COMM_SELF,ierr);
  return PETSC_TRUE;
}


PetscErrorCode t1(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 12, cells = 0;
  PetscInt       m[] = {mx,mx,mx};
  FVDA           fv;
  PetscBool      found = PETSC_FALSE;
  Vec            Q,Ql,fv_coor_local,geometry_coorl;
  DIMap          map_l2rl,map_rl2l;
  PetscInt       c;
  FVReconstructionCell rcell;
  const PetscReal      *_Q,*_fv_coor,*_geom_coor;
  PetscReal            dx[3];
  PetscBool            view = PETSC_FALSE;
  PetscReal            l2Q,volOmega;
  PetscInt             dm_nel,dm_nen;
  const PetscInt       *dm_element,*element;
  PetscReal            cell_coor[3*DACELL3D_VERTS];
  PetscInt             q,nqp,i;
  PetscReal            *q_xi,*q_w;
  PetscReal            N[8][8],gradN_xi[8][3][8],gradN_x[8][8],gradN_y[8][8],gradN_z[8][8],detJ[8];
  PetscLogDouble       t0,t1,dt[2];
  
  
  ierr = PetscOptionsGetBool(NULL,NULL,"-view",&view,NULL);CHKERRQ(ierr);
  found = PETSC_FALSE; ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&found);CHKERRQ(ierr);
  if (found) {
    m[0] = mx;
    m[1] = mx;
    m[2] = mx;
  }
  found = PETSC_FALSE; ierr = PetscOptionsGetInt(NULL,NULL,"-my",&cells,&found);CHKERRQ(ierr);
  if (found) { m[1] = cells; }
  found = PETSC_FALSE; ierr = PetscOptionsGetInt(NULL,NULL,"-mz",&cells,&found);CHKERRQ(ierr);
  if (found) { m[2] = cells; }

  dx[0] = 2.0 / ((PetscReal)m[0]);
  dx[1] = 2.0 / ((PetscReal)m[1]);
  dx[2] = 2.0 / ((PetscReal)m[2]);
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  {
    Vec vertex_coor_geometry,coor;
    ierr = FVDAGetGeometryCoordinates(fv,&vertex_coor_geometry);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&coor);CHKERRQ(ierr);
    ierr = VecCopy(coor,vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"k*",1);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(fv->dm_fv,&Q);CHKERRQ(ierr);

  ierr = FVDAVecTraverse(fv,Q,0.0,0,func_eval,NULL);CHKERRQ(ierr);
  
  if (view) {
    ierr = FVDAView_CellData(fv,Q,PETSC_FALSE,"ex9_xcell");CHKERRQ(ierr);
  }
  
  ierr = DIMapCreate_FVDACell_RankLocalToLocal(fv,&map_rl2l);CHKERRQ(ierr);
  ierr = DIMapCreate_FVDACell_LocalToRankLocal(fv,&map_l2rl);CHKERRQ(ierr);

  /* Pass through just to check no errors occur */
  for (c=0; c<fv->ncells; c++) {
    PetscInt l;
    ierr = DIMapApply(map_rl2l,c,&l);CHKERRQ(ierr);
  }
  
  ierr = DMGetCoordinatesLocal(fv->dm_fv,&fv_coor_local);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_fv,&Ql);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_fv,Q,INSERT_VALUES,Ql);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fv_coor_local,&_fv_coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Ql,&_Q);CHKERRQ(ierr);

  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  QuadratureCreateGauss_2pnt_3D(&nqp,&q_xi,&q_w);
  for (q=0; q<nqp; q++) {
    /* ptatin3d uses different element ordering to petsc's returned from DMDAGetElements() */
    /* Hence use the FV EvalBasis methods */
    //P3D_ConstructNi_Q1_3D(&q_xi[3*q],N[q]);
    //P3D_ConstructGNi_Q1_3D(&q_xi[3*q],gradN_xi[q]);
    
    EvaluateBasis_Q1_3D(&q_xi[3*q],N[q]);
    EvaluateBasisDerivative_Q1_3D(&q_xi[3*q],gradN_xi[q]);
  }
  
  /* compute error wrt piece-wise constant data */
  volOmega = 0.0;
  l2Q = 0.0;
  for (c=0; c<fv->ncells; c++) {
    PetscInt l;
    PetscReal Q_exact,Q_fv;

    ierr = DIMapApply(map_rl2l,c,&l);CHKERRQ(ierr);

    Q_fv = _Q[l];

    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    
    P3D_evaluate_geometry_elementQ1(nqp,cell_coor,gradN_xi,detJ,gradN_x,gradN_y,gradN_z);
    for (q=0; q<nqp; q++) {
      PetscReal q_x[] = {0,0,0};
      for (i=0; i<DACELL3D_Q1_SIZE; i++) {
        q_x[0] += N[q][i] * cell_coor[3*i+0];
        q_x[1] += N[q][i] * cell_coor[3*i+1];
        q_x[2] += N[q][i] * cell_coor[3*i+2];
      }
      ierr = func(q_x,&Q_exact);CHKERRQ(ierr);
      
      volOmega += 1.0 * detJ[q];
      l2Q += (Q_exact - Q_fv)*(Q_exact - Q_fv) * detJ[q];
    }
  }
  PetscPrintf(fv->comm,"Test function: %D\n",test_function_type);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&volOmega,1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  PetscPrintf(fv->comm,"Domain(volume)       %1.4e\n",volOmega);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&l2Q,1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  l2Q = PetscSqrtReal(l2Q);
  PetscPrintf(fv->comm,"Q(L2)                %1.4e\n",l2Q);
  
#if 0
  for (c=0; c<fv->ncells; c++) {
    PetscInt l,ii,jj,kk;
    PetscReal xc[3],ival,Qref;
    
    ierr = DIMapApply(map_rl2l,c,&l);CHKERRQ(ierr);
    
    ierr = FVReconstructionP1Create(&rcell,fv,l,_fv_coor,_Q);CHKERRQ(ierr);
    
    xc[0] = _fv_coor[3*l+0];
    xc[1] = _fv_coor[3*l+1];
    xc[2] = _fv_coor[3*l+2];
    Qref = _Q[l];
    //ierr = FVReconstructionP1Interpolate(&rcell,xc,&ival);CHKERRQ(ierr);
    //printf(" ival %g : Q0 %g\n",ival,Qref);

    printf("c %d -> l %d\n",c,l);
    for (kk=0; kk<4; kk++) {
      for (jj=0; jj<4; jj++) {
        for (ii=0; ii<4; ii++) {
          xc[0] = _fv_coor[3*l+0] - 0.5 * dx[0] + (dx[0]/(4.0-1.0))*ii;
          xc[1] = _fv_coor[3*l+1] - 0.5 * dx[1] + (dx[1]/(4.0-1.0))*jj;
          xc[2] = _fv_coor[3*l+2] - 0.5 * dx[2] + (dx[2]/(4.0-1.0))*kk;
          
          ierr = FVReconstructionP1Interpolate(&rcell,xc,&ival);CHKERRQ(ierr);
          printf(" ival %g : Q0 %g\n",ival,Qref);
          
    }}}
  }
#endif

  dt[0] = dt[1] = 0.0;
  
  /* compute error wrt P1 reconstructed data */
  l2Q = 0.0;
  for (c=0; c<fv->ncells; c++) {
    PetscInt l;
    PetscReal Q_exact,Q_fv;

    ierr = DIMapApply(map_rl2l,c,&l);CHKERRQ(ierr);

    PetscTime(&t0);
    ierr = FVReconstructionP1Create(&rcell,fv,l,_fv_coor,_Q);CHKERRQ(ierr);
    PetscTime(&t1);
    dt[0] += t1 - t0;
    
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    
    P3D_evaluate_geometry_elementQ1(nqp,cell_coor,gradN_xi,detJ,gradN_x,gradN_y,gradN_z);
    for (q=0; q<nqp; q++) {
      PetscReal q_x[] = {0,0,0};
      for (i=0; i<DACELL3D_Q1_SIZE; i++) {
        q_x[0] += N[q][i] * cell_coor[3*i+0];
        q_x[1] += N[q][i] * cell_coor[3*i+1];
        q_x[2] += N[q][i] * cell_coor[3*i+2];
      }
      ierr = func(q_x,&Q_exact);CHKERRQ(ierr);
      PetscTime(&t0);
      ierr = FVReconstructionP1Interpolate(&rcell,q_x,&Q_fv);CHKERRQ(ierr);
      PetscTime(&t1);
      dt[1] += t1 - t0;
      
      l2Q += (Q_exact - Q_fv)*(Q_exact - Q_fv) * detJ[q];
    }
  }
  ierr = MPI_Allreduce(MPI_IN_PLACE,&l2Q,1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  l2Q = PetscSqrtReal(l2Q);
  PetscPrintf(fv->comm,"Q[reconstructed](L2) %1.4e\n",l2Q);

  PetscPrintf(PETSC_COMM_WORLD,"FVReconstructionP1Create: time %1.4e (sec) [total]\n",dt[0]);
  PetscPrintf(PETSC_COMM_WORLD,"FVReconstructionP1Create: time %1.4e (sec) [avg]\n",dt[0]/((PetscReal)fv->ncells));
  PetscPrintf(PETSC_COMM_WORLD,"FVReconstructionP1Interpolate: time %1.4e (sec) [total]\n",dt[1]);
  PetscPrintf(PETSC_COMM_WORLD,"FVReconstructionP1Interpolate: time %1.4e (sec) [avg]\n",dt[1]/((PetscReal)fv->ncells * nqp));

  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&geometry_coorl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fv_coor_local,&_fv_coor);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(Ql,&_Q);CHKERRQ(ierr);
  ierr = VecDestroy(&Ql);CHKERRQ(ierr);
  ierr = VecDestroy(&Q);CHKERRQ(ierr);
  
  ierr = DIMapDestroy(&map_rl2l);CHKERRQ(ierr);
  ierr = DIMapDestroy(&map_l2rl);CHKERRQ(ierr);
  
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  ierr = PetscFree(q_xi);CHKERRQ(ierr);
  ierr = PetscFree(q_w);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode t2(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 12, cells = 0;
  PetscInt       m[] = {mx,mx,mx};
  FVDA           fv;
  PetscBool      found = PETSC_FALSE;
  Vec            Q,Ql,fv_coor_local,geometry_coorl;
  Vec            gradQ,scalar,grad[3];
  DIMap          map_l2rl,map_rl2l;
  PetscInt       c;
  FVReconstructionCell rcell[3];
  const PetscReal      *_gradQ[3],*_fv_coor,*_geom_coor;
  PetscReal            dx[3];
  PetscBool            view = PETSC_FALSE;
  PetscReal            l2Q[3],volOmega;
  PetscInt             dm_nel,dm_nen;
  const PetscInt       *dm_element,*element;
  PetscReal            cell_coor[3*DACELL3D_VERTS];
  PetscInt             q,nqp,i;
  PetscReal            *q_xi,*q_w;
  PetscReal            N[8][8],gradN_xi[8][3][8],gradN_x[8][8],gradN_y[8][8],gradN_z[8][8],detJ[8];
  FVArray              Qa,gradQa;
  PetscLogDouble       t0,t1;
  
  ierr = PetscOptionsGetBool(NULL,NULL,"-view",&view,NULL);CHKERRQ(ierr);
  found = PETSC_FALSE; ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&found);CHKERRQ(ierr);
  if (found) {
    m[0] = mx;
    m[1] = mx;
    m[2] = mx;
  }
  found = PETSC_FALSE; ierr = PetscOptionsGetInt(NULL,NULL,"-my",&cells,&found);CHKERRQ(ierr);
  if (found) { m[1] = cells; }
  found = PETSC_FALSE; ierr = PetscOptionsGetInt(NULL,NULL,"-mz",&cells,&found);CHKERRQ(ierr);
  if (found) { m[2] = cells; }
  
  dx[0] = 2.0 / ((PetscReal)m[0]);
  dx[1] = 2.0 / ((PetscReal)m[1]);
  dx[2] = 2.0 / ((PetscReal)m[2]);
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  {
    Vec vertex_coor_geometry,coor;
    ierr = FVDAGetGeometryCoordinates(fv,&vertex_coor_geometry);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&coor);CHKERRQ(ierr);
    ierr = VecCopy(coor,vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"k*",1);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(fv->dm_fv,&Q);CHKERRQ(ierr);

  {
    PetscInt Qm,QM;

    ierr = VecGetSize(Q,&QM);CHKERRQ(ierr);
    ierr = VecGetLocalSize(Q,&Qm);CHKERRQ(ierr);
    ierr = VecCreate(fv->comm,&gradQ);CHKERRQ(ierr);
    ierr = VecSetSizes(gradQ,Qm*3,QM*3);CHKERRQ(ierr);
    ierr = VecSetBlockSize(gradQ,3);CHKERRQ(ierr);
    ierr = VecSetFromOptions(gradQ);CHKERRQ(ierr);
    ierr = VecSetUp(gradQ);CHKERRQ(ierr);
  }
  
  ierr = DMCreateGlobalVector(fv->dm_fv,&scalar);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_fv,&grad[0]);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_fv,&grad[1]);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_fv,&grad[2]);CHKERRQ(ierr);

  
  ierr = FVDAVecTraverse(fv,Q,0.0,0,func_eval,NULL);CHKERRQ(ierr);
  
  if (view) {
    ierr = FVDAView_CellData(fv,Q,PETSC_FALSE,"ex9_xcell");CHKERRQ(ierr);
  }
  
  ierr = DIMapCreate_FVDACell_RankLocalToLocal(fv,&map_rl2l);CHKERRQ(ierr);
  ierr = DIMapCreate_FVDACell_LocalToRankLocal(fv,&map_l2rl);CHKERRQ(ierr);
  
  /* Pass through just to check no errors occur */
  for (c=0; c<fv->ncells; c++) {
    PetscInt l;
    ierr = DIMapApply(map_rl2l,c,&l);CHKERRQ(ierr);
  }
  
  ierr = DMGetCoordinatesLocal(fv->dm_fv,&fv_coor_local);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_fv,&Ql);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_fv,Q,INSERT_VALUES,Ql);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fv_coor_local,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  
  QuadratureCreateGauss_2pnt_3D(&nqp,&q_xi,&q_w);
  for (q=0; q<nqp; q++) {
    EvaluateBasis_Q1_3D(&q_xi[3*q],N[q]);
    EvaluateBasisDerivative_Q1_3D(&q_xi[3*q],gradN_xi[q]);
  }
  
  ierr = FVArrayCreateFromVec(FVPRIMITIVE_CELL,Q,&Qa);CHKERRQ(ierr);
  ierr = FVArrayCreateFromVec(FVPRIMITIVE_CELL,gradQ,&gradQa);CHKERRQ(ierr);
  
  PetscTime(&t0);
  ierr = FVDAGradientProjectViaReconstruction(fv,Qa,gradQa);CHKERRQ(ierr);
  PetscTime(&t1);
  PetscPrintf(PETSC_COMM_WORLD,"FVDAGradientProjectViaReconstruction: time %1.4e (sec)\n",t1-t0);
  
  ierr = FVArrayDestroy(&gradQa);CHKERRQ(ierr);
  ierr = FVArrayDestroy(&Qa);CHKERRQ(ierr);

  ierr = VecZeroEntries(scalar);CHKERRQ(ierr);
  ierr = VecStrideGather(gradQ,0,scalar,INSERT_VALUES);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_fv,scalar,INSERT_VALUES,grad[0]);CHKERRQ(ierr);
  ierr = VecGetArrayRead(grad[0],&_gradQ[0]);CHKERRQ(ierr);

  ierr = VecZeroEntries(scalar);CHKERRQ(ierr);
  ierr = VecStrideGather(gradQ,1,scalar,INSERT_VALUES);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_fv,scalar,INSERT_VALUES,grad[1]);CHKERRQ(ierr);
  ierr = VecGetArrayRead(grad[1],&_gradQ[1]);CHKERRQ(ierr);

  ierr = VecZeroEntries(scalar);CHKERRQ(ierr);
  ierr = VecStrideGather(gradQ,2,scalar,INSERT_VALUES);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_fv,scalar,INSERT_VALUES,grad[2]);CHKERRQ(ierr);
  ierr = VecGetArrayRead(grad[2],&_gradQ[2]);CHKERRQ(ierr);

  /* compute error wrt piece-wise constant data */
  volOmega = 0.0;
  l2Q[0] = 0.0;
  l2Q[1] = 0.0;
  l2Q[2] = 0.0;
  for (c=0; c<fv->ncells; c++) {
    PetscInt l;
    PetscReal gradQ_exact[3],gradQ_fv[3];
    
    ierr = DIMapApply(map_rl2l,c,&l);CHKERRQ(ierr);
    
    gradQ_fv[0] = _gradQ[0][l];
    gradQ_fv[1] = _gradQ[1][l];
    gradQ_fv[2] = _gradQ[2][l];
    
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    
    P3D_evaluate_geometry_elementQ1(nqp,cell_coor,gradN_xi,detJ,gradN_x,gradN_y,gradN_z);
    for (q=0; q<nqp; q++) {
      PetscReal q_x[] = {0,0,0};
      for (i=0; i<DACELL3D_Q1_SIZE; i++) {
        q_x[0] += N[q][i] * cell_coor[3*i+0];
        q_x[1] += N[q][i] * cell_coor[3*i+1];
        q_x[2] += N[q][i] * cell_coor[3*i+2];
      }
      ierr = grad_func(q_x,gradQ_exact);CHKERRQ(ierr);
      
      volOmega += 1.0 * detJ[q];
      l2Q[0] += (gradQ_exact[0] - gradQ_fv[0])*(gradQ_exact[0] - gradQ_fv[0]) * detJ[q];
      l2Q[1] += (gradQ_exact[1] - gradQ_fv[1])*(gradQ_exact[1] - gradQ_fv[1]) * detJ[q];
      l2Q[2] += (gradQ_exact[2] - gradQ_fv[2])*(gradQ_exact[2] - gradQ_fv[2]) * detJ[q];
    }
  }
  PetscPrintf(fv->comm,"Test function gradient: %D\n",test_function_type);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&volOmega,1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  PetscPrintf(fv->comm,"Domain(volume)              %1.4e\n",volOmega);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&l2Q[0],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&l2Q[1],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&l2Q[2],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  l2Q[0] = PetscSqrtReal(l2Q[0]);
  l2Q[1] = PetscSqrtReal(l2Q[1]);
  l2Q[2] = PetscSqrtReal(l2Q[2]);
  PetscPrintf(fv->comm,"gradQ[0](L2)                %1.4e\n",l2Q[0]);
  PetscPrintf(fv->comm,"gradQ[1](L2)                %1.4e\n",l2Q[1]);
  PetscPrintf(fv->comm,"gradQ[2](L2)                %1.4e\n",l2Q[2]);

  /* compute error wrt P1 reconstructed data */
  l2Q[0] = 0.0;
  l2Q[1] = 0.0;
  l2Q[2] = 0.0;
  for (c=0; c<fv->ncells; c++) {
    PetscInt l;
    PetscReal gradQ_exact[3],gradQ_fv[3];
    
    ierr = DIMapApply(map_rl2l,c,&l);CHKERRQ(ierr);
    
    ierr = FVReconstructionP1Create(&rcell[0],fv,l,_fv_coor,_gradQ[0]);CHKERRQ(ierr);
    ierr = FVReconstructionP1Create(&rcell[1],fv,l,_fv_coor,_gradQ[1]);CHKERRQ(ierr);
    ierr = FVReconstructionP1Create(&rcell[2],fv,l,_fv_coor,_gradQ[2]);CHKERRQ(ierr);
    
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    
    P3D_evaluate_geometry_elementQ1(nqp,cell_coor,gradN_xi,detJ,gradN_x,gradN_y,gradN_z);
    for (q=0; q<nqp; q++) {
      PetscReal q_x[] = {0,0,0};
      for (i=0; i<DACELL3D_Q1_SIZE; i++) {
        q_x[0] += N[q][i] * cell_coor[3*i+0];
        q_x[1] += N[q][i] * cell_coor[3*i+1];
        q_x[2] += N[q][i] * cell_coor[3*i+2];
      }
      ierr = grad_func(q_x,gradQ_exact);CHKERRQ(ierr);
      
      ierr = FVReconstructionP1Interpolate(&rcell[0],q_x,&gradQ_fv[0]);CHKERRQ(ierr);
      ierr = FVReconstructionP1Interpolate(&rcell[1],q_x,&gradQ_fv[1]);CHKERRQ(ierr);
      ierr = FVReconstructionP1Interpolate(&rcell[2],q_x,&gradQ_fv[2]);CHKERRQ(ierr);
      
      l2Q[0] += (gradQ_exact[0] - gradQ_fv[0])*(gradQ_exact[0] - gradQ_fv[0]) * detJ[q];
      l2Q[1] += (gradQ_exact[1] - gradQ_fv[1])*(gradQ_exact[1] - gradQ_fv[1]) * detJ[q];
      l2Q[2] += (gradQ_exact[2] - gradQ_fv[2])*(gradQ_exact[2] - gradQ_fv[2]) * detJ[q];
    }
  }
  ierr = MPI_Allreduce(MPI_IN_PLACE,&l2Q[0],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&l2Q[1],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&l2Q[2],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  l2Q[0] = PetscSqrtReal(l2Q[0]);
  l2Q[1] = PetscSqrtReal(l2Q[1]);
  l2Q[2] = PetscSqrtReal(l2Q[2]);
  PetscPrintf(fv->comm,"gradQ[0][reconstructed](L2) %1.4e\n",l2Q[0]);
  PetscPrintf(fv->comm,"gradQ[1][reconstructed](L2) %1.4e\n",l2Q[1]);
  PetscPrintf(fv->comm,"gradQ[2][reconstructed](L2) %1.4e\n",l2Q[2]);
  
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&geometry_coorl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fv_coor_local,&_fv_coor);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(grad[0],&_gradQ[0]);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(grad[1],&_gradQ[1]);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(grad[2],&_gradQ[2]);CHKERRQ(ierr);
  
  
  ierr = VecDestroy(&Ql);CHKERRQ(ierr);
  ierr = VecDestroy(&Q);CHKERRQ(ierr);
  ierr = VecDestroy(&grad[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&grad[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&grad[2]);CHKERRQ(ierr);
  ierr = VecDestroy(&gradQ);CHKERRQ(ierr);
  ierr = VecDestroy(&scalar);CHKERRQ(ierr);
  
  ierr = DIMapDestroy(&map_rl2l);CHKERRQ(ierr);
  ierr = DIMapDestroy(&map_l2rl);CHKERRQ(ierr);
  
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  ierr = PetscFree(q_xi);CHKERRQ(ierr);
  ierr = PetscFree(q_w);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


const char doc[] = {
"[FVDA test] Verifies functionality associated with dense integer map and cell reconstructions.\n" \
};
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       tid = 0;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,doc);if (ierr) return ierr;
  test_function_type = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-test_func",&test_function_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-tid",&tid,NULL);CHKERRQ(ierr);
  switch (tid) {
    case 0:
      ierr = t1();CHKERRQ(ierr);
      break;
    case 1:
      ierr = t2();CHKERRQ(ierr);
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Valid values for -tid {0,1}");
      break;
  }
  ierr = PetscFinalize();
  return ierr;
}
