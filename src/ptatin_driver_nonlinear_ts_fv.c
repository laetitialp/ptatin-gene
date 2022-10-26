
static const char help[] = "Prototype pTatin3D driver using finite volume transport discretisation\n\n";

#include "petsc/private/dmdaimpl.h"

#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin_init.h"
#include "ptatin_log.h"

#include "material_point_utils.h"
#include "material_point_std_utils.h"
#include "material_point_popcontrol.h"
#include "ptatin_models.h"
#include "ptatin_utils.h"
#include "stokes_form_function.h"
#include "stokes_operators.h"
#include "stokes_operators_mf.h"
#include "stokes_assembly.h"
#include "dmda_element_q2p1.h"
#include "dmda_duplicate.h"
#include "dmda_project_coords.h"
#include "dmda_update_coords.h"
#include "monitors.h"
#include "mp_advection.h"
#include "mesh_update.h"

#include "ptatin3d_energy.h"
#include "energy_assembly.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>

#define MAX_MG_LEVELS 20

typedef enum { OP_TYPE_REDISC_ASM=0, OP_TYPE_REDISC_MF, OP_TYPE_GALERKIN } OperatorType;

typedef struct {
  PetscInt     nlevels;
  OperatorType *level_type;
  Mat          *operatorA11;
  Mat          *operatorB11;
  DM           *dav_hierarchy;
  Mat          *interpolation_v;
  Mat          *interpolation_eta;
  Quadrature   *volQ;
  BCList       *u_bclist;
  SurfBCList   *surf_bclist;
  IS           *is_stokes_field;
} AuuMultiLevelCtx;


PetscErrorCode SNESGetKSP_(SNES snes,SNES *this_snes,KSP *this_ksp)
{
  PetscBool is_ngmres = PETSC_FALSE;
  PetscErrorCode ierr;

  *this_snes = NULL;
  *this_ksp  = NULL;
  ierr = PetscObjectTypeCompare((PetscObject)snes,SNESNGMRES,&is_ngmres);CHKERRQ(ierr);

  if (is_ngmres) {
    ierr = SNESGetNPC(snes,this_snes);CHKERRQ(ierr);
  } else {
    *this_snes = snes;
  }
  ierr = SNESGetKSP(*this_snes,this_ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESComposeWithMGCtx(SNES snes,AuuMultiLevelCtx *mgctx)
{
  PetscErrorCode ierr;
  PetscContainer container;

  PetscFunctionBegin;
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)snes),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,(void*)mgctx);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)snes,"AuuMultiLevelCtx",(PetscObject)container);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SNESDestroyMGCtx(SNES snes)
{
  PetscErrorCode ierr;
  PetscContainer container;

  PetscFunctionBegin;

  container = NULL;
  ierr = PetscObjectQuery((PetscObject)snes,"AuuMultiLevelCtx",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dStokesKSPConfigureFSGMG(KSP ksp,PetscInt nlevels,Mat operatorA11[],Mat operatorB11[],Mat interpolation_v[],DM dav_hierarchy[])
{
  PetscInt k,nsplits;
  PC       pc,pc_i;
  KSP      *sub_ksp,ksp_coarse,ksp_smoother;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCFieldSplitGetSubKSP(pc,&nsplits,&sub_ksp);CHKERRQ(ierr);

  ierr = KSPSetDM(sub_ksp[0],dav_hierarchy[nlevels-1]);CHKERRQ(ierr);
  ierr = KSPSetDMActive(sub_ksp[0],PETSC_FALSE);CHKERRQ(ierr);

  ierr = KSPGetPC(sub_ksp[0],&pc_i);CHKERRQ(ierr);
  ierr = PCSetType(pc_i,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc_i,nlevels,NULL);CHKERRQ(ierr);
  ierr = PCMGSetType(pc_i,PC_MG_MULTIPLICATIVE);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc_i,PC_MG_GALERKIN_NONE);CHKERRQ(ierr);
  ierr = PCSetDM(pc_i,NULL);CHKERRQ(ierr);

  for( k=1; k<nlevels; k++ ){
    ierr = PCMGSetInterpolation(pc_i,k,interpolation_v[k]);CHKERRQ(ierr);
  }

  /* drop the operators in - i presume this will also need to be performed inside the jacobian each time the operators are modified */
  /* No - it looks like PCSetUp_MG will call set operators on all levels if the SetOperators was called on the finest, which should/is done by the SNES */
  ierr = PCMGGetCoarseSolve(pc_i,&ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_coarse,operatorA11[0],operatorA11[0]);CHKERRQ(ierr);

  ierr = KSPSetDM(ksp_coarse,dav_hierarchy[0]);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp_coarse,PETSC_FALSE);CHKERRQ(ierr);

  for( k=1; k<nlevels; k++ ){
    PetscBool use_low_order_geometry = PETSC_FALSE;

    ierr = PCMGGetSmoother(pc_i,k,&ksp_smoother);CHKERRQ(ierr);

    // use A for smoother, B for residual
    ierr = PetscOptionsGetBool(NULL,NULL,"-use_low_order_geometry",&use_low_order_geometry,NULL);CHKERRQ(ierr);
    if (use_low_order_geometry==PETSC_TRUE) {
      ierr = KSPSetOperators(ksp_smoother,operatorB11[k],operatorB11[k]);CHKERRQ(ierr);
      //ierr = KSPSetOperators(ksp_smoother,operatorA11[k],operatorB11[k]);CHKERRQ(ierr);
    } else {
      // Use A for smoother, lo
      ierr = KSPSetOperators(ksp_smoother,operatorA11[k],operatorA11[k]);CHKERRQ(ierr);
    }
    ierr = KSPSetDM(ksp_smoother,dav_hierarchy[k]);CHKERRQ(ierr);
    ierr = KSPSetDMActive(ksp_smoother,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFree(sub_ksp);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dStokesBuildMeshHierarchy(DM dav,PetscInt nlevels,DM dav_hierarchy[])
{
  PetscErrorCode ierr;
  DM *coarsened_list;
  PetscInt k;

  PetscFunctionBegin;

  /* set up mg */
  dav->ops->coarsenhierarchy = DMCoarsenHierarchy2_DA;

  dav_hierarchy[ nlevels-1 ] = dav;
  ierr = PetscObjectReference((PetscObject)dav);CHKERRQ(ierr);

  /* Coarsen nlevels - 1 times, and add levels into list so that level 0 is the coarsest */
  ierr = PetscMalloc(sizeof(DM)*(nlevels-1),&coarsened_list);CHKERRQ(ierr);
  ierr = DMCoarsenHierarchy(dav,nlevels-1,coarsened_list);CHKERRQ(ierr);
  for (k=0; k<nlevels-1; k++) {
    dav_hierarchy[ nlevels-2-k ] = coarsened_list[k];
  }
  PetscFree(coarsened_list);

  /* Set all dav's to be of type Q2 */
  for (k=0; k<nlevels-1; k++) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)dav_hierarchy[k],"stk_velocity_");CHKERRQ(ierr);
    ierr = DMDASetElementType_Q2(dav_hierarchy[k]);CHKERRQ(ierr);
  }

  /* inject coordinates */
  ierr = DMDARestrictCoordinatesHierarchy(dav_hierarchy,nlevels);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dStokesReportMeshHierarchy(PetscInt nlevels,DM dav_hierarchy[])
{
  PetscErrorCode ierr;
  PetscInt       k,lmx,lmy,lmz;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Report mesh sizes */
  for (k=0; k<nlevels; k++) {
    ierr = DMDAGetSizeElementQ2(dav_hierarchy[k],&lmx,&lmy,&lmz);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"         level [%2D]: global Q2 elements (%D x %D x %D) \n", k,lmx,lmy,lmz );
  }

  /*
  for (k=0; k<nlevels; k++) {

    ierr = DMDAGetElements_pTatinQ2P1(dav_hierarchy[k],&nels,&nen,&els);CHKERRQ(ierr);
    ierr = DMDAGetLocalSizeElementQ2(dav_hierarchy[k],&lmx,&lmy,&lmz);CHKERRQ(ierr);
    if (rank<10) {
      PetscPrintf(PETSC_COMM_SELF,"[r%4D]: level [%2D]: local Q2 elements  (%D x %D x %D) \n", rank, k,lmx,lmy,lmz );
    }
  }
  */
  /*
  for (k=0; k<nlevels; k++) {
    ierr = DMDAGetElements_pTatinQ2P1(dav_hierarchy[k],&nels,&nen,&els);CHKERRQ(ierr);

    ierr = DMDAGetCornersElementQ2(dav_hierarchy[k],&si,&sj,&sk,&lmx,&lmy,&lmz);CHKERRQ(ierr);
    si = si/2;
    sj = sj/2;
    sk = sk/2;
    if (rank<10) {
      PetscPrintf(PETSC_COMM_SELF,"[r%4D]: level [%2D]: element range [%D - %D] x [%D - %D] x [%D - %D] \n", rank, k,si,si+lmx-1,sj,sj+lmy-1,sk,sk+lmz-1 );
    }
  }
  */

  for (k=0; k<nlevels; k++) {
    PetscInt mp,np,pp,*_mx,*_my,*_mz,ii,jj,kk;

    ierr = DMDAGetOwnershipRangesElementQ2(dav_hierarchy[k],&mp,&np,&pp,NULL,NULL,NULL,&_mx,&_my,&_mz);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"level [%2D]: [total cores %4D]: np-I [%4D]: element range I [ ", k,size,mp );
    for (ii=0; ii<mp; ii++) {
      PetscPrintf(PETSC_COMM_WORLD,"%4D", _mx[ii] );
      if (ii != mp-1) { PetscPrintf(PETSC_COMM_WORLD,", "); }
    }PetscPrintf(PETSC_COMM_WORLD," ]\n");

    PetscPrintf(PETSC_COMM_WORLD,"                                np-J [%4D]: element range J [ ",np);
    for (jj=0; jj<np; jj++) {
      PetscPrintf(PETSC_COMM_WORLD,"%4D", _my[jj] );
      if (jj != np-1) { PetscPrintf(PETSC_COMM_WORLD,", "); }
    }PetscPrintf(PETSC_COMM_WORLD," ]\n");

    PetscPrintf(PETSC_COMM_WORLD,"                                np-K [%4D]: element range K [ ",pp);
    for (kk=0; kk<pp; kk++) {
      PetscPrintf(PETSC_COMM_WORLD,"%4D", _mz[kk] );
      if (kk != pp-1) { PetscPrintf(PETSC_COMM_WORLD,", "); }
    }PetscPrintf(PETSC_COMM_WORLD," ]\n");

    ierr = PetscFree(_mx);CHKERRQ(ierr);
    ierr = PetscFree(_my);CHKERRQ(ierr);
    ierr = PetscFree(_mz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3dCreateStokesOperators(PhysCompStokes stokes_ctx,IS is_stokes_field[],
                                             PetscInt nlevels,DM dav_hierarchy[],Mat interpolation_v[],
                                             BCList u_bclist[],Quadrature volQ[],SurfBCList surf_bclist[],
                                             OperatorType level_type[],
                                             Mat *_A,Mat operatorA11[],Mat *_B,Mat operatorB11[])
{
  Mat            A,B;
  DM             dap;
  PetscInt       k,max;
  PetscBool      flg;
  PetscInt       _level_type[MAX_MG_LEVELS];
  static int     been_here = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dap = stokes_ctx->dap;

  /* A operator */
  ierr = StokesQ2P1CreateMatrix_Operator(stokes_ctx,&A);CHKERRQ(ierr);
  /* memory saving - only need daU IF you want to split A11 into A11uu,A11vv,A11ww */
  {
    MatStokesMF mf;

    ierr = MatShellGetMatStokesMF(A,&mf);CHKERRQ(ierr);
    ierr = DMDestroy(&mf->daU);CHKERRQ(ierr);
    mf->daU = NULL;
    ierr = ISDestroy(&mf->isU);CHKERRQ(ierr);
    ierr = ISDestroy(&mf->isV);CHKERRQ(ierr);
    ierr = ISDestroy(&mf->isW);CHKERRQ(ierr);
  }

  /* B operator */
  {
    Mat         Aup,Apu,Spp,bA[2][2];
    MatStokesMF StkCtx;

    ierr = MatShellGetMatStokesMF(A,&StkCtx);CHKERRQ(ierr);

    /* Schur complement */
    //ierr = DMSetMatType(dap,MATSBAIJ);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dap,&Spp);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(Spp,"S*_");CHKERRQ(ierr);
    ierr = MatSetOption(Spp,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Spp);CHKERRQ(ierr);

    /* A12 */
    ierr = StokesQ2P1CreateMatrix_MFOperator_A12(StkCtx,&Aup);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(Aup,"Bup_");CHKERRQ(ierr);
    ierr = MatSetFromOptions(Aup);CHKERRQ(ierr);

    /* A21 */
    ierr = StokesQ2P1CreateMatrix_MFOperator_A21(StkCtx,&Apu);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(Apu,"Bpu_");CHKERRQ(ierr);
    ierr = MatSetFromOptions(Apu);CHKERRQ(ierr);

    /* nest */
    bA[0][0] = NULL; bA[0][1] = Aup;
    bA[1][0] = Apu;  bA[1][1] = Spp;

    ierr = MatCreateNest(PETSC_COMM_WORLD,2,is_stokes_field,2,is_stokes_field,&bA[0][0],&B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* tidy up - hand back destruction to B */
    ierr = MatDestroy(&Aup);CHKERRQ(ierr);
    ierr = MatDestroy(&Apu);CHKERRQ(ierr);
    ierr = MatDestroy(&Spp);CHKERRQ(ierr);
  }

  /* A11 operator */
  /* defaults */
  _level_type[0] = (PetscInt)OP_TYPE_REDISC_ASM;
  for (k=1; k<nlevels; k++) {
    _level_type[k] = (PetscInt)OP_TYPE_REDISC_MF;
  }

  max = nlevels;
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-A11_operator_type",_level_type,&max,&flg);CHKERRQ(ierr);
  for (k=nlevels-1; k>=0; k--) {
    level_type[k] = (OperatorType)_level_type[k];
  }
  for (k=nlevels-1; k>=0; k--) {

    switch (level_type[k]) {

      case OP_TYPE_REDISC_ASM:
      {
        Mat Auu;
        PetscBool same1 = PETSC_FALSE,same2 = PETSC_FALSE,same3 = PETSC_FALSE;
        Vec X;
        MatNullSpace nullsp;

        /* use -stk_velocity_da_mat_type sbaij or -Buu_da_mat_type sbaij */
        if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: Re-discretisation :: assembled operator \n", k);
        //ierr = DMSetMatType(dav_hierarchy[k],MATSBAIJ);CHKERRQ(ierr);
        ierr = DMCreateMatrix(dav_hierarchy[k],&Auu);CHKERRQ(ierr);
        ierr = MatSetOptionsPrefix(Auu,"Buu_");CHKERRQ(ierr);
        ierr = MatSetFromOptions(Auu);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)Auu,MATSBAIJ,&same1);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)Auu,MATSEQSBAIJ,&same2);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)Auu,MATMPISBAIJ,&same3);CHKERRQ(ierr);
        if (same1||same2||same3) {
          ierr = MatSetOption(Auu,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
        }
        /* should move assembly into jacobian */
        ierr = MatZeroEntries(Auu);CHKERRQ(ierr);
        ierr = MatAssemble_StokesA_AUU(Auu,dav_hierarchy[k],u_bclist[k],volQ[k],surf_bclist[k]);CHKERRQ(ierr);

        operatorA11[k] = Auu;
        operatorB11[k] = Auu;
        ierr = PetscObjectReference((PetscObject)Auu);CHKERRQ(ierr);
        ierr = DMGetCoordinates(dav_hierarchy[k],&X);CHKERRQ(ierr);
        ierr = MatNullSpaceCreateRigidBody(X,&nullsp);CHKERRQ(ierr);
        ierr = MatSetNearNullSpace(Auu,nullsp);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);

      }
        break;

      case OP_TYPE_REDISC_MF:
      {
        Mat Auu;
        MatA11MF mf,A11Ctx;

        if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: Re-discretisation :: matrix free operator \n", k);
        ierr = MatA11MFCreate(&A11Ctx);CHKERRQ(ierr);
        ierr = MatA11MFSetup(A11Ctx,dav_hierarchy[k],volQ[k],u_bclist[k],surf_bclist[k]);CHKERRQ(ierr);

        ierr = StokesQ2P1CreateMatrix_MFOperator_A11(A11Ctx,&Auu);CHKERRQ(ierr);
        /* memory saving - only need daU IF you want to split A11 into A11uu,A11vv,A11ww */
        ierr = MatShellGetMatA11MF(Auu,&mf);CHKERRQ(ierr);
        ierr = DMDestroy(&mf->daU);CHKERRQ(ierr);
        mf->daU = NULL;
        ierr = ISDestroy(&mf->isU);CHKERRQ(ierr);
        ierr = ISDestroy(&mf->isV);CHKERRQ(ierr);
        ierr = ISDestroy(&mf->isW);CHKERRQ(ierr);
        /* --- */
        operatorA11[k] = Auu;

        {
          PetscBool use_low_order_geometry = PETSC_FALSE;

          ierr = PetscOptionsGetBool(NULL,NULL,"-use_low_order_geometry",&use_low_order_geometry,NULL);CHKERRQ(ierr);
          if (use_low_order_geometry==PETSC_TRUE) {
            Mat Buu;

            if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Activiting low order A11 operator \n");
            ierr = StokesQ2P1CreateMatrix_MFOperator_A11LowOrder(A11Ctx,&Buu);CHKERRQ(ierr);
            /* memory saving - only need daU IF you want to split A11 into A11uu,A11vv,A11ww */
            ierr = MatShellGetMatA11MF(Buu,&mf);CHKERRQ(ierr);
            ierr = DMDestroy(&mf->daU);CHKERRQ(ierr);
            mf->daU = NULL;
            ierr = ISDestroy(&mf->isU);CHKERRQ(ierr);
            ierr = ISDestroy(&mf->isV);CHKERRQ(ierr);
            ierr = ISDestroy(&mf->isW);CHKERRQ(ierr);
            /* --- */
            operatorB11[k] = Buu;

          } else {
            operatorB11[k] = Auu;
            ierr = PetscObjectReference((PetscObject)Auu);CHKERRQ(ierr);
          }
        }

        ierr = MatA11MFDestroy(&A11Ctx);CHKERRQ(ierr);
      }
        break;

      case OP_TYPE_GALERKIN:
      {
        Mat Auu;
        Vec X;
        MatNullSpace nullsp;

        if (k==nlevels-1) {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Cannot use galerkin coarse grid on the finest level");
        }

        if (!been_here) PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: Galerkin :: assembled operator \n", k);

        /* should move coarse grid assembly into jacobian */
        ierr = MatPtAP(operatorA11[k+1],interpolation_v[k+1],MAT_INITIAL_MATRIX,1.0,&Auu);CHKERRQ(ierr);

        operatorA11[k] = Auu;
        operatorB11[k] = Auu;
        ierr = PetscObjectReference((PetscObject)Auu);CHKERRQ(ierr);
        ierr = DMGetCoordinates(dav_hierarchy[k],&X);CHKERRQ(ierr);
        ierr = MatNullSpaceCreateRigidBody(X,&nullsp);CHKERRQ(ierr);
        ierr = MatSetBlockSize(Auu,3);CHKERRQ(ierr);
        ierr = MatSetNearNullSpace(Auu,nullsp);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
      }
        break;

      default:
        break;
    }
  }

  /* Set fine A11 into nest */
  ierr = MatNestSetSubMat(B,0,0,operatorA11[nlevels-1]);CHKERRQ(ierr);

  *_A = A;
  *_B = B;

  been_here = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_StokesMGAuu(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  pTatinCtx         user;
  AuuMultiLevelCtx  *mlctx;
  DM                stokes_pack,dau,dap;
  PhysCompStokes    stokes;
  Vec               Uloc,Ploc;
  PetscScalar       *LA_Uloc,*LA_Ploc;
  PetscBool         is_mffd = PETSC_FALSE;
  PetscBool         is_nest = PETSC_FALSE;
  PetscBool         is_shell = PETSC_FALSE;
  PetscContainer    container;
  PetscInt          k;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  user = (pTatinCtx)ctx;

  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;

  ierr = PetscObjectQuery((PetscObject)snes,"AuuMultiLevelCtx",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"No data with name \"AuuMultiLevelCtx\" was composed with SNES");
  ierr = PetscContainerGetPointer(container,(void*)&mlctx);CHKERRQ(ierr);


  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  ierr = DMCompositeScatter(stokes_pack,X,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);

  /* nonlinearitiers: markers => quad points */
  ierr = pTatin_EvaluateRheologyNonlinearities(user,dau,LA_Uloc,dap,LA_Ploc);CHKERRQ(ierr);

  /* interpolate coefficients */
  {
    int               npoints;
    DataField         PField_std;
    DataField         PField_stokes;
    MPntStd           *mp_std;
    MPntPStokes       *mp_stokes;

    DataBucketGetDataFieldByName(user->materialpoint_db, MPntStd_classname     , &PField_std);
    DataBucketGetDataFieldByName(user->materialpoint_db, MPntPStokes_classname , &PField_stokes);

    DataBucketGetSizes(user->materialpoint_db,&npoints,NULL,NULL);
    mp_std    = PField_std->data; /* should write a function to do this */
    mp_stokes = PField_stokes->data; /* should write a function to do this */

    ierr = SwarmUpdateGaussPropertiesLocalL2Projection_Q1_MPntPStokes_Hierarchy(user->coefficient_projection_type,npoints,mp_std,mp_stokes,mlctx->nlevels,mlctx->interpolation_eta,mlctx->dav_hierarchy,mlctx->volQ,NULL,NULL);CHKERRQ(ierr);
  }

  /* operator */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMFFD, &is_mffd);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATNEST, &is_nest);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSHELL,&is_shell);CHKERRQ(ierr);

  if (is_nest) {
    Mat Auu;

    ierr = MatCreateSubMatrix(A,mlctx->is_stokes_field[0],mlctx->is_stokes_field[0],MAT_INITIAL_MATRIX,&Auu);CHKERRQ(ierr);

    is_shell = PETSC_FALSE;
    ierr = PetscObjectTypeCompare((PetscObject)Auu,MATSHELL,&is_shell);CHKERRQ(ierr);
    if (!is_shell) {
      ierr = MatZeroEntries(Auu);CHKERRQ(ierr);
      ierr = MatAssemble_StokesA_AUU(Auu,dau,user->stokes_ctx->u_bclist,user->stokes_ctx->volQ,user->stokes_ctx->surf_bclist);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&Auu);CHKERRQ(ierr);
  }
  /* If shell, do nothing */
  /* If mffd,  do nothing */

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* preconditioner operator for Jacobian */
  {
    Mat Buu,Bpp;

    ierr = MatCreateSubMatrix(B,mlctx->is_stokes_field[0],mlctx->is_stokes_field[0],MAT_INITIAL_MATRIX,&Buu);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(B,mlctx->is_stokes_field[1],mlctx->is_stokes_field[1],MAT_INITIAL_MATRIX,&Bpp);CHKERRQ(ierr);

    is_shell = PETSC_FALSE;
    ierr = PetscObjectTypeCompare((PetscObject)Buu,MATSHELL,&is_shell);CHKERRQ(ierr);
    if (!is_shell) {
      ierr = MatZeroEntries(Buu);CHKERRQ(ierr);
      ierr = MatAssemble_StokesA_AUU(Buu,dau,user->stokes_ctx->u_bclist,user->stokes_ctx->volQ,user->stokes_ctx->surf_bclist);CHKERRQ(ierr);
    }

    is_shell = PETSC_FALSE;
    ierr = PetscObjectTypeCompare((PetscObject)Bpp,MATSHELL,&is_shell);CHKERRQ(ierr);
    if (!is_shell) {
      ierr = MatZeroEntries(Bpp);CHKERRQ(ierr);
      ierr = MatAssemble_StokesPC_ScaledMassMatrix(Bpp,dau,dap,user->stokes_ctx->p_bclist,user->stokes_ctx->volQ);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&Buu);CHKERRQ(ierr);
    ierr = MatDestroy(&Bpp);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Buu preconditioner for all other levels in the hierarchy */
  {
    PetscBool use_low_order_geometry;
    SNES      this_snes;
    KSP       ksp,*sub_ksp,ksp_smoother;
    PC        pc,pc_i;
    PetscInt  nsplits;

    use_low_order_geometry = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-use_low_order_geometry",&use_low_order_geometry,NULL);CHKERRQ(ierr);

    ierr = SNESGetKSP_(snes,&this_snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCFieldSplitGetSubKSP(pc,&nsplits,&sub_ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(sub_ksp[0],&pc_i);CHKERRQ(ierr);

    for (k=mlctx->nlevels-2; k>=0; k--) {
      /* fetch smoother */
      if (k == 0) {
        ierr = PCMGGetCoarseSolve(pc_i,&ksp_smoother);CHKERRQ(ierr);
      } else {
        ierr = PCMGGetSmoother(pc_i,k,&ksp_smoother);CHKERRQ(ierr);
      }

      switch (mlctx->level_type[k]) {

        case OP_TYPE_REDISC_ASM:
        {
          ierr = MatZeroEntries(mlctx->operatorB11[k]);CHKERRQ(ierr);
          ierr = MatAssemble_StokesA_AUU(mlctx->operatorB11[k],mlctx->dav_hierarchy[k],mlctx->u_bclist[k],mlctx->volQ[k],mlctx->surf_bclist[k]);CHKERRQ(ierr);

          ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          /* hack for nested coarse solver */
          {
            KSP ksp_nested;
            PC pc_smoother;
            PetscBool is_nested_ksp;

            ierr = KSPGetPC(ksp_smoother,&pc_smoother);CHKERRQ(ierr);
            is_nested_ksp = PETSC_FALSE;
            ierr = PetscObjectTypeCompare((PetscObject)pc_smoother,PCKSP,&is_nested_ksp);CHKERRQ(ierr);
            if (is_nested_ksp) {
              ierr = PCKSPGetKSP(pc_smoother,&ksp_nested);CHKERRQ(ierr);
              ierr = KSPSetOperators(ksp_nested,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
            }
          }

          /* no low order assembly */
          /*
          if (use_low_order_geometry==PETSC_TRUE) {
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          } else {
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorA11[k]);CHKERRQ(ierr);
          }
          */
        }
          break;

        case OP_TYPE_REDISC_MF:
        {
          if (use_low_order_geometry == PETSC_TRUE) {
            //  ierr = KSPSetOperators(ksp_smoother,operatorB11[k],operatorB11[k]);CHKERRQ(ierr);
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          } else {
            //  ierr = KSPSetOperators(ksp_smoother,operatorA11[k],operatorA11[k]);CHKERRQ(ierr);
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorA11[k]);CHKERRQ(ierr);
          }
        }
          break;

        case OP_TYPE_GALERKIN:
        {
          Mat Auu_k;

          if (k == mlctx->nlevels-1) {
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Cannot use galerkin coarse grid on the finest level");
          }
          PetscPrintf(PETSC_COMM_WORLD,"Level [%D]: Coarse grid type :: Galerkin :: assembled operator \n", k);

          /*
          ierr = MatPtAP(mlctx->operatorA11[k+1],mlctx->interpolation_v[k+1],MAT_INITIAL_MATRIX,1.0,&Auu_k);CHKERRQ(ierr);
          ierr = KSPSetOperators(ksp_smoother,Auu_k,Auu_k);CHKERRQ(ierr);
          mlctx->operatorA11[k] = Auu_k;
          mlctx->operatorB11[k] = Auu_k;
          ierr = PetscObjectReference((PetscObject)Auu_k);CHKERRQ(ierr);
          */
          Auu_k = mlctx->operatorA11[k];
          ierr = MatPtAP(mlctx->operatorA11[k+1],mlctx->interpolation_v[k+1],MAT_REUSE_MATRIX,1.0,&Auu_k);CHKERRQ(ierr);
          ierr = KSPSetOperators(ksp_smoother,Auu_k,Auu_k);CHKERRQ(ierr);
          mlctx->operatorB11[k] = Auu_k;
        }
          break;

        default:
          break;
      }
    }
    PetscFree(sub_ksp);

    /* push operators */
    for (k=mlctx->nlevels-1; k>=0; k--) {
      SNES this_snes;

      ierr = SNESGetKSP_(snes,&this_snes,&ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCFieldSplitGetSubKSP(pc,&nsplits,&sub_ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(sub_ksp[0],&pc_i);CHKERRQ(ierr);

      if (k == 0) {
        ierr = PCMGGetCoarseSolve(pc_i,&ksp_smoother);CHKERRQ(ierr);
      } else {
        ierr = PCMGGetSmoother(pc_i,k,&ksp_smoother);CHKERRQ(ierr);
      }

      switch (mlctx->level_type[k]) {

        case OP_TYPE_REDISC_ASM:
        {
          ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          /* hack for nested coarse solver */
          {
            KSP ksp_nested;
            PC pc_smoother;
            PetscBool is_nested_ksp;

            ierr = KSPGetPC(ksp_smoother,&pc_smoother);CHKERRQ(ierr);
            is_nested_ksp = PETSC_FALSE;
            ierr = PetscObjectTypeCompare((PetscObject)pc_smoother,PCKSP,&is_nested_ksp);CHKERRQ(ierr);
            if (is_nested_ksp) {
              ierr = PCKSPGetKSP(pc_smoother,&ksp_nested);CHKERRQ(ierr);
              ierr = KSPSetOperators(ksp_nested,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
            }
          }
        }
          break;

        case OP_TYPE_REDISC_MF:
        {
          if (use_low_order_geometry == PETSC_TRUE) {
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorB11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
          } else {
            ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorA11[k]);CHKERRQ(ierr);
          }
        }
          break;

        case OP_TYPE_GALERKIN:
        {
          ierr = KSPSetOperators(ksp_smoother,mlctx->operatorA11[k],mlctx->operatorB11[k]);CHKERRQ(ierr);
        }
          break;
      }
      PetscFree(sub_ksp);
    }

  }

  {
    PetscBool mg_dump_coarse = PETSC_FALSE;
    char filename[PETSC_MAX_PATH_LEN];
    PetscInt snes_it;
    PetscViewer viewer;

    PetscOptionsGetBool(NULL,NULL,"-ptatin_mg_dump_coarse_operator",&mg_dump_coarse,0);
    SNESGetIterationNumber(snes,&snes_it);

    if (mg_dump_coarse) {
      if (mlctx->level_type[0] != OP_TYPE_REDISC_MF) {

        PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/mg_coarse_operatorA_step%D_snes%D.mat",user->outputpath,user->step,snes_it);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);
        MatView(mlctx->operatorA11[0],viewer);
        PetscViewerDestroy(&viewer);
        if (mlctx->operatorA11[0] != mlctx->operatorB11[0]) {
          PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/mg_coarse_operatorB_step%D_snes%D.mat",user->outputpath,user->step,snes_it);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);
          MatView(mlctx->operatorB11[0],viewer);
          PetscViewerDestroy(&viewer);
        }
      }
    }
  }

  /* clean up */
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3d_nonlinear_viscous_forward_model_driver_v1(int argc,char **argv)
{
  pTatinCtx         user;
  pTatinModel       model;
  PhysCompStokes    stokes;
  //PhysCompEnergy    energy = NULL;
  PhysCompEnergyFV  energyfv = NULL;
  DM              multipys_pack,dav,dap;
  Mat       A,B;
  Vec       X,F;
  IS        *is_stokes_field;
  SNES      snes;
  KSP       ksp;
  PC        pc;
  DM             dav_hierarchy[MAX_MG_LEVELS];
  OperatorType   level_type[MAX_MG_LEVELS];
  Mat            operatorA11[MAX_MG_LEVELS],operatorB11[MAX_MG_LEVELS];
  Mat            interpolation_v[MAX_MG_LEVELS],interpolation_eta[MAX_MG_LEVELS];
  PetscInt       k,nlevels,step;
  Quadrature     volQ[MAX_MG_LEVELS];
  BCList         u_bclist[MAX_MG_LEVELS];
  SurfBCList     surf_bclist[MAX_MG_LEVELS];
  SurfaceQuadrature surfQ[MAX_MG_LEVELS];
  MeshFacetInfo     mfi[MAX_MG_LEVELS];
  AuuMultiLevelCtx mlctx;
  PetscInt         newton_its,picard_its;
  PetscBool        active_energy;
  RheologyType     init_rheology_type;
  PetscBool        monitor_stages = PETSC_FALSE,write_icbc = PETSC_FALSE;
  PetscBool        activate_quasi_newton_coord_update = PETSC_FALSE;
  DataBucket       materialpoint_db;
  PetscLogDouble   time[2];
  PetscReal        surface_displacement_max = 1.0e32;
  PetscReal        dt_factor = 10.0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  ierr = pTatin3dCreateContext(&user);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(user);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor_stages",&monitor_stages,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_quasi_newton_coordinate_update",&activate_quasi_newton_coord_update,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt_max_surface_displacement",&surface_displacement_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-ptatin_driver_write_icbc",&write_icbc,NULL);CHKERRQ(ierr);

  /* Register all models */
  ierr = pTatinModelRegisterAll();CHKERRQ(ierr);
  /* Load model, call an initialization routines */
  ierr = pTatinModelLoad(user);CHKERRQ(ierr);
  ierr = pTatinGetModel(user,&model);CHKERRQ(ierr);

  ierr = pTatinModel_Initialize(model,user);CHKERRQ(ierr);

  /* Generate physics modules */
  ierr = pTatin3d_PhysCompStokesCreate(user);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Generated vel/pressure mesh --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* Pack all physics together */
  /* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
  ierr = PetscObjectReference((PetscObject)stokes->stokes_pack);CHKERRQ(ierr);
  user->pack = stokes->stokes_pack;

  /* fetch some local variables */
  multipys_pack = user->pack;
  dav           = stokes->dav;
  dap           = stokes->dap;

  /* IF I DON'T DO THIS, THE IS's OBTAINED FROM DMCompositeGetGlobalISs() are wrong !! */
  {
    Vec X;

    ierr = DMGetGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
  }
  ierr = DMCompositeGetGlobalISs(multipys_pack,&is_stokes_field);CHKERRQ(ierr);

  /* work vector for solution and residual */
  ierr = DMCreateGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = pTatinPhysCompAttachData_Stokes(user,X);CHKERRQ(ierr);

  ierr = pTatin3dCreateMaterialPoints(user,dav);CHKERRQ(ierr);
  ierr = pTatinGetMaterialPoints(user,&materialpoint_db,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Generated material points --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* mesh geometry */
  ierr = pTatinModel_ApplyInitialMeshGeometry(model,user);CHKERRQ(ierr);

  ierr = pTatinLogBasicDMDA(user,"Velocity",dav);CHKERRQ(ierr);
  ierr = pTatinLogBasicDMDA(user,"Pressure",dap);CHKERRQ(ierr);

  ierr = PhysCompStokesUpdateSurfaceQuadratureGeometry(user->stokes_ctx);CHKERRQ(ierr);

  /* generate energy solver */
  /* NOTE - Generating the thermal solver here will ensure that the initial geometry on the mechanical model is copied */
  /* NOTE - Calling pTatinPhysCompActivate_Energy() after pTatin3dCreateMaterialPoints() is essential */
  {
    PetscBool load_energy = PETSC_FALSE;

    PetscOptionsGetBool(NULL,NULL,"-activate_energy",&load_energy,NULL);
    if (load_energy) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"FV support only");
    ierr = pTatinPhysCompActivate_Energy(user,load_energy);CHKERRQ(ierr);
    ierr = pTatinContextValid_Energy(user,&active_energy);CHKERRQ(ierr);
  }
  {
    PetscBool load_energy = PETSC_FALSE;
    
    PetscOptionsGetBool(NULL,NULL,"-activate_energyfv",&load_energy,NULL);
    ierr = pTatinPhysCompActivate_EnergyFV(user,load_energy);CHKERRQ(ierr);
    ierr = pTatinContextValid_EnergyFV(user,&active_energy);CHKERRQ(ierr);
  }
  
  if (active_energy) {
    DM dmfv;
    
    ierr = pTatinGetContext_EnergyFV(user,&energyfv);CHKERRQ(ierr);

    ierr = FVDAGetDM(energyfv->fv,&dmfv);CHKERRQ(ierr);
    ierr = pTatinLogBasicDMDA(user,"EnergyFV",dmfv);CHKERRQ(ierr);

    //ierr = pTatinLogBasicDMDA(user,"Energy",energy->daT);CHKERRQ(ierr);
    //ierr = DMCreateGlobalVector(energy->daT,&T);CHKERRQ(ierr);
    //ierr = pTatinPhysCompAttachData_Energy(user,T,NULL);CHKERRQ(ierr);
    ierr = pTatinCtxAttachModelData(user,"PhysCompEnergy_T",(void*)energyfv->T);CHKERRQ(ierr);
    
    //ierr = DMCreateGlobalVector(energy->daT,&f);CHKERRQ(ierr);
    //ierr = DMSetMatType(energy->daT,MATAIJ);CHKERRQ(ierr);
    //ierr = DMCreateMatrix(energy->daT,&JE);CHKERRQ(ierr);
    //ierr = MatSetFromOptions(JE);CHKERRQ(ierr);

    pTatinGetRangeCurrentMemoryUsage(NULL);
  }

  /* interpolate material point coordinates (needed if mesh was modified) */
  ierr = MaterialPointCoordinateSetUp(user,dav);CHKERRQ(ierr);

  /* material geometry */
  ierr = pTatinModel_ApplyInitialMaterialGeometry(model,user);CHKERRQ(ierr);
  if (active_energy) {
    //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Requires update for FV support");
    PetscPrintf(PETSC_COMM_WORLD,"********* <FV SUPPORT NOTE> IS THIS REQUIRED?? pTatinPhysCompEnergy_MPProjectionQ1 ****************\n");
    
    ierr = EnergyFVEvaluateCoefficients(user,0.0,energyfv,NULL,NULL);CHKERRQ(ierr);
    
    ierr = pTatinPhysCompEnergyFV_MPProjection(energyfv,user);CHKERRQ(ierr);
    
    ierr = FVDACellPropertyProjectToFace_HarmonicMean(energyfv->fv,"k","k");CHKERRQ(ierr);
  }
  DataBucketView(PetscObjectComm((PetscObject)multipys_pack), materialpoint_db,"MaterialPoints StokesCoefficients",DATABUCKET_VIEW_STDOUT);

  /* boundary conditions */
  ierr = pTatinModel_ApplyBoundaryCondition(model,user);CHKERRQ(ierr);

  /* setup mg */
  nlevels = 1;
  PetscOptionsGetInt(NULL,NULL,"-dau_nlevels",&nlevels,0);
  PetscPrintf(PETSC_COMM_WORLD,"Mesh size (%D x %D x %D) : MG levels %D  \n", user->mx,user->my,user->mz,nlevels );
  ierr = pTatin3dStokesBuildMeshHierarchy(dav,nlevels,dav_hierarchy);CHKERRQ(ierr);
  ierr = pTatin3dStokesReportMeshHierarchy(nlevels,dav_hierarchy);CHKERRQ(ierr);
  ierr = pTatinLogNote(user,"  [Velocity multi-grid hierarchy]");CHKERRQ(ierr);
  for (k=nlevels-1; k>=0; k--) {
    char name[PETSC_MAX_PATH_LEN];
    PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"vel_dmda_Lv%D",k);
    ierr = pTatinLogBasicDMDA(user,name,dav_hierarchy[k]);CHKERRQ(ierr);
  }

  /* Coarse grid setup: Define interpolation operators for velocity space */
  interpolation_v[0] = NULL;
  for (k=0; k<nlevels-1; k++) {
    ierr = DMCreateInterpolation(dav_hierarchy[k],dav_hierarchy[k+1],&interpolation_v[k+1],NULL);CHKERRQ(ierr);
  }

  /* Coarse grid setup: Define interpolation operators for scalar space */
  interpolation_eta[0] = NULL;
  for (k=1; k<nlevels; k++) {
    ierr = MatMAIJRedimension(interpolation_v[k],1,&interpolation_eta[k]);CHKERRQ(ierr);
  }
  PetscPrintf(PETSC_COMM_WORLD,"Generated velocity mesh hierarchy --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* Coarse grid setup: Define material properties on gauss points */
  for (k=0; k<nlevels-1; k++) {
    PetscInt ncells,lmx,lmy,lmz;
    PetscInt np_per_dim;

    np_per_dim = 3;
    ierr = DMDAGetLocalSizeElementQ2(dav_hierarchy[k],&lmx,&lmy,&lmz);CHKERRQ(ierr);
    ncells = lmx * lmy * lmz;
    ierr = VolumeQuadratureCreate_GaussLegendreStokes(3,np_per_dim,ncells,&volQ[k]);CHKERRQ(ierr);
  }
  volQ[nlevels-1] = stokes->volQ;
  PetscPrintf(PETSC_COMM_WORLD,"Generated quadrature point hierarchy --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* Coarse grid setup: Define boundary conditions */
  for (k=0; k<nlevels-1; k++) {
    ierr = DMDABCListCreate(dav_hierarchy[k],&u_bclist[k]);CHKERRQ(ierr);
  }
  u_bclist[nlevels-1] = stokes->u_bclist;

  /* define surface info and quadrature on coarse grids */
  for (k=0; k<nlevels-1; k++) {
    ierr = MeshFacetInfoCreate2(dav_hierarchy[k],&mfi[k]);CHKERRQ(ierr);
    ierr = SurfaceQuadratureCreate_GaussLegendreStokes(dav_hierarchy[k],&surfQ[k]);CHKERRQ(ierr);
  }
  mfi[nlevels-1] = user->stokes_ctx->mfi;
  surfQ[nlevels-1] = user->stokes_ctx->surfQ;

  for (k=0; k<nlevels-1; k++) {
    ierr = SurfaceQuadratureGeometryUpdate_Stokes(surfQ[k],mfi[k]);CHKERRQ(ierr);
  }

  /* define surface boundary list on coarse grids */
  for (k=0; k<nlevels-1; k++) {
    ierr = SurfBCListCreate(mfi[k]->dm,surfQ[k],mfi[k],&surf_bclist[k]);CHKERRQ(ierr);
  }
  surf_bclist[nlevels-1] = user->stokes_ctx->surf_bclist;

  // populate
  {
    PetscInt nc;
    SurfBCList surf_fine = surf_bclist[nlevels-1];
    PetscPrintf(PETSC_COMM_WORLD,"[SurfBCList hierarchy]\n");
    for (k=0; k<nlevels-1; k++) {
      PetscPrintf(PETSC_COMM_WORLD,"  level %d\n",k);
      // for all methods in fine level
      for (nc=0; nc<surf_fine->sc_nreg; nc++) {
        SurfaceConstraint sc_coarse = NULL;
        SurfaceConstraint sc_coarse_A11 = NULL;
        PetscPrintf(PETSC_COMM_WORLD,"    checking for \"%s\"\n",surf_fine->sc_list[nc]->name);
        // query coarse
        ierr = SurfBCListGetConstraint(surf_bclist[k],surf_fine->sc_list[nc]->name,&sc_coarse);CHKERRQ(ierr);
        if (!sc_coarse) {
          PetscPrintf(PETSC_COMM_WORLD,"    not found... duplicating\n");
          ierr = SurfaceConstraintDuplicateOperatorA11(surf_fine->sc_list[nc],mfi[k],surfQ[k],&sc_coarse_A11);CHKERRQ(ierr);
          if (sc_coarse_A11) { // constraint supports A11
            PetscBool inserted = PETSC_FALSE;
            PetscPrintf(PETSC_COMM_WORLD,"      fine level duplicate supports A11... inserting into level\n");
            ierr = SurfBCListInsertConstraint(surf_bclist[k],sc_coarse_A11,&inserted);CHKERRQ(ierr);
            if (inserted) {
              PetscPrintf(PETSC_COMM_WORLD,"        insertion successful\n");
            }
          } else {
            PetscPrintf(PETSC_COMM_WORLD,"      fine level duplicate does not supports A11... ignoring\n");
          }
        }
      }
    }
  }
  
  /* Coarse grid setup: Configure boundary conditions */
  ierr = pTatinModel_ApplyBoundaryConditionMG(nlevels,u_bclist,surf_bclist,dav_hierarchy,model,user);CHKERRQ(ierr);

  /* set all pointers into mg context */
  mlctx.is_stokes_field     = is_stokes_field;
  mlctx.nlevels             = nlevels;
  mlctx.dav_hierarchy       = dav_hierarchy;
  mlctx.interpolation_v     = interpolation_v;
  mlctx.interpolation_eta   = interpolation_eta;
  mlctx.volQ                = volQ;
  mlctx.u_bclist            = u_bclist;
  mlctx.surf_bclist         = surf_bclist;

  /* ============================================== */
  /* configure stokes opertors */
  ierr = pTatin3dCreateStokesOperators(stokes,is_stokes_field,
                                       nlevels,dav_hierarchy,interpolation_v,u_bclist,volQ,surf_bclist,level_type,
                                       &A,operatorA11,&B,operatorB11);CHKERRQ(ierr);
  mlctx.level_type  = level_type;
  mlctx.operatorA11 = operatorA11;
  mlctx.operatorB11 = operatorB11;
  /* ============================================== */
  PetscPrintf(PETSC_COMM_WORLD,"Generated stokes operators --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* initial condition */
  ierr = pTatinModel_ApplyInitialSolution(model,user,X);CHKERRQ(ierr);

  /* initial viscosity  */
  ierr = pTatinModel_ApplyInitialStokesVariableMarkers(model,user,X);CHKERRQ(ierr);

  /* update markers = >> gauss points */
  {
    int               npoints;
    DataField         PField_std;
    DataField         PField_stokes;
    MPntStd           *mp_std;
    MPntPStokes       *mp_stokes;

    DataBucketGetDataFieldByName(user->materialpoint_db, MPntStd_classname     , &PField_std);
    DataBucketGetDataFieldByName(user->materialpoint_db, MPntPStokes_classname , &PField_stokes);

    DataBucketGetSizes(user->materialpoint_db,&npoints,NULL,NULL);
    mp_std    = PField_std->data; /* should write a function to do this */
    mp_stokes = PField_stokes->data; /* should write a function to do this */

    ierr = SwarmUpdateGaussPropertiesLocalL2Projection_Q1_MPntPStokes_Hierarchy(user->coefficient_projection_type,npoints,mp_std,mp_stokes,nlevels,interpolation_eta,dav_hierarchy,volQ,surfQ,mfi);CHKERRQ(ierr);
  }

  /* Set boundary conditions again after viscosity update on surface quadrature points */
  ierr = pTatinModel_ApplyBoundaryCondition(model,user);CHKERRQ(ierr);
  ierr = pTatinModel_ApplyBoundaryConditionMG(nlevels,u_bclist,surf_bclist,dav_hierarchy,model,user);CHKERRQ(ierr);

  /* insert boundary conditions into solution vector */
  {
    Vec velocity,pressure;

    ierr = DMCompositeGetAccess(multipys_pack,X,&velocity,&pressure);CHKERRQ(ierr);
    ierr = BCListInsert(stokes->u_bclist,velocity);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(multipys_pack,X,&velocity,&pressure);CHKERRQ(ierr);

    if (active_energy) {
      //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Requires update for FV support");
      //ierr = BCListInsert(energy->T_bclist,T);CHKERRQ(ierr);
    }
  }

  /* output ic */
  if (write_icbc) {
    ierr = pTatinModel_Output(model,user,X,"icbc");CHKERRQ(ierr);
  }

  PetscPrintf(PETSC_COMM_WORLD,"   [[ COMPUTING FLOW FIELD FOR STEP : %D ]]\n", 0 );

  {
    PetscInt snes_its;

    PetscPrintf(PETSC_COMM_WORLD,"PreLinearSolve(0) --> ");
    pTatinGetRangeCurrentMemoryUsage(NULL);
    /* --------------------------------------------------------- */
    /* Define operators */
    // DONE ABOVE //
    /* --------------------------------------------------------- */
    /* Define non-linear solver */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    if (!activate_quasi_newton_coord_update) {
      ierr = SNESSetFunction(snes,F,FormFunction_Stokes,user);CHKERRQ(ierr);
    } else {
      ierr = SNESSetFunction(snes,F,FormFunction_Stokes_QuasiNewtonX,user);CHKERRQ(ierr);
    }

    // activate mffd via -snes_mf_operator
    ierr = SNESSetJacobian(snes,A,B,FormJacobian_StokesMGAuu,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    /* force MG context into SNES */
    ierr = SNESComposeWithMGCtx(snes,&mlctx);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);

    /*
       Set non-zero initial guess as we only perform 1 non-linear iteration and the
       user may have provided a initial guess for velocity, pressure in their model
     */
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

    ierr = pTatin_Stokes_ActivateMonitors(user,snes);CHKERRQ(ierr);

    /* configure for fieldsplit */
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"u",is_stokes_field[0]);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"p",is_stokes_field[1]);CHKERRQ(ierr);

    /* configure uu split for galerkin multi-grid */
    ierr = pTatin3dStokesKSPConfigureFSGMG(ksp,nlevels,operatorA11,operatorB11,interpolation_v,dav_hierarchy);CHKERRQ(ierr);

    ierr = pTatinLogBasic(user);CHKERRQ(ierr);

    /* --------------------------------------------------------- */

    SNESGetTolerances(snes,0,0,0,&snes_its,0);

    /* switch to linear rheology to use the viscosity set on marker by the initialStokeVariables */
    init_rheology_type = user->rheology_constants.rheology_type;
    user->rheology_constants.rheology_type = RHEOLOGY_VISCOUS;

    /* do a linear solve */
    SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1,PETSC_DEFAULT);
    PetscPrintf(PETSC_COMM_WORLD,"   --------- LINEAR STAGE ---------\n");
    PetscTime(&time[0]);
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
    PetscTime(&time[1]);
    ierr = pTatinLogBasicSNES(user,"Stokes[LinearStage]",snes);CHKERRQ(ierr);
    ierr = pTatinLogBasicCPUtime(user,"Stokes[LinearStage]",time[1]-time[0]);CHKERRQ(ierr);
    ierr = pTatinLogBasicStokesSolution(user,multipys_pack,X);CHKERRQ(ierr);
    ierr = pTatinLogBasicStokesSolutionResiduals(user,snes,multipys_pack,X);CHKERRQ(ierr);
    ierr = pTatinLogPetscLog(user,"Stokes[LinearStage]");CHKERRQ(ierr);
    if (monitor_stages) {
      ierr = pTatinModel_Output(model,user,X,"linear_stage");CHKERRQ(ierr);
    }
    PetscPrintf(PETSC_COMM_WORLD,"PostLinearSolve(0) --> ");
    pTatinGetRangeCurrentMemoryUsage(NULL);

    /* tidy up assembled operators */
    for (k=0; k<nlevels; k++) {
      ierr = MatDestroy(&operatorA11[k]);CHKERRQ(ierr);
      ierr = MatDestroy(&operatorB11[k]);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = SNESDestroyMGCtx(snes);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);

    /* do a picard solve */
    /* switch to non-linear rheology */
    user->rheology_constants.rheology_type = init_rheology_type;

    PetscPrintf(PETSC_COMM_WORLD,"PrePicardSolve(0) --> ");
    pTatinGetRangeCurrentMemoryUsage(NULL);
    /* --------------------------------------------------------- */
    /* Define operators */
    ierr = pTatin3dCreateStokesOperators(stokes,is_stokes_field,
                                         nlevels,dav_hierarchy,interpolation_v,u_bclist,volQ,surf_bclist,level_type,
                                         &A,operatorA11,&B,operatorB11);CHKERRQ(ierr);
    /* --------------------------------------------------------- */
    /* Define non-linear solver */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    if (!activate_quasi_newton_coord_update) {
      ierr = SNESSetFunction(snes,F,FormFunction_Stokes,user);CHKERRQ(ierr);
    } else {
      ierr = SNESSetFunction(snes,F,FormFunction_Stokes_QuasiNewtonX,user);CHKERRQ(ierr);
    }
    // activate mffd via -snes_mf_operator
    ierr = SNESSetJacobian(snes,A,B,FormJacobian_StokesMGAuu,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    /* force MG context into SNES */
    ierr = SNESComposeWithMGCtx(snes,&mlctx);CHKERRQ(ierr);

    /* configure KSP */
    //ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);

    /* monitors */
    ierr = pTatin_Stokes_ActivateMonitors(user,snes);CHKERRQ(ierr);

    /* configure for fieldsplit */
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"u",is_stokes_field[0]);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"p",is_stokes_field[1]);CHKERRQ(ierr);
    /* mg */
    ierr = pTatin3dStokesKSPConfigureFSGMG(ksp,nlevels,operatorA11,operatorB11,interpolation_v,dav_hierarchy);CHKERRQ(ierr);
    ierr = pTatinLogBasic(user);CHKERRQ(ierr);

    /* --------------------------------------------------------- */

    picard_its = snes_its;
    PetscOptionsGetInt(NULL,NULL,"-picard_its",&picard_its,NULL);
    SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,picard_its,PETSC_DEFAULT);

    PetscPrintf(PETSC_COMM_WORLD,"   --------- PICARD STAGE ---------\n");
    PetscTime(&time[0]);
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
    PetscTime(&time[1]);
    ierr = pTatinLogBasicSNES(user,"Stokes[PicardStage]",snes);CHKERRQ(ierr);
    ierr = pTatinLogBasicCPUtime(user,"Stokes[PicardStage]",time[1]-time[0]);CHKERRQ(ierr);
    ierr = pTatinLogBasicStokesSolution(user,multipys_pack,X);CHKERRQ(ierr);
    ierr = pTatinLogBasicStokesSolutionResiduals(user,snes,multipys_pack,X);CHKERRQ(ierr);
    ierr = pTatinLogPetscLog(user,"Stokes[PicardStage]");CHKERRQ(ierr);
    if (monitor_stages) {
      ierr = pTatinModel_Output(model,user,X,"picard_stage");CHKERRQ(ierr);
    }
    PetscPrintf(PETSC_COMM_WORLD,"PostPicardSolve(0) --> ");
    pTatinGetRangeCurrentMemoryUsage(NULL);

    /* tidy up assembled operators */
    for (k=0; k<nlevels; k++) {
      ierr = MatDestroy(&operatorA11[k]);CHKERRQ(ierr);
      ierr = MatDestroy(&operatorB11[k]);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = SNESDestroyMGCtx(snes);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  }

  newton_its = 0;
  PetscOptionsGetInt(NULL,NULL,"-newton_its",&newton_its,NULL);
  if (newton_its>0) {
    SNES snes_newton;

    /* Define operators */
    ierr = pTatin3dCreateStokesOperators(stokes,is_stokes_field,
                                         nlevels,dav_hierarchy,interpolation_v,u_bclist,volQ,surf_bclist,level_type,
                                         &A,operatorA11,&B,operatorB11);CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes_newton);CHKERRQ(ierr);
    //ierr = SNESSetApplicationContext(snes_newton,(void*)user);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(snes_newton,"n_");CHKERRQ(ierr);
    if (!activate_quasi_newton_coord_update) {
      ierr = SNESSetFunction(snes_newton,F,FormFunction_Stokes,user);CHKERRQ(ierr);
    } else {
      ierr = SNESSetFunction(snes_newton,F,FormFunction_Stokes_QuasiNewtonX,user);CHKERRQ(ierr);
    }
    // Force mffd
    ierr = SNESSetJacobian(snes_newton,A,B,FormJacobian_StokesMGAuu,user);CHKERRQ(ierr);

    //ierr = SNESStokesPCSetOptions_A(snes_newton);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes_newton);CHKERRQ(ierr);

    /* compose */
    ierr = SNESComposeWithMGCtx(snes_newton,&mlctx);CHKERRQ(ierr);

    /* configure KSP */
    ierr = SNESGetKSP(snes_newton,&ksp);CHKERRQ(ierr);

    ierr = pTatin_Stokes_ActivateMonitors(user,snes_newton);CHKERRQ(ierr);

    /* configure for fieldsplit */
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"u",is_stokes_field[0]);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"p",is_stokes_field[1]);CHKERRQ(ierr);

    /* configure uu split for galerkin multi-grid */
    ierr = pTatin3dStokesKSPConfigureFSGMG(ksp,nlevels,operatorA11,operatorB11,interpolation_v,dav_hierarchy);CHKERRQ(ierr);

    SNESSetTolerances(snes_newton,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,newton_its,PETSC_DEFAULT);
    PetscPrintf(PETSC_COMM_WORLD,"   --------- NEWTON STAGE ---------\n");
    PetscTime(&time[0]);
    ierr = SNESSolve(snes_newton,NULL,X);CHKERRQ(ierr);
    PetscTime(&time[1]);
    ierr = pTatinLogBasicSNES(user,"Stokes[NewtonStage]",snes_newton);CHKERRQ(ierr);
    ierr = pTatinLogBasicCPUtime(user,"Stokes[NewtonStage]",time[1]-time[0]);CHKERRQ(ierr);
    ierr = pTatinLogBasicStokesSolution(user,multipys_pack,X);CHKERRQ(ierr);
    ierr = pTatinLogBasicStokesSolutionResiduals(user,snes_newton,multipys_pack,X);CHKERRQ(ierr);
    ierr = pTatinLogPetscLog(user,"Stokes[NewtonStage]");CHKERRQ(ierr);
    if (monitor_stages) {
      ierr = pTatinModel_Output(model,user,X,"newton_stage");CHKERRQ(ierr);
    }

    /* tidy up assembled operators */
    for (k=0; k<nlevels; k++) {
      ierr = MatDestroy(&operatorA11[k]);CHKERRQ(ierr);
      ierr = MatDestroy(&operatorB11[k]);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = SNESDestroyMGCtx(snes_newton);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes_newton);CHKERRQ(ierr);
  }

  /* dump */
  if (write_icbc) {
    ierr = pTatinModel_Output(model,user,X,"step000000");CHKERRQ(ierr);
  }

  /* compute timestep */
  user->dt = 1.0e32;
  {
    Vec velocity,pressure;
    PetscReal timestep;

    ierr = DMCompositeGetAccess(multipys_pack,X,&velocity,&pressure);CHKERRQ(ierr);
    ierr = SwarmUpdatePosition_ComputeCourantStep(dav_hierarchy[nlevels-1],velocity,&timestep);CHKERRQ(ierr);
    ierr = pTatin_SetTimestep(user,"StkCourant",timestep);CHKERRQ(ierr);

    ierr = UpdateMeshGeometry_ComputeSurfaceCourantTimestep(dav_hierarchy[nlevels-1],velocity,surface_displacement_max,&timestep);CHKERRQ(ierr);
    ierr = pTatin_SetTimestep(user,"StkSurfaceCourant",timestep);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(multipys_pack,X,&velocity,&pressure);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"  timestep[stokes] dt_courant = %1.4e \n", user->dt );

  }
  /* first time step, enforce to be super small */
  user->dt = user->dt * 1.0e-10;

#if 0 // SUPG - DO THINGS DIFFERENT FOR FV
  /* initialise the energy solver */
  if (active_energy) {
    PetscReal timestep;

    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Requires update for FV support");
    ierr = pTatinPhysCompEnergy_Initialise(energy,T);CHKERRQ(ierr);

    /* first time this is called we REQUIRE that a valid time step is chosen */
    energy->dt = user->dt;
    ierr = pTatinPhysCompEnergy_UpdateALEVelocity(stokes,X,energy,energy->dt);CHKERRQ(ierr);
    ierr = pTatinPhysCompEnergy_ComputeTimestep(energy,energy->Told,&timestep);CHKERRQ(ierr);

    /*
     Note - we cannot use the time step for energy equation here.
     It seems silly, but to compute the adf-diff time step, we need to the ALE velocity,
     however to compute the ALE velocity we need to know the timestep.
     */
    PetscPrintf(PETSC_COMM_WORLD,"  timestep[adv-diff] dt_courant = %1.4e \n", timestep );
    energy->dt = user->dt;
  }
#endif

  if (active_energy) {
    ierr = pTatinPhysCompEnergyFV_Initialise(energyfv,energyfv->T);CHKERRQ(ierr);
    
    energyfv->dt = user->dt;
    PetscPrintf(PETSC_COMM_WORLD,"  timestep[adv-diff] dt_courant = <UNAVAILABLE BUT NOT REQUIRED>\n");
  }

  user->step = 1;
  user->time = user->time + user->dt;
  if (active_energy) {
    energyfv->time = user->time;
  }

  /* TIME STEP */
  for (step=1; step <= user->nsteps; step++) {
    char      stepname[PETSC_MAX_PATH_LEN];
    Vec       velocity,pressure,q2_coor_k,fv_coor_k;
    PetscReal timestep;

    PetscPrintf(PETSC_COMM_WORLD,"<<----------------------------------------------------------------------------------------------->>\n");
    PetscPrintf(PETSC_COMM_WORLD,"   [[ EXECUTING TIME STEP : %D ]]\n", step );
    PetscPrintf(PETSC_COMM_WORLD,"     dt    : %1.4e \n", user->dt );
    PetscPrintf(PETSC_COMM_WORLD,"     time  : %1.4e \n", user->time );

    ierr = pTatinLogBasic(user);CHKERRQ(ierr);

    
    /* update mesh coordinates then restore */
    {
      DM  cdm;
      Vec tmp,q2_coor;

      ierr = DMGetCoordinateDM(dav,&cdm);CHKERRQ(ierr);
      ierr = DMGetCoordinates(dav,&q2_coor);CHKERRQ(ierr);
      
      ierr = VecDuplicate(q2_coor,&tmp);CHKERRQ(ierr);
      ierr = VecCopy(q2_coor,tmp);CHKERRQ(ierr);
      
      ierr = DMCreateGlobalVector(cdm,&q2_coor_k);CHKERRQ(ierr);
      ierr = pTatinModel_UpdateMeshGeometry(model,user,X);CHKERRQ(ierr);
      ierr = PhysCompStokesUpdateSurfaceQuadratureGeometry(user->stokes_ctx);CHKERRQ(ierr);

      ierr = DMGetCoordinates(dav,&q2_coor);CHKERRQ(ierr);
      ierr = VecCopy(q2_coor,q2_coor_k);CHKERRQ(ierr);
      ierr = VecCopy(tmp,q2_coor);CHKERRQ(ierr);
      ierr = DMDAUpdateGhostedCoordinates(dav);CHKERRQ(ierr);
      
      ierr = VecDestroy(&tmp);CHKERRQ(ierr);
    }

    
    
    /* solve energy equation with FV + ALE */

    // [FV EXTENSION] (1) Evaluate thermal properties on material points, project onto FV cell / cell faces
    if (active_energy) {
      ierr = EnergyFVEvaluateCoefficients(user,0.0,energyfv,NULL,NULL);CHKERRQ(ierr);
      
      ierr = pTatinPhysCompEnergyFV_MPProjection(energyfv,user);CHKERRQ(ierr);
      
      ierr = FVDACellPropertyProjectToFace_HarmonicMean(energyfv->fv,"k","k");CHKERRQ(ierr);
    }

    // [FV EXTENSION] (1) Interpolate Q2 velocity onto vertices of FV geometry mesh, (2) interpolate the nodal velocity onto the cell faces
    if (active_energy) {
      PetscTime(&time[0]);
      ierr = DMCompositeGetAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
      
      // (1)
      ierr = PhysCompEnergyFVInterpolateMacroQ2ToSubQ1(dav,velocity,energyfv,energyfv->dmv,energyfv->velocity);CHKERRQ(ierr);
      
      ierr = DMCompositeRestoreAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
      
      // (2)
      ierr = PhysCompEnergyFVInterpolateNormalVectorToFace(energyfv,energyfv->velocity,"v.n");CHKERRQ(ierr);
      
      ierr = PhysCompEnergyFVInterpolateVectorToFace(energyfv,energyfv->velocity,"v");CHKERRQ(ierr);
      PetscTime(&time[1]);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV-InterpV",time[1]-time[0]);CHKERRQ(ierr);
    }

    
    // [FV EXTENSION] Compute ALE velocity
    if (active_energy) {
      DM  dm_fv_geometry;
      Vec fv_vertex_coor_geometry;
      
      PetscTime(&time[0]);
      ierr = FVDAGetGeometryDM(energyfv->fv,&dm_fv_geometry);CHKERRQ(ierr);
      ierr = FVDAGetGeometryCoordinates(energyfv->fv,&fv_vertex_coor_geometry);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(dm_fv_geometry,&fv_coor_k);CHKERRQ(ierr);
      
      ierr = PhysCompEnergyFVInterpolateMacroQ2ToSubQ1(dav,q2_coor_k,energyfv,dm_fv_geometry,fv_coor_k);CHKERRQ(ierr);
      {
        Vec x_target;
        
        ierr = FVDAAccessData_ALE(energyfv->fv,NULL,NULL,&x_target);CHKERRQ(ierr);
        ierr = VecCopy(fv_coor_k,x_target);CHKERRQ(ierr);
      }
      ierr = pTatinPhysCompEnergyFV_ComputeALEVelocity(dm_fv_geometry,fv_vertex_coor_geometry,fv_coor_k,user->dt,energyfv->velocity);CHKERRQ(ierr); /* note we re-use storage for velocity here */
      
      ierr = PhysCompEnergyFVInterpolateVectorToFace(energyfv,energyfv->velocity,"xDot");CHKERRQ(ierr);
      //ierr = PhysCompEnergyFVInterpolateNormalVectorToFace(energyfv,energyfv->velocity,"xDot.n");CHKERRQ(ierr);
      PetscTime(&time[1]);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV-ALE-ComputeV",time[1]-time[0]);CHKERRQ(ierr);
    }

    // [FV EXTENSION] Make v.n define on each face compatible (e.g. ensure it satisfies \int v.n dS = \int div(v) dV = 0
    if (active_energy) {
      KSP ksp_pp;
      Vec source,fv_vertex_coor_geometry;
      
      ierr = FVDAGetGeometryCoordinates(energyfv->fv,&fv_vertex_coor_geometry);CHKERRQ(ierr);
      ierr = VecDuplicate(energyfv->T,&source);CHKERRQ(ierr);
      
      ierr = FVDAPPCompatibleVelocityCreate(energyfv->fv,&ksp_pp);CHKERRQ(ierr);
      
      PetscTime(&time[0]);
      ierr = VecZeroEntries(source);CHKERRQ(ierr); /* div(u) = 0 */
      ierr = FVDAPostProcessCompatibleVelocity(energyfv->fv,"v","v.n",source,ksp_pp);CHKERRQ(ierr);
      PetscTime(&time[1]);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV-PostProc-v",time[1]-time[0]);CHKERRQ(ierr);

      PetscTime(&time[0]);
      ierr = VecZeroEntries(source);CHKERRQ(ierr);
      ierr = pTatinPhysCompEnergyFV_ComputeALESource(energyfv->fv,fv_vertex_coor_geometry,fv_coor_k,energyfv->dt,source,PETSC_TRUE);CHKERRQ(ierr);
      ierr = FVDAPostProcessCompatibleVelocity(energyfv->fv,"xDot","xDot.n",source,ksp_pp);CHKERRQ(ierr);
      PetscTime(&time[1]);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV-PostProc-xDot",time[1]-time[0]);CHKERRQ(ierr);

      ierr = KSPDestroy(&ksp_pp);CHKERRQ(ierr);
      ierr = VecDestroy(&source);CHKERRQ(ierr);
    }

    // [FV EXTENSION] Combine (v - v_mesh)
    {
      FVDA      fv = energyfv->fv;
      PetscInt  f,nfaces;
      PetscReal *vdotn,*xDotdotn;
      
      ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyByNameArray(fv,"v.n",&vdotn);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyByNameArray(fv,"xDot.n",&xDotdotn);CHKERRQ(ierr);
      
      for (f=0; f<nfaces; f++) {
        vdotn[f] = vdotn[f] - xDotdotn[f];
      }
    }

    /* solve energy equation */
    if (active_energy) {
      PetscReal *dt;
      Vec       E,Ek,Ekk;
      
      E    = energyfv->T;
      Ek   = energyfv->Told;
      //ierr = FVDAAccessData_TimeDep(energyfv->fv,&dt,&Ekk);CHKERRQ(ierr);
      ierr = FVDAAccessData_ALE(energyfv->fv,&dt,&Ekk,NULL);CHKERRQ(ierr);
      *dt = energyfv->dt;
      
      /* Push current state into old state */
      ierr = VecCopy(E,Ek);CHKERRQ(ierr);
      ierr = VecCopy(E,Ekk);CHKERRQ(ierr);
      
      /* Three choices */
      {
        const PetscReal range[] = {0.0,1.0e32};
        
        //ierr = EnergyFV_RK1(energyfv->snes,range,energyfv->time,energyfv->dt,energyfv->Told,energyfv->T);CHKERRQ(ierr);
        
        // high-order no limiting (negative temperature)
        //ierr = EnergyFV_RK2SSP(energyfv->snes,NULL,energyfv->time,energyfv->dt,energyfv->Told,energyfv->T);CHKERRQ(ierr);
        
        //ierr = EnergyFV_RK2SSP(energyfv->snes,range,energyfv->time,energyfv->dt,energyfv->Told,energyfv->T);CHKERRQ(ierr);
        
        PetscTime(&time[0]);
        ierr = SNESSolve(energyfv->snes,NULL,energyfv->T);CHKERRQ(ierr);
        PetscTime(&time[1]);
        
        ierr = pTatinLogBasicSNES(user,"EnergyFV",energyfv->snes);CHKERRQ(ierr);
        ierr = pTatinLogBasicCPUtime(user,"EnergyFV-Solve",time[1]-time[0]);CHKERRQ(ierr);
      }
      
    }
    
    /* update marker time dependent terms */
    /* e.g. e_plastic^1 = e_plastic^0 + dt * [ strain_rate_inv(u^0) ] */
    /*
     NOTE: for a consistent forward difference time integration we evaluate u^0 at x^0
     - thus this update is performed BEFORE we advect the markers
     */
    ierr = pTatin_UpdateCoefficientTemporalDependence_Stokes(user,X);CHKERRQ(ierr);
    
    /* update marker positions */
    ierr = DMCompositeGetAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
    ierr = MaterialPointStd_UpdateGlobalCoordinates(user->materialpoint_db,dav_hierarchy[nlevels-1],velocity,user->dt);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);


    
    /* Q2: update mesh coordinates */
    {
      Vec q2_coor;
      
      ierr = DMGetCoordinates(dav,&q2_coor);CHKERRQ(ierr);
      
      ierr = VecCopy(q2_coor_k,q2_coor);CHKERRQ(ierr);
      ierr = DMDAUpdateGhostedCoordinates(dav);CHKERRQ(ierr);
    }
    
    /* FV: update mesh coordinates */
    if (active_energy) {
      Vec fv_vertex_coor_geometry;
      
      ierr = FVDAGetGeometryCoordinates(energyfv->fv,&fv_vertex_coor_geometry);CHKERRQ(ierr);
      ierr = VecCopy(fv_coor_k,fv_vertex_coor_geometry);CHKERRQ(ierr);
      
      ierr = PhysCompEnergyFVUpdateGeometry(energyfv,stokes);CHKERRQ(ierr);
    }
    
    
    
    /* update mesh coordinate hierarchy */
    ierr = DMDARestrictCoordinatesHierarchy(dav_hierarchy,nlevels);CHKERRQ(ierr);

    /* 3 Update local coordinates and communicate */
    ierr = MaterialPointStd_UpdateCoordinates(user->materialpoint_db,dav_hierarchy[nlevels-1],user->materialpoint_ex);CHKERRQ(ierr);

    /* 3a - Add material */
    ierr = pTatinModel_ApplyMaterialBoundaryCondition(model,user);CHKERRQ(ierr);
    //if ( (step%5 == 0) || (step == 1) ) {
    //ierr = pTatinModel_ApplyMaterialBoundaryCondition(model,user);CHKERRQ(ierr);
    //}

    /* add / remove points if cells are over populated or depleted of points */
    ierr = MaterialPointPopulationControl_v1(user);CHKERRQ(ierr);

    /* update markers = >> gauss points */
    {
      int               npoints;
      DataField         PField_std;
      DataField         PField_stokes;
      MPntStd           *mp_std;
      MPntPStokes       *mp_stokes;

      DataBucketGetDataFieldByName(user->materialpoint_db, MPntStd_classname     , &PField_std);
      DataBucketGetDataFieldByName(user->materialpoint_db, MPntPStokes_classname , &PField_stokes);

      DataBucketGetSizes(user->materialpoint_db,&npoints,NULL,NULL);
      mp_std    = PField_std->data; /* should write a function to do this */
      mp_stokes = PField_stokes->data; /* should write a function to do this */

      ierr = SwarmUpdateGaussPropertiesLocalL2Projection_Q1_MPntPStokes_Hierarchy(user->coefficient_projection_type,npoints,mp_std,mp_stokes,nlevels,interpolation_eta,dav_hierarchy,volQ,surfQ,mfi);CHKERRQ(ierr);
    }
    
    
    
    
    
    /* Update boundary conditions */
    /* Fine level setup */
    ierr = pTatinModel_ApplyBoundaryCondition(model,user);CHKERRQ(ierr);
    /* Coarse grid setup: Configure boundary conditions */
    ierr = pTatinModel_ApplyBoundaryConditionMG(nlevels,u_bclist,surf_bclist,dav_hierarchy,model,user);CHKERRQ(ierr);


    /* solve stokes */
    /* a) configure stokes opertors */
    ierr = pTatin3dCreateStokesOperators(stokes,is_stokes_field,
                                         nlevels,dav_hierarchy,interpolation_v,u_bclist,volQ,surf_bclist,level_type,
                                         &A,operatorA11,&B,operatorB11);CHKERRQ(ierr);

    /* b) create solver */
    /* Define non-linear solver */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    //ierr = SNESSetApplicationContext(snes,(void*)user);CHKERRQ(ierr);

    if (!activate_quasi_newton_coord_update) {
      ierr = SNESSetFunction(snes,F,FormFunction_Stokes,user);CHKERRQ(ierr);
    } else {
      ierr = SNESSetFunction(snes,F,FormFunction_Stokes_QuasiNewtonX,user);CHKERRQ(ierr);
    }
    ierr = SNESSetJacobian(snes,A,B,FormJacobian_StokesMGAuu,user);CHKERRQ(ierr);

    //ierr = SNESStokesPCSetOptions_A(snes);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    /* force MG context into SNES */
    ierr = SNESComposeWithMGCtx(snes,&mlctx);CHKERRQ(ierr);

    /* configure KSP */
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);

    /* initial condition used */
    //ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

    /* monitors */
    ierr = pTatin_Stokes_ActivateMonitors(user,snes);CHKERRQ(ierr);
    {
      PetscBool cvg_test_set;

      cvg_test_set = PETSC_FALSE;
      ierr = PetscOptionsGetBool(NULL,NULL,"-stokes_snes_converged_upstol",&cvg_test_set,NULL);CHKERRQ(ierr);
      if (cvg_test_set) { ierr = SNESStokes_SetConvergenceTest_UPstol(snes,user);CHKERRQ(ierr); }
    }

    /* c) configure for fieldsplit */
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"u",is_stokes_field[0]);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"p",is_stokes_field[1]);CHKERRQ(ierr);

    /* configure uu split for galerkin multi-grid */
    ierr = pTatin3dStokesKSPConfigureFSGMG(ksp,nlevels,operatorA11,operatorB11,interpolation_v,dav_hierarchy);CHKERRQ(ierr);

    /* insert boundary conditions into solution vector */
    ierr = DMCompositeGetAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
    ierr = BCListInsert(stokes->u_bclist,velocity);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);

    /* e) solve mechanical model */
    PetscPrintf(PETSC_COMM_WORLD,"   [[ COMPUTING FLOW FIELD FOR STEP : %D ]]\n", step );
    PetscTime(&time[0]);
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
    PetscTime(&time[1]);
    ierr = pTatinLogBasicSNES(user,"Stokes",snes);CHKERRQ(ierr);
    ierr = pTatinLogBasicCPUtime(user,"Stokes",time[1]-time[0]);CHKERRQ(ierr);
    ierr = pTatinLogBasicStokesSolution(user,multipys_pack,X);CHKERRQ(ierr);
    ierr = pTatinLogBasicStokesSolutionResiduals(user,snes,multipys_pack,X);CHKERRQ(ierr);
    //ierr = pTatinLogPetscLog(user,"Stokes");CHKERRQ(ierr);

    /* output */
    if (step%user->output_frequency == 0) {
      PetscSNPrintf(stepname,PETSC_MAX_PATH_LEN-1,"step%1.6D",step);
      ierr = pTatinModel_Output(model,user,X,stepname);CHKERRQ(ierr);
    }

    /* compute timestep */
    user->dt = 1.0e32;
    ierr = DMCompositeGetAccess(multipys_pack,X,&velocity,&pressure);CHKERRQ(ierr);
    ierr = SwarmUpdatePosition_ComputeCourantStep(dav_hierarchy[nlevels-1],velocity,&timestep);CHKERRQ(ierr);
    timestep = timestep/dt_factor;
    ierr = pTatin_SetTimestep(user,"StkCourant",timestep);CHKERRQ(ierr);

    ierr = UpdateMeshGeometry_ComputeSurfaceCourantTimestep(dav_hierarchy[nlevels-1],velocity,surface_displacement_max,&timestep);CHKERRQ(ierr);
    ierr = pTatin_SetTimestep(user,"StkSurfaceCourant",timestep);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(multipys_pack,X,&velocity,&pressure);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"  timestep_stokes[%D] dt_courant = %1.4e \n", step,user->dt );
    if (active_energy) {
      PetscReal timestep;

      {
        
        ierr = DMCompositeGetAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
        
        // (1)
        ierr = PhysCompEnergyFVInterpolateMacroQ2ToSubQ1(dav,velocity,energyfv,energyfv->dmv,energyfv->velocity);CHKERRQ(ierr);
        
        ierr = pTatinPhysCompEnergyFV_ComputeAdvectiveTimestep(energyfv,energyfv->velocity,&timestep);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"  PhysCompEnergyFV_ComputeAdvectiveTimestep[%D] dt_courant = %1.4e \n", step,timestep );
        
        ierr = DMCompositeRestoreAccess(user->pack,X,&velocity,&pressure);CHKERRQ(ierr);
      }
      
      ierr = pTatin_SetTimestep(user,"AdvDiffCourant",timestep);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"  timestep_advdiff[%D] dt_courant = %1.4e \n", step,user->dt );
      
      energyfv->dt   = user->dt;
    }

    /* update time */
    user->step++;
    user->time = user->time + user->dt;
    if (active_energy) {
      energyfv->time = user->time;
    }

    /* tidy up */
    if (active_energy) {
      ierr = VecDestroy(&fv_coor_k);CHKERRQ(ierr);
      ierr = VecDestroy(&q2_coor_k);CHKERRQ(ierr);
    }
    for (k=0; k<nlevels; k++) {
      ierr = MatDestroy(&operatorA11[k]);CHKERRQ(ierr);
      ierr = MatDestroy(&operatorB11[k]);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = SNESDestroyMGCtx(snes);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  }

  /* Clean up */
  for (k=0; k<nlevels-1; k++) {
    ierr = BCListDestroy(&u_bclist[k]);CHKERRQ(ierr);
    ierr = QuadratureDestroy(&volQ[k]);CHKERRQ(ierr);
  }
  for (k=0; k<nlevels; k++) {
    if (interpolation_v[k]) {
      ierr = MatDestroy(&interpolation_v[k]);CHKERRQ(ierr);
    }
    if (interpolation_eta[k]) {
      ierr = MatDestroy(&interpolation_eta[k]);CHKERRQ(ierr);
    }
    ierr = DMDestroy(&dav_hierarchy[k]);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&is_stokes_field[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&is_stokes_field[1]);CHKERRQ(ierr);
  ierr = PetscFree(is_stokes_field);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = pTatin3dDestroyContext(&user);

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscBool experimental_driver,experimental_driver1;
  PetscErrorCode ierr;
  PetscMPIInt rank;

  ierr = pTatinInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);

  experimental_driver = PETSC_FALSE;
  PetscOptionsGetBool(NULL,NULL,"-experimental",&experimental_driver,NULL);

  experimental_driver1 = PETSC_FALSE;
  PetscOptionsGetBool(NULL,NULL,"-nonlinear_driver_v1",&experimental_driver1,NULL);

  if (experimental_driver) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"-nonlinear_driver_v1 only");
  } else if (experimental_driver1) {
    ierr = pTatin3d_nonlinear_viscous_forward_model_driver_v1(argc,argv);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"-nonlinear_driver_v1 only");
  }

  ierr = pTatinGetRangeMaximumMemoryUsage(NULL);CHKERRQ(ierr);

  ierr = pTatinFinalize();CHKERRQ(ierr);
  return 0;
}
