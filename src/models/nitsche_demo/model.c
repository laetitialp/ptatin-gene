#include "petsc.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"

#include "dmda_bcs.h"
#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include "dmda_iterator.h"
#include "mesh_deformation.h"
#include "mesh_update.h"
#include "dmda_remesh.h"
#include "dmda_element_q2p1.h"
#include "material_point_std_utils.h"
#include "material_point_popcontrol.h"
#include "output_material_points.h"
#include "xdmf_writer.h"
#include "output_material_points_p0.h"
#include "private/quadrature_impl.h"
#include "quadrature.h"
#include "QPntSurfCoefStokes_def.h"

#include "mesh_entity.h"
#include "surface_constraint.h"
#include "surfbclist.h"


PetscInt boundary_conditon_type = 0;

static PetscErrorCode ModelInitialize(pTatinCtx c,void *ctx)
{
  RheologyConstants *rheology;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  rheology                = &c->rheology_constants;
  rheology->rheology_type = RHEOLOGY_VISCOUS;
  rheology->const_eta0[0] = 1.0;
  rheology->const_eta0[1] = 1.0;
  rheology->const_rho0[0] = 1.0;
  rheology->const_rho0[1] = 2.1;
  rheology->const_rho0[0] = 0.0;
  rheology->const_rho0[1] = 0.2;
  rheology->nphases_active = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-bc",&boundary_conditon_type,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode bctype_no_slip(DM dav,BCList bclist)
{
  PetscErrorCode ierr;
  PetscScalar zero = 0.0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode const_ud(Facet F,const PetscReal qp_coor[],PetscReal uD[],void *data)
{
  PetscReal *input = (PetscReal*)data;
  uD[0] = input[0];
  uD[1] = input[1];
  uD[2] = input[2];
  PetscFunctionReturn(0);
}

PetscErrorCode bctype_no_slip_nitsche(SurfBCList surflist,PetscBool insert_if_not_fouund)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscErrorCode    ierr;

  
  ierr = SurfBCListGetConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_fouund) {
      ierr = SurfBCListAddConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_DIRICHLET);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  
  {
    PetscInt       nsides;
    HexElementFace sides[] = { HEX_FACE_Nxi, HEX_FACE_Pxi, HEX_FACE_Neta, HEX_FACE_Nzeta, HEX_FACE_Pzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }
  
  SURFC_CHKSETVALS(SC_NITSCHE_DIRICHLET,const_ud);
  {
    PetscReal uD_c[] = {0.0, 0.0, 0.0};
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)const_ud,(void*)uD_c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


static PetscErrorCode const_udotn(Facet F,const PetscReal qp_coor[],PetscReal udotn[],void *data)
{
  PetscReal *input = (PetscReal*)data;
  udotn[0] = input[0];
  PetscFunctionReturn(0);
}

PetscErrorCode bctype_slip_nitsche(SurfBCList surflist,PetscBool insert_if_not_fouund)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscErrorCode    ierr;
  
  
  ierr = SurfBCListGetConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_fouund) {
      ierr = SurfBCListAddConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_NAVIER_SLIP);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  
  {
    PetscInt       nsides;
    HexElementFace sides[] = { HEX_FACE_Nxi, HEX_FACE_Pxi, HEX_FACE_Neta, HEX_FACE_Nzeta, HEX_FACE_Pzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }
  
  SURFC_CHKSETVALS(SC_NITSCHE_NAVIER_SLIP,const_ud);
  {
    PetscReal uD_c[] = {0.0};
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)const_udotn,(void*)uD_c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition(pTatinCtx user,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);


  switch (boundary_conditon_type) {
    case 0:
      ierr = bctype_no_slip(user->stokes_ctx->dav,user->stokes_ctx->u_bclist);CHKERRQ(ierr);
      break;

    case 1:
    {
      PhysCompStokes stokes;
      
      ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
      ierr = bctype_no_slip_nitsche(stokes->surf_bclist,PETSC_TRUE);CHKERRQ(ierr);
    }
      break;

    case 2:
    {
      PhysCompStokes stokes;
      
      ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
      ierr = bctype_slip_nitsche(stokes->surf_bclist,PETSC_TRUE);CHKERRQ(ierr);
    }
      break;

    default:
      break;
  }

  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryConditionMG(PetscInt nl,BCList bclist[],SurfBCList surf_bclist[],DM dav[],pTatinCtx user,void *ctx)
{
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  
  switch (boundary_conditon_type) {
    case 0:
      for (n=0; n<nl; n++) {
        ierr = bctype_no_slip(dav[n],bclist[n]);CHKERRQ(ierr);
      }
      break;

    case 1:
      for (n=0; n<nl-1; n++) {
        ierr = bctype_no_slip_nitsche(surf_bclist[n],PETSC_FALSE);CHKERRQ(ierr);
      }
      break;

    case 2:
      for (n=0; n<nl-1; n++) {
        ierr = bctype_slip_nitsche(surf_bclist[n],PETSC_FALSE);CHKERRQ(ierr);
      }
      break;

    default:
      break;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMeshGeometry(pTatinCtx c,void *ctx)
{
  PetscErrorCode ierr;
  PetscReal Lx,Ly,Lz;

  PetscFunctionBegin;
  Lx = 1.0;
  Ly = 1.0;
  Lz = 1.0;
  ierr = DMDASetUniformCoordinates(c->stokes_ctx->dav,0.0,Lx, 0.0,Ly, 0.0,Lz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ApplyInitialMaterialGeometry_SingleInclusion(pTatinCtx c)
{
  int                    p,n_mp_points;
  DataBucket             db;
  DataField              PField_std,PField_stokes;
  int                    phase;
  PetscReal              Lx,Ly,Lz,gmin[3],gmax[3],origin[3],length[3],eta0,eta1,rho0,rho1;
  RheologyConstants      *rheology;
  PetscErrorCode         ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = DMGetBoundingBox(c->stokes_ctx->dav,gmin,gmax);CHKERRQ(ierr);
  
  Lx = gmax[0] - gmin[0];
  Ly = gmax[1] - gmin[1];
  Lz = gmax[2] - gmin[2];
  
  rheology                = &c->rheology_constants;
  rheology->rheology_type = RHEOLOGY_VISCOUS;
  
  eta0 = rheology->const_eta0[0] = 1.0;
  eta1 = rheology->const_eta0[1] = 1.0;
  
  rho0 = rheology->const_rho0[0] = 1.0e3;
  rho1 = rheology->const_rho0[1] = 1.001e3;


  origin[0] = 0.5 * Lx;
  origin[1] = 0.5 * Ly;
  origin[2] = 0.5 * Lz;
  
  /* spheriod diameter or box length */
  length[0] = 0.3 * Lx;
  length[1] = 0.3 * Ly;
  length[2] = 0.3 * Lz;
  

  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokes_classname,&PField_stokes);
  DataFieldGetAccess(PField_stokes);
  DataFieldVerifyAccess(PField_stokes,sizeof(MPntPStokes));

  DataBucketGetSizes(db,&n_mp_points,0,0);

  for (p=0; p<n_mp_points; p++) {
    MPntStd     *material_point;
    MPntPStokes *mpprop_stokes;
    double      *position;
    double      eta,rho;

    DataFieldAccessPoint(PField_std,p,   (void**)&material_point);
    DataFieldAccessPoint(PField_stokes,p,(void**)&mpprop_stokes);

    /* Access using the getter function provided for you (recommeneded for beginner user) */
    MPntStdGetField_global_coord(material_point,&position);

    phase = 0;
    eta =  eta0;
    rho = -rho0;

    { //if (data->is_sphere) {
      double rx = (position[0] - origin[0])/(0.5*length[0]);
      double ry = (position[1] - origin[1])/(0.5*length[1]);
      double rz = (position[2] - origin[2])/(0.5*length[2]);

      if ( rx*rx + ry*ry + rz*rz < 1.0 ) {
        phase = 1;
        eta =  eta1;
        rho = -rho1;
      }
    }

    /* user the setters provided for you */
    MPntStdSetField_phase_index(material_point,phase);

    MPntPStokesSetField_eta_effective(mpprop_stokes,eta);
    MPntPStokesSetField_density(mpprop_stokes,rho);
  }

  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_stokes);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry(pTatinCtx c,void *ctx)
{
  PetscErrorCode ierr;
  ierr = ApplyInitialMaterialGeometry_SingleInclusion(c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutput(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  SurfaceQuadratureViewParaview_Stokes(c->stokes_ctx,"./","surfq");

  ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  // testing
  //ierr = pTatin3d_ModelOutputLite_Velocity_Stokes(c,X,prefix);CHKERRQ(ierr);
  //ierr = pTatinOutputLiteMeshVelocitySlicedPVTS(c->stokes_ctx->stokes_pack,c->outputpath,prefix);CHKERRQ(ierr);
  //ierr = ptatin3d_StokesOutput_VelocityXDMF(c,X,prefix);CHKERRQ(ierr);
  // testing
  //ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  ierr = pTatin3d_ModelOutput_MPntStd(c,prefix);CHKERRQ(ierr);
  // tests for alternate (output/load)ing of "single file marker" formats
  //ierr = SwarmDataWriteToPetscVec(c->materialpoint_db,prefix);CHKERRQ(ierr);
  //ierr = SwarmDataLoadFromPetscVec(c->materialpoint_db,prefix);CHKERRQ(ierr);
  /*
     {
     MaterialPointVariable vars[] = { MPV_viscosity, MPV_density, MPV_region };
  //ierr = pTatin3dModelOutput_MarkerCellFieldsP0_ParaView(c,sizeof(vars)/sizeof(MaterialPointVariable),vars,PETSC_TRUE,prefix);CHKERRQ(ierr);
  ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(c,PETSC_TRUE,sizeof(vars)/sizeof(MaterialPointVariable),vars,prefix);CHKERRQ(ierr);
  }
  */
  PetscFunctionReturn(0);
}


PetscErrorCode pTatinModelRegister_NitscheDemo(void)
{
  pTatinModel m;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"nitsche_demo");CHKERRQ(ierr);

  /* Set model data */

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelInitialCondition_ViscousSinker);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_ViscousSinker);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_ViscousSinker);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_ViscousSinker);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
