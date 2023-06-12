#include "petsc.h"
#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "dmda_bcs.h"
#include "element_utils_q1.h"
#include "dmda_element_q1.h"
#include "dmda_element_q2p1.h"
#include "quadrature.h"
#include "dmda_checkpoint.h"
#include "data_bucket.h"
#include "dmdae.h"
#include "fvda_private.h"
#include "ptatin_log.h"
#include "dmda_update_coords.h"
#include "mesh_update.h"
#include "dmda_remesh.h"
#include "stokes_output.h"

PetscErrorCode CheckGravityOnPoint(pTatinCtx ptatin)
{
  PhysCompStokes        stokes;
  QPntVolCoefStokes     *all_gausspoints,*cell_gausspoints;
  DM                    stokes_pack,dau,dap,cda;
  Vec                   gcoords;
  PetscReal             *LA_gcoords;
  PetscReal             elcoords[3*Q2_NODES_PER_EL_3D];
  const PetscInt        *elnidx_u;
  PetscInt              e,q,nel,nqp,nen_u;
  PetscInt              d,k;
  PetscReal             Ni[Q2_NODES_PER_EL_3D],qp_coor[3],position[3],grav[3];
  PetscErrorCode        ierr;
  
  PetscFunctionBegin;

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);

  nqp = stokes->volQ->npoints;

  /* Get Stokes DMs */
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  /* Element-nodes connectivity */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);

  /* Loop over elements */
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /* Get element coordinates */
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);
    /* Loop over quadrature points */
    for (q=0; q<nqp; q++) {

      /* Get quadrature point coordinates */
      for (d=0; d<NSD; d++) {
        qp_coor[d] = stokes->volQ->q_xi_coor[3*q + d];
      }
      /* Construct Q2 interpolation function */
      pTatin_ConstructNi_Q2_3D( qp_coor, Ni );

      /* Interpolate quadrature point global coords */
      position[0] = position[1] = position[2] = 0.0;
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        for (d=0; d<NSD; d++) {
          position[d] += Ni[k] * elcoords[3*k + d];
        }
      }

      QPntVolCoefStokesSetField_rho_effective(&cell_gausspoints[q],1.0);
      ierr = pTatinGetGravityPointWiseVector(ptatin,e,position,qp_coor,grav);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode GenerateSphericalMesh()
{
  Vec X;
  IS        *is_stokes_field;
  Vec       velocity,pressure;
  pTatinCtx      ptatin = NULL;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscReal      O[3],L[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatin3dCreateContext(&ptatin);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(ptatin);CHKERRQ(ierr);

  O[0] = M_PI / 3.0; //0.0;
  O[1] = 63.75-12.0; //6375.0e3-1200e3;//650.0e3;
  O[2] = -M_PI / 6.0; //0.0; 

  L[0] = M_PI / 1.5; //M_PI;
  L[1] = 63.75; //6375.0e3;
  L[2] = M_PI / 6.0; //2.0*M_PI;

  PetscPrintf(PETSC_COMM_WORLD,"Box: Ox, Lx = [ %f, %f ]\n",O[0],L[0]);
  PetscPrintf(PETSC_COMM_WORLD,"Box: Oy, Ly = [ %f, %f ]\n",O[1],L[1]);
  PetscPrintf(PETSC_COMM_WORLD,"Box: Oz, Lz = [ %f, %f ]\n",O[2],L[2]);

  ptatin->mx = 8;
  ptatin->my = 8;
  ptatin->mz = 8;
  // Create a Q2 mesh 
  ierr = pTatin3d_PhysCompStokesCreate(ptatin);CHKERRQ(ierr);

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformSphericalToCartesianCoordinates(dav,O[0],L[0],O[1],L[1],O[2],L[2]);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);

  {
    Vec X;

    ierr = DMGetGlobalVector(stokes_pack,&X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(stokes_pack,&X);CHKERRQ(ierr);
  }
  ierr = DMCompositeGetGlobalISs(stokes_pack,&is_stokes_field);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(stokes_pack,&X);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);

  /* Gravity */
  {
    Gravity gravity;
    GravityType gtype;
    PetscReal gvec[] = {0.0,-9.8,0.0},scaling;
    PetscReal gmag = -9.8;

    gtype = GRAVITY_RADIAL_CONSTANT; //GRAVITY_CONSTANT;
    scaling = 2.0;

    ierr = pTatinCreateGravity(ptatin,gtype);CHKERRQ(ierr);
    ierr = pTatinGetGravityCtx(ptatin,&gravity);CHKERRQ(ierr);

    switch (gtype) {
      case GRAVITY_CONSTANT:
        {
          PetscReal grav[3];
          PetscReal mag;

          ierr = GravitySet_Constant(gravity,gvec);CHKERRQ(ierr);
          ierr = GravityGet_ConstantVector(gravity,grav);CHKERRQ(ierr);
          ierr = GravityGet_ConstantMagnitude(gravity,&mag);CHKERRQ(ierr);
          PetscPrintf(PETSC_COMM_WORLD,"gvec = (%f, %f, %f), mag = %f\n",grav[0],grav[1],grav[2],mag);
        }
        break;

      case GRAVITY_RADIAL_CONSTANT:
        ierr = GravitySet_RadialConstant(gravity,gmag);CHKERRQ(ierr);
        break;

      default:
        break;
    }
    ierr = GravityScale(gravity,scaling);CHKERRQ(ierr);
    ierr = pTatinQuadratureSetGravity(ptatin);

    /* Set density on quadrature points */
    ierr = CheckGravityOnPoint(ptatin);
    /* Set rho*g on quadrature points */
    ierr = pTatinQuadratureUpdateGravity(ptatin);CHKERRQ(ierr);
    /* output volume quadrature points */
    ierr = VolumeQuadratureViewParaview_Stokes(stokes,ptatin->outputpath,"def");CHKERRQ(ierr);
  }

  {
    PetscViewer viewer;
    char        fname[256];

    sprintf(fname,"spherical_domain.vts");

    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(velocity,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(stokes_pack,&X);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = pTatin3dDestroyContext(&ptatin);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = GenerateSphericalMesh();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}