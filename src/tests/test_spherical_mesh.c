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
#include "dmda_iterator.h"

static PetscErrorCode CheckGravityOnPoint(pTatinCtx ptatin)
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

static PetscErrorCode InitialTopography(DM da, PetscReal perturbation)
{
  MPI_Comm       comm;
  DM             cda;
  Vec            coord;
  PetscScalar   *LA_coords;
  PetscInt       i, j, k, d, M, N, P, istart, isize, jstart, jsize, kstart, ksize, dim, nidx;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(da, &dim, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  ierr = PetscObjectGetComm((PetscObject)da, &comm);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, &istart, &jstart, &kstart, &isize, &jsize, &ksize);CHKERRQ(ierr); 
  ierr = DMGetCoordinateDM(da, &cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = VecGetArray(coord,&LA_coords);CHKERRQ(ierr);

  if (jstart + jsize == N) { // surface of the mesh
    j = jsize - 1;
    for (k=0; k<ksize; k++) {
      for (i=0; i<isize; i++) {
        PetscReal coor_3d[3],elevation,theta,phi,A;

        nidx = i + j*isize + k*jsize*isize;

        elevation = 0.0;
        for (d=0; d<3; d++) {
          coor_3d[d] = LA_coords[3*nidx + d];
          elevation += coor_3d[d] * coor_3d[d];
        }
        elevation = PetscSqrtReal(elevation);
        elevation += perturbation;

        A  = coor_3d[1] * coor_3d[1];
        A += coor_3d[2] * coor_3d[2];
        A  = PetscSqrtReal(A);
        
        theta = PetscAtan2Real( A          , coor_3d[0] );
        phi   = PetscAtan2Real( coor_3d[2] , coor_3d[1] );

        if ( coor_3d[0] <= -20.0 && coor_3d[0] >= -25.0) {
          LA_coords[3*nidx + 0] = elevation * PetscCosReal(theta);
          LA_coords[3*nidx + 1] = elevation * PetscSinReal(theta) * PetscCosReal(phi) ; 
          LA_coords[3*nidx + 2] = elevation * PetscSinReal(theta) * PetscSinReal(phi);
        }
      }
    }
  }

  ierr = VecRestoreArray(coord, &LA_coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscBool TangentRadialVelocity(PetscScalar position[], PetscScalar *value, void *ctx)
{
  PetscInt  dim = *((PetscInt*)ctx);
  PetscInt  d;
  PetscBool impose=PETSC_TRUE;
  PetscReal position_norm,normal[3],radial_u[3];

  PetscFunctionBegin;

  position_norm = 0.0;
  for (d=0; d<3; d++) {
    position_norm += position[d] * position[d];
  }
  position_norm = PetscSqrtReal(position_norm);

  /* Radial normal vector */
  for (d=0; d<3; d++) {
    if (position_norm > 1.0e-17) {
      normal[d] = position[d] / position_norm;
    } else {
      normal[d] = 0.0;
    }
  }

  /* Compute the cross product with unit z */
  radial_u[0] = normal[1];
  radial_u[1] = -normal[0];
  radial_u[2] = 0.0;

  *value = radial_u[ dim ];

  PetscFunctionReturn(impose);
}

static PetscErrorCode AdvectionVelocity(DM dav, Vec velocity)
{
  PetscInt       component;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Initialize to zero the velocity vector */
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);

  /* x component */
  component = 0;
  ierr = DMDAVecTraverse3d(dav,velocity,component,TangentRadialVelocity,(void*)&component);CHKERRQ(ierr);
  /* y component */
  component = 1;
  ierr = DMDAVecTraverse3d(dav,velocity,component,TangentRadialVelocity,(void*)&component);CHKERRQ(ierr);
  /* z component */
  component = 2;
  ierr = DMDAVecTraverse3d(dav,velocity,component,TangentRadialVelocity,(void*)&component);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode WritePVD(char fname[], PetscInt nsteps)
{
  FILE* fp = NULL;
  PetscInt n;

  PetscFunctionBegin;
  if ((fp = fopen ( fname, "w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",fname );
  }

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  fprintf(fp, "<Collection>\n");
  for (n=0; n<nsteps; n++) {
    fprintf(fp, "  <DataSet timestep=\"%f\" file=\"spherical_domain_%d.vts\"/>\n",(float)n,n);
  }
  fprintf(fp, "</Collection>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose( fp );
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
  PetscReal      O[3],L[3],dt;
  PetscInt       step;
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

  ptatin->mx = 16;
  ptatin->my = 8;
  ptatin->mz = 16;
  // Create a Q2 mesh 
  ierr = pTatin3d_PhysCompStokesCreate(ptatin);CHKERRQ(ierr);

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformSphericalToCartesianCoordinates(dav,O[0],L[0],O[1],L[1],O[2],L[2]);CHKERRQ(ierr);

  ierr = InitialTopography(dav,1.0);CHKERRQ(ierr);

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
  
  /* Advection velocity */
  ierr = AdvectionVelocity(dav,velocity);CHKERRQ(ierr);

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
    //ierr = VolumeQuadratureViewParaview_Stokes(stokes,ptatin->outputpath,"def");CHKERRQ(ierr);
  }

  for (step=0; step<25; step++) {

    dt = 1.0;
    {
      PetscViewer viewer;
      char        fname[256];

      sprintf(fname,"%s/spherical_domain_%d.vts",ptatin->outputpath,step);

      ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
      ierr = VecView(velocity,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,dt,PETSC_TRUE);CHKERRQ(ierr);
  }

  {
    char pvd_name[256];

    sprintf(pvd_name,"%s/spherical_domain.pvd",ptatin->outputpath);
    ierr = WritePVD(pvd_name,step);CHKERRQ(ierr);
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