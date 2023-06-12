/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2012
 **        Dave A. May [dave.may@erdw.ethz.ch]
 **        Institute of Geophysics
 **        ETH Zürich
 **        Sonneggstrasse 5
 **        CH-8092 Zürich
 **        Switzerland
 **
 **    project:    pTatin3d
 **    filename:   stokes_output.c
 **
 **
 **    pTatin3d is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published
 **    by the Free Software Foundation, either version 3 of the License,
 **    or (at your option) any later version.
 **
 **    pTatin3d is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 **    See the GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with pTatin3d. If not, see <http://www.gnu.org/licenses/>.
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ @*/


#include "petsc.h"
#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "dmda_element_q2p1.h"
#include "quadrature.h"
#include "dmda_checkpoint.h"
#include "element_type_Q2.h"
#include "mesh_entity.h"
#include "output_paraview.h"
#include "QPntVolCoefStokes_def.h"
#include "QPntSurfCoefStokes_def.h"


/* surface quadrature point viewer */
PetscErrorCode _SurfaceQuadratureViewParaviewVTU_Stokes(SurfaceQuadrature surfQ,PetscInt start,PetscInt end,MeshFacetInfo mfi,const char name[])
{
  PetscErrorCode ierr;
  PetscInt fe,n,e,k,ngp,npoints;
  QPntSurfCoefStokes *all_qpoint;
  QPntSurfCoefStokes *cell_qpoint;
  FILE* fp = NULL;
  double *normal,*tangent1,*tangent2,xp,yp,zp;
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

  PetscFunctionBegin;
  if ((fp = fopen ( name, "w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
  }

  da = mfi->dm;
  element = mfi->element;
  
  ngp = surfQ->npoints;
  nfaces = end - start;
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
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  fprintf(fp, "  <UnstructuredGrid>\n");
  fprintf(fp, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\" >\n",npoints32,npoints32);

  /* POINT COORDS */
  fprintf(fp, "    <Points>\n");
  fprintf(fp, "      <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (fe=start; fe<end; fe++) {
    PetscInt face_id;
    
    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);

    e = mfi->facet_cell_index[fe];
    face_id = mfi->facet_label[fe];
    
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

      fprintf(fp, "      %1.4e %1.4e %1.4e \n", xp, yp, zp );
    }
  }
  fprintf(fp, "      </DataArray>\n");
  fprintf(fp, "    </Points>\n");

  /* POINT-DATA HEADER - OPEN */
  fprintf(fp, "    <PointData>\n");

  /* POINT-DATA FIELDS */

  /* normals */
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"normal\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (fe=start; fe<end; fe++) {
    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      qpoint = &cell_qpoint[n];

      QPntSurfCoefStokesGetField_surface_normal(qpoint,&normal);
      fprintf(fp, "      %1.4e %1.4e %1.4e\n", normal[0],normal[1],normal[2] );
    }
  }
  fprintf(fp, "      </DataArray>\n");

  /* tangent */
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"tangent1\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (fe=start; fe<end; fe++) {
    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      qpoint = &cell_qpoint[n];

      QPntSurfCoefStokesGetField_surface_tangent1(qpoint,&tangent1);
      fprintf(fp, "      %1.4e %1.4e %1.4e\n", tangent1[0], tangent1[1], tangent1[2]);
    }
  }
  fprintf(fp, "      </DataArray>\n");

  /* tangent */
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"tangent2\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (fe=start; fe<end; fe++) {
    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      qpoint = &cell_qpoint[n];

      QPntSurfCoefStokesGetField_surface_tangent2(qpoint,&tangent2);
      fprintf(fp, "      %1.4e %1.4e %1.4e\n", tangent2[0], tangent2[1], tangent2[2]);
    }
  }
  fprintf(fp, "      </DataArray>\n");

  /* eta/rho */
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"eta\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (fe=start; fe<end; fe++) {
    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      double field;
      qpoint = &cell_qpoint[n];
      
      QPntSurfCoefStokesGetField_viscosity(qpoint,&field);
      fprintf(fp, "      %1.4e \n", field );
    }
  }
  fprintf(fp, "      </DataArray>\n");

  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"rho\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (fe=start; fe<end; fe++) {
    ierr =  SurfaceQuadratureGetCellData_Stokes(surfQ,all_qpoint,fe,&cell_qpoint);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      double field;
      qpoint = &cell_qpoint[n];
      
      QPntSurfCoefStokesGetField_density(qpoint,&field);
      fprintf(fp, "      %1.4e \n", field );
    }
  }
  fprintf(fp, "      </DataArray>\n");

  
  /* POINT-DATA HEADER - CLOSE */
  fprintf(fp, "    </PointData>\n");



  /* UNSTRUCTURED GRID DATA */
  fprintf(fp, "    <Cells>\n");

  // connectivity //
  fprintf(fp, "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ", c);
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");

  // offsets //
  fprintf(fp, "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ", (c+1));
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");

  // types //
  fprintf(fp, "      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ", 1);
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");

  fprintf(fp, "    </Cells>\n");


  /* VTU HEADER - CLOSE */
  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </UnstructuredGrid>\n");
  fprintf(fp, "</VTKFile>\n");

  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  fclose(fp);
  PetscFunctionReturn(0);
}

PetscErrorCode _SurfaceQuadratureViewParaviewPVTU_Stokes(const char prefix[],const char name[])
{
  PetscErrorCode ierr;
  FILE* fp = NULL;
  PetscMPIInt nproc;
  PetscInt i,fe;
  char *sourcename;

  PetscFunctionBegin;
  if ((fp = fopen ( name, "w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
  }

  /* PVTU HEADER - OPEN */
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  /* define size of the nodal mesh based on the cell DM */
  fprintf(fp, "  <PUnstructuredGrid GhostLevel=\"0\">\n" ); /* note overlap = 0 */

  /* POINT COORDS */
  fprintf(fp, "    <PPoints>\n");
  fprintf(fp, "      <PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\"/>\n");
  fprintf(fp, "    </PPoints>\n");

  /* CELL-DATA HEADER - OPEN */
  fprintf(fp, "    <PCellData>\n");
  /* CELL-DATA HEADER - CLOSE */
  fprintf(fp, "    </PCellData>\n");

  /* POINT-DATA HEADER - OPEN */
  fprintf(fp, "    <PPointData>\n");
  /* POINT-DATA FIELDS */
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"normal\" NumberOfComponents=\"3\"/>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"tangent1\" NumberOfComponents=\"3\"/>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"tangent2\" NumberOfComponents=\"3\"/>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"eta\" NumberOfComponents=\"1\"/>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"rho\" NumberOfComponents=\"1\"/>\n");

  /* POINT-DATA HEADER - CLOSE */
  fprintf(fp, "    </PPointData>\n");


  /* PVTU write sources */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRQ(ierr);
  for (i=0; i<nproc; i++) {
    for (fe=0; fe<HEX_EDGES; fe++) {
            int i32,fe32;

            PetscMPIIntCast(i,&i32);
            PetscMPIIntCast(fe,&fe32);

      if (asprintf( &sourcename, "%s_face%.2d-subdomain%1.5d.vtu", prefix, fe32,i32 ) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
      fprintf( fp, "    <Piece Source=\"%s\"/>\n",sourcename);
      free(sourcename);
    }
  }


  /* PVTU HEADER - CLOSE */
  fprintf(fp, "  </PUnstructuredGrid>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose( fp );
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceQuadratureViewParaview_Stokes(PhysCompStokes ctx,const char path[],const char prefix[])
{
  PetscInt e;
  char *vtkfilename,*filename;
  PetscMPIInt rank;
  char *appended;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  for (e=0; e<HEX_EDGES; e++) {
        int e32;

        PetscMPIIntCast(e,&e32);
    if (asprintf(&appended,"%s_face%.2d",prefix,e32) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = pTatinGenerateParallelVTKName(appended,"vtu",&vtkfilename);CHKERRQ(ierr);
    if (path) {
      if (asprintf(&filename,"%s/%s",path,vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    } else {
      if (asprintf(&filename,"./%s",vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    }

    ierr = _SurfaceQuadratureViewParaviewVTU_Stokes(ctx->surfQ,
                                                    ctx->mfi->facet_label_offset[e],
                                                    ctx->mfi->facet_label_offset[e+1],
                                                    ctx->mfi,filename);CHKERRQ(ierr);
    free(filename);
    free(vtkfilename);
    free(appended);
  }

  if (asprintf(&appended,"%s_allfaces",prefix) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = pTatinGenerateVTKName(appended,"pvtu",&vtkfilename);CHKERRQ(ierr);
  if (path) {
    if (asprintf(&filename,"%s/%s",path,vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  } else {
    if (asprintf(&filename,"./%s",vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  }
  if (rank==0) { /* not we are a bit tricky about which name we pass in here to define the edge data sets */
    ierr = _SurfaceQuadratureViewParaviewPVTU_Stokes(prefix,filename);CHKERRQ(ierr);
  }
  free(filename);
  free(vtkfilename);
  free(appended);

  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceQuadratureViewParaview_Stokes2(SurfaceQuadrature surfQ, MeshFacetInfo mfi, const char path[], const char prefix[])
{
  PetscInt e;
  char *vtkfilename,*filename;
  PetscMPIInt rank;
  char *appended;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  for (e=0; e<HEX_EDGES; e++) {
    int e32;
    
    PetscMPIIntCast(e,&e32);
    if (asprintf(&appended,"%s_face%.2d",prefix,e32) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = pTatinGenerateParallelVTKName(appended,"vtu",&vtkfilename);CHKERRQ(ierr);
    if (path) {
      if (asprintf(&filename,"%s/%s",path,vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    } else {
      if (asprintf(&filename,"./%s",vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    }
    
    ierr = _SurfaceQuadratureViewParaviewVTU_Stokes(surfQ,
                                                    mfi->facet_label_offset[e],
                                                    mfi->facet_label_offset[e+1],
                                                    mfi,filename);CHKERRQ(ierr);
    free(filename);
    free(vtkfilename);
    free(appended);
  }
  
  if (asprintf(&appended,"%s_allfaces",prefix) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = pTatinGenerateVTKName(appended,"pvtu",&vtkfilename);CHKERRQ(ierr);
  if (path) {
    if (asprintf(&filename,"%s/%s",path,vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  } else {
    if (asprintf(&filename,"./%s",vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  }
  if (rank == 0) { /* not we are a bit tricky about which name we pass in here to define the edge data sets */
    ierr = _SurfaceQuadratureViewParaviewPVTU_Stokes(prefix,filename);CHKERRQ(ierr);
  }
  free(filename);
  free(vtkfilename);
  free(appended);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _VolumeQuadratureViewParaviewVTU_Stokes(PhysCompStokes stokes,const char name[])
{
  
  PetscInt          n,e,k,d,ngp,npoints;
  Quadrature        volQ;
  QPntVolCoefStokes *all_gausspoints,*cell_gausspoints;
  FILE*             fp = NULL;
  DM                stokes_pack,dau,dap,cda;
  Vec               gcoords;
  PetscScalar       *LA_gcoords;
  double            elcoords[3*Q2_NODES_PER_EL_3D];
  double            Ni[Q2_NODES_PER_EL_3D];
  double            *gravity_vector,*momentum_rhs;
  double            eta,rho,Fp,qp_coor[3],xp[3];
  const PetscInt    *elnidx;
  PetscInt          nel,nen;
  int               c,npoints32;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if ((fp = fopen ( name, "w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
  }

  /* Get Stokes DMs */
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  /* Get volume quadrature data structure */
  volQ = stokes->volQ;
  /* Get number of quadrature points per cell */
  ngp  = volQ->npoints;

  /* Get quadrature points data */
  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* Element-nodes connectivity */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen,&elnidx);CHKERRQ(ierr);
  /* Total number of quadrature points */
  npoints = nel * ngp;
  PetscMPIIntCast(npoints,&npoints32);
  
  /* VTU HEADER - OPEN */
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  fprintf(fp, "  <UnstructuredGrid>\n");
  fprintf(fp, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\" >\n",npoints32,npoints32);

  /* POINT COORDS */
  fprintf(fp, "    <Points>\n");
  fprintf(fp, "      <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  /* Loop over elements */
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    /* Get element coordinates */
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*e],LA_gcoords);CHKERRQ(ierr);
    /* Loop over quadrature points */
    for (n=0; n<ngp; n++) {

      for (d=0; d<NSD; d++) {
        qp_coor[d] = volQ->q_xi_coor[3*n + d];
      }
      /* Construct Q2 interpolation function */
      pTatin_ConstructNi_Q2_3D( qp_coor, Ni );

      xp[0] = xp[1] = xp[2] = 0.0;
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        for (d=0; d<NSD; d++) {
          xp[d] += Ni[k] * elcoords[3*k + d];
        }
      }
      fprintf(fp, "      %1.4e %1.4e %1.4e \n", xp[0], xp[1], xp[2] );
    }
  }
  fprintf(fp, "      </DataArray>\n");
  fprintf(fp, "    </Points>\n");

  /* POINT-DATA HEADER - OPEN */
  fprintf(fp, "    <PointData>\n");

  /* POINT-DATA FIELDS */

  /* viscosity */
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"eta_effective\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      /* Get viscosity on quadrature point */
      QPntVolCoefStokesGetField_eta_effective(&cell_gausspoints[n],&eta); 
      fprintf(fp, "      %1.4e \n", eta );
    }
  }
  fprintf(fp, "      </DataArray>\n");

  /* density */
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"rho_effective\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      /* Get viscosity on quadrature point */
      QPntVolCoefStokesGetField_rho_effective(&cell_gausspoints[n],&rho); 
      fprintf(fp, "      %1.4e \n", rho );
    }
  }
  fprintf(fp, "      </DataArray>\n");

  /* momentum rhs */
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"momentum_rhs\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      /* Get viscosity on quadrature point */
      QPntVolCoefStokesGetField_momentum_rhs(&cell_gausspoints[n],&momentum_rhs);
      fprintf(fp, "      %1.4e %1.4e %1.4e\n", momentum_rhs[0],momentum_rhs[1],momentum_rhs[2] );
    }
  }
  fprintf(fp, "      </DataArray>\n");
  
  /* continuity_rhs */
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"continuity_rhs\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (e=0; e<nel; e++) {
    /* Get cell quadrature points data structure */
    ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);
    for (n=0; n<ngp; n++) {
      /* Get viscosity on quadrature point */
      QPntVolCoefStokesGetField_continuity_rhs(&cell_gausspoints[n],&Fp); 
      fprintf(fp, "      %1.4e \n", Fp );
    }
  }
  fprintf(fp, "      </DataArray>\n");

  
  /* POINT-DATA HEADER - CLOSE */
  fprintf(fp, "    </PointData>\n");



  /* UNSTRUCTURED GRID DATA */
  fprintf(fp, "    <Cells>\n");

  // connectivity //
  fprintf(fp, "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ", c);
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");

  // offsets //
  fprintf(fp, "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ", (c+1));
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");

  // types //
  fprintf(fp, "      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  fprintf(fp,"      ");
  for (c=0; c<npoints32; c++) {
    fprintf(fp,"%d ", 1);
  }
  fprintf(fp,"\n");
  fprintf(fp, "      </DataArray>\n");

  fprintf(fp, "    </Cells>\n");


  /* VTU HEADER - CLOSE */
  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </UnstructuredGrid>\n");
  fprintf(fp, "</VTKFile>\n");

  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  fclose(fp);
  PetscFunctionReturn(0);
}

PetscErrorCode _VolumeQuadratureViewParaviewPVTU_Stokes(const char prefix[],const char name[])
{
  PetscErrorCode ierr;
  FILE* fp = NULL;
  PetscMPIInt nproc;
  PetscInt i,fe;
  char *sourcename;

  PetscFunctionBegin;
  if ((fp = fopen ( name, "w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
  }

  /* PVTU HEADER - OPEN */
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  /* define size of the nodal mesh based on the cell DM */
  fprintf(fp, "  <PUnstructuredGrid GhostLevel=\"0\">\n" ); /* note overlap = 0 */

  /* POINT COORDS */
  fprintf(fp, "    <PPoints>\n");
  fprintf(fp, "      <PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\"/>\n");
  fprintf(fp, "    </PPoints>\n");

  /* CELL-DATA HEADER - OPEN */
  fprintf(fp, "    <PCellData>\n");
  /* CELL-DATA HEADER - CLOSE */
  fprintf(fp, "    </PCellData>\n");

  /* POINT-DATA HEADER - OPEN */
  fprintf(fp, "    <PPointData>\n");
  /* POINT-DATA FIELDS */
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"eta_effective\" NumberOfComponents=\"1\"/>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"rho_effective\" NumberOfComponents=\"1\"/>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"momentum_rhs\" NumberOfComponents=\"3\"/>\n");
  fprintf(fp,"      <PDataArray type=\"Float32\" Name=\"continuity_rhs\" NumberOfComponents=\"1\"/>\n");

  /* POINT-DATA HEADER - CLOSE */
  fprintf(fp, "    </PPointData>\n");


  /* PVTU write sources */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRQ(ierr);
  for (i=0; i<nproc; i++) {
    for (fe=0; fe<HEX_EDGES; fe++) {
            int i32,fe32;

            PetscMPIIntCast(i,&i32);
            PetscMPIIntCast(fe,&fe32);

      if (asprintf( &sourcename, "%s_face%.2d-subdomain%1.5d.vtu", prefix, fe32,i32 ) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
      fprintf( fp, "    <Piece Source=\"%s\"/>\n",sourcename);
      free(sourcename);
    }
  }


  /* PVTU HEADER - CLOSE */
  fprintf(fp, "  </PUnstructuredGrid>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose( fp );
  PetscFunctionReturn(0);
}

PetscErrorCode VolumeQuadratureViewParaview_Stokes(PhysCompStokes stokes, const char path[], const char prefix[])
{
  char *vtkfilename,*filename;
  PetscMPIInt rank;
  char *appended;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (asprintf(&appended,"%s",prefix) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = pTatinGenerateParallelVTKName(appended,"vtu",&vtkfilename);CHKERRQ(ierr);
  if (path) {
    if (asprintf(&filename,"%s/%s",path,vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  } else {
    if (asprintf(&filename,"./%s",vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  }

  ierr = _VolumeQuadratureViewParaviewVTU_Stokes(stokes,filename);CHKERRQ(ierr);

  free(filename);
  free(vtkfilename);
  free(appended);
  
  if (asprintf(&appended,"%s",prefix) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  ierr = pTatinGenerateVTKName(appended,"pvtu",&vtkfilename);CHKERRQ(ierr);
  if (path) {
    if (asprintf(&filename,"%s/%s",path,vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  } else {
    if (asprintf(&filename,"./%s",vtkfilename) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
  }
  if (rank == 0) { /* not we are a bit tricky about which name we pass in here to define the edge data sets */
    ierr = _VolumeQuadratureViewParaviewPVTU_Stokes(prefix,filename);CHKERRQ(ierr);
  }
  free(filename);
  free(vtkfilename);
  free(appended);
  
  PetscFunctionReturn(0);
}