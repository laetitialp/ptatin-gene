/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2012, 
 **        Dave A. May [dave.may@erdw.ethz.ch]
 **        Geophysical Fluid Dynamics, 
 **        Department of Earth Sciences,
 **        ETH Zürich,
 **        Sonneggstrasse 5,
 **        CH-8092 Zurich,
 **        Switzerland
 **
 **    Project:       pTatin3d
 **    Filename:      material_point_std_utils.c
 **
 **
 **    pTatin3d is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published by
 **    the Free Software Foundation, either version 3 of the License, or
 **    (at your option) any later version.
 **
 **    pTatin3d is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 **    GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with pTatin3d.  If not, see <http://www.gnu.org/licenses/>.
 **
 **
 **    $Id$
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@*/


#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "ptatin3d.h"
#include "element_type_Q2.h"
#include "dmda_element_q2p1.h"
#include "swarm_fields.h"
#include "data_exchanger.h"
#include "MPntStd_def.h"
#include "ptatin3d_stokes.h"
#include "output_paraview.h"


#undef __FUNCT__
#define __FUNCT__ "SwarmMPntStd_AssignUniquePointIdentifiers"
PetscErrorCode SwarmMPntStd_AssignUniquePointIdentifiers(MPI_Comm comm,DataBucket db,int start_pid,int end_pid)
{
	DataField    PField;
	long int     np_local, np_global, max_local, max;
	int          rank,p,L;
	
	
	PetscFunctionBegin;
	
	MPI_Comm_rank(comm,&rank);
	DataBucketGetDataFieldByName(db,MPntStd_classname,&PField);
	DataFieldGetAccess(PField);
	DataFieldVerifyAccess( PField,sizeof(MPntStd));
	
	DataBucketGetSizes(db,&L,0,0);
	
	/* find max pid presently in the system */
	max_local = 0;
	for (p=0; p<L; p++) {
		MPntStd *marker;
		DataFieldAccessPoint(PField,p,(void**)&marker);
		
		if ( marker->pid > max_local ) {
			max_local = marker->pid;
		}
	}
	MPI_Allreduce( &max_local, &max, 1, MPI_LONG, MPI_MAX, comm );
	PetscPrintf(PETSC_COMM_WORLD,"SwarmMPntStd_AssignUniquePointIdentifiers : max_pid = %ld \n",max);
	max = max + 1;
	
	/* give particles a unique identifier */
	np_local = (end_pid-start_pid);

	MPI_Scan( &np_local, &np_global, 1, MPI_LONG, MPI_SUM, comm );
	//printf("rank %d : np_local = %ld, np_global = %ld \n",rank,np_local,np_global);
	for (p=start_pid; p<end_pid; p++) {
		MPntStd *marker;
		DataFieldAccessPoint(PField,p,(void**)&marker);
		
		//marker->pid = max + (np_global-1) - (np_local-1-p);
		marker->pid = max + (np_global-np_local) + (p-start_pid);
		//printf("assigning %d -> pid = %ld \n", p, marker->pid );
	}
	
	
	DataFieldRestoreAccess(PField);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SwarmMPntStd_CoordAssignment_LatticeLayout3d"
PetscErrorCode SwarmMPntStd_CoordAssignment_LatticeLayout3d(DM da,PetscInt Nxp[],PetscReal perturb,DataBucket db)
{
	DataField    PField;
	PetscInt     e,mE,nE,pE;
  DM           cda;
  Vec          gcoords;
  PetscScalar  *LA_coords;
  PetscScalar  el_coords[Q2_NODES_PER_EL_3D*NSD];
	int          ncells,np_per_cell;
	PetscInt     nel,nen;
	const PetscInt     *elnidx;
	PetscInt     p,k,pi,pj,pk;
	PetscReal    dxi,deta,dzeta;
	long int     np_local, np_global;
	int          rank;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscOptionsGetReal(PETSC_NULL,"-lattice_layout_perturb", &perturb, PETSC_NULL );
	PetscOptionsGetInt(PETSC_NULL,"-lattice_layout_Nx", &Nxp[0], PETSC_NULL );
	PetscOptionsGetInt(PETSC_NULL,"-lattice_layout_Ny", &Nxp[1], PETSC_NULL );
	PetscOptionsGetInt(PETSC_NULL,"-lattice_layout_Nz", &Nxp[2], PETSC_NULL );
	
	ierr = DMDAGetElements_pTatinQ2P1(da,&nel,&nen,&elnidx);CHKERRQ(ierr);
	
	// re-size //
	ncells = nel;
	np_per_cell = Nxp[0] * Nxp[1] * Nxp[2];
	DataBucketSetSizes(db,np_per_cell*ncells,-1);
	
	if (perturb<0.0) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Cannot use a negative perturbation");
	}
	if (perturb>1.0) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Cannot use a perturbation greater than 1.0");
	}
	
  /* setup for coords */
  ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_coords);CHKERRQ(ierr);
	
	
	DataBucketGetDataFieldByName(db,MPntStd_classname,&PField);
	DataFieldGetAccess(PField);
	DataFieldVerifyAccess( PField,sizeof(MPntStd));
	
	dxi   = 2.0/(PetscReal)Nxp[0];
	deta  = 2.0/(PetscReal)Nxp[1];
	dzeta  = 2.0/(PetscReal)Nxp[2];
	
	p = 0;
	for (e = 0; e < ncells; e++) {
		/* get coords for the element */
		ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&elnidx[nen*e],LA_coords);CHKERRQ(ierr);
		
		for (pk=0; pk<Nxp[2]; pk++) {
			for (pj=0; pj<Nxp[1]; pj++) {
				for (pi=0; pi<Nxp[0]; pi++) {
					MPntStd *marker;
					double xip_rand[NSD],xp_rand[NSD],Ni[Q2_NODES_PER_EL_3D];
					
					/* random between -1 <= xi,eta,zeta <= 1 */
					xip_rand[0] = 2.0*(rand()/(RAND_MAX+1.0)) - 1.0;
					xip_rand[1] = 2.0*(rand()/(RAND_MAX+1.0)) - 1.0;
					xip_rand[2] = 2.0*(rand()/(RAND_MAX+1.0)) - 1.0;
					
					xip_rand[0] = perturb * dxi    * xip_rand[0];
					xip_rand[1] = perturb * deta   * xip_rand[1];
					xip_rand[2] = perturb * dzeta  * xip_rand[2];
					
					xip_rand[0] += -1.0 + dxi    * (pi + 0.5);
					xip_rand[1] += -1.0 + deta   * (pj + 0.5);
					xip_rand[2] += -1.0 + dzeta  * (pk + 0.5);
					
					if (fabs(xip_rand[0]) > 1.0) {
						SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"fabs(x-point coord) greater than 1.0");
					}
					if (fabs(xip_rand[1]) > 1.0) {
						SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"fabs(y-point coord) greater than 1.0");
					}
					if (fabs(xip_rand[2]) > 1.0) {
						SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"fabs(z-point coord) greater than 1.0");
					}
					
					pTatin_ConstructNi_Q2_3D(xip_rand,Ni);
					
					xp_rand[0] = xp_rand[1] = xp_rand[2] = 0.0;
					for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
						xp_rand[0] += Ni[k] * el_coords[NSD*k+0];
						xp_rand[1] += Ni[k] * el_coords[NSD*k+1];
						xp_rand[2] += Ni[k] * el_coords[NSD*k+2];
					}
					
					DataFieldAccessPoint(PField,p,(void**)&marker);
					
					marker->coor[0] = xp_rand[0];
					marker->coor[1] = xp_rand[1];
					marker->coor[2] = xp_rand[2];
					
					marker->xi[0] = xip_rand[0];
					marker->xi[1] = xip_rand[1];
					marker->xi[2] = xip_rand[2];
					
					marker->wil    = e;
					marker->pid    = 0;
					
					p++;
				}
			}
		}		
	}
	
	DataFieldRestoreAccess(PField);
	
	np_local = np_per_cell * ncells;
	ierr = SwarmMPntStd_AssignUniquePointIdentifiers(((PetscObject)da)->comm,db,0,np_local);CHKERRQ(ierr);
	
  ierr = VecRestoreArray(gcoords,&LA_coords);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SwarmMPntStd_CoordAssignment_RandomLayout3d"
PetscErrorCode SwarmMPntStd_CoordAssignment_RandomLayout3d(DM da,PetscInt nPerCell,DataBucket db)
{
	DataField    PField;
	PetscInt     e,mE,nE,pE;
  DM           cda;
  Vec          gcoords;
  PetscScalar  *LA_coords;
  PetscScalar  el_coords[Q2_NODES_PER_EL_3D*NSD];
	int          ncells,np_per_cell;
	PetscInt     nel,nen;
	const PetscInt *elnidx;
	PetscInt     p,k,pi;
	long int     np_local, np_global;
	int          rank;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	
	PetscOptionsGetInt(PETSC_NULL,"-random_layout_Np", &nPerCell, PETSC_NULL );
	
	// re-size //
	ierr = DMDAGetElements_pTatinQ2P1(da,&nel,&nen,&elnidx);CHKERRQ(ierr);
	ncells = nel;
	np_per_cell = nPerCell;
	DataBucketSetSizes(db,np_per_cell*ncells,-1);
	
  /* setup for coords */
  ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_coords);CHKERRQ(ierr);
	
	
	DataBucketGetDataFieldByName(db,MPntStd_classname,&PField);
	DataFieldGetAccess(PField);
	DataFieldVerifyAccess( PField,sizeof(MPntStd));
	
	p = 0;
	for (e = 0; e < ncells; e++) {
		/* get coords for the element */
		ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&elnidx[nen*e],LA_coords);CHKERRQ(ierr);
		
		for (pi=0; pi<np_per_cell; pi++) {
			MPntStd *marker;
			double xip_rand[NSD],xp_rand[NSD],Ni[Q2_NODES_PER_EL_3D];
			
			/* random between -1 <= xi,eta,zeta <= 1 */
			xip_rand[0] = 2.0*(rand()/(RAND_MAX+1.0)) - 1.0;
			xip_rand[1] = 2.0*(rand()/(RAND_MAX+1.0)) - 1.0;
			xip_rand[2] = 2.0*(rand()/(RAND_MAX+1.0)) - 1.0;
			
			pTatin_ConstructNi_Q2_3D(xip_rand,Ni);
			
			xp_rand[0] = xp_rand[1] = xp_rand[2] = 0.0;
			for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
				xp_rand[0] += Ni[k] * el_coords[NSD*k+0];
				xp_rand[1] += Ni[k] * el_coords[NSD*k+1];
				xp_rand[2] += Ni[k] * el_coords[NSD*k+2];
			}
			
			DataFieldAccessPoint(PField,p,(void**)&marker);
			
			marker->coor[0] = xp_rand[0];
			marker->coor[1] = xp_rand[1];
			marker->coor[2] = xp_rand[2];
			
			marker->xi[0] = xip_rand[0];
			marker->xi[1] = xip_rand[1];
			marker->xi[2] = xip_rand[2];
			
			marker->wil    = e;
			marker->pid    = p;
			
			p++;
		}
		
	}
	
	np_local = np_per_cell * ncells;
	ierr = SwarmMPntStd_AssignUniquePointIdentifiers(((PetscObject)da)->comm,db,0,np_local);CHKERRQ(ierr);
	
	DataFieldRestoreAccess(PField);
	ierr = DMDAVecRestoreArray(cda,gcoords,&LA_coords);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SwarmView_MPntStd_VTKascii"
PetscErrorCode SwarmView_MPntStd_VTKascii(DataBucket db,const char name[])
{
	PetscMPIInt rank;
	FILE *vtk_fp;
	PetscInt k;
	int npoints;
	PetscLogDouble t0,t1;
	DataField PField;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	ierr = PetscGetTime(&t0);CHKERRQ(ierr);
	
	if ((vtk_fp = fopen ( name, "w")) == NULL)  {
		SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
	}
	
	fprintf( vtk_fp, "<?xml version=\"1.0\"?>\n");
	
#ifdef WORDSIZE_BIGENDIAN
	fprintf( vtk_fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
	fprintf( vtk_fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
	
	fprintf( vtk_fp, "\t<UnstructuredGrid>\n" );
	
	DataBucketGetSizes(db,&npoints,PETSC_NULL,PETSC_NULL);
	fprintf( vtk_fp, "\t\t<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",npoints,npoints );
	
	
	fprintf( vtk_fp, "\n");
	fprintf( vtk_fp, "\t\t\t<Cells>\n");
	
	// connectivity //
	fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
	fprintf( vtk_fp, "\t\t\t\t");
	for(k=0;k<npoints;k++) {
		fprintf( vtk_fp,"%d ",k);
	}
	fprintf( vtk_fp, "\n");
	fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");	
	
	// offsets //
	fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
	fprintf( vtk_fp, "\t\t\t\t");
	for(k=0;k<npoints;k++) {
		fprintf( vtk_fp,"%d ",k+1);
	}
	fprintf( vtk_fp, "\n");
	fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
	
	// types //
	fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
	fprintf( vtk_fp, "\t\t\t\t");
	for(k=0;k<npoints;k++) {
		fprintf( vtk_fp,"1 "); // VTK_VERTEX //
	}
	fprintf( vtk_fp, "\n");
	fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
	
	fprintf( vtk_fp, "\t\t\t</Cells>\n");
	
	fprintf( vtk_fp, "\n");
	fprintf( vtk_fp, "\t\t\t<CellData>\n");
	fprintf( vtk_fp, "\t\t\t</CellData>\n");
	fprintf( vtk_fp, "\n");
	
	
	DataBucketGetDataFieldByName(db, MPntStd_classname ,&PField);
	DataFieldGetAccess(PField);
	DataFieldVerifyAccess( PField,sizeof(MPntStd));
	
	
	/* point coordinates */
	fprintf( vtk_fp, "\t\t\t<Points>\n");
	
	/* copy coordinates */
	fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
	for(k=0;k<npoints;k++) {
		MPntStd *marker;
		double *coords;
		
		DataFieldAccessPoint(PField,k,(void**)&marker);
		
		
		/* extract coords from your data type */
		//coords = elasticParticle->pos;
		MPntStdGetField_global_coord( marker,&coords );
		
		fprintf( vtk_fp,"\t\t\t\t\t%lf %lf %lf \n",coords[0],coords[1],coords[2]);
	}
	fprintf( vtk_fp, "\t\t\t\t</DataArray>\n");
	
	fprintf( vtk_fp, "\t\t\t</Points>\n");
	fprintf( vtk_fp, "\n");
	
	DataFieldRestoreAccess(PField);
	
	/* point data BEGIN */
	fprintf( vtk_fp, "\t\t\t<PointData>\n");
	
	/* auto generated shit goes here */
	{
		MPntStd *marker = PField->data; /* should write a function to do this */
		
		MPntStdVTKWriteAsciiAllFields(vtk_fp,(const int)npoints,(const MPntStd*)marker );
	}
	fprintf( vtk_fp, "\t\t\t</PointData>\n");
	fprintf( vtk_fp, "\n");
	/* point data END */
	
	
	fprintf( vtk_fp, "\t\t</Piece>\n");
	fprintf( vtk_fp, "\t</UnstructuredGrid>\n");
	
	fprintf( vtk_fp, "</VTKFile>\n");
	
	if( vtk_fp!= NULL ) {
		fclose( vtk_fp );
		vtk_fp = NULL;
	}
	
	ierr = PetscGetTime(&t1);CHKERRQ(ierr);
#ifdef PROFILE_TIMING
	PetscPrintf(PETSC_COMM_WORLD,"VTKWriter(%s): Time %1.4e sec\n",__FUNCT__,t1-t0);
#endif
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SwarmView_MPntStd_VTKappended_binary"
PetscErrorCode SwarmView_MPntStd_VTKappended_binary(DataBucket db,const char name[])
{
	PetscMPIInt rank;
	FILE *vtk_fp;
	PetscInt k;
	int npoints;
	PetscLogDouble t0,t1;
	DataField PField;
	int byte_offset,length;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	ierr = PetscGetTime(&t0);CHKERRQ(ierr);
	
	if ((vtk_fp = fopen ( name, "w")) == NULL)  {
		SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
	}

	DataBucketGetDataFieldByName(db, MPntStd_classname ,&PField);
	
	fprintf( vtk_fp, "<?xml version=\"1.0\"?>\n");
	
#ifdef WORDSIZE_BIGENDIAN
	fprintf( vtk_fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
	fprintf( vtk_fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
	
	fprintf( vtk_fp, "\t<UnstructuredGrid>\n" );
	
	DataBucketGetSizes(db,&npoints,PETSC_NULL,PETSC_NULL);
	fprintf( vtk_fp, "\t\t<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",npoints,npoints );
	
	
	fprintf( vtk_fp, "\n");
	fprintf( vtk_fp, "\t\t\t<Cells>\n");
	
	byte_offset = 0;
	
	// connectivity //
	fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"%d\" />\n",byte_offset);
  byte_offset = byte_offset + sizeof(int) + npoints * sizeof(int);
	
	// offsets //
	fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"%d\" />\n",byte_offset);
  byte_offset = byte_offset + sizeof(int) + npoints * sizeof(int);
	
	// types //
	fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"%d\" />\n",byte_offset);
  byte_offset = byte_offset + sizeof(int) + npoints * sizeof(unsigned char);
	
	fprintf( vtk_fp, "\t\t\t</Cells>\n");
	
	fprintf( vtk_fp, "\n");
	fprintf( vtk_fp, "\t\t\t<CellData>\n");
	fprintf( vtk_fp, "\t\t\t</CellData>\n");
	fprintf( vtk_fp, "\n");
	
	fprintf( vtk_fp, "\t\t\t<Points>\n");
	
	/* coordinates */
	fprintf( vtk_fp, "\t\t\t\t<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\" />\n",byte_offset);
  byte_offset = byte_offset + sizeof(int) + npoints * 3 * sizeof(double);
	
	fprintf( vtk_fp, "\t\t\t</Points>\n");
	fprintf( vtk_fp, "\n");
	
	/* point data BEGIN */
	fprintf( vtk_fp, "\t\t\t<PointData>\n");
	/* auto generated shit for the header goes here */
	{
		MPntStd *marker = PField->data; /* should write a function to do this */

		MPntStdVTKWriteBinaryAppendedHeaderAllFields(vtk_fp,&byte_offset,npoints,marker);
	}
	fprintf( vtk_fp, "\t\t\t</PointData>\n");
	fprintf( vtk_fp, "\n");
	/* point data END */

	fprintf( vtk_fp, "\t\t</Piece>\n");
	fprintf( vtk_fp, "\t</UnstructuredGrid>\n");

	/* WRITE APPENDED DATA HERE */
	fprintf( vtk_fp,"\t<AppendedData encoding=\"raw\">\n");
	fprintf( vtk_fp,"_");

	/* connectivity, offsets, types, coords */
	////////////////////////////////////////////////////////
	/* write connectivity */
	length = sizeof(int)*npoints;
	fwrite( &length,sizeof(int),1,vtk_fp);
	for (k=0; k<npoints; k++) {
		int idx = k;
		fwrite( &idx, sizeof(int),1, vtk_fp );
	}
	////////////////////////////////////////////////////////
	/* write offset */
	length = sizeof(int)*npoints;
	fwrite( &length,sizeof(int),1,vtk_fp);
	for (k=0; k<npoints; k++) {
		int idx = k+1;
		fwrite( &idx, sizeof(int),1, vtk_fp );
	}
	////////////////////////////////////////////////////////
	/* write types */
	length = sizeof(unsigned char)*npoints;
	fwrite( &length,sizeof(int),1,vtk_fp);
	for (k=0; k<npoints; k++) {
		unsigned char idx = 1; /* VTK_VERTEX */
		fwrite( &idx, sizeof(unsigned char),1, vtk_fp );
	}
	////////////////////////////////////////////////////////
	/* write coordinates */
	DataFieldGetAccess(PField);
	DataFieldVerifyAccess( PField,sizeof(MPntStd));

	length = sizeof(double)*npoints*3;
	fwrite( &length,sizeof(int),1,vtk_fp);
	for (k=0; k<npoints; k++) {
		MPntStd *marker;
		double  *coor;
		double  coords_k[] = {0.0, 0.0, 0.0};
		
		DataFieldAccessPoint(PField,k,(void**)&marker);
		MPntStdGetField_global_coord(marker,&coor);
		coords_k[0] = coor[0];
		coords_k[1] = coor[1];
		coords_k[2] = coor[2];
		
		fwrite( coords_k, sizeof(double), 3, vtk_fp );
	}
	DataFieldRestoreAccess(PField);
	
	/* auto generated shit for the marker data goes here */
	{
		MPntStd *marker = PField->data;
		MPntStdVTKWriteBinaryAppendedDataAllFields(vtk_fp,npoints,marker);
	}
		
	fprintf( vtk_fp,"\n\t</AppendedData>\n");
	
	fprintf( vtk_fp, "</VTKFile>\n");
	
	if( vtk_fp!= NULL ) {
		fclose( vtk_fp );
		vtk_fp = NULL;
	}
	
	ierr = PetscGetTime(&t1);CHKERRQ(ierr);
#ifdef PROFILE_TIMING
	PetscPrintf(PETSC_COMM_WORLD,"VTKWriter(%s): Time %1.4e sec\n",__FUNCT__,t1-t0);
#endif
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "__SwarmView_MPntStd_PVTU"
PetscErrorCode __SwarmView_MPntStd_PVTU(const char prefix[],const char name[])
{
	PetscMPIInt nproc,rank;
	FILE *vtk_fp;
	PetscInt i;
	char *sourcename;
	
	PetscFunctionBegin;
	
	if ((vtk_fp = fopen ( name, "w")) == NULL)  {
		SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
	}
	
	/* (VTK) generate pvts header */
	fprintf( vtk_fp, "<?xml version=\"1.0\"?>\n");
	
#ifdef WORDSIZE_BIGENDIAN
	fprintf( vtk_fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
	fprintf( vtk_fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
	
	/* define size of the nodal mesh based on the cell DM */
	fprintf( vtk_fp, "  <PUnstructuredGrid GhostLevel=\"0\">\n" ); /* note overlap = 0 */
	
	/* DUMP THE CELL REFERENCES */
	fprintf( vtk_fp, "    <PCellData>\n");
	fprintf( vtk_fp, "    </PCellData>\n");
	
	///////////////
	fprintf( vtk_fp, "    <PPoints>\n");
	fprintf( vtk_fp, "      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\"/>\n");
	fprintf( vtk_fp, "    </PPoints>\n");
	///////////////
	
	///////////////
  fprintf(vtk_fp, "    <PPointData>\n");
	MPntStdPVTUWriteAllPPointDataFields(vtk_fp);
  fprintf(vtk_fp, "    </PPointData>\n");
	///////////////
	
	/* write out the parallel information */
	MPI_Comm_size(PETSC_COMM_WORLD,&nproc);
	for (i=0; i<nproc; i++) {
		asprintf( &sourcename, "%s-subdomain%1.5d.vtu", prefix, i );
		fprintf( vtk_fp, "    <Piece Source=\"%s\"/>\n",sourcename);
		free(sourcename);
	}
	
	/* close the file */
	fprintf( vtk_fp, "  </PUnstructuredGrid>\n");
	fprintf( vtk_fp, "</VTKFile>\n");
	
	if(vtk_fp!=NULL){
		fclose( vtk_fp );
		vtk_fp = NULL;
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SwarmOutputParaView_MPntStd"
PetscErrorCode SwarmOutputParaView_MPntStd(DataBucket db,const char path[],const char prefix[])
{ 
	char *vtkfilename,*filename;
	PetscMPIInt rank;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	
	ierr = pTatinGenerateParallelVTKName(prefix,"vtu",&vtkfilename);CHKERRQ(ierr);
	if (path) {
		asprintf(&filename,"%s/%s",path,vtkfilename);
	} else {
		asprintf(&filename,"./%s",vtkfilename);
	}

//#ifdef __VTK_ASCII__
//	ierr = SwarmView_MPntStd_VTKascii( db,filename );CHKERRQ(ierr);
//#endif
//#ifndef __VTK_ASCII__
	ierr = SwarmView_MPntStd_VTKappended_binary(db,filename);CHKERRQ(ierr);
//#endif
	free(filename);
	free(vtkfilename);
	
	ierr = pTatinGenerateVTKName(prefix,"pvtu",&vtkfilename);CHKERRQ(ierr);
	if (path) {
		asprintf(&filename,"%s/%s",path,vtkfilename);
	} else {
		asprintf(&filename,"./%s",vtkfilename);
	}
	
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	if (rank==0) {
		ierr = __SwarmView_MPntStd_PVTU( prefix, filename );CHKERRQ(ierr);
	}
	free(filename);
	free(vtkfilename);
	
	PetscFunctionReturn(0);
}

