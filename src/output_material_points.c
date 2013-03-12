
#include "petsc.h"

#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"

#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPStokesPl_def.h"
#include "MPntPEnergy_def.h"


#include "dmda_duplicate.h"
#include "dmda_element_q2p1.h"
#include "swarm_fields.h"
#include "output_paraview.h"
#include "element_type_Q2.h"
#include "element_utils_q2.h"
#include "element_utils_q1.h"
#include "quadrature.h"

#include "output_material_points.h"


const char *MaterialPointVariableName[] =  {
	"region",
  "viscosity", 
  "density", 
  "plastic_strain", 
  "yield_indicator", 
  "diffusivity", 
  "heat_source", 
  0 
};

const char *MaterialPointVariableParaviewDataType[] =  {
	"Int32",
  "Float64", 
  "Float64", 
  "Float32", 
  "Int16", 
  "Float64", 
  "Float64", 
  0
};

#undef __FUNCT__
#define __FUNCT__ "_write_float"
PetscErrorCode _write_float(FILE *vtk_fp,const PetscInt mx,const PetscInt my,const PetscInt mz,float LA_cell[])
{
	PetscInt i,j,k;
	
	PetscFunctionBegin;
	for (k=0; k<mz; k++) {
		for (j=0; j<my; j++) {
			for (i=0; i<mx; i++) {
				int idx = i + j*(mx) + k*(mx)*(my);
				
				fprintf( vtk_fp,"      %1.6e \n", LA_cell[idx]);
			}
		}
	}
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "_write_double"
PetscErrorCode _write_double(FILE *vtk_fp,const PetscInt mx,const PetscInt my,const PetscInt mz,double LA_cell[])
{
	PetscInt i,j,k;
	
	PetscFunctionBegin;
	for (k=0; k<mz; k++) {
		for (j=0; j<my; j++) {
			for (i=0; i<mx; i++) {
				int idx = i + j*(mx) + k*(mx)*(my);
				
				fprintf( vtk_fp,"      %1.6e \n", LA_cell[idx]);
			}
		}
	}
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "_write_int"
PetscErrorCode _write_int(FILE *vtk_fp,const PetscInt mx,const PetscInt my,const PetscInt mz,int LA_cell[])
{
	PetscInt i,j,k;
	
	PetscFunctionBegin;
	for (k=0; k<mz; k++) {
		for (j=0; j<my; j++) {
			for (i=0; i<mx; i++) {
				int idx = i + j*(mx) + k*(mx)*(my);
				
				fprintf( vtk_fp,"      %d \n", LA_cell[idx]);
			}
		}
	}
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "_write_short"
PetscErrorCode _write_short(FILE *vtk_fp,const PetscInt mx,const PetscInt my,const PetscInt mz,short LA_cell[])
{
	PetscInt i,j,k;
	
	PetscFunctionBegin;
	for (k=0; k<mz; k++) {
		for (j=0; j<my; j++) {
			for (i=0; i<mx; i++) {
				int idx = i + j*(mx) + k*(mx)*(my);
				
				fprintf( vtk_fp,"      %d \n", LA_cell[idx]);
			}
		}
	}
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "_check_for_empty_cells_double"
PetscErrorCode _check_for_empty_cells_double(const PetscInt mx,const PetscInt my,const PetscInt mz,int cell_count[],double LA_cell[])
{
	int uei,uej,uek;
	int e,ei,ej,ek,eid2,ii,jj,kk;
	int constant_conversion_occurred;
	PetscFunctionBegin;
	
	constant_conversion_occurred = 0;
	for (e=0; e<mx*my*mz; e++) {
		if (cell_count[e] == 0) {
			double local_LA_cell;
			int    local_cell_count;
			
			constant_conversion_occurred = 1;
			
			/* convert e into q2 eidx */
			ek   = e/(mx*my);
			eid2 = e - ek * (mx*my);
			ej   = eid2/mx;
			ei   = eid2 - ej * mx;

			uei = ei/2;
			uej = ej/2;
			uek = ek/2;
			
			/* traverse the q2 cell and try a new average */
			local_LA_cell = 0.0;
			local_cell_count = 0;
			for (kk=0; kk<2; kk++) {
				for (jj=0; jj<2; jj++) {
					for (ii=0; ii<2; ii++) {
						int cidx,ci,cj,ck;
						
						ci = 2*ei + ii;
						cj = 2*ej + jj;
						ck = 2*ek + kk;
						
						cidx = ci + cj*mx + ck*mx*my;
						local_LA_cell += LA_cell[cidx];
						local_cell_count += cell_count[cidx];
					}
				}
			}
			/* set the same values on the 8 sub cells */
			for (kk=0; kk<2; kk++) {
				for (jj=0; jj<2; jj++) {
					for (ii=0; ii<2; ii++) {
						int cidx,ci,cj,ck;
						
						ci = 2*ei + ii;
						cj = 2*ej + jj;
						ck = 2*ek + kk;

						cidx = ci + cj*mx + ck*mx*my;
						LA_cell[cidx]    = local_LA_cell;
						cell_count[cidx] = local_cell_count;
					}
				}
			}
			
			
		}
	}
	
	if (constant_conversion_occurred == 1) {
		for (e=0; e<mx*my*mz; e++) {
			if (LA_cell[e] == 0) {
				SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cell contains zero markers");
			}
		}
	}	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "_check_for_empty_cells_float"
PetscErrorCode _check_for_empty_cells_float(const PetscInt mx,const PetscInt my,const PetscInt mz,int cell_count[],float LA_cell[])
{
	int uei,uej,uek;
	int e,ei,ej,ek,eid2,ii,jj,kk;
	int constant_conversion_occurred;
	PetscFunctionBegin;
	
	constant_conversion_occurred = 0;
	for (e=0; e<mx*my*mz; e++) {
		if (cell_count[e] == 0) {
			float local_LA_cell;
			int   local_cell_count;
			
			constant_conversion_occurred = 1;
			
			/* convert e into q2 eidx */
			ek   = e/(mx*my);
			eid2 = e - ek * (mx*my);
			ej   = eid2/mx;
			ei   = eid2 - ej * mx;
			
			uei = ei/2;
			uej = ej/2;
			uek = ek/2;
			
			/* traverse the q2 cell and try a new average */
			local_LA_cell = 0.0;
			local_cell_count = 0;
			for (kk=0; kk<2; kk++) {
				for (jj=0; jj<2; jj++) {
					for (ii=0; ii<2; ii++) {
						int cidx,ci,cj,ck;
						
						ci = 2*ei + ii;
						cj = 2*ej + jj;
						ck = 2*ek + kk;
						
						cidx = ci + cj*mx + ck*mx*my;
						local_LA_cell += LA_cell[cidx];
						local_cell_count += cell_count[cidx];
					}
				}
			}
			/* set the same values on the 8 sub cells */
			for (kk=0; kk<2; kk++) {
				for (jj=0; jj<2; jj++) {
					for (ii=0; ii<2; ii++) {
						int cidx,ci,cj,ck;
						
						ci = 2*ei + ii;
						cj = 2*ej + jj;
						ck = 2*ek + kk;
						
						cidx = ci + cj*mx + ck*mx*my;
						LA_cell[cidx]    = local_LA_cell;
						cell_count[cidx] = local_cell_count;
					}
				}
			}
			
			
		}
	}
	
	if (constant_conversion_occurred == 1) {
		for (e=0; e<mx*my*mz; e++) {
			if (LA_cell[e] == 0) {
				SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cell contains zero markers");
			}
		}
	}	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "_compute_cell_value_double"
PetscErrorCode _compute_cell_value_double(DataBucket db,MaterialPointVariable variable,const PetscInt mx,const PetscInt my,const PetscInt mz,double LA_cell[])
{
	int *cell_count;
	int e,ueid,ueid2,umx,umy,umz,uei,uej,uek;
	double *xi_p;
	int ei,ej,ek,eidx;
	double var;
	int p,n_mp;
	MPAccess X;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	
	umx = mx/2;
	umy = my/2;
	umz = mz/2;
	
	ierr = PetscMalloc(sizeof(int)*mx*my*mz,&cell_count);CHKERRQ(ierr);
	ierr = PetscMemzero(cell_count,sizeof(int)*mx*my*mz);CHKERRQ(ierr);
	
	DataBucketGetSizes(db,&n_mp,PETSC_NULL,PETSC_NULL);
	
	ierr = MaterialPointGetAccess(db,&X);CHKERRQ(ierr);
	
	for (p=0; p<n_mp; p++) {
		ierr = MaterialPointGet_local_element_index(X,p,&ueid);CHKERRQ(ierr);
		ierr = MaterialPointGet_local_coord(X,p,&xi_p);CHKERRQ(ierr);
		
		uek   = ueid/(umx*umy);
		ueid2 = ueid - uek * (umx*umy);
		uej   = ueid2/umx;
		uei   = ueid2 - uej * umx;
		
		ei = 2*uei;
		ej = 2*uej;
		ek = 2*uek;
		
		if (xi_p[0] > 0.0) { ei++; }
		if (xi_p[1] > 0.0) { ej++; }
		if (xi_p[2] > 0.0) { ek++; }
		
		eidx = ei + ej*mx + ek*mx*my;
		if (eidx >= mx*my*mz) {
			SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"eidx is too large");
		}
		
		switch (variable) {
			case MPV_viscosity:
				ierr = MaterialPointGet_viscosity(X,p,&var);CHKERRQ(ierr);
				break;
			case MPV_density:
				ierr = MaterialPointGet_density(X,p,&var);CHKERRQ(ierr);
				break;
			case MPV_diffusivity:
				ierr = MaterialPointGet_diffusivity(X,p,&var);CHKERRQ(ierr);
				break;
			case MPV_heat_source:
				ierr = MaterialPointGet_heat_source(X,p,&var);CHKERRQ(ierr);
				break;
		}
		
		LA_cell[eidx] += var;
		cell_count[eidx]++;
	}
	
	ierr = _check_for_empty_cells_double(mx,my,mz,cell_count,LA_cell);CHKERRQ(ierr);
	
	for (e=0; e<mx*my*mz; e++) {
		LA_cell[e] = LA_cell[e] / ( (double)(cell_count[e]) );
	}
	
	ierr = MaterialPointRestoreAccess(db,&X);CHKERRQ(ierr);
	ierr = PetscFree(cell_count);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "_compute_cell_value_float"
PetscErrorCode _compute_cell_value_float(DataBucket db,MaterialPointVariable variable,const PetscInt mx,const PetscInt my,const PetscInt mz,float LA_cell[])
{
	int *cell_count;
	int e,ueid,ueid2,umx,umy,umz,uei,uej,uek;
	double *xi_p;
	int ei,ej,ek,eidx;
	float var;
	int p,n_mp;
	MPAccess X;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	
	umx = mx/2;
	umy = my/2;
	umz = mz/2;
	
	ierr = PetscMalloc(sizeof(int)*mx*my*mz,&cell_count);CHKERRQ(ierr);
	ierr = PetscMemzero(cell_count,sizeof(int)*mx*my*mz);CHKERRQ(ierr);
	
	DataBucketGetSizes(db,&n_mp,PETSC_NULL,PETSC_NULL);
	
	ierr = MaterialPointGetAccess(db,&X);CHKERRQ(ierr);
	
	for (p=0; p<n_mp; p++) {
		ierr = MaterialPointGet_local_element_index(X,p,&ueid);CHKERRQ(ierr);
		ierr = MaterialPointGet_local_coord(X,p,&xi_p);CHKERRQ(ierr);
		
		uek   = ueid/(umx*umy);
		ueid2 = ueid - uek * (umx*umy);
		uej   = ueid2/umx;
		uei   = ueid2 - uej * umx;
		
		ei = 2*uei;
		ej = 2*uej;
		ek = 2*uek;
		
		if (xi_p[0] > 0.0) { ei++; }
		if (xi_p[1] > 0.0) { ej++; }
		if (xi_p[2] > 0.0) { ek++; }
		
		eidx = ei + ej*mx + ek*mx*my;
		if (eidx >= mx*my*mz) {
			SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"eidx is too large");
		}
		
		switch (variable) {
			case MPV_plastic_strain:
				ierr = MaterialPointGet_plastic_strain(X,p,&var);CHKERRQ(ierr);
				break;
		}
		
		LA_cell[eidx] += var;
		cell_count[eidx]++;
	}
	
	ierr = _check_for_empty_cells_float(mx,my,mz,cell_count,LA_cell);CHKERRQ(ierr);
	
	for (e=0; e<mx*my*mz; e++) {
		LA_cell[e] = LA_cell[e] / ( (float)(cell_count[e]) );
	}
	
	ierr = MaterialPointRestoreAccess(db,&X);CHKERRQ(ierr);
	ierr = PetscFree(cell_count);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pTatinOutputParaViewMarkerFields_VTS"
PetscErrorCode pTatinOutputParaViewMarkerFields_VTS(DM dau,DataBucket material_points,const int nvars,const MaterialPointVariable vars[],const char name[])
{
	PetscErrorCode ierr;
	DM cda;
	Vec gcoords;
	DMDACoor3d ***LA_gcoords;	
	PetscInt mx,my,mz;
	PetscInt i,j,k,esi,esj,esk;
	FILE*	vtk_fp = NULL;
	PetscInt gsi,gsj,gsk,gm,gn,gp;
	int t;
	int    *i_LA_cell;
	short  *s_LA_cell;
	float  *f_LA_cell;
	double *d_LA_cell;
	
	PetscFunctionBegin;
	if ((vtk_fp = fopen ( name, "w")) == NULL)  {
		SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
	}
	
	
	ierr = DMDAGetGhostCorners(dau,&gsi,&gsj,&gsk,&gm,&gn,&gp);CHKERRQ(ierr);
	ierr = DMDAGetCornersElementQ2(dau,&esi,&esj,&esk,&mx,&my,&mz);CHKERRQ(ierr);
	
	ierr = DMDAGetCoordinateDA(dau,&cda);CHKERRQ(ierr);
	ierr = DMDAGetGhostedCoordinates(dau,&gcoords);CHKERRQ(ierr);
	ierr = DMDAVecGetArray(cda,gcoords,&LA_gcoords);CHKERRQ(ierr);
	
	
	/* VTS HEADER - OPEN */	
	fprintf( vtk_fp, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
	fprintf( vtk_fp, "  <StructuredGrid WholeExtent=\"%d %d %d %d %d %d\">\n", esi,esi+2*mx+1-1, esj,esj+2*my+1-1, esk,esk+2*mz+1-1);
	fprintf( vtk_fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n", esi,esi+2*mx+1-1, esj,esj+2*my+1-1, esk,esk+2*mz+1-1);
	
	/* VTS COORD DATA */	
	fprintf( vtk_fp, "    <Points>\n");
	fprintf( vtk_fp, "      <DataArray Name=\"coords\" type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
	for (k=esk; k<esk+2*mz+1; k++) {
		for (j=esj; j<esj+2*my+1; j++) {
			for (i=esi; i<esi+2*mx+1; i++) {
				fprintf( vtk_fp,"      %1.6e %1.6e %1.6e\n", (float)LA_gcoords[k][j][i].x, (float)LA_gcoords[k][j][i].y, (float)LA_gcoords[k][j][i].z );
			}
		}
	}
	fprintf( vtk_fp, "      </DataArray>\n");
	fprintf( vtk_fp, "    </Points>\n");
	
	/* VTS CELL DATA */	
	fprintf( vtk_fp, "    <CellData>\n");
	
	if (nvars == -1) {
		const char *mpv_name;
		
		t = 0;
		mpv_name = MaterialPointVariableName[t];
		while (mpv_name != PETSC_NULL) {
			
			fprintf( vtk_fp, "      <DataArray Name=\"%s\" type=\"%s\" NumberOfComponents=\"1\" format=\"ascii\">\n",MaterialPointVariableName[t],MaterialPointVariableParaviewDataType[t]);
			
			switch (t) {
					
				case MPV_region:
					ierr = PetscMalloc(sizeof(int)*2*mx*2*my*2*mz,&i_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(i_LA_cell,sizeof(int)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					PetscPrintf(PETSC_COMM_WORLD,"MPV_region -> writer not yet completed\n");
					ierr = _write_int(vtk_fp,2*mx,2*my,2*mz,i_LA_cell);CHKERRQ(ierr);
					ierr = PetscFree(i_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_viscosity:
					ierr = PetscMalloc(sizeof(double)*2*mx*2*my*2*mz,&d_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(d_LA_cell,sizeof(double)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					ierr = _compute_cell_value_double(material_points,MPV_viscosity,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					ierr = _write_double(vtk_fp,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);

					ierr = PetscFree(d_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_density:
					ierr = PetscMalloc(sizeof(double)*2*mx*2*my*2*mz,&d_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(d_LA_cell,sizeof(double)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					ierr = _compute_cell_value_double(material_points,MPV_density,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					ierr = _write_double(vtk_fp,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);

					ierr = PetscFree(d_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_plastic_strain:
					ierr = PetscMalloc(sizeof(float)*2*mx*2*my*2*mz,&f_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(f_LA_cell,sizeof(float)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					ierr = _compute_cell_value_float(material_points,MPV_plastic_strain,2*mx,2*my,2*mz,f_LA_cell);CHKERRQ(ierr);
					ierr = _write_float(vtk_fp,2*mx,2*my,2*mz,f_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(f_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_yield_indicator:
					ierr = PetscMalloc(sizeof(short)*2*mx*2*my*2*mz,&s_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(s_LA_cell,sizeof(short)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					PetscPrintf(PETSC_COMM_WORLD,"MPV_yield_indicator -> writer not yet completed\n");
					ierr = _write_short(vtk_fp,2*mx,2*my,2*mz,s_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(s_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_diffusivity:
					ierr = PetscMalloc(sizeof(double)*2*mx*2*my*2*mz,&d_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(d_LA_cell,sizeof(double)*2*mx*2*my*2*mz);CHKERRQ(ierr);

					ierr = _compute_cell_value_double(material_points,MPV_diffusivity,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					ierr = _write_double(vtk_fp,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(d_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_heat_source:
					ierr = PetscMalloc(sizeof(double)*2*mx*2*my*2*mz,&d_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(d_LA_cell,sizeof(double)*2*mx*2*my*2*mz);CHKERRQ(ierr);

					ierr = _compute_cell_value_double(material_points,MPV_heat_source,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					ierr = _write_double(vtk_fp,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(d_LA_cell);CHKERRQ(ierr);
					break;
			}
			
			fprintf( vtk_fp, "      </DataArray>\n");

			t++;
			mpv_name = MaterialPointVariableName[t];
		}
	} else {
		for (t=0; t<nvars; t++) {
			MaterialPointVariable idx = vars[t];
			
			fprintf( vtk_fp, "      <DataArray Name=\"%s\" type=\"%s\" NumberOfComponents=\"1\" format=\"ascii\">\n",MaterialPointVariableName[idx],MaterialPointVariableParaviewDataType[idx]);
			
			switch (idx) {
					
				case MPV_region:
					ierr = PetscMalloc(sizeof(int)*2*mx*2*my*2*mz,&i_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(i_LA_cell,sizeof(int)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					PetscPrintf(PETSC_COMM_WORLD,"MPV_region -> writer not yet completed\n");
					ierr = _write_int(vtk_fp,2*mx,2*my,2*mz,i_LA_cell);CHKERRQ(ierr);
					ierr = PetscFree(i_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_viscosity:
					ierr = PetscMalloc(sizeof(double)*2*mx*2*my*2*mz,&d_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(d_LA_cell,sizeof(double)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					ierr = _compute_cell_value_double(material_points,MPV_viscosity,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					ierr = _write_double(vtk_fp,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(d_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_density:
					ierr = PetscMalloc(sizeof(double)*2*mx*2*my*2*mz,&d_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(d_LA_cell,sizeof(double)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					ierr = _compute_cell_value_double(material_points,MPV_density,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					ierr = _write_double(vtk_fp,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(d_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_plastic_strain:
					ierr = PetscMalloc(sizeof(float)*2*mx*2*my*2*mz,&f_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(f_LA_cell,sizeof(float)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					ierr = _compute_cell_value_float(material_points,MPV_plastic_strain,2*mx,2*my,2*mz,f_LA_cell);CHKERRQ(ierr);
					ierr = _write_float(vtk_fp,2*mx,2*my,2*mz,f_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(f_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_yield_indicator:
					ierr = PetscMalloc(sizeof(short)*2*mx*2*my*2*mz,&s_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(s_LA_cell,sizeof(short)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					PetscPrintf(PETSC_COMM_WORLD,"MPV_yield_indicator -> writer not yet completed\n");
					ierr = _write_short(vtk_fp,2*mx,2*my,2*mz,s_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(s_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_diffusivity:
					ierr = PetscMalloc(sizeof(double)*2*mx*2*my*2*mz,&d_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(d_LA_cell,sizeof(double)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					ierr = _compute_cell_value_double(material_points,MPV_diffusivity,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					ierr = _write_double(vtk_fp,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(d_LA_cell);CHKERRQ(ierr);
					break;
					
				case MPV_heat_source:
					ierr = PetscMalloc(sizeof(double)*2*mx*2*my*2*mz,&d_LA_cell);CHKERRQ(ierr);
					ierr = PetscMemzero(d_LA_cell,sizeof(double)*2*mx*2*my*2*mz);CHKERRQ(ierr);
					
					ierr = _compute_cell_value_double(material_points,MPV_heat_source,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					ierr = _write_double(vtk_fp,2*mx,2*my,2*mz,d_LA_cell);CHKERRQ(ierr);
					
					ierr = PetscFree(d_LA_cell);CHKERRQ(ierr);
					break;
			}
			
			
			fprintf( vtk_fp, "      </DataArray>\n");
		}
	}
	
	
	fprintf( vtk_fp, "    </CellData>\n");
	
	/* VTS NODAL DATA */
	fprintf( vtk_fp, "    <PointData>\n");
	fprintf( vtk_fp, "    </PointData>\n");
	
	/* VTS HEADER - CLOSE */	
	fprintf( vtk_fp, "    </Piece>\n");
	fprintf( vtk_fp, "  </StructuredGrid>\n");
	fprintf( vtk_fp, "</VTKFile>\n");
	
	ierr = DMDAVecRestoreArray(cda,gcoords,&LA_gcoords);CHKERRQ(ierr);
	
	fclose( vtk_fp );
	
	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "pTatinOutputParaViewMarkerFields_PVTS"
PetscErrorCode pTatinOutputParaViewMarkerFields_PVTS(DM dau,const int nvars,const MaterialPointVariable vars[],const char prefix[],const char name[])
{
	PetscErrorCode ierr;
	FILE*	vtk_fp = NULL;
	PetscInt M,N,P,swidth;
	PetscMPIInt rank;
	int t;
	
	PetscFunctionBegin;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	vtk_fp = NULL;
	if (rank==0) {
		if ((vtk_fp = fopen ( name, "w")) == NULL)  {
			SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file %s",name );
		}
	}
	
	
	/* VTS HEADER - OPEN */	
	if(vtk_fp) fprintf( vtk_fp, "<?xml version=\"1.0\"?>\n");
	if(vtk_fp) fprintf( vtk_fp, "<VTKFile type=\"PStructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
	
	DMDAGetInfo( dau, 0, &M,&N,&P, 0,0,0, 0,&swidth, 0,0,0, 0 );
	if(vtk_fp) fprintf( vtk_fp, "  <PStructuredGrid GhostLevel=\"%d\" WholeExtent=\"%d %d %d %d %d %d\">\n", swidth, 0,M-1, 0,N-1, 0,P-1 ); /* note overlap = 1 for Q1 */
	
	/* VTS COORD DATA */	
	if(vtk_fp) fprintf( vtk_fp, "    <PPoints>\n");
	if(vtk_fp) fprintf( vtk_fp, "      <PDataArray type=\"Float32\" Name=\"coords\" NumberOfComponents=\"3\"/>\n");
	if(vtk_fp) fprintf( vtk_fp, "    </PPoints>\n");
	
	
	/* VTS CELL DATA */	
	if(vtk_fp) fprintf( vtk_fp, "    <PCellData>\n");
	if (nvars == -1) {
		const char *mpv_name;

		t = 0;
		mpv_name = MaterialPointVariableName[t];
		while (mpv_name != PETSC_NULL) {
			if(vtk_fp) fprintf( vtk_fp, "      <PDataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"1\"/>\n",MaterialPointVariableParaviewDataType[t],MaterialPointVariableName[t]);
			t++;
			mpv_name = MaterialPointVariableName[t];
		}
	} else {
		for (t=0; t<nvars; t++) {
			MaterialPointVariable idx = vars[t];
			
			if(vtk_fp) fprintf( vtk_fp, "      <PDataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"1\"/>\n",MaterialPointVariableParaviewDataType[idx],MaterialPointVariableName[idx]);
		}
	}

	if(vtk_fp) fprintf( vtk_fp, "    </PCellData>\n");
	
	/* VTS NODAL DATA */
	if(vtk_fp) fprintf( vtk_fp, "    <PPointData>\n");
	if(vtk_fp) fprintf( vtk_fp, "    </PPointData>\n");
	
	/* write out the parallel information */
	ierr = DAQ2PieceExtendForGhostLevelZero(vtk_fp,2,dau,prefix);CHKERRQ(ierr);
	
	/* VTS HEADER - CLOSE */	
	if(vtk_fp) fprintf( vtk_fp, "  </PStructuredGrid>\n");
	if(vtk_fp) fprintf( vtk_fp, "</VTKFile>\n");
	
	if(vtk_fp) fclose( vtk_fp );
	PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "pTatinOutputParaViewMarkerFields"
PetscErrorCode pTatinOutputParaViewMarkerFields(DM pack,DataBucket material_points,const int nvars,const MaterialPointVariable vars[],const char path[],const char prefix[])
{
	char *vtkfilename,*filename;
	PetscMPIInt rank;
	DM dau,dap;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;

	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	ierr = pTatinGenerateParallelVTKName(prefix,"vts",&vtkfilename);CHKERRQ(ierr);
	if (path) {
		asprintf(&filename,"%s/%s",path,vtkfilename);
	} else {
		asprintf(&filename,"./%s",vtkfilename);
	}

	ierr = DMCompositeGetEntries(pack,&dau,&dap);CHKERRQ(ierr);
	
	ierr = pTatinOutputParaViewMarkerFields_VTS(dau,material_points,nvars,vars,filename);CHKERRQ(ierr);

	free(filename);
	free(vtkfilename);
	
	ierr = pTatinGenerateVTKName(prefix,"pvts",&vtkfilename);CHKERRQ(ierr);
	if (path) {
		asprintf(&filename,"%s/%s",path,vtkfilename);
	} else {
		asprintf(&filename,"./%s",vtkfilename);
	}

	ierr = pTatinOutputParaViewMarkerFields_PVTS(dau,nvars,vars,prefix,filename);CHKERRQ(ierr);
	
	free(filename);
	free(vtkfilename);
	
	PetscFunctionReturn(0);
}

