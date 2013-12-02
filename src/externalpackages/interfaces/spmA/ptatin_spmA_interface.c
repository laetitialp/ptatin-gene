
#include "ptatin3d.h"

#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "dmda_redundant.h"
#include "dmda_remesh.h"

#ifdef PTATIN_HAVE_SPMA
#include "spmA.h"
#endif

/* 
SEQ
 Copy data
 
MPI
 special: no decomp in J
 
 
 general:
 Find ranks containing jmax
 Gather sizes on output_rank
 Send data
 Reshuffle data into natural ordering
*/
#undef __FUNCT__
#define __FUNCT__ "ptatin3d_DMDAAllGatherCoorJMax"
PetscErrorCode ptatin3d_DMDAAllGatherCoorJMax(DM dm,PetscMPIInt output_rank,long int *_nx,long int *_nz,double *_x[],double *_y[],double *_z[])
{
	DM         da_surf;
	double     *x,*y,*z;
	PetscInt        nx,ny,nz,si,sj,sk,si_p,ei_p,sj_p,ej_p,sk_p,ek_p;
	PetscMPIInt     rank,nproc;
	PetscErrorCode ierr;
		
	PetscFunctionBegin;

	
	ierr = MPI_Comm_size(((PetscObject)dm)->comm,&nproc);CHKERRQ(ierr);
	ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
	
	ierr = DMDAGetInfo( dm,0,&nx,&ny,&nz,0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
	ierr = DMDAGetCorners( dm, &si,&sj,&sk, 0,0,0 );CHKERRQ(ierr);

	/* general gather list of nodes to collect */
	si_p = 0;
	ei_p = nx+1;
	
	sk_p = 0;
	ek_p = nz+1;
	
	sj_p = ny;
	ej_p = ny+1;
	
	/* on all non-"output_rank" ranks, just fetch a local piece of the domain so no uneccessary communication occurs */
	if (rank != output_rank) {
		si_p = si;
		ei_p = si+1;
		
		sk_p = sk;
		ek_p = sk+1;
		
		sj_p = sj;
		ej_p = sj+1;
	}
	ierr = DMDACreate3dRedundant( dm, si_p,ei_p, sj_p,ej_p, sk_p,ek_p, 1, &da_surf );CHKERRQ(ierr);

	x = y = z = PETSC_NULL;
	if (rank == output_rank) {
		DM         cda_surf;
		Vec        coor_surf;
		DMDACoor3d ***LA_coor_surf;
		PetscInt   i,j,k;
		
		/* copy out coordinates */
		PetscMalloc(sizeof(double)*nx*nz,&x);
		PetscMalloc(sizeof(double)*nx*nz,&y);
		PetscMalloc(sizeof(double)*nx*nz,&z);
		
		ierr = DMDAGetCoordinates(da_surf,&coor_surf);CHKERRQ(ierr);
		ierr = DMDAGetCoordinateDA(da_surf,&cda_surf);CHKERRQ(ierr);
		ierr = DMDAVecGetArray(cda_surf,coor_surf,&LA_coor_surf);CHKERRQ(ierr);
		j = 0;
		for (k=0; k<nz; k++) {
			for (i=0; i<nx; i++) {
				PetscInt idx;
				
				idx = i + k*nx;
				x[idx] = LA_coor_surf[k][j][i].x;

				y[idx] = LA_coor_surf[k][j][i].y;
				
				z[idx] = LA_coor_surf[k][j][i].z;
			}
		}
		ierr = DMDAVecRestoreArray(cda_surf,coor_surf,&LA_coor_surf);CHKERRQ(ierr);
	}
	
	ierr = DMDestroy(&da_surf);CHKERRQ(ierr);

	*_nx = -1;
	*_nz = -1;
	if (rank == output_rank) {
		*_nx = (long int)nx;
		*_nx = (long int)nz;
	}
	*_x  = x;
	*_y  = y;
	*_z  = z;
	
	
	PetscFunctionReturn(0);
}

/*
 Copy the chunk each rank requires into temporary array
 Send chunk
 Recieve and insert
*/
#undef __FUNCT__
#define __FUNCT__ "ptatin3d_DMDAAllScatterCoorJMax"
PetscErrorCode ptatin3d_DMDAAllScatterCoorJMax(PetscMPIInt intput_rank,double ymax[],DM dm)
{
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscFunctionReturn(0);
}


/* 
 Interpolate mechanical model surface onto lem grid.
*/
#undef __FUNCT__
#define __FUNCT__ "ptatin3d_SEQLEMHelper_InterpolateM2L"
PetscErrorCode ptatin3d_SEQLEMHelper_InterpolateM2L(DM m_surf,DM l_surf)
{
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ptatin3d_SEQLEMHelper_InterpolateL2M"
PetscErrorCode ptatin3d_SEQLEMHelper_InterpolateL2M(DM l_surf,DM m_surf)
{
	PetscErrorCode ierr;
	
	
	PetscFunctionReturn(0);
}


/* 
 * ----------------------------------------------------------------------------------------------- *
 SPMA specfic functionality is embedded inside this #if statment
*/
#ifdef PTATIN_HAVE_SPMA

int spmA_InitialiseTopo_pTatin3d(SPMAData *spm,int nx,int nz,double x[],double z[],double h[])
{
	int i,j,ii;
	double xpos,ypos;
	
	for (i=0; i<spm->nx; i++) {
		for (j=0; j<spm->ny; j++) {
			int nid;
			double h_interp,sep,smax;
			
			nid = spmA_NID(i,j,spm->nx);
			xpos = spm->x[nid];
			ypos = spm->y[nid];
			
			/* interpolate height */
			smax = 1.0e32;
			for (ii=0; ii<nx*nz; ii++) {
				
				sep =  (x[ii]-xpos)*(x[ii]-xpos);
				sep += (z[ii]-ypos)*(z[ii]-ypos);
				if (sep < smax) {
					h_interp = h[ii];
				}
			}
			
			spm->h_old[nid] = h_interp;
		}
	}
	
	return 1;
}


#undef __FUNCT__
#define __FUNCT__ "_ptatin3d_ApplyLandscapeEvolutionModel_SPMA"
PetscErrorCode _ptatin3d_ApplyLandscapeEvolutionModel_SPMA(pTatinCtx pctx,Vec X)
{
	PhysCompStokes  stokes;
	DM              stokes_pack,dau,dap;
	PetscReal       dt_mechanical;
	PetscMPIInt     rank,spm_rank,nproc;
	SPMAData *spm;
	PetscReal min[3],max[3];
	double Lx,Lz;
	int      ie;
	double dt,dt_final;
	long int nx,ny,nz;
	double *x,*y,*z;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	ierr = pTatinGetStokesContext(pctx,&stokes);CHKERRQ(ierr);
	stokes_pack = stokes->stokes_pack;
	ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
	
	ierr = MPI_Comm_size(((PetscObject)dau)->comm,&nproc);CHKERRQ(ierr);
	ierr = MPI_Comm_rank(((PetscObject)dau)->comm,&rank);CHKERRQ(ierr);
	spm_rank = nproc - 1;
	
	ierr= DMDAGetBoundingBox(dau,min,max);CHKERRQ(ierr);
	Lx = (double)( max[0] - min[0] );
	Lz = (double)( max[2] - min[2] );

	
	/* fetch the mechincal model surface on the desired core */
	ierr = ptatin3d_DMDAAllGatherCoorJMax(dau,spm_rank,&nx,&nz,&x,&y,&z);CHKERRQ(ierr);
	
	/* set time step for spm based on mechanical timestep */
	ierr = pTatinGetTimestep(pctx,&dt_mechanical);CHKERRQ(ierr);
	
	dt_final = dt_mechanical;
	dt       = dt_mechanical / 40.0;
	
	if (rank == spm_rank) {
		/* --- ---------------------- --- */
		/* --- call spm functionality --- */
		ie = spmA_New(&spm);
		ie = spmA_Initialise(spm,(int)2*nx+1,(int)2*nz+1,(double)Lx,(double)Lz,1.0,0.0,0.0,dt,dt_final);

		ie = spmA_InitialiseTopo_pTatin3d(spm,nx,nz,x,z,y);
		//ie = spmA_InitialiseUplift_pTatin3d(spm);
		spm->output_frequency = 100;

		ie = spmA_OutputIC(spm,"pt3d2spma.dat");
		/*
		ie = spmA_Apply(spm);
		ie = spmA_Output(spm,"test.dat");
		ie = spmA_Destroy(&spm);
		*/
		/* --- ---------------------- --- */
	}


#if 0
	/* push update suface value into sequential dmda coordinate vector */
	ierr = DMDAVecGetArray(cda_surf,coor_surf,&LA_coor_surf);CHKERRQ(ierr);
	j = 0;
	for (k=0; k<nz; k++) {
		for (i=0; i<nx; i++) {
			PetscInt idx;
			
			idx = i + k*nx;
			LA_coor_surf[k][j][i].y = z[idx];
		}
	}
	ierr = DMDAVecRestoreArray(cda_surf,coor_surf,&LA_coor_surf);CHKERRQ(ierr);
#endif	

	/* scatter new surface back to parallel mesh */
	
	if (x) { PetscFree(x); }
	if (y) { PetscFree(y); }
	if (z) { PetscFree(z); }
	
	
	PetscFunctionReturn(0);
}

#endif
/* ----------------------------------------------------------------------------------------------- */


#undef __FUNCT__
#define __FUNCT__ "ptatin3d_ApplyLandscapeEvolutionModel_SPMA"
PetscErrorCode ptatin3d_ApplyLandscapeEvolutionModel_SPMA(pTatinCtx pctx,Vec X)
{
	PetscErrorCode ierr;
	
	PetscFunctionBegin;

#ifdef PTATIN_HAVE_SPMA
	ierr = _ptatin3d_ApplyLandscapeEvolutionModel_SPMA(pctx,X);CHKERRQ(ierr);
#else
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"pTatind3D must be compiled with external package <SPMA: A simple FD landscape evolution model>");
#endif
	
	PetscFunctionReturn(0);
}


