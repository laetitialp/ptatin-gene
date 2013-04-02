
#define _GNU_SOURCE
#include "petsc.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin_models.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include "dmda_update_coords.h"
#include "dmda_element_q2p1.h"
#include "mesh_update.h"
#include "dmda_remesh.h"
#include "output_material_points.h"
#include "mesh_quality_metrics.h"

#include "model_basin_comp_ctx.h"
#include "model_utils.h"



/*
#undef __FUNCT__
#define __FUNCT__ "save_mesh"
PetscErrorCode save_mesh(DM dav,const char name[])
{
	Vec             coord, slice;
	DM              cda;
	PetscViewer     viewer;
	PetscInt        M, N, P, i,j;
	PetscScalar     *slice_a;
	DMDACoor3d      ***LA_coord;
	PetscMPIInt     size;
	PetscErrorCode  ierr;
	
	PetscFunctionBegin;
	
	ierr = MPI_Comm_size(((PetscObject)dav)->comm,&size);CHKERRQ(ierr);
	if (size != 1) {
		PetscPrintf(PETSC_COMM_WORLD,"WARNING: save_mesh() is only valid if nproc == 1\n");
		PetscFunctionReturn(0);
	}

	ierr = DMDAGetInfo(dav,0,&M,&N,&P,0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF,2*M*N,&slice);CHKERRQ(ierr);
	ierr = VecGetArray(slice, &slice_a);CHKERRQ(ierr);
	ierr = DMDAGetCoordinateDA(dav,&cda);CHKERRQ(ierr);
	ierr = DMDAGetCoordinates(dav,&coord);CHKERRQ(ierr);
	ierr = DMDAVecGetArray(cda,coord,&LA_coord);CHKERRQ(ierr);
	
	for(j=0; j < N; ++j){
		for(i = 0; i<M; ++i){
			slice_a[2*(j*M + i)] = LA_coord[0][j][i].x;
			slice_a[2*(j*M + i) +1] = LA_coord[0][j][i].y;
		}
	}
	ierr = VecRestoreArray(slice, &slice_a);CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(cda,coord,&LA_coord);CHKERRQ(ierr);
	
	ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
	ierr = VecView(slice,viewer);CHKERRQ(ierr);
	
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);	
	ierr = VecDestroy(&slice);CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
*/


#undef __FUNCT__
#define __FUNCT__ "ModelInitialize_BasinComp"
PetscErrorCode ModelInitialize_BasinComp(pTatinCtx c,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscInt n_int,n;
	PetscBool flg;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;

	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);

	/* assign defaults */
	data->max_layers = 100;
	
	PetscOptionsGetInt(PETSC_NULL,"-model_basin_comp_n_interfaces",&data->n_interfaces,&flg);
	if (!flg) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide the number of interfaces including the top and bottom boundaries (-model_basin_comp_n_interfaces)");
	}
    
	PetscOptionsGetReal(PETSC_NULL,"-model_basin_comp_Lx",&data->Lx,&flg);
	if (!flg) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide the length along the x direction (-model_basin_comp_Lx)");
	}
	
	PetscOptionsGetReal(PETSC_NULL,"-model_basin_comp_Ly",&data->Ly,&flg);
	if (!flg) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide the length along the y direction (-model_basin_comp_Ly)");
	}
    
	n_int = data->n_interfaces;
	PetscOptionsGetRealArray(PETSC_NULL,"-model_basin_comp_interface_heights_f",data->interface_heights_f,&n_int,&flg);
	if (!flg) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide front interface heights relative from the base of the model including the top and bottom boundaries. Interface heights taken from the top of the slope (-model_basin_comp_interface_heights_f)");
	}
	if (n_int != data->n_interfaces) {
        	    //printf("------>%d %f   %f    %f\n",n_int, data->interface_heights[0], data->interface_heights[1], data->interface_heights[2]);
		SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide %d front interface heights relative from the base of the model including the top and bottom boundaries (-model_basin_comp_interface_heights_f)",data->n_interfaces);

    }
    
	n_int = data->n_interfaces;
	PetscOptionsGetRealArray(PETSC_NULL,"-model_basin_comp_interface_heights_b",data->interface_heights_b,&n_int,&flg);
	if (!flg) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide back interface heights relative from the base of the model including the top and bottom boundaries. Interface heights taken from the top of the slope (-model_basin_comp_interface_heights_b)");
	}
	if (n_int != data->n_interfaces) {
        //printf("------>%d %f   %f    %f\n",n_int, data->interface_heights[0], data->interface_heights[1], data->interface_heights[2]);
		SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide %d back interface heights relative from the base of the model including the top and bottom boundaries (-model_basin_comp_interface_heights_b)",data->n_interfaces);
        
    }    
	data->Lz = data->interface_heights_f[data->n_interfaces-1];

    PetscOptionsGetIntArray(PETSC_NULL,"-model_basin_comp_layer_res_k",data->layer_res_k,&n_int,&flg);
	if (!flg) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide layer resolution list (-model_basin_comp_layer_res_k)");
	}
	if (n_int != data->n_interfaces-1) {
		SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide %d layer resolutions (-model_basin_comp_layer_res_k)",data->n_interfaces-1);
	}
    
	n_int = data->max_layers;
	PetscOptionsGetRealArray(PETSC_NULL,"-model_basin_comp_layer_eta",data->eta,&n_int,&flg);
	if (!flg) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide layer viscosity list (-model_basin_comp_layer_eta)");
	}
	if (n_int != data->n_interfaces-1) {
		SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide %d layer viscosity (-model_basin_comp_layer_eta)",data->n_interfaces-1);
	}
	
	n_int = data->max_layers;
	PetscOptionsGetRealArray(PETSC_NULL,"-model_basin_comp_layer_rho",data->rho,&n_int,&flg);
	if (!flg) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide layer density list (-model_basin_comp_layer_rho)");
	}
	if (n_int != data->n_interfaces-1) {
		SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"User must provide %d layer density (-model_basin_comp_layer_rho)",data->n_interfaces-1);
	}
	
    /* define the mesh size the z-direction for the global problem */
	c->mz = 0;
	for (n=0; n<data->n_interfaces-1; n++) {
		c->mz += data->layer_res_k[n];
	}
    
	data->bc_type = 0; /* 0 use vx compression ; 1 use exx compression */
	data->exx             = -1.0e-3;
	data->vx_commpression = 1.0;
	
	/* parse from command line or input file */
	ierr = PetscOptionsGetInt(PETSC_NULL,"-model_basin_comp_bc_type",&data->bc_type,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(PETSC_NULL,"-model_basin_comp_exx",&data->exx,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(PETSC_NULL,"-model_basin_comp_vx",&data->vx_commpression,&flg);CHKERRQ(ierr);
	
	PetscPrintf(PETSC_COMM_WORLD,"ModelReport: \"Basin Compression\"\n");
	PetscPrintf(PETSC_COMM_WORLD," Domain: [0 , %1.4e] x [0 , %1.4e] x [0 , %1.4e]\n", data->Lx,data->Ly,data->Lz );
	PetscPrintf(PETSC_COMM_WORLD," Mesh:   %.4D x %.4D x %.4D \n", c->mx,c->my,c->mz ); 
    
    n=data->n_interfaces-1;
    	/*
    PetscPrintf(PETSC_COMM_WORLD," ---------------------------- z = %1.4e ----------------------------\n",data->interface_heights[n]);
    PetscPrintf(PETSC_COMM_WORLD,"|\n"); 
    PetscPrintf(PETSC_COMM_WORLD,"|      eta = %1.4e , rho = %1.4e , my = %.4D \n",data->eta[n-1],data->rho[n-1],data->layer_res_k[n-1]);
    PetscPrintf(PETSC_COMM_WORLD,"|\n");
    //PetscPrintf(PETSC_COMM_WORLD,"-\n -\n  -\n   -\n    -\n");

    for (n=data->n_interfaces-1; n>=1; n--) {
		PetscPrintf(PETSC_COMM_WORLD," ---------------------------- z = %1.4e ----------------------------\n",data->interface_heights[n]);
		PetscPrintf(PETSC_COMM_WORLD,"|\n"); 
		PetscPrintf(PETSC_COMM_WORLD,"|      eta = %1.4e , rho = %1.4e , my = %.4D \n",data->eta[n-1],data->rho[n-1],data->layer_res_k[n-1]);
		PetscPrintf(PETSC_COMM_WORLD,"|\n");
	}
	//PetscPrintf(PETSC_COMM_WORLD,"|\n");
	PetscPrintf(PETSC_COMM_WORLD," ---------------------------- z = %1.4e ----------------------------\n",data->interface_heights[0],data->layer_res_k[0]);
	*/
	
	PetscFunctionReturn(0);
}




#undef __FUNCT__
#define __FUNCT__ "BoundaryCondition_BasinComp"
PetscErrorCode BoundaryCondition_BasinComp(DM dav,BCList bclist,pTatinCtx c,ModelBasinCompCtx *data)
{
	PetscReal         exx, zero = 0.0, vx_E=0.0, vx_W = 0.0;
	PetscErrorCode    ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	
	exx = data->exx;
    vx_E=-data->vx_commpression;
    vx_W = data->vx_commpression;
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    
	if (data->bc_type == 0) {
		/* compression east/west in the x-direction (0) [east-west] using constant velocity */

        ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&vx_W);CHKERRQ(ierr);
		ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&vx_E);CHKERRQ(ierr);
	} else if (data->bc_type == 1) {
		/* compression east/west in the x-direction (0) [east-west] using constant strain rate */
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Not Implemented yet!");
        
		//ierr = DirichletBC_ApplyDirectStrainRate(bclist,dav,exx,0);CHKERRQ(ierr);
	} else {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unknonwn boundary condition type");
	}
    

    
	/* free slip south (base) */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr); 

	
	/* free surface north */
	/* do nothing! */
	;
    
    /* free slip lateral */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "ModelApplyBoundaryCondition_BasinComp"
PetscErrorCode ModelApplyBoundaryCondition_BasinComp(pTatinCtx c,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscReal         exx;
	BCList            bclist;
	DM                dav;
	PetscErrorCode    ierr;

	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);

	
	exx = data->exx;

	bclist = c->stokes_ctx->u_bclist;
	dav    = c->stokes_ctx->dav;
	ierr = BoundaryCondition_BasinComp(dav,bclist,c,data);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyBoundaryConditionMG_BasinComp"
PetscErrorCode ModelApplyBoundaryConditionMG_BasinComp(PetscInt nl,BCList bclist[],DM dav[],pTatinCtx user,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscInt n;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	for (n=0; n<nl; n++) {
		/* Define boundary conditions for each level in the MG hierarchy */
		ierr = BoundaryCondition_BasinComp(dav[n],bclist[n],user,data);CHKERRQ(ierr);
	}	
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyMaterialBoundaryCondition_BasinComp"
PetscErrorCode ModelApplyMaterialBoundaryCondition_BasinComp(pTatinCtx c,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "BasinCompSetMeshGeometry"
PetscErrorCode BasinCompSetMeshGeometry(DM dav, void *ctx)
{
	PetscErrorCode ierr;
    ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscInt i,j,k,si,sj,sk,nx,ny,nz,M,N,P, kinter_max, kinter_min, interf;
    PetscScalar *dzs, a_b, a_t;
    PetscReal *interface_heights_f, *interface_heights_b, Ly;
    PetscInt *layer_res_k, n_interfaces;

	DM cda;
	Vec coord;
	DMDACoor3d ***LA_coord;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
    
    interface_heights_f = data->interface_heights_f;//front heights
    interface_heights_b = data->interface_heights_b;//back heights
    layer_res_k = data->layer_res_k;
    n_interfaces = data->n_interfaces;
    Ly = data->Ly;
    
	ierr = DMDAGetInfo(dav,0,&M,&N,&P,0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
	ierr = DMDAGetCorners(dav,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);
	ierr = DMDAGetCoordinateDA(dav,&cda);CHKERRQ(ierr);
	ierr = DMDAGetCoordinates(dav,&coord);CHKERRQ(ierr);
	ierr = DMDAVecGetArray(cda,coord,&LA_coord);CHKERRQ(ierr);
    

    ierr = PetscMalloc(ny*sizeof(PetscScalar), &dzs);CHKERRQ(ierr);
    
    kinter_max = 0;
	for(interf = 0; interf < n_interfaces-1; interf++){ 
        kinter_min = kinter_max;
        kinter_max += 2*layer_res_k[interf];
        for(i=si; i<si+nx; i++){
            a_b = (interf == 0)?0.0:(interface_heights_b[interf] - interface_heights_f[interf])/Ly;
            a_t = (interface_heights_b[interf+1] - interface_heights_f[interf+1])/Ly;
            for(j = sj; j<ny+sj; j++){
                
                dzs[j-sj] = ((a_t*LA_coord[sk][j][i].y + interface_heights_f[interf+1]) - (a_b*LA_coord[sk][j][i].y + interface_heights_f[interf]))/(PetscReal)(2.0*layer_res_k[interf]);
            }
            for(j=sj; j<sj+ny; j++){
                PetscScalar h;
                h = (a_b*LA_coord[sk][j][i].y + interface_heights_f[interf]);
                for(k=sk;k<sk+nz;k++){
                    if((k <= kinter_max)&&(k >= kinter_min)){
                        LA_coord[k][j][i].z = h + (PetscReal)dzs[j-sj]*(k-kinter_min); 
                        
                    }   
                }
            }
        }
        
    }
    
	ierr = DMDAVecRestoreArray(cda,coord,&LA_coord);CHKERRQ(ierr);
	ierr = DMDAUpdateGhostedCoordinates(dav);CHKERRQ(ierr);
	ierr = PetscFree(dzs);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BasinCompSetPerturbedInterfaces"
PetscErrorCode BasinCompSetPerturbedInterfaces(DM dav, void *ctx)
{
	PetscErrorCode ierr;
    ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscInt i,j,si,sj,sk,nx,ny,nz,M,N,P, interf, kinter, rank;
	PetscScalar random, dz_f, dz_b;
    PetscReal *interface_heights_f, *interface_heights_b;
    PetscInt *layer_res_k, n_interfaces;
    PetscReal amp;
	DM cda;
	Vec coord;
	DMDACoor3d ***LA_coord;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
    
    interface_heights_f = data->interface_heights_f;
    interface_heights_b = data->interface_heights_b;
    layer_res_k = data->layer_res_k;
    n_interfaces = data->n_interfaces;
    amp = data->amp;
    
	ierr = DMDAGetInfo(dav,0,&M,&N,&P,0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
	ierr = DMDAGetCorners(dav,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);
	ierr = DMDAGetCoordinateDA(dav,&cda);CHKERRQ(ierr);
	ierr = DMDAGetCoordinates(dav,&coord);CHKERRQ(ierr);
	ierr = DMDAVecGetArray(cda,coord,&LA_coord);CHKERRQ(ierr);
	
	
	/*Perturbes the interface for cylindrical folding*/
    /*Perturbes the interface for cylindrical folding*/
    kinter = 0;
    MPI_Comm_rank(((PetscObject)dav)->comm,&rank);
	for(interf = 1; interf < n_interfaces-1; interf++){
		kinter += 2*layer_res_k[interf-1];
		PetscPrintf(PETSC_COMM_WORLD,"jinter = %d (max=%d)\n", kinter,N-1 );
        srand(rank*interf+2);//The seed changes with the interface and the process process.

		if ( (kinter>=sk) && (kinter<sk+nz) ) {
			
			dz_f = 0.5*((interface_heights_f[interf+1] - interface_heights_f[interf])/(PetscScalar)(layer_res_k[interf]) + (interface_heights_f[interf] - interface_heights_f[interf-1])/(PetscScalar)(layer_res_k[interf-1]) );
			dz_b = 0.5*((interface_heights_b[interf+1] - interface_heights_b[interf])/(PetscScalar)(layer_res_k[interf]) + (interface_heights_b[interf] - interface_heights_b[interf-1])/(PetscScalar)(layer_res_k[interf-1]) );            
            
            for(i = si; i<si+nx; i++) {
                
				if((sj+ny == N) && (sj == 0)){
                    j=sj+ny-1;
                    random = 2.0 * rand()/(RAND_MAX+1.0) - 1.0; 
					LA_coord[kinter][j][i].z += amp * dz_b * random;
                    j=0;
                    random = 2.0 * rand()/(RAND_MAX+1.0) - 1.0; 
                    LA_coord[kinter][j][i].z += amp * dz_f * random;                    
				}else if ((sj+ny == N) || (sj == 0)){
                    PetscReal dz = 0.0;
                    j = (sj == 0)?0:(sj+ny-1);
                    dz = (sj == 0)?dz_f:dz_b;
                    random = 2.0 * rand()/(RAND_MAX+1.0) - 1.0; 
					LA_coord[kinter][j][i].z += amp * dz * random;

				}
			}
			
		}
	}
    
	ierr = DMDAVecRestoreArray(cda,coord,&LA_coord);CHKERRQ(ierr);
	ierr = DMDAUpdateGhostedCoordinates(dav);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}




#undef __FUNCT__
#define __FUNCT__ "InitialMaterialGeometryMaterialPoints_BasinComp"
PetscErrorCode InitialMaterialGeometryMaterialPoints_BasinComp(pTatinCtx c,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	int                    p,n_mp_points;
	DataBucket             db;
	DataField              PField_std,PField_stokes;
	PetscErrorCode ierr;
			
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
			
			
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
		//double      *position;
		PetscReal      eta,rho;
		PetscInt    phase;
		PetscInt    layer, kmaxlayer, kminlayer;
		PetscInt    I, J, K;
		
		DataFieldAccessPoint(PField_std,p,   (void**)&material_point);
		DataFieldAccessPoint(PField_stokes,p,(void**)&mpprop_stokes);
		/* Access using the getter function provided for you (recommeneded for beginner user) */
		//MPntStdGetField_global_coord(material_point,&position)

    MPntGetField_global_element_IJKindex(c->stokes_ctx->dav,material_point, &I, &J, &K);
		phase = -1;
		eta =  0.0;
		rho = 0.0;
		kmaxlayer = kminlayer = 0;
		layer = 0;
		// gets the global element index (i,j,k)
		//....
		
		//Set the properties
		while( (phase == -1) && (layer < data->n_interfaces-1) ){
			kmaxlayer += data->layer_res_k[layer];
			
			if( (K<kmaxlayer) && (K>=kminlayer) ){
				phase = layer + 1;
				eta = data->eta[layer];
				rho = data->rho[layer];

				rho = -rho * GRAVITY;
			}
			kminlayer += data->layer_res_k[layer];
			layer++;
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

#undef __FUNCT__
#define __FUNCT__ "InitialMaterialGeometryQuadraturePoints_BasinComp"
PetscErrorCode InitialMaterialGeometryQuadraturePoints_BasinComp(pTatinCtx c,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	int                    p,n_mp_points;
	DataBucket             db;
	DataField              PField_std,PField_stokes;
	PhysCompStokes         user;
	QPntVolCoefStokes      *all_gausspoints,*cell_gausspoints;
	PetscInt               nqp,qp;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	
	/* define properties on material points */
	db = c->materialpoint_db;
	DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
	DataFieldGetAccess(PField_std);
	DataFieldVerifyAccess(PField_std,sizeof(MPntStd));
	
	DataBucketGetDataFieldByName(db,MPntPStokes_classname,&PField_stokes);
	DataFieldGetAccess(PField_stokes);
	DataFieldVerifyAccess(PField_stokes,sizeof(MPntPStokes));
	
	
	DataBucketGetSizes(db,&n_mp_points,0,0);
	

	/* get the quadrature points */
	user = c->stokes_ctx;
	ierr = VolumeQuadratureGetAllCellData_Stokes(user->volQ,&all_gausspoints);CHKERRQ(ierr);
	nqp = user->volQ->npoints;
	
	for (p=0; p<n_mp_points; p++) {
		MPntStd     *material_point;
		MPntPStokes *mpprop_stokes;
		//double      *position;
		PetscReal      eta,rho;
		PetscInt    phase;
		PetscInt    layer, kmaxlayer, kminlayer, localeid_p;
		PetscInt    I, J, K;
		
		DataFieldAccessPoint(PField_std,p,   (void**)&material_point);
		DataFieldAccessPoint(PField_stokes,p,(void**)&mpprop_stokes);
		
    MPntGetField_global_element_IJKindex(c->stokes_ctx->dav,material_point, &I, &J, &K);

		//Set the properties
		phase = -1;
		eta =  0.0;
		rho = 0.0;
		kmaxlayer = kminlayer = 0;
		layer = 0;
		while( (phase == -1) && (layer < data->n_interfaces-1) ){
			kmaxlayer += data->layer_res_k[layer];
			
			if( (K<kmaxlayer) && (K>=kminlayer) ){
				phase = layer + 1;
				eta = data->eta[layer];
				rho = data->rho[layer];
			}
			kminlayer += data->layer_res_k[layer];
			layer++;
		}

		
		MPntStdGetField_local_element_index(material_point,&localeid_p);
		ierr = VolumeQuadratureGetCellData_Stokes(user->volQ,all_gausspoints,localeid_p,&cell_gausspoints);CHKERRQ(ierr);
		
		for (qp=0; qp<nqp; qp++) {
			cell_gausspoints[qp].eta  = eta;
			cell_gausspoints[qp].rho  = rho;

			cell_gausspoints[qp].Fu[0] = 0.0;
			cell_gausspoints[qp].Fu[1] = -rho * GRAVITY;
			cell_gausspoints[qp].Fu[2] = 0.0;

			cell_gausspoints[qp].Fp = 0.0;
		}		
		
	}
	
	DataFieldRestoreAccess(PField_std);
	DataFieldRestoreAccess(PField_stokes);
	
	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ModelApplyInitialMaterialGeometry_BasinComp"
PetscErrorCode ModelApplyInitialMaterialGeometry_BasinComp(pTatinCtx c,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	int                    p,n_mp_points;
	DataBucket             db;
	DataField              PField_std,PField_stokes;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	ierr = InitialMaterialGeometryMaterialPoints_BasinComp(c,ctx);CHKERRQ(ierr);
	ierr = InitialMaterialGeometryQuadraturePoints_BasinComp(c,ctx);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

		
		
		
#undef __FUNCT__
#define __FUNCT__ "ModelApplyInitialMeshGeometry_BasinComp"
PetscErrorCode ModelApplyInitialMeshGeometry_BasinComp(pTatinCtx c,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscReal         Lx,Ly,dx,dy,dz,Lz;
	PetscInt          mx,my,mz, itf;
	PetscReal         amp,factor;
  char              mesh_outputfile[PETSC_MAX_PATH_LEN];
	PetscErrorCode    ierr;

	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);

	/* step 1 - create structured grid */
	Lx = data->Lx;
	Ly = data->Ly;
    Lz = data->Lz;
	
	mx = c->mx; 
	my = c->my; 
	mz = c->mz; 
	
	dx = Lx / ((PetscReal)mx);
	dy = Ly / ((PetscReal)my);
	dz = Lz / ((PetscReal)mz);

	ierr = DMDASetUniformCoordinates(c->stokes_ctx->dav, 0.0,Lx, 0.0,Ly, data->interface_heights_f[0],Lz);CHKERRQ(ierr);
	factor = 0.1;
	ierr = PetscOptionsGetReal(PETSC_NULL,"-model_basin_comp_amp_factor",&factor,PETSC_NULL);CHKERRQ(ierr);
	amp = factor * 1.0; /* this is internal scaled by dy inside BasinCompSetPerturbedInterfaces() */
	if ( (amp < 0.0) || (amp >1.0) ) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-model_basin_comp_amp_factor must be 0 < amp < 1");
	}
	data->amp = amp;
	/* step 2 - define two interfaces and perturb coords along the interface */
	ierr = BasinCompSetMeshGeometry(c->stokes_ctx->dav, data);CHKERRQ(ierr);
	ierr = BasinCompSetPerturbedInterfaces(c->stokes_ctx->dav, data);CHKERRQ(ierr);
    
	ierr = DMDABilinearizeQ2Elements(c->stokes_ctx->dav);CHKERRQ(ierr);
    
    ierr = sprintf(mesh_outputfile, "%s/mesh_t_0.dat",c->outputpath);
	ierr = save_mesh(c->stokes_ctx->dav,mesh_outputfile);CHKERRQ(ierr);
    
	PetscFunctionReturn(0);
}

/*

0/ Full lagrangian update
1/ Check mesh quality metrics
2/ If mesh quality metrics are not satisfied (on the first failure only)
 a) set projection type to Q1
 
3/ set advection vel = 0
4/ remesh
5/ Check mesh quality metrics
 
 
*/
#undef __FUNCT__
#define __FUNCT__ "ModelApplyUpdateMeshGeometry_BasinComp"
PetscErrorCode ModelApplyUpdateMeshGeometry_BasinComp(pTatinCtx c,Vec X,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscReal      step;
	PhysCompStokes stokes;
	DM             stokes_pack,dav,dap;
	Vec            velocity,pressure;
	PetscInt       M,N,P;
	PetscInt           metric_L = 5; 
	MeshQualityMeasure metric_list[] = { MESH_QUALITY_ASPECT_RATIO, MESH_QUALITY_DISTORTION, MESH_QUALITY_DIAGONAL_RATIO, MESH_QUALITY_VERTEX_ANGLE, MESH_QUALITY_FACE_AREA_RATIO };
	PetscReal          value[100];
	PetscBool          remesh;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);

	ierr = pTatinGetTimestep(c,&step);CHKERRQ(ierr);
	ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);

	stokes_pack = stokes->stokes_pack;
	ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
	ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
	
	ierr = UpdateMeshGeometry_FullLagrangian(dav,velocity,step);CHKERRQ(ierr);
	
	ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
	
	/* check mesh quality */
	ierr = DMDAComputeMeshQualityMetricList(dav,metric_L,metric_list,value);CHKERRQ(ierr);
	remesh = PETSC_FALSE;
	if (value[0] > 2.0) {
		remesh = PETSC_TRUE;
	}
	if ( (value[1] < 0.7) || (value[1] > 1.0)) {
		remesh = PETSC_TRUE;
	}
	PetscPrintf(PETSC_COMM_WORLD,"  Mesh metrics \"MESH_QUALITY_ASPECT_RATIO\"    %1.4e \n", value[0]);
	PetscPrintf(PETSC_COMM_WORLD,"  Mesh metrics \"MESH_QUALITY_DISTORTION\"      %1.4e \n", value[1]);
	PetscPrintf(PETSC_COMM_WORLD,"  Mesh metrics \"MESH_QUALITY_DIAGONAL_RATIO\"  %1.4e \n", value[2]);
	PetscPrintf(PETSC_COMM_WORLD,"  Mesh metrics \"MESH_QUALITY_VERTEX_ANGLE\"    %1.4e \n", value[3]);
	PetscPrintf(PETSC_COMM_WORLD,"  Mesh metrics \"MESH_QUALITY_FACE_AREA_RATIO\" %1.4e \n", value[4]);
	
	
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]] Remeshing currently deactivated \n", __FUNCT__);
	
#if 0
	/* activate marker interpolation */	
	if (remesh) {
		c->coefficient_projection_type = 1;
		
		ierr = DMDAGetInfo(dav,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
		ierr = DMDARemeshSetUniformCoordinatesBetweenJLayers3d(dav,0,N);CHKERRQ(ierr);
	}
#endif	
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelInitialCondition_BasinComp"
PetscErrorCode ModelInitialCondition_BasinComp(pTatinCtx c,Vec X,void *ctx)
{
    /*
	ModelFolding2dCtx *data = (ModelFolding2dCtx*)ctx;
	DM stokes_pack,dau,dap;
	Vec velocity,pressure;
	PetscReal rho0;
	DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
	DMDAVecTraverse3d_InterpCtx IntpCtx;*/
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	/*
	
	stokes_pack = c->stokes_ctx->stokes_pack;
	
	ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
	ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
    
	ierr = VecZeroEntries(velocity);CHKERRQ(ierr);
	ierr = VecZeroEntries(pressure);CHKERRQ(ierr);
	ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
    */
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelOutput_BasinComp"
PetscErrorCode ModelOutput_BasinComp(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	//char           name[256];
	DataBucket     materialpoint_db;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
	
	{
		const int                   nf = 2;
		const MaterialPointVariable mp_prop_list[] = { MPV_viscosity, MPV_density }; 
		
		ierr = pTatinGetMaterialPoints(c,&materialpoint_db,PETSC_NULL);CHKERRQ(ierr);
		//sprintf(name,"%s_mpoints_cell",prefix);
		//ierr = pTatinOutputParaViewMarkerFields(c->stokes_ctx->stokes_pack,materialpoint_db,nf,mp_prop_list,c->outputpath,name);CHKERRQ(ierr);
		ierr = pTatin3d_ModelOutput_MarkerCellFields(c,nf,mp_prop_list,prefix);CHKERRQ(ierr);
	}	
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelDestroy_BasinComp"
PetscErrorCode ModelDestroy_BasinComp(pTatinCtx c,void *ctx)
{
	ModelBasinCompCtx *data = (ModelBasinCompCtx*)ctx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	/* Free contents of structure */
	
	/* Free structure */
	ierr = PetscFree(data);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pTatinModelRegister_BasinComp"
PetscErrorCode pTatinModelRegister_BasinComp(void)
{
	ModelBasinCompCtx *data;
	pTatinModel m,model;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	
	/* Allocate memory for the data structure for this model */
	ierr = PetscMalloc(sizeof(ModelBasinCompCtx),&data);CHKERRQ(ierr);
	ierr = PetscMemzero(data,sizeof(ModelBasinCompCtx));CHKERRQ(ierr);
	
	/* register user model */
	ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

	/* Set name, model select via -ptatin_model NAME */
	ierr = pTatinModelSetName(m,"basin_comp");CHKERRQ(ierr);

	/* Set model data */
	ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);
	
	/* Set function pointers */
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_BasinComp);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_BasinComp);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_BasinComp);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_BasinComp);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_BasinComp);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_BasinComp);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_BasinComp);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_BasinComp);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_BasinComp);CHKERRQ(ierr);
	
	/* Insert model into list */
	ierr = pTatinModelRegister(m);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}
