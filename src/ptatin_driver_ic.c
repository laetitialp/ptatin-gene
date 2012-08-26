
static const char help[] = "Stokes solver using Q2-Pm1 mixed finite elements.\n"
"3D prototype of the (p)ragmatic version of Tatin. (pTatin3d_v0.0)\n\n";


#include "ptatin3d.h"
#include "private/ptatin_impl.h"

#include "material_point_utils.h"
#include "material_point_std_utils.h"
#include "ptatin_models.h"
#include "ptatin_utils.h"
#include "stokes_form_function.h"
#include "stokes_operators.h"

#undef __FUNCT__  
#define __FUNCT__ "pTatin3d_material_points_check_ic"
PetscErrorCode pTatin3d_material_points_check_ic(int argc,char **argv)
{
	DM              multipys_pack,dav,dap;
	PetscErrorCode ierr;
	pTatinCtx user;

	PetscFunctionBegin;
	
	ierr = pTatin3dCreateContext(&user);CHKERRQ(ierr);
	ierr = pTatin3dParseOptions(user);CHKERRQ(ierr);

	/* Register all models */
	ierr = pTatinModelRegisterAll();CHKERRQ(ierr);
	/* Load model, call an initialization routines */
	ierr = pTatinModelLoad(user);CHKERRQ(ierr);
	
	ierr = pTatinModel_Initialize(user->model,user);CHKERRQ(ierr);
	
	/* Generate physics modules */
	ierr = pTatin3d_PhysCompStokesCreate(user);CHKERRQ(ierr);

	/* Pack all physics together */
	/* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
	ierr = PetscObjectReference((PetscObject)user->stokes_ctx->stokes_pack);CHKERRQ(ierr);
	user->pack = user->stokes_ctx->stokes_pack;

	/* fetch some local variables */
	multipys_pack = user->pack;
	dav           = user->stokes_ctx->dav;
	dap           = user->stokes_ctx->dap;
	
	ierr = pTatin3dCreateMaterialPoints(user,dav);CHKERRQ(ierr);
	
	/* mesh geometry */
	ierr = pTatinModel_ApplyInitialMeshGeometry(user->model,user);CHKERRQ(ierr);
	
	/* interpolate point coordinates (needed if mesh was modified) */
	//ierr = QuadratureStokesCoordinateSetUp(user->stokes_ctx->Q,dav);CHKERRQ(ierr);
	//for (e=0; e<QUAD_EDGES; e++) {
	//	ierr = SurfaceQuadratureStokesGeometrySetUp(user->stokes_ctx->surfQ[e],dav);CHKERRQ(ierr);
	//}
	/* interpolate material point coordinates (needed if mesh was modified) */
	ierr = MaterialPointCoordinateSetUp(user,dav);CHKERRQ(ierr);
	
	/* material geometry */
	ierr = pTatinModel_ApplyInitialMaterialGeometry(user->model,user);CHKERRQ(ierr);
	
	/* boundary conditions */
	ierr = pTatinModel_ApplyBoundaryCondition(user->model,user);CHKERRQ(ierr);

	/* test form function */
	{
		Vec  X,F;
		SNES snes;
		Mat  JMF;
		
		ierr = DMCreateGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
		ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
		
		ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
		ierr = SNESSetFunction(snes,F,FormFunction_Stokes,user);CHKERRQ(ierr);  
		ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
		ierr = SNESDestroy(&snes);CHKERRQ(ierr);
		ierr = VecDestroy(&X);CHKERRQ(ierr);
		ierr = VecDestroy(&F);CHKERRQ(ierr);
	}
	
	/* test data bucket viewer */
	DataBucketView(((PetscObject)multipys_pack)->comm, user->materialpoint_db,"materialpoint_stokes",DATABUCKET_VIEW_STDOUT);
	DataBucketView(PETSC_COMM_SELF, user->material_constants,"material_constants",DATABUCKET_VIEW_STDOUT);
	

	/* write out the initial condition */
	{
		Vec X;
		
		ierr = DMGetGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
		ierr = pTatinModel_Output(user->model,user,X,"icbc");CHKERRQ(ierr);

	}
	
	/* test generic viewer */
	{
		const int nf = 1;
		const MaterialPointField mp_prop_list[] = { MPField_Std }; 
		ierr = SwarmViewGeneric_ParaView(user->materialpoint_db,nf,mp_prop_list,user->outputpath,"test_MPStd");CHKERRQ(ierr);
	}
	{
		const int nf = 1;
		const MaterialPointField mp_prop_list[] = { MPField_Stokes }; 
		ierr = SwarmViewGeneric_ParaView(user->materialpoint_db,nf,mp_prop_list,user->outputpath,"test_MPStokes");CHKERRQ(ierr);
	}
	
	{
		const int nf = 2;
		const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes }; 
		ierr = SwarmViewGeneric_ParaView(user->materialpoint_db,nf,mp_prop_list,user->outputpath,"test_MPStd_MPStokes");CHKERRQ(ierr);
	}
	
	
	
	ierr = pTatin3dDestroyContext(&user);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
	PetscErrorCode ierr;
	
	ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
	
	ierr = pTatinWriteOptionsFile(PETSC_NULL);CHKERRQ(ierr);
	
	ierr = pTatin3d_material_points_check_ic(argc,argv);CHKERRQ(ierr);
	
	ierr = PetscFinalize();CHKERRQ(ierr);
	return 0;
}
