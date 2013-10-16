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
 **    Filename:      submarinelavaflow_ops.c
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
 **    $Id:$
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@*/

/*
 
 Submarine Lava Flow project. Joint with Eunseo Choi.
 To be presented in AGU 2013, poster title
 "Numerical investigation of the morphological transition of submarine lava flow due to slope change"
 
*/


#define _GNU_SOURCE
#include "petsc.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin_utils.h"
#include "ptatin_models.h"
#include "ptatin3d_stokes.h"
#include "ptatin3d_energy.h"
#include "energy_output.h"

#include "dmda_bcs.h"
#include "swarm_fields.h"
#include "dmda_iterator.h"
#include "geometry_object.h"
#include "geometry_object_evaluator.h"
#include "rheology.h"
#include "material_constants.h"

#include "submarinelavaflow_ctx.h"

#define geom_eps 1.0e-8

#undef __FUNCT__
#define __FUNCT__ "ModelInitialize_SubmarineLavaFlow"
PetscErrorCode ModelInitialize_SubmarineLavaFlow(pTatinCtx c,void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PetscBool flg;
	PetscReal x0[3],Lx[3];
	GeometryObject domain,lava,crust,G;
	DataBucket materialconstants;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;

	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);


	/* set flag for thermal solver to be switched on */
	ierr = PetscOptionsInsertString("-activate_energy");CHKERRQ(ierr);
	
	
	//ierr = PetscOptionsGetReal(PETSC_NULL,"-model_submarinelavaflow_param1",&data->param1,&flg);CHKERRQ(ierr);

	/* define geometry here so everyone can use it */
	ierr = GeometryObjectCreate("domain",&domain);CHKERRQ(ierr);
	x0[0] = -geom_eps;      x0[1] = -geom_eps;      x0[2] = -geom_eps;
	Lx[0] = 1.0 + geom_eps; Lx[1] = 1.0 + geom_eps; Lx[2] = 1.0 + geom_eps; 
	ierr = GeometryObjectSetType_BoxCornerReference(domain,x0,Lx);CHKERRQ(ierr);
	
	ierr = GeometryObjectCreate("lava_region",&lava);CHKERRQ(ierr);
	x0[0] = -geom_eps; x0[1] = -geom_eps; x0[2] = 0.5;
	ierr = GeometryObjectSetType_Cylinder(lava,x0,0.25,1.0+geom_eps,ROTATE_AXIS_Z);CHKERRQ(ierr);
	
	ierr = GeometryObjectCreate("outercrust",&G);CHKERRQ(ierr);
	x0[0] = -geom_eps; x0[1] = -geom_eps; x0[2] = 0.5;
	ierr = GeometryObjectSetType_Cylinder(G,x0,0.35,1.0+geom_eps,ROTATE_AXIS_Z);CHKERRQ(ierr);
	
	ierr = GeometryObjectCreate("crust_region",&crust);CHKERRQ(ierr);
	ierr = GeometryObjectSetType_SetOperation(crust,GeomType_SetComplement,x0,G,lava);CHKERRQ(ierr);
	ierr = GeometryObjectDestroy(&G);CHKERRQ(ierr);
	
	data->go[0] = domain;
	data->go[1] = lava;
	data->go[2] = crust;
	data->ngo = 3; 
	
	
	ierr = GeometryObjectEvalCreate("domain",&data->go_thermal_ic[0]);CHKERRQ(ierr);
	ierr = GeometryObjectEvalCreate("lava_region",&data->go_thermal_ic[1]);CHKERRQ(ierr);
	ierr = GeometryObjectEvalCreate("crust_region",&data->go_thermal_ic[2]);CHKERRQ(ierr);
	
	ierr = GeometryObjectEvalSetGeometryObject(data->go_thermal_ic[0],domain);CHKERRQ(ierr);
	ierr = GeometryObjectEvalSetGeometryObject(data->go_thermal_ic[1],lava);CHKERRQ(ierr);
	ierr = GeometryObjectEvalSetGeometryObject(data->go_thermal_ic[2],crust);CHKERRQ(ierr);
	
	ierr = GeometryObjectEvalSetRegionValue(data->go_thermal_ic[0],0.0);CHKERRQ(ierr);
	ierr = GeometryObjectEvalSetRegionValue(data->go_thermal_ic[1],1100.0);CHKERRQ(ierr);
	ierr = GeometryObjectEvalSetRegionValue(data->go_thermal_ic[2],1200.0);CHKERRQ(ierr);
	
	/* material properties */
	ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
	MaterialConstantsSetDefaults(materialconstants);
	/* water */
	MaterialConstantsSetValues_MaterialType(materialconstants,0,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);		
	MaterialConstantsSetValues_ViscosityConst(materialconstants,0,1.0);

	/* lava types */
	MaterialConstantsSetValues_MaterialType(materialconstants,1,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);		
	MaterialConstantsSetValues_ViscosityConst(materialconstants,1,1.0);
	MaterialConstantsSetValues_MaterialType(materialconstants,2,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);		
	MaterialConstantsSetValues_ViscosityConst(materialconstants,2,1.0);
	
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ApplyBasalVelocityJBC"
PetscBool ApplyBasalVelocityJBC(PetscScalar pos[],PetscScalar *val,void *ctx)
{
	PetscReal *vy_values;
	PetscReal x,y,z;
	
	PetscFunctionBegin;
	vy_values = (PetscReal*)ctx;
	
	x   = pos[0];
	y   = pos[1];
	z   = pos[2];
	
	if (x <= 0.2) {
		*val = vy_values[0];
	} else {
		*val = vy_values[1];
	}
	return PETSC_TRUE;
}

#undef __FUNCT__
#define __FUNCT__ "SubmarineLavaFlow_VelocityBC"
PetscErrorCode SubmarineLavaFlow_VelocityBC(BCList bclist,DM dav,pTatinCtx c,SubmarineLavaFlowCtx *data)
{
	PetscReal val_V;
	PetscReal basal_val_V[2];
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	basal_val_V[0] = 0.3;
	basal_val_V[1] = 0.0;
	ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,ApplyBasalVelocityJBC,(void*)basal_val_V);CHKERRQ(ierr);

	val_V = 0.0;
	ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&val_V);CHKERRQ(ierr);
	ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&val_V);CHKERRQ(ierr);

	ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&val_V);CHKERRQ(ierr);
	ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&val_V);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ApplyBasalThermalBC"
PetscBool ApplyBasalThermalBC(PetscScalar pos[],PetscScalar *val,void *ctx)
{
	PetscReal *base_values;
	PetscReal x,y,z;
	

	base_values = (PetscReal*)ctx;
	
	x   = pos[0];
	y   = pos[1];
	z   = pos[2];
	
	if (x <= 0.2) {
		*val = base_values[0];
	} else {
		*val = base_values[1];
	}
	return PETSC_TRUE;
}

#undef __FUNCT__
#define __FUNCT__ "SubmarineLavaFlow_EnergyBC"
PetscErrorCode SubmarineLavaFlow_EnergyBC(BCList bclist,DM daT,pTatinCtx c,SubmarineLavaFlowCtx *data)
{
	PetscReal val_T;
	PetscReal basal_val_T[2];
	PetscErrorCode ierr;
	
	
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	basal_val_T[0] = 1100.0;
	basal_val_T[1] = 0.0;
	//ierr = DMDABCListTraverse3d(bclist,daT,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&val_T);CHKERRQ(ierr);
	ierr = DMDABCListTraverse3d(bclist,daT,DMDABCList_JMIN_LOC,0,ApplyBasalThermalBC,(void*)basal_val_T);CHKERRQ(ierr);
	
	val_T = 0.0;
	ierr = DMDABCListTraverse3d(bclist,daT,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&val_T);CHKERRQ(ierr);		
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyBoundaryConditionMG_SubmarineLavaFlow"
PetscErrorCode ModelApplyBoundaryConditionMG_SubmarineLavaFlow(PetscInt nl,BCList bclist[],DM dav[],pTatinCtx c,void *ctx)
{
	PetscInt n;
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	for (n=0; n<nl; n++) {
		/* Define boundary conditions for each level in the MG hierarchy */
		ierr = SubmarineLavaFlow_VelocityBC(bclist[n],dav[n],c,data);CHKERRQ(ierr);
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyBoundaryCondition_SubmarineLavaFlow"
PetscErrorCode ModelApplyBoundaryCondition_SubmarineLavaFlow(pTatinCtx c,void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PhysCompStokes            stokes;
	DM                        stokes_pack,dav,dap;
	PhysCompEnergy            energy;
	PetscBool                 energy_active;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	/* velocity */
	ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
	stokes_pack = stokes->stokes_pack;
	ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
	
	ierr = SubmarineLavaFlow_VelocityBC(stokes->u_bclist,dav,c,data);CHKERRQ(ierr);
	
	/* energy */
	ierr = pTatinContextValid_Energy(c,&energy_active);CHKERRQ(ierr);
	if (energy_active) {
		ierr   = pTatinGetContext_Energy(c,&energy);CHKERRQ(ierr);
		ierr = SubmarineLavaFlow_EnergyBC(energy->T_bclist,energy->daT,c,data);CHKERRQ(ierr);
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyMaterialBoundaryCondition_SubmarineLavaFlow"
PetscErrorCode ModelApplyMaterialBoundaryCondition_SubmarineLavaFlow(pTatinCtx c,void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyInitialMeshGeometry_SubmarineLavaFlow"
PetscErrorCode ModelApplyInitialMeshGeometry_SubmarineLavaFlow(pTatinCtx c,void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PhysCompStokes       stokes;
	DM                   stokes_pack,dav,dap;
	PetscErrorCode       ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);

	ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
	stokes_pack = stokes->stokes_pack;
	ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
	ierr = DMDASetUniformCoordinates(dav,0.0,1.0,0.0,1.0,0.0,0.1);CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyInitialMaterialGeometry_SubmarineLavaFlow"
PetscErrorCode ModelApplyInitialMaterialGeometry_SubmarineLavaFlow(pTatinCtx c,void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PetscInt       k;
	int            p,n_mp_points;
	DataBucket     material_point_db;
	MPAccess       mpX;
	int            inside;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	
	ierr = pTatinGetMaterialPoints(c,&material_point_db,PETSC_NULL);CHKERRQ(ierr);
	DataBucketGetSizes(material_point_db,&n_mp_points,0,0);
	ierr = MaterialPointGetAccess(material_point_db,&mpX);CHKERRQ(ierr);
	for (p=0; p<n_mp_points; p++) {
		double  *position;
		int     phase_index;
		
		/* Access using the getter function provided for you (recommeneded for beginner user) */
		ierr = MaterialPointGet_global_coord(mpX,p,&position);CHKERRQ(ierr);

		ierr = GeometryObjectPointInside(data->go[0],position,&inside);CHKERRQ(ierr);
		if (inside) {
			ierr = MaterialPointSet_phase_index(mpX,p,0);CHKERRQ(ierr);
		} else {
			SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Point appears to be outside domain");
		}
			
		for (k=1; k<data->ngo; k++) {
			
			ierr = GeometryObjectPointInside(data->go[k],position,&inside);CHKERRQ(ierr);
			if (inside == 1) {
				ierr = MaterialPointSet_phase_index(mpX,p,k);CHKERRQ(ierr);
				break;
			}
		}

		ierr = MaterialPointSet_viscosity(mpX,p,1.0e-5);CHKERRQ(ierr);
		ierr = MaterialPointSet_density(mpX,p,-0.0*9.8);CHKERRQ(ierr);
		
		ierr = MaterialPointGet_phase_index(mpX,p,&phase_index);CHKERRQ(ierr);
		if (phase_index != 0) {
			ierr = MaterialPointSet_viscosity(mpX,p,1.0);CHKERRQ(ierr);
			ierr = MaterialPointSet_density(mpX,p,-1600.0*9.8);CHKERRQ(ierr);
		}

		
	}
	ierr = MaterialPointRestoreAccess(material_point_db,&mpX);CHKERRQ(ierr);
	
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EvaluateSubmarineLavaFlow_EnergyIC"
PetscBool EvaluateSubmarineLavaFlow_EnergyIC(PetscScalar pos[],PetscScalar *icvalue,void *data)
{
	PetscInt k;
	GeometryObjectEval *go_thermal_ic;
	PetscScalar  value;
	PetscBool assigned;
	PetscErrorCode ierr;
	
	go_thermal_ic = (GeometryObjectEval*)data;
	for (k=0; k<3; k++) {
		ierr = GeometryObjectEvaluateRegionValue(go_thermal_ic[k],pos,&value,&assigned);CHKERRQ(ierr);
		if (assigned) {
			*icvalue = value;
		}
	}
	
	return PETSC_TRUE;
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyInitialSolution_SubmarineLavaFlow"
PetscErrorCode ModelApplyInitialSolution_SubmarineLavaFlow(pTatinCtx c,Vec X,void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PetscBool energy_active;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	ierr = pTatinContextValid_Energy(c,&energy_active);CHKERRQ(ierr);
	if (energy_active) {
		PhysCompEnergy energy;
		Vec            temperature;
		DM             daT;
		PetscReal      coeffs[8];
		
		ierr = pTatinGetContext_Energy(c,&energy);CHKERRQ(ierr);
		ierr = pTatinPhysCompGetData_Energy(c,&temperature,PETSC_NULL);CHKERRQ(ierr);
		daT  = energy->daT;

		ierr = DMDAVecTraverse3d(daT,temperature,0,EvaluateSubmarineLavaFlow_EnergyIC,(void*)data->go_thermal_ic);CHKERRQ(ierr);
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelApplyUpdateMeshGeometry_SubmarineLavaFlow"
PetscErrorCode ModelApplyUpdateMeshGeometry_SubmarineLavaFlow(pTatinCtx c,Vec X,void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelOutput_SubmarineLavaFlow"
PetscErrorCode ModelOutput_SubmarineLavaFlow(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PetscBool energy_active,output_markers;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);

	ierr = pTatin3d_ModelOutputLite_Velocity_Stokes(c,X,prefix);CHKERRQ(ierr);

	ierr = pTatinContextValid_Energy(c,&energy_active);CHKERRQ(ierr);
	if (energy_active) {
		PhysCompEnergy energy;
		Vec            temperature;
		
		ierr = pTatinGetContext_Energy(c,&energy);CHKERRQ(ierr);
		ierr = pTatinPhysCompGetData_Energy(c,&temperature,PETSC_NULL);CHKERRQ(ierr);
		
		ierr = pTatin3d_ModelOutput_Temperature_Energy(c,temperature,prefix);CHKERRQ(ierr);
	}

	output_markers = PETSC_TRUE;
/*
	if (output_markers) {
		DataBucket  materialpoint_db;
		const int   nf = 1;
		const MaterialPointField mp_prop_list[] = { MPField_Stokes };
		//  Write out just std, stokes and plastic variables
		//const int nf = 4;
		//const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl, MPField_Energy };
		char mp_file_prefix[256];
		
		ierr = pTatinGetMaterialPoints(c,&materialpoint_db,PETSC_NULL);CHKERRQ(ierr);
		sprintf(mp_file_prefix,"%s_mpoints",prefix);
		ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
	}
*/
	if (output_markers) {
		ierr = pTatin3d_ModelOutput_MPntStd(c,prefix);CHKERRQ(ierr);
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelDestroy_SubmarineLavaFlow"
PetscErrorCode ModelDestroy_SubmarineLavaFlow(pTatinCtx c,void *ctx)
{
	SubmarineLavaFlowCtx *data = (SubmarineLavaFlowCtx*)ctx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", __FUNCT__);
	
	/* Free contents of structure */
	
	/* Free structure */
	ierr = PetscFree(data);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pTatinModelRegister_SubmarineLavaFlow"
PetscErrorCode pTatinModelRegister_SubmarineLavaFlow(void)
{
	SubmarineLavaFlowCtx *data;
	pTatinModel m,model;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	
	/* Allocate memory for the data structure for this model */
	ierr = PetscMalloc(sizeof(SubmarineLavaFlowCtx),&data);CHKERRQ(ierr);
	ierr = PetscMemzero(data,sizeof(SubmarineLavaFlowCtx));CHKERRQ(ierr);
	
	/* set initial values for model parameters */
	PetscMemzero(data->go,sizeof(GeometryObject)*10);
	
	/* register user model */
	ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

	/* Set name, model select via -ptatin_model NAME */
	ierr = pTatinModelSetName(m,"submarinelavaflow");CHKERRQ(ierr);

	/* Set model data */
	ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);
	
	/* Set function pointers */
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_SubmarineLavaFlow);CHKERRQ(ierr);
	ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_SubmarineLavaFlow);CHKERRQ(ierr);
	
	/* Insert model into list */
	ierr = pTatinModelRegister(m);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}