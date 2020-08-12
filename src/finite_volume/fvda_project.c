
#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_private.h>
#include <fvda_utils.h>


PetscErrorCode FVDACellPropertyProjectToFace_HarmonicMean(FVDA fv,const char cell_field[],const char face_field[])
{
  PetscErrorCode  ierr;
  DM              dm;
  Vec             field,fieldl;
  const PetscReal *_field;
  PetscReal       *face_data = NULL;
  PetscInt        c,f,bs;
  

  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = FVDACellPropertyGetInfo(fv,cell_field,NULL,NULL,&bs);
  if (bs != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only cell properties with block-size 1 are supported");
  
  /* push cell_field data into vec */
  ierr = DMCreateGlobalVector(dm,&field);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&fieldl);CHKERRQ(ierr);
  {
    const PetscReal *cell_data = NULL;
    PetscReal       *f;
    
    ierr = FVDAGetCellPropertyByNameArrayRead(fv,cell_field,&cell_data);CHKERRQ(ierr);
    ierr = VecGetArray(field,&f);CHKERRQ(ierr);
    for (c=0; c<fv->ncells; c++) {
      f[c] = cell_data[c];
    }
    ierr = VecRestoreArray(field,&f);CHKERRQ(ierr);
  }
  
  /* scatter to local space */
  ierr = DMGlobalToLocal(dm,field,INSERT_VALUES,fieldl);CHKERRQ(ierr);
  
  /* traverse faces, get f+, f- and average them */
  ierr = FVDAGetFacePropertyByNameArray(fv,face_field,&face_data);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fieldl,&_field);CHKERRQ(ierr);
  for (f=0; f<fv->nfaces; f++) {
    PetscInt c_m,c_p;
    PetscReal avg = 0;
    
    c_m = fv->face_fv_map[2*f+0];
    c_p = fv->face_fv_map[2*f+1];
    if (c_p >= 0) {
      avg = 1.0/_field[c_m] + 1.0/_field[c_p];
      avg = 2.0 / avg;
    } else {
      avg = _field[c_m];
    }
    face_data[f] = avg;
  }
  ierr = VecRestoreArrayRead(fieldl,&_field);CHKERRQ(ierr);
  ierr = VecDestroy(&field);CHKERRQ(ierr);
  ierr = VecDestroy(&fieldl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDACellPropertyProjectToFace_ArithmeticMean(FVDA fv,const char cell_field[],const char face_field[])
{
  PetscErrorCode  ierr;
  DM              dm;
  Vec             field,fieldl;
  const PetscReal *_field;
  PetscReal       *face_data = NULL;
  PetscInt        c,f,bs;
  

  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = FVDACellPropertyGetInfo(fv,cell_field,NULL,NULL,&bs);
  if (bs != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only cell properties with block-size 1 are supported");
  
  /* push cell_field data into vec */
  ierr = DMCreateGlobalVector(dm,&field);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&fieldl);CHKERRQ(ierr);
  {
    const PetscReal *cell_data = NULL;
    PetscReal       *f;
    
    ierr = FVDAGetCellPropertyByNameArrayRead(fv,cell_field,&cell_data);CHKERRQ(ierr);
    ierr = VecGetArray(field,&f);CHKERRQ(ierr);
    for (c=0; c<fv->ncells; c++) {
      f[c] = cell_data[c];
    }
    ierr = VecRestoreArray(field,&f);CHKERRQ(ierr);
  }
  
  /* scatter to local space */
  ierr = DMGlobalToLocal(dm,field,INSERT_VALUES,fieldl);CHKERRQ(ierr);
  
  /* traverse faces, get f+, f- and average them */
  ierr = FVDAGetFacePropertyByNameArray(fv,face_field,&face_data);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fieldl,&_field);CHKERRQ(ierr);
  for (f=0; f<fv->nfaces; f++) {
    PetscInt c_m,c_p;
    PetscReal avg = 0;
    
    c_m = fv->face_fv_map[2*f+0];
    c_p = fv->face_fv_map[2*f+1];
    if (c_p >= 0) {
      avg = 0.5 * (_field[c_m] + _field[c_p]);
    } else {
      avg = _field[c_m];
    }
    face_data[f] = avg;
  }
  ierr = VecRestoreArrayRead(fieldl,&_field);CHKERRQ(ierr);
  ierr = VecDestroy(&field);CHKERRQ(ierr);
  ierr = VecDestroy(&fieldl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 
 avg_type = {0,1,2}
 0: arhithmetic
 1: harmonic
 2: geometric
 
 volume_weighted = PETSC_TRUE will perform weighted averaging using the cell volume
 
*/
PetscErrorCode FVDACellPropertyProjectToFace_GeneralizedMean(FVDA fv,const char cell_field[],const char face_field[],PetscInt avg_type,PetscBool volume_weighted)
{
  PetscErrorCode  ierr;
  DM              dm;
  Vec             field,fieldl;
  const PetscReal *_field,*_vol;
  PetscReal       *face_data = NULL;
  PetscInt        c,f,bs;
  Vec             voll = NULL;

  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = FVDACellPropertyGetInfo(fv,cell_field,NULL,NULL,&bs);
  if (bs != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only cell properties with block-size 1 are supported");
  
  /* push cell_field data into vec */
  ierr = DMCreateGlobalVector(dm,&field);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&fieldl);CHKERRQ(ierr);
  {
    const PetscReal *cell_data = NULL;
    PetscReal       *f;
    
    ierr = FVDAGetCellPropertyByNameArrayRead(fv,cell_field,&cell_data);CHKERRQ(ierr);
    ierr = VecGetArray(field,&f);CHKERRQ(ierr);
    for (c=0; c<fv->ncells; c++) {
      f[c] = cell_data[c];
    }
    ierr = VecRestoreArray(field,&f);CHKERRQ(ierr);
  }
  
  /* scatter to local space */
  ierr = DMGlobalToLocal(dm,field,INSERT_VALUES,fieldl);CHKERRQ(ierr);
  
  ierr = DMCreateLocalVector(dm,&voll);CHKERRQ(ierr);

  if (volume_weighted) {
    PetscReal         *f;
    Vec               vol = NULL;
    Vec               geometry_coorl;
    const PetscScalar *_geom_coor;
    PetscInt          dm_nel,dm_nen;
    const PetscInt    *dm_element,*element;
    PetscReal         cell_coor[3*DACELL3D_VERTS],cellvol = 0;
    
    ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
    ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
    
    ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(dm,&vol);CHKERRQ(ierr);
    ierr = VecGetArray(vol,&f);CHKERRQ(ierr);
    for (c=0; c<fv->ncells; c++) {
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
      
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateCellVolume3d(cell_coor,&cellvol);

      f[c] = cellvol;
    }
    ierr = VecRestoreArray(vol,&f);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dm,vol,INSERT_VALUES,voll);CHKERRQ(ierr);
    
    ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
    ierr = VecDestroy(&vol);CHKERRQ(ierr);
  } else {
    ierr = VecSet(voll,1.0);CHKERRQ(ierr);
  }
  
  
  /* traverse faces, get f+, f- and average them */
  ierr = FVDAGetFacePropertyByNameArray(fv,face_field,&face_data);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fieldl,&_field);CHKERRQ(ierr);
  ierr = VecGetArrayRead(voll,&_vol);CHKERRQ(ierr);
  
  switch (avg_type) {
    case 0:
      for (f=0; f<fv->nfaces; f++) {
        PetscInt c_m,c_p;
        PetscReal avg = 0;
        
        c_m = fv->face_fv_map[2*f+0];
        c_p = fv->face_fv_map[2*f+1];
        if (c_p >= 0) {
          avg = (_vol[c_m] * _field[c_m] + _vol[c_p] * _field[c_p]) / (_vol[c_m] + _vol[c_p]);
        } else {
          avg = _field[c_m];
        }
        face_data[f] = avg;
      }
      break;
      
    case 1:
      for (f=0; f<fv->nfaces; f++) {
        PetscInt c_m,c_p;
        PetscReal avg = 0;
        
        c_m = fv->face_fv_map[2*f+0];
        c_p = fv->face_fv_map[2*f+1];
        if (c_p >= 0) {
          avg = _vol[c_m]/_field[c_m] + _vol[c_p]/_field[c_p];
          avg = (_vol[c_m] + _vol[c_p]) / avg;
        } else {
          avg = _field[c_m];
        }
        face_data[f] = avg;
      }
      break;
      
    case 2:
      for (f=0; f<fv->nfaces; f++) {
        PetscInt c_m,c_p;
        PetscReal avg = 0;
        
        c_m = fv->face_fv_map[2*f+0];
        c_p = fv->face_fv_map[2*f+1];
        if (c_p >= 0) {
          avg = PetscSqrtReal(_vol[c_m]*_field[c_m] * _vol[c_p]*_field[c_p]);
          avg = avg / PetscSqrtReal(_vol[c_m] * _vol[c_p]);
        } else {
          avg = _field[c_m];
        }
        face_data[f] = avg;
      }
      break;
      
    default:
      break;
  }
  
  ierr = VecRestoreArrayRead(voll,&_vol);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fieldl,&_field);CHKERRQ(ierr);
  ierr = VecDestroy(&voll);CHKERRQ(ierr);
  ierr = VecDestroy(&field);CHKERRQ(ierr);
  ierr = VecDestroy(&fieldl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAFieldSetUpProjectToVertex_Q1(FVDA fv,DM *dmf,Vec *field)
{
  PetscInt       nel,nen;
  const PetscInt *e;
  PetscErrorCode ierr;

  
  PetscFunctionBegin;
  ierr = DMDACreateCompatibleDMDA(fv->dm_geometry,1,dmf);CHKERRQ(ierr);
  ierr = DMDASetElementType(*dmf,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
  ierr = DMDAGetElements(*dmf,&nel,&nen,&e);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(*dmf,field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAFieldProjectToVertex_Q1(FVDA fv,Vec fv_field,DM dmf,Vec field)
{
  PetscErrorCode    ierr;
  Vec               geometry_coorl;
  const PetscScalar *_geom_coor,*_fv_field;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS];
  Vec               sum,fieldl,suml;
  PetscReal         *_field,*_sum,dV;
  PetscInt          c,i;


  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dmf,&sum);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmf,&fieldl);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmf,&suml);CHKERRQ(ierr);

  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(field);CHKERRQ(ierr);CHKERRQ(ierr); /* initialize input */
  ierr = VecZeroEntries(sum);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = VecZeroEntries(fieldl);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = VecZeroEntries(suml);CHKERRQ(ierr);CHKERRQ(ierr);
  
  ierr = VecGetArrayRead(fv_field,&_fv_field);CHKERRQ(ierr);
  ierr = VecGetArray(fieldl,&_field);CHKERRQ(ierr);
  ierr = VecGetArray(suml,&_sum);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    for (i=0; i<8; i++) {
      _field[ element[i] ] += _fv_field[c] * dV;
      _sum[ element[i] ]   += dV;
    }
  }
  ierr = VecRestoreArray(suml,&_sum);CHKERRQ(ierr);
  ierr = VecRestoreArray(fieldl,&_field);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fv_field,&_fv_field);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  ierr = DMLocalToGlobal(dmf,fieldl,ADD_VALUES,field);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dmf,suml,ADD_VALUES,sum);CHKERRQ(ierr);

  ierr = VecPointwiseDivide(field,field,sum);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmf,&fieldl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmf,&suml);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmf,&sum);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* 
 Compute grad(Q) at the cell centre using Gauss's theorem.
 
 From Gauss theorem we have
 
 \int_V \vec w . grad(Q) dV = -\int_V div(\vec w) Q dV + \int_S \vec w . \vec n Q dS
 
 for some test function \vec w and where \vec n = (n_x, n_y, n_z)
 
 To approximate the gradients, we will use
 \vec w^(1) = (1,0,0)
 \vec w^(2) = (0,1,0)
 \vec w^(3) = (0,0,1)
 
 e.g.
 \int_V \vec w^(1) . grad(Q) dV = -\int_V div(\vec w^(1)) Q dV + \int_S \vec w^(1) . n Q dS
 ==>
 \int_V {\partial Q}/{\partial x} dV = \int_S n_x Q dS
 
 We will approximate all integrals with a 1 point quadrature rule, leading to
 
 Q^c_{,x} vol(c) = \sum_{f=1}^{6} (n_x)_{f} Q_{f} area(face(f))

 Q^c_{,x}  = (1 / vol(c)) [\sum_{f=1}^{6} (n_x)_{f} Q_{f} area(face(f))]

 where 
   vol(c) is the volume of cell c
   f is the index of each face of he hex cell
   area(face(f)) is the area of the face, f
   Q^c_{,x} is the cell average approximation to {\partial Q}/{\partial x}
 
 Q_{f} is approximated via interpolating across cells with a common face
 If the face is on a boundary, we will have to resort to using the boundary conditions
 
*/
PetscErrorCode FVDAGradientProject(FVDA fv,Vec Q,Vec gradQ)
{
  PetscErrorCode  ierr;
  PetscReal       cell_coor[3 * DACELL3D_Q1_SIZE];
  Vec             coorl,Ql;
  const PetscReal *_geom_coor,*_Q;
  PetscInt        c,f,fb,dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  PetscReal       dS,Q_m,Q_p,Q_f,dV;
  PetscInt        c_m,c_p,cellid,cl_m,cl_p;
  PetscReal       *_gradQ,*normal;
  
  
  PetscFunctionBegin;
  ierr = VecZeroEntries(gradQ);CHKERRQ(ierr);
  ierr = VecGetArray(gradQ,&_gradQ);CHKERRQ(ierr);

  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_fv,&Ql);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_fv,Q,INSERT_VALUES,Ql);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Ql,&_Q);CHKERRQ(ierr);

  /* interior faces */
  for (f=0; f<fv->nfaces; f++) {
    if (fv->face_location[f] == DAFACE_BOUNDARY) continue;

    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    c_m = fv->face_fv_map[2*f+0];
    cl_m = fv->face_element_map[2*f+0];
    Q_m = _Q[c_m];
    
    c_p = fv->face_fv_map[2*f+1];
    cl_p = fv->face_element_map[2*f+1];
    Q_p = _Q[c_p];
   
    /* TODO - perform interpolation */
    Q_f = 0.5 * (Q_m + Q_p); /* hack - this should be an actual interpolation - will be fine on uniform grids */
    
    normal = &fv->face_normal[3*f];
    
    if (cl_m >= 0) {
      _gradQ[3 * cl_m + 0] += Q_f * normal[0] * dS; // cell[-]
      _gradQ[3 * cl_m + 1] += Q_f * normal[1] * dS; // cell[-]
      _gradQ[3 * cl_m + 2] += Q_f * normal[2] * dS; // cell[-]
    }
    if (cl_p >= 0) {
      _gradQ[3 * cl_p + 0] -= Q_f * normal[0] * dS; // cell[+]
      _gradQ[3 * cl_p + 1] -= Q_f * normal[1] * dS; // cell[+]
      _gradQ[3 * cl_p + 2] -= Q_f * normal[2] * dS; // cell[+]
    }
    
  }
  
#if 0
  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    FVFluxType bctype;
    PetscReal  bcvalue;
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    bctype = fv->boundary_flux[fb];
    bcvalue = fv->boundary_value[fb];

    c_m = fv->face_fv_map[2*f+0];
    cl_m = fv->face_element_map[2*f+0];

    normal = &fv->face_normal[3*f];

    Q_f = 0;
    
    switch (bctype) {
        
      case FVFLUX_DIRICHLET_CONSTRAINT:
        Q_f = bcvalue;
        break;
        
      case FVFLUX_NEUMANN_CONSTRAINT:
        Q_f = bcvalue / 1.0; /* broken */
        break;
        
      default:
        break;
    }
    
    if (cl_m >= 0) {
      _gradQ[3 * cl_m + 0] += Q_f * normal[0] * dS; // cell[-]
      
      _gradQ[3 * cl_m + 1] += Q_f * normal[1] * dS; // cell[-]
      
      _gradQ[3 * cl_m + 2] += Q_f * normal[2] * dS; // cell[-]
    }
  }
#endif

  for (fb=0; fb<fv->nfaces_boundary; fb++) {
    PetscInt   f = fv->face_idx_boundary[fb];
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);
    
    c_m = fv->face_fv_map[2*f+0];
    cl_m = fv->face_element_map[2*f+0];
    normal = &fv->face_normal[3*f];
    
    Q_f = _Q[c_m];
    if (cl_m >= 0) {
      _gradQ[3 * cl_m + 0] += Q_f * normal[0] * dS; // cell[-]
      _gradQ[3 * cl_m + 1] += Q_f * normal[1] * dS; // cell[-]
      _gradQ[3 * cl_m + 2] += Q_f * normal[2] * dS; // cell[-]
    }
  }

  /* normalize */
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);

    _gradQ[3 * c + 0] = _gradQ[3 * c + 0] / dV;
    _gradQ[3 * c + 1] = _gradQ[3 * c + 1] / dV;
    _gradQ[3 * c + 2] = _gradQ[3 * c + 2] / dV;
  }
  
  ierr = VecRestoreArrayRead(Ql,&_Q);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_fv,&Ql);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  ierr = VecRestoreArray(gradQ,&_gradQ);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 On each face, perform a reconstruction
*/
PetscErrorCode FVDAGradientProjectViaReconstruction(FVDA fv,FVArray Q,FVArray gradQ)
{
  PetscErrorCode  ierr;
  PetscReal       cell_coor[3 * DACELL3D_Q1_SIZE];
  Vec             coorl,Qg,Ql,gradg,grad[3],fv_coorl;
  const PetscReal *_geom_coor,*_Q,*_fv_coor;
  PetscInt        c,f,dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  PetscReal       dS,dV;
  PetscInt        c_m,c_p,cellid;
  PetscReal       *_gradQ,*normal,*_grad[3];
  PetscReal       *coeff,_coeff[3];
  PetscInt        n_neigh,neigh[27*5];
  
  PetscFunctionBegin;
  if (Q->type != FVPRIMITIVE_CELL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Q must be type FVPRIMITIVE_CELL");
  if (gradQ->type != FVPRIMITIVE_CELL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Q must be type FVPRIMITIVE_CELL");

  ierr = FVArrayZeroEntries(gradQ);CHKERRQ(ierr);
  _gradQ = gradQ->v;
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(fv->dm_fv,&fv_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fv_coorl,&_fv_coor);CHKERRQ(ierr);

  /* VecCreateMPIWithArray() will work with COMM_SELF, but the code is more logical as written */
  {
    PetscMPIInt commsize;
    ierr = MPI_Comm_size(fv->comm,&commsize);CHKERRQ(ierr);
    if (commsize == 1) {
      ierr = VecCreateSeqWithArray(fv->comm,Q->bs,Q->len,(const PetscScalar*)Q->v,&Qg);CHKERRQ(ierr);
    } else {
      ierr = VecCreateMPIWithArray(fv->comm,Q->bs,Q->len,PETSC_DECIDE,(const PetscScalar*)Q->v,&Qg);CHKERRQ(ierr);
    }
  }
  
  ierr = DMGetLocalVector(fv->dm_fv,&Ql);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_fv,Qg,INSERT_VALUES,Ql);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Ql,&_Q);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(fv->dm_fv,&gradg);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_fv,&grad[0]);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_fv,&grad[1]);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_fv,&grad[2]);CHKERRQ(ierr);

  ierr = PetscCalloc1(fv->ncells*3,&coeff);CHKERRQ(ierr);
  {
    PetscInt e,fv_start[3],fv_range[3],fv_start_local[3],fv_ghost_offset[3],fv_ghost_range[3];
    
    ierr = DMDAGetCorners(fv->dm_fv,&fv_start[0],&fv_start[1],&fv_start[2],&fv_range[0],&fv_range[1],&fv_range[2]);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(fv->dm_fv,&fv_start_local[0],&fv_start_local[1],&fv_start_local[2],&fv_ghost_range[0],&fv_ghost_range[1],&fv_ghost_range[2]);CHKERRQ(ierr);
    fv_ghost_offset[0] = fv_start[0] - fv_start_local[0];
    fv_ghost_offset[1] = fv_start[1] - fv_start_local[1];
    fv_ghost_offset[2] = fv_start[2] - fv_start_local[2];
    
    for (e=0; e<fv->ncells; e++) {
      PetscInt cijk[3];
      
      ierr = _cart_convert_index_to_ijk(e,(const PetscInt*)fv_range,cijk);CHKERRQ(ierr);
      cijk[0] += fv_ghost_offset[0];
      cijk[1] += fv_ghost_offset[1];
      cijk[2] += fv_ghost_offset[2];
      
      ierr = _cart_convert_ijk_to_index((const PetscInt*)cijk,(const PetscInt*)fv_ghost_range,&c);CHKERRQ(ierr);
      
      ierr = FVDAGetReconstructionStencil_AtCell(fv,c,&n_neigh,neigh);CHKERRQ(ierr);
      ierr = setup_coeff(fv,c,n_neigh,(const PetscInt*)neigh,_fv_coor,_Q,_coeff);CHKERRQ(ierr);
      coeff[3*e+0] = _coeff[0];
      coeff[3*e+1] = _coeff[1];
      coeff[3*e+2] = _coeff[2];
    }
  }
  
  /* interior faces - average */
  /* exterior faces - evaluate */

  ierr = VecGetArray(grad[0],&_grad[0]);CHKERRQ(ierr);
  ierr = VecGetArray(grad[1],&_grad[1]);CHKERRQ(ierr);
  ierr = VecGetArray(grad[2],&_grad[2]);CHKERRQ(ierr);
  
  for (f=0; f<fv->nfaces; f++) {
    PetscReal val,Qhr;
    
    c_m = fv->face_fv_map[2*f+0];
    c_p = fv->face_fv_map[2*f+1];

    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[f],cell_coor,&dS);

    normal = &fv->face_normal[3*f];
    
    if (fv->face_location[f] == DAFACE_BOUNDARY) {
      PetscInt cl = fv->face_element_map[2*f+0];

      ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,&_fv_coor[3*c_m],_Q,&coeff[3*cl],&Qhr);CHKERRQ(ierr);

      // cell[-]
      for (PetscInt d=0; d<3; d++) {
        val = Qhr * normal[d] * dS;
        //ierr = VecSetValue(grad[d],c_m,val,ADD_VALUES);CHKERRQ(ierr);
        _grad[d][c_m] += val;
      }
      
    } else {
      PetscInt cl_m = fv->face_element_map[2*f+0];
      PetscInt cl_p = fv->face_element_map[2*f+1];
      
      if (cl_m >= 0) {
        ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_m,&_fv_coor[3*c_m],_Q,&coeff[3*cl_m],&Qhr);CHKERRQ(ierr);
        
        // cell[-]
        for (PetscInt d=0; d<3; d++) {
          val = 0.5 * Qhr * normal[d] * dS;
          //ierr = VecSetValue(grad[d],c_m,val,ADD_VALUES);CHKERRQ(ierr);
          _grad[d][c_m] += val;
        }
        
        // cell[+]
        for (PetscInt d=0; d<3; d++) {
          val = -0.5 * Qhr * normal[d] * dS;
          //ierr = VecSetValue(grad[d],c_p,val,ADD_VALUES);CHKERRQ(ierr);
          _grad[d][c_p] += val;
        }
      }
      
      if (cl_p >= 0) {
        ierr = FVDAReconstructP1Evaluate(fv,&fv->face_centroid[3*f],c_p,&_fv_coor[3*c_p],_Q,&coeff[3*cl_p],&Qhr);CHKERRQ(ierr);

        // cell[-]
        for (PetscInt d=0; d<3; d++) {
          val = 0.5 * Qhr * normal[d] * dS;
          //ierr = VecSetValue(grad[d],c_m,val,ADD_VALUES);CHKERRQ(ierr);
          _grad[d][c_m] += val;
        }

        // cell[+]
        for (PetscInt d=0; d<3; d++) {
          val = -0.5 * Qhr * normal[d] * dS;
          //ierr = VecSetValue(grad[d],c_p,val,ADD_VALUES);CHKERRQ(ierr);
          _grad[d][c_p] += val;
        }
      }
    }
  }
  /*
  for (PetscInt d=0; d<3; d++) {
    ierr = VecAssemblyBegin(grad[d]);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(grad[d]);CHKERRQ(ierr);
  }
  */
  ierr = VecRestoreArray(grad[0],&_grad[0]);CHKERRQ(ierr);
  ierr = VecRestoreArray(grad[1],&_grad[1]);CHKERRQ(ierr);
  ierr = VecRestoreArray(grad[2],&_grad[2]);CHKERRQ(ierr);
  
/*
  // slow variant which stupidly recomputes the cell volume 3x times //
  for (PetscInt d=0; d<3; d++) {
    const PetscReal *_g;
    
    ierr = VecZeroEntries(gradg);CHKERRQ(ierr);
    ierr = DMLocalToGlobal(fv->dm_fv,grad[d],ADD_VALUES,gradg);CHKERRQ(ierr);
    ierr = VecGetArrayRead(gradg,&_g);CHKERRQ(ierr);
    
    // normalize //
    for (c=0; c<fv->ncells; c++) {
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateCellVolume3d(cell_coor,&dV);
      
      _gradQ[3 * c + d] = _g[c] / dV;
    }

    ierr = VecRestoreArrayRead(gradg,&_g);CHKERRQ(ierr);
  }
*/

  {
    
    /* insert 1/volume into gradQ storage */
    for (c=0; c<fv->ncells; c++) {
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateCellVolume3d(cell_coor,&dV);
      
      for (PetscInt d=0; d<3; d++) {
        _gradQ[3 * c + d] = 1.0 / dV;
      }
    }
    
    for (PetscInt d=0; d<3; d++) {
      const PetscReal *_g;
      
      ierr = VecZeroEntries(gradg);CHKERRQ(ierr);
      ierr = DMLocalToGlobal(fv->dm_fv,grad[d],ADD_VALUES,gradg);CHKERRQ(ierr);
      ierr = VecGetArrayRead(gradg,&_g);CHKERRQ(ierr);
      
      /* scale 1/|v| by gradQ estimate */
      for (c=0; c<fv->ncells; c++) {
        _gradQ[3 * c + d] *= _g[c];
      }
      
      ierr = VecRestoreArrayRead(gradg,&_g);CHKERRQ(ierr);
    }
    
  }
  
  
  ierr = PetscFree(coeff);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Ql,&_Q);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_fv,&Ql);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fv_coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  ierr = VecDestroy(&Qg);CHKERRQ(ierr);

  for (PetscInt d=0; d<3; d++) {
    ierr = VecDestroy(&grad[d]);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&gradg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode FVDAFieldProjectReconstructionToVertex_Q1(FVDA fv,Vec fv_field,PetscReal min,PetscReal max,DM dmf,Vec field)
{
  PetscErrorCode    ierr;
  Vec               geometry_coorl,fv_fieldl,fv_coorl;
  const PetscScalar *_geom_coor,*_fv_field,*_fv_coor;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS];
  Vec               sum,fieldl,suml;
  PetscReal         *_field,*_sum;
  PetscInt          c,i;
  PetscInt          n_neigh,neigh[27];
  PetscReal         coeff[3];
  PetscInt          *cell2fvcell;
  
  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(fv->dm_fv,&fv_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fv_coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_fv,&fv_fieldl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_fv,fv_field,INSERT_VALUES,fv_fieldl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fv_fieldl,&_fv_field);CHKERRQ(ierr);
  
  ierr = DMGetGlobalVector(dmf,&sum);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmf,&fieldl);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmf,&suml);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(field);CHKERRQ(ierr);CHKERRQ(ierr); /* initialize input */
  ierr = VecZeroEntries(sum);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = VecZeroEntries(fieldl);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = VecZeroEntries(suml);CHKERRQ(ierr);CHKERRQ(ierr);
  
  ierr = VecGetArray(fieldl,&_field);CHKERRQ(ierr);
  ierr = VecGetArray(suml,&_sum);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(fv->ncells,&cell2fvcell);CHKERRQ(ierr);
  {
    PetscInt f;
    
    for (f=0; f<fv->nfaces; f++) {
      PetscInt clocal_p,clocal_m;
      clocal_m = fv->face_element_map[2*f+0];
      clocal_p = fv->face_element_map[2*f+1];
      
      if (clocal_m >= fv->ncells) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"clocal_minus > ncells");
      if (clocal_p >= fv->ncells) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"clocal_plus > ncells");
      
      if (clocal_m >= 0) { cell2fvcell[clocal_m] = fv->face_fv_map[2*f+0]; }
      if (clocal_p >= 0) { cell2fvcell[clocal_p] = fv->face_fv_map[2*f+1]; }
    }
  }
  
  for (c=0; c<fv->ncells; c++) {
    PetscInt c_fv;
    
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    
    /* TODO - need to convert c into dm_fv (local) numbering */
    c_fv = cell2fvcell[c];
    
    ierr = FVDAGetReconstructionStencil_AtCell(fv,c_fv,&n_neigh,neigh);CHKERRQ(ierr);
    ierr = setup_coeff(fv,c_fv,n_neigh,(const PetscInt*)neigh,_fv_coor,_fv_field,coeff);CHKERRQ(ierr);
    
    for (i=0; i<8; i++) {
      PetscReal Qhr;
      
      ierr = FVDAReconstructP1Evaluate(fv,&cell_coor[3*i],c_fv,(const PetscReal*)&_fv_coor[3*c_fv],_fv_field,coeff,&Qhr);CHKERRQ(ierr);
      
      if (Qhr > max) { Qhr = max; }
      if (Qhr < min) { Qhr = min; }
      
      _field[ element[i] ] += Qhr;
      _sum[ element[i] ]   += 1.0;
    }
  }
  ierr = VecRestoreArray(suml,&_sum);CHKERRQ(ierr);
  ierr = VecRestoreArray(fieldl,&_field);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fv_fieldl,&_fv_field);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fv_coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dmf,fieldl,ADD_VALUES,field);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dmf,suml,ADD_VALUES,sum);CHKERRQ(ierr);
  
  ierr = VecPointwiseDivide(field,field,sum);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmf,&suml);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmf,&fieldl);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmf,&sum);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_fv,&fv_fieldl);CHKERRQ(ierr);
  
  ierr = PetscFree(cell2fvcell);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

