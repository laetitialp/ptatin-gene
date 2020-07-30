
#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_private.h>
#include <fvda_utils.h>


/*
 Computes
   v = (x1 - x0)/dt
 pointwise
 
 - dmg should define the cell geometry (vertex) mesh
 - The vectors x0, x1, v are defined on dmg
 
*/
PetscErrorCode FVDAALEComputeMeshVelocity(DM dmg,Vec x0,Vec x1,PetscReal dt,Vec v)
{
  PetscErrorCode  ierr;
  PetscInt        k,d,len;
  const PetscReal *_x0,*_x1;
  PetscReal       *_v;

  
  PetscFunctionBegin;
  ierr = VecGetLocalSize(x0,&len);CHKERRQ(ierr);
  len = len / 3;
  ierr = VecGetArrayRead(x0,&_x0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x1,&_x1);CHKERRQ(ierr);
  ierr = VecGetArray(v,&_v);CHKERRQ(ierr);
  for (k=0; k<len; k++) {
    for (d=0; d<3; d++) {
      _v[3*k+d] = (_x1[3*k+d] - _x0[3*k+d])/dt;
    }
  }
  ierr = VecRestoreArray(v,&_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x1,&_x1);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x0,&_x0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Compute an average velocity on each control face.
 
 Performs a simple pointwise approximation to define the velocity vector on each control volume face.
 This is equivalent to performing a integral average approximated with a one point quadrature rule.
 
 [Alg]
 * loop over faces
 * pick a valid element
 * extract all cell velocities (vertex)
 * get local cell indices associated with face (and valid element)
 * extract face velocity values
 * average face velocities (sum and divide by 4)
 * set avg onto fv face property storage
 
 - dmg should define the cell geometry (vertex) mesh
 - x0 and v are define on dmg
 - x0 defines the vertex mesh coordinates (not used here)
 - face_vec_name is textual name where the face property will be stored in the FVDA object
*/
PetscErrorCode FVDAALEComputeFaceAverageVelocity_Interpolate(DM dmg,Vec x0,Vec v,FVDA fv,const char face_vec_name[])
{
  PetscErrorCode  ierr;
  Vec             vl;
  const PetscReal *_vl,*x_face;
  PetscReal       *v_face;
  DACellFace      cell_face_label;
  PetscInt        fidx[DACELL3D_FACE_VERTS];
  PetscInt        f,i,cellid;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_v[3*DACELL3D_VERTS];
  
  
  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dmg,&vl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmg,v,INSERT_VALUES,vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vl,&_vl);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArray(fv,face_vec_name,&v_face);CHKERRQ(ierr);
  ierr = FVDAGetFaceInfo(fv,NULL,NULL,NULL,NULL,&x_face);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  for (f=0; f<fv->nfaces; f++) {
    PetscReal avg_v[] = {0,0,0};
    
    cell_face_label = fv->face_type[f];
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    
    for (i=0; i<DACELL3D_VERTS; i++) {
      cell_v[3*i+0] = _vl[3*element[i]+0];
      cell_v[3*i+1] = _vl[3*element[i]+1];
      cell_v[3*i+2] = _vl[3*element[i]+2];
    }
    
    ierr = DACellGeometry3d_GetFaceIndices(NULL,cell_face_label,fidx);CHKERRQ(ierr);
    
    for (i=0; i<DACELL3D_FACE_VERTS; i++) {
      avg_v[0] += cell_v[3*fidx[i]  ];
      avg_v[1] += cell_v[3*fidx[i]+1];
      avg_v[2] += cell_v[3*fidx[i]+2];
    }
    avg_v[0] = avg_v[0] * 0.25;
    avg_v[1] = avg_v[1] * 0.25;
    avg_v[2] = avg_v[2] * 0.25;
    
    v_face[3*f+0] = avg_v[0];
    v_face[3*f+1] = avg_v[1];
    v_face[3*f+2] = avg_v[2];
  }
  
  ierr = VecRestoreArrayRead(vl,&_vl);CHKERRQ(ierr);
  ierr = VecDestroy(&vl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
