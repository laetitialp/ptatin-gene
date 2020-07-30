
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda.h>
#include <fvda_utils.h>

PetscErrorCode default_setter(FVDA fv,
                              DACellFace face,
                              PetscInt nfaces,
                              const PetscReal coor[],
                              const PetscReal normal[],
                              const PetscInt cell[],
                              PetscReal time,
                              FVFluxType flux[],
                              PetscReal bcvalue[],
                              void *ctx)
{
  PetscInt f;
  
  for (f=0; f<nfaces; f++) {
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode t1_default(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 12;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  //ierr = FVDAView_CellGeom_local(fv);CHKERRQ(ierr);
  //ierr = FVDAView_BFaceGeom_local(fv);CHKERRQ(ierr);
  ierr = FVDAView_FaceGeom_local(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr);
  
  ierr = FVDARegisterCellProperty(fv,"rho_cp",1);CHKERRQ(ierr);
  //ierr = FVDARegisterCellProperty(fv,"k",1);CHKERRQ(ierr);
  //ierr = FVDARegisterCellProperty(fv,"Q",1);CHKERRQ(ierr);
  
  ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  
  //ierr = FVDAView_JSON(fv,NULL,NULL);CHKERRQ(ierr);
  //ierr = FVDAView_JSON(fv,NULL,"stepA");CHKERRQ(ierr);
  ierr = FVDAView_JSON(fv,"./jout","stepA");CHKERRQ(ierr);
  {
    Vec Q;
    DMCreateGlobalVector(fv->dm_fv,&Q);
    ierr = PetscVecWriteJSON(Q,0,"thisvec");CHKERRQ(ierr);
    ierr = FVDAView_Heavy(fv,"./jout","stepA");CHKERRQ(ierr);
  }
  
  ierr = FVDAViewStatistics(fv,PETSC_TRUE);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* mesh warp */
/* mappings */
typedef struct _n_CCmplx CCmplx;
struct _n_CCmplx {
  PetscReal real;
  PetscReal imag;
};

CCmplx CCmplxPow(CCmplx a,PetscReal n)
{
  CCmplx b;
  PetscReal r,theta;
  r      = PetscSqrtReal(a.real*a.real + a.imag*a.imag);
  theta  = PetscAtan2Real(a.imag,a.real);
  b.real = PetscPowReal(r,n) * PetscCosReal(n*theta);
  b.imag = PetscPowReal(r,n) * PetscSinReal(n*theta);
  return b;
}
CCmplx CCmplxExp(CCmplx a)
{
  CCmplx b;
  b.real = PetscExpReal(a.real) * PetscCosReal(a.imag);
  b.imag = PetscExpReal(a.real) * PetscSinReal(a.imag);
  return b;
}
CCmplx CCmplxSqrt(CCmplx a)
{
  CCmplx b;
  PetscReal r,theta;
  r      = PetscSqrtReal(a.real*a.real + a.imag*a.imag);
  theta  = PetscAtan2Real(a.imag,a.real);
  b.real = PetscSqrtReal(r) * PetscCosReal(0.5*theta);
  b.imag = PetscSqrtReal(r) * PetscSinReal(0.5*theta);
  return b;
}
CCmplx CCmplxAdd(CCmplx a,CCmplx c)
{
  CCmplx b;
  b.real = a.real +c.real;
  b.imag = a.imag +c.imag;
  return b;
}
PetscScalar CCmplxRe(CCmplx a)
{
  return (PetscScalar)a.real;
}
PetscScalar CCmplxIm(CCmplx a)
{
  return (PetscScalar)a.imag;
}

PetscErrorCode DAApplyConformalMapping(DM da,PetscInt idx)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscInt       sx,nx,sy,ny,sz,nz,dim;
  Vec            Gcoords;
  PetscScalar    *XX;
  PetscScalar    xx,yy,zz;
  DM             cda;
  
  
  PetscFunctionBeginUser;
  if (idx == 1) { /* dam break */
    ierr = DMDASetUniformCoordinates(da, -1.0,1.0, -1.0,1.0, -1.0,1.0);CHKERRQ(ierr);
  } else if (idx == 2) { /* stagnation in a corner */
    ierr = DMDASetUniformCoordinates(da, -1.0,1.0, 0.0,1.0, -1.0,1.0);CHKERRQ(ierr);
  } else if (idx == 3) { /* nautilis */
    ierr = DMDASetUniformCoordinates(da, -1.0,1.0, -1.0,1.0, -1.0,1.0);CHKERRQ(ierr);
  } else if (idx == 4) {
    ierr = DMDASetUniformCoordinates(da, -1.0,1.0, -1.0,1.0, -1.0,1.0);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"idx must be {1,2,3,4}");
  
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&Gcoords);CHKERRQ(ierr);
  
  ierr = VecGetArray(Gcoords,&XX);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, &dim, 0,0,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Gcoords,&n);CHKERRQ(ierr);
  n    = n / dim;
  
  for (i=0; i<n; i++) {
    if ((dim == 3) && (idx != 2)) {
      PetscScalar Ni[8];
      PetscScalar xi   = XX[dim*i];
      PetscScalar eta  = XX[dim*i+1];
      PetscScalar zeta = XX[dim*i+2];
      PetscScalar xn[] = {-1.0,1.0,-1.0,1.0,   -1.0,1.0,-1.0,1.0  };
      PetscScalar yn[] = {-1.0,-1.0,1.0,1.0,   -1.0,-1.0,1.0,1.0  };
      PetscScalar zn[] = {-0.1,-4.0,-0.2,-1.0,  0.1,4.0,0.2,1.0  };
      PetscInt    p;
      
      Ni[0] = 0.125*(1.0-xi)*(1.0-eta)*(1.0-zeta);
      Ni[1] = 0.125*(1.0+xi)*(1.0-eta)*(1.0-zeta);
      Ni[2] = 0.125*(1.0-xi)*(1.0+eta)*(1.0-zeta);
      Ni[3] = 0.125*(1.0+xi)*(1.0+eta)*(1.0-zeta);
      
      Ni[4] = 0.125*(1.0-xi)*(1.0-eta)*(1.0+zeta);
      Ni[5] = 0.125*(1.0+xi)*(1.0-eta)*(1.0+zeta);
      Ni[6] = 0.125*(1.0-xi)*(1.0+eta)*(1.0+zeta);
      Ni[7] = 0.125*(1.0+xi)*(1.0+eta)*(1.0+zeta);
      
      xx = yy = zz = 0.0;
      for (p=0; p<8; p++) {
        xx += Ni[p]*xn[p];
        yy += Ni[p]*yn[p];
        zz += Ni[p]*zn[p];
      }
      XX[dim*i]   = xx;
      XX[dim*i+1] = yy;
      XX[dim*i+2] = zz;
    }
    
    if (idx == 1) {
      CCmplx zeta,t1,t2;
      
      xx = XX[dim*i]   - 0.8;
      yy = XX[dim*i+1] + 1.5;
      
      zeta.real = PetscRealPart(xx);
      zeta.imag = PetscRealPart(yy);
      
      t1 = CCmplxPow(zeta,-1.0);
      t2 = CCmplxAdd(zeta,t1);
      
      XX[dim*i]   = CCmplxRe(t2);
      XX[dim*i+1] = CCmplxIm(t2);
    } else if (idx == 2) {
      CCmplx zeta,t1;
      
      xx = XX[dim*i];
      yy = XX[dim*i+1];
      zeta.real = PetscRealPart(xx);
      zeta.imag = PetscRealPart(yy);
      
      t1 = CCmplxSqrt(zeta);
      XX[dim*i]   = CCmplxRe(t1);
      XX[dim*i+1] = CCmplxIm(t1);
    } else if (idx == 3) {
      CCmplx zeta,t1,t2;
      
      xx = XX[dim*i]   - 0.8;
      yy = XX[dim*i+1] + 1.5;
      
      zeta.real   = PetscRealPart(xx);
      zeta.imag   = PetscRealPart(yy);
      t1          = CCmplxPow(zeta,-1.0);
      t2          = CCmplxAdd(zeta,t1);
      XX[dim*i]   = CCmplxRe(t2);
      XX[dim*i+1] = CCmplxIm(t2);
      
      xx          = XX[dim*i];
      yy          = XX[dim*i+1];
      zeta.real   = PetscRealPart(xx);
      zeta.imag   = PetscRealPart(yy);
      t1          = CCmplxExp(zeta);
      XX[dim*i]   = CCmplxRe(t1);
      XX[dim*i+1] = CCmplxIm(t1);
      
      xx          = XX[dim*i] + 0.4;
      yy          = XX[dim*i+1];
      zeta.real   = PetscRealPart(xx);
      zeta.imag   = PetscRealPart(yy);
      t1          = CCmplxPow(zeta,2.0);
      XX[dim*i]   = CCmplxRe(t1);
      XX[dim*i+1] = CCmplxIm(t1);
    } else if (idx == 4) {
      PetscScalar Ni[4];
      PetscScalar xi   = XX[dim*i];
      PetscScalar eta  = XX[dim*i+1];
      PetscScalar xn[] = {0.0,2.0,0.2,3.5};
      PetscScalar yn[] = {-1.3,0.0,2.0,4.0};
      PetscInt    p;
      
      Ni[0] = 0.25*(1.0-xi)*(1.0-eta);
      Ni[1] = 0.25*(1.0+xi)*(1.0-eta);
      Ni[2] = 0.25*(1.0-xi)*(1.0+eta);
      Ni[3] = 0.25*(1.0+xi)*(1.0+eta);
      
      xx = yy = 0.0;
      for (p=0; p<4; p++) {
        xx += Ni[p]*xn[p];
        yy += Ni[p]*yn[p];
      }
      XX[dim*i]   = xx;
      XX[dim*i+1] = yy;
    }
  }
  ierr = VecRestoreArray(Gcoords,&XX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode t1_geom(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 24;
  const PetscInt m[] = {mx,mx,4};
  FVDA           fv;
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  
  {
    PetscInt geom_index = 1;
    Vec      coor,coorl;
    DM       dm2 = fv->dm_geometry;
    DM       cdm;
    
    ierr = PetscOptionsGetInt(NULL,NULL,"-geom_id",&geom_index,NULL);CHKERRQ(ierr);
    ierr = DAApplyConformalMapping(dm2,geom_index);CHKERRQ(ierr);
    
    ierr = DMGetCoordinateDM(dm2,&cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dm2,&coor);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm2,&coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(cdm,coor,INSERT_VALUES,coorl);CHKERRQ(ierr);
  }
  
  {
    Vec gcoor;
    
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  //ierr = FVDAView_CellGeom_local(fv);CHKERRQ(ierr);
  //ierr = FVDAView_BFaceGeom_local(fv);CHKERRQ(ierr);
  ierr = FVDAView_FaceGeom_local(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr);
  
  ierr = FVDARegisterCellProperty(fv,"rho_cp",1);CHKERRQ(ierr);
  ierr = FVDARegisterCellProperty(fv,"k",1);CHKERRQ(ierr);
  ierr = FVDARegisterCellProperty(fv,"Q",1);CHKERRQ(ierr);
  
  ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  
  ierr = FVDAViewStatistics(fv,PETSC_FALSE);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode t1_usergeom(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 8,dim;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  DM             dmg;
  PetscInt       Nv[]={0,0,0},mi[]={0,0,0},ncells,nen;
  const PetscInt *e;

  
  
  ierr = DMDACreate3d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                      DMDA_STENCIL_BOX,
                      mx+1,mx+1,4+1,
                      PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                      3,
                      1,
                      NULL,NULL,NULL,&dmg);CHKERRQ(ierr);

  ierr = DMSetFromOptions(dmg);CHKERRQ(ierr);
  ierr = DMSetUp(dmg);CHKERRQ(ierr);

  ierr = DMDASetElementType(dmg,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
  ierr = DMDAGetElements(dmg,&ncells,&nen,&e);CHKERRQ(ierr);

  
  ierr = DMDASetUniformCoordinates(dmg,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  ierr = DAApplyConformalMapping(dmg,3);CHKERRQ(ierr);
  {
    Vec coor,coorl;
    DM  cdm;
    
    ierr = DMGetCoordinateDM(dmg,&cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dmg,&coor);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dmg,&coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(cdm,coor,INSERT_VALUES,coorl);CHKERRQ(ierr);
  }

  
  ierr = FVDACreate(PetscObjectComm((PetscObject)dmg),&fv);CHKERRQ(ierr);
  
  ierr = DMGetDimension(dmg,&dim);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,dim);CHKERRQ(ierr);
  
  ierr = DMDAGetInfo(dmg,NULL,&Nv[0],&Nv[1],&Nv[2],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  Nv[0]--;
  Nv[1]--;
  Nv[2]--;
  ierr = DMDAGetElementsSizes(dmg,&mi[0],&mi[1],&mi[2]);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,m,Nv);CHKERRQ(ierr);

  ierr = FVDASetGeometryDM(fv,dmg);CHKERRQ(ierr);
  {
    Vec gcoor;
    
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)gcoor);CHKERRQ(ierr);
    fv->vertex_coor_geometry = gcoor;
  }
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  
  //ierr = FVDAView_CellGeom_local(fv);CHKERRQ(ierr);
  //ierr = FVDAView_BFaceGeom_local(fv);CHKERRQ(ierr);
  ierr = FVDAView_FaceGeom_local(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr);
  
  ierr = FVDARegisterCellProperty(fv,"rho_cp",1);CHKERRQ(ierr);
  ierr = FVDARegisterCellProperty(fv,"k",1);CHKERRQ(ierr);
  ierr = FVDARegisterCellProperty(fv,"Q",1);CHKERRQ(ierr);
  
  ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_FALSE,0.0,default_setter,NULL);CHKERRQ(ierr);
  
  ierr = FVDAViewStatistics(fv,PETSC_TRUE);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  ierr = DMDestroy(&dmg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

const char doc[] = {
"[FVDA test] Verifies setup of FVDA and structure and mesh geometry.\n" \
"  -tid 0: Test FVDA setup with coordinate aligned geometry\n" \
"  -tid 1: Test FVDA setup with deformed geometry\n"\
"    -geom_id {1,2,3,4}\n"\
"      1 -> Geometry description: \"dam break\"\n"\
"      2 -> Geometry description: \"stagnation in corner\"\n"\
"      3 -> Geometry description: \"nautilis\"\n"\
"      4 -> Geometry description: \"stealth bomber\"\n"\
"  -tid 2: Test FVDA user defined geometry\n"\
"  -tid 3: Test FVDA user specified partition size\n"\
};
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       tid = 0;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,doc);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-tid",&tid,NULL);CHKERRQ(ierr);
  switch (tid) {
    case 0:
      ierr = t1_default();CHKERRQ(ierr);
      break;
    case 1:
      ierr = t1_geom();CHKERRQ(ierr);
      break;
    case 2:
      ierr = t1_usergeom();CHKERRQ(ierr);
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Valid values for -tid {0,1,2}");
      break;
  }
  ierr = PetscFinalize();
  return ierr;
}
