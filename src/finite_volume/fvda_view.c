
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_private.h>
#include <fvda_utils.h>


PetscErrorCode FVDAViewStatistics(FVDA fv,PetscBool collective)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       k,blocksize;
  PetscMPIInt    commsize,commrank;
  
  
  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)fv->dm_fv);
  ierr = MPI_Comm_size(comm,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&commrank);CHKERRQ(ierr);
  PetscPrintf(comm,"[FVDA View]\n");
  if (!fv->setup) {
    PetscPrintf(comm,"  Warning: FVDASetUp() not called - report may be incomplete\n");
  }

  PetscPrintf(comm,"[FVDA sizes]\n");
  PetscPrintf(comm,"  mx,my,mz  %D x %D x %D (global)\n",fv->Mi[0],fv->Mi[1],fv->Mi[2]);
  
  if (!collective) {
    PetscPrintf(comm,"  mx,my,mz  %D x %D x %D (local)\n",fv->mi[0],fv->mi[1],fv->mi[2]);
    PetscPrintf(comm,"  #cells    %D\n",fv->ncells);
    PetscPrintf(comm,"  #faces    %D (total)\n",fv->nfaces);
    PetscPrintf(comm,"  #faces    %D (interior)\n",fv->nfaces_interior);
    PetscPrintf(comm,"  #faces    %D (boundary)\n",fv->nfaces_boundary);
  } else {
    PetscSynchronizedPrintf(comm,"  [rank %d]\n",(int)commrank);
    PetscSynchronizedPrintf(comm,"    mx,my,mz  %D x %D x %D (local)\n",fv->mi[0],fv->mi[1],fv->mi[2]);
    PetscSynchronizedPrintf(comm,"    #cells    %D\n",fv->ncells);
    PetscSynchronizedPrintf(comm,"    #faces    %D (total)\n",fv->nfaces);
    PetscSynchronizedPrintf(comm,"    #faces    %D (interior)\n",fv->nfaces_interior);
    PetscSynchronizedPrintf(comm,"    #faces    %D (boundary)\n",fv->nfaces_boundary);
    PetscSynchronizedFlush(comm,PETSC_STDOUT);
  }
  
  PetscPrintf(comm,"  #auxiliary cell fields %D\n",fv->ncoeff_cell);
  for (k=0; k<fv->ncoeff_cell; k++) {
    blocksize = fv->cell_coeff_size[k] / fv->ncells;
    PetscPrintf(comm,"    [%D] (\"%s\") blocksize %D\n",k,fv->cell_coeff_name[k],blocksize);
  }

  PetscPrintf(comm,"  #auxiliary face fields %D\n",fv->ncoeff_face);
  for (k=0; k<fv->ncoeff_face; k++) {
    blocksize = fv->face_coeff_size[k] / fv->nfaces;
    PetscPrintf(comm,"    [%D] (\"%s\") blocksize %D\n",k,fv->face_coeff_name[k],blocksize);
  }

  PetscPrintf(comm,"[FVDA memory usage]\n");
  if (collective) {
    PetscPrintf(comm,"  ** Values are collective over comm.size = %d ** \n",(int)commsize);
  }

/*
 PetscPrintf(comm,"  face_normal      %1.2e (MB)\n",sizeof(PetscReal)*fv->nfaces*3 * 1.0e-6);
 PetscPrintf(comm,"  face_centroid    %1.2e (MB)\n",sizeof(PetscReal)*fv->nfaces*3 * 1.0e-6);
 PetscPrintf(comm,"  boundary_flux    %1.2e (MB)\n",sizeof(FVFluxType)*fv->nfaces_boundary * 1.0e-6);
 PetscPrintf(comm,"  boundary_value   %1.2e (MB)\n",sizeof(PetscReal)*fv->nfaces_boundary * 1.0e-6);
 
 PetscPrintf(comm,"  face_element_map %1.2e (MB)\n",sizeof(PetscInt)*fv->nfaces*2 * 1.0e-6);
 PetscPrintf(comm,"  face_type        %1.2e (MB)\n",sizeof(DACellFace)*fv->nfaces * 1.0e-6);
 PetscPrintf(comm,"  face_loc         %1.2e (MB)\n",sizeof(DACellFaceLocation)*fv->nfaces * 1.0e-6);
 
 PetscPrintf(comm,"  face_fv_map      %1.2e (MB)\n",sizeof(PetscInt)*2*fv->nfaces * 1.0e-6);
 PetscPrintf(comm,"  face_id_inter.   %1.2e (MB)\n",sizeof(PetscInt)*fv->nfaces_interior * 1.0e-6);
 PetscPrintf(comm,"  face_id_bound.   %1.2e (MB)\n",sizeof(PetscInt)*fv->nfaces_boundary * 1.0e-6);
 
 for (k=0; k<fv->ncoeff_cell; k++) {
   PetscPrintf(comm,"  cell_coefficient[%D] (\"%s\") %1.2e (MB)\n",k,fv->cell_coeff_name[k],sizeof(PetscReal)*fv->cell_coeff_size[k] * 1.0e-6);
 }
 
 for (k=0; k<fv->ncoeff_face; k++) {
   PetscPrintf(comm,"  face_coefficient[%D] (\"%s\") %1.2e (MB)\n",k,fv->face_coeff_name[k],sizeof(PetscReal)*fv->face_coeff_size[k] * 1.0e-6);
 }
*/
  {
    double *mem,total = 0;
    int    cnt,k;
    
    ierr = PetscCalloc1(10,&mem);CHKERRQ(ierr);
    mem[0] = sizeof(PetscReal)*fv->nfaces*3 * 1.0e-6; // face_normal
    mem[1] = sizeof(PetscReal)*fv->nfaces*3 * 1.0e-6; // face_centroid
    mem[2] = sizeof(FVFluxType)*fv->nfaces_boundary * 1.0e-6; // boundary_flux
    mem[3] = sizeof(PetscReal)*fv->nfaces_boundary * 1.0e-6; // boundary_value
    
    mem[4] = sizeof(PetscInt)*fv->nfaces*2 * 1.0e-6; // face_element_map
    mem[5] = sizeof(DACellFace)*fv->nfaces * 1.0e-6; // face_type
    mem[6] = sizeof(DACellFaceLocation)*fv->nfaces * 1.0e-6; // face_loc
    
    mem[7] = sizeof(PetscInt)*2*fv->nfaces * 1.0e-6; // face_fv_map
    mem[8] = sizeof(PetscInt)*fv->nfaces_interior * 1.0e-6; // face_id_inter.
    mem[9] = sizeof(PetscInt)*fv->nfaces_boundary * 1.0e-6; // face_id_bound.
    
    if (collective) {
      ierr = MPI_Allreduce(MPI_IN_PLACE,mem,10,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    }
    total = 0; for (k=0; k<10; k++) { total += mem[k]; }
    
    PetscPrintf(comm,"  face_normal      %1.2e (MB)\n",mem[0]);
    PetscPrintf(comm,"  face_centroid    %1.2e (MB)\n",mem[1]);
    PetscPrintf(comm,"  boundary_flux    %1.2e (MB)\n",mem[2]);
    PetscPrintf(comm,"  boundary_value   %1.2e (MB)\n",mem[3]);

    PetscPrintf(comm,"  face_element_map %1.2e (MB)\n",mem[4]);
    PetscPrintf(comm,"  face_type        %1.2e (MB)\n",mem[5]);
    PetscPrintf(comm,"  face_loc         %1.2e (MB)\n",mem[6]);
    
    PetscPrintf(comm,"  face_fv_map      %1.2e (MB)\n",mem[7]);
    PetscPrintf(comm,"  face_id_inter.   %1.2e (MB)\n",mem[8]);
    PetscPrintf(comm,"  face_id_bound.   %1.2e (MB)\n",mem[9]);
    PetscPrintf(comm,"  total <fv-internal>     %1.2e (MB)\n",total);
    ierr = PetscFree(mem);CHKERRQ(ierr);
    
    ierr = PetscCalloc1(fv->ncoeff_cell + fv->ncoeff_face,&mem);CHKERRQ(ierr);
    cnt = 0;
    for (k=0; k<fv->ncoeff_cell; k++) {
      mem[cnt] = sizeof(PetscReal)*fv->cell_coeff_size[k] * 1.0e-6;
      cnt++;
    }
    for (k=0; k<fv->ncoeff_face; k++) {
      mem[cnt] = sizeof(PetscReal)*fv->face_coeff_size[k] * 1.0e-6;
      cnt++;
    }
    
    if (collective) {
      ierr = MPI_Allreduce(MPI_IN_PLACE,mem,fv->ncoeff_cell + fv->ncoeff_face,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    }

    cnt = 0;
    for (k=0; k<fv->ncoeff_cell; k++) {
      PetscPrintf(comm,"  cell_coefficient[%D] (\"%s\") %1.2e (MB)\n",k,fv->cell_coeff_name[k],mem[cnt]);
      cnt++;
    }
    
    for (k=0; k<fv->ncoeff_face; k++) {
      PetscPrintf(comm,"  face_coefficient[%D] (\"%s\") %1.2e (MB)\n",k,fv->face_coeff_name[k],mem[cnt]);
      cnt++;
    }
    
    total = 0; for (k=0; k<fv->ncoeff_cell; k++) { total += mem[k]; }
    PetscPrintf(comm,"  total <cells>           %1.2e (MB)\n",total);
    total = 0; for (k=0; k<fv->ncoeff_face; k++) { total += mem[k+fv->ncoeff_cell]; }
    PetscPrintf(comm,"  total <faces>           %1.2e (MB)\n",total);
    total = 0; for (k=0; k<cnt; k++) { total += mem[k]; }
    PetscPrintf(comm,"  total                   %1.2e (MB)\n",total);
    ierr = PetscFree(mem);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

/*
 Generate vtu
 cell
 cell centroid
 face normal vectors (only on exterior facets)
 face_centroid (only on exterior facets)
*/
PetscErrorCode FVDAView_CellGeom_local(FVDA fv)
{
  PetscErrorCode  ierr;
  FILE            *fp = NULL;
  char            name[PETSC_MAX_PATH_LEN];
  PetscMPIInt     rank;
  Vec             coorl;
  const PetscReal *_coorl;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element;
  PetscInt        Nv,i,c;
  int             npoints,ncells,offset;
  
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"fv-cellgeom-r%d.vtu",(int)rank);CHKERRQ(ierr);
  if ((fp = fopen (name,"w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open new VTU file %s",name);
  }
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = VecGetSize(coorl,&Nv);CHKERRQ(ierr);
  Nv = Nv / 3;
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  
  /* insert cells */
  npoints = (int)Nv;
  ncells = (int)dm_nel;
  
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  
  fprintf(fp,"<UnstructuredGrid>\n");
  fprintf(fp,"  <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",npoints,ncells);
  
  fprintf(fp,"    <Points>\n");
  fprintf(fp,"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (i=0; i<Nv; i++) {
    fprintf(fp,"%+1.6e %+1.6e %+1.6e ",_coorl[3*i],_coorl[3*i+1],_coorl[3*i+2]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Points>\n");
  
  fprintf(fp,"    <PointData>\n");
  fprintf(fp,"    </PointData>\n");
  fprintf(fp,"    <CellData>\n");
  fprintf(fp,"    </CellData>\n");
  
  fprintf(fp,"    <Cells>\n");
  /* connectivity */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  for (c=0; c<dm_nel; c++) {
    int cellvmap[8];
    
    cellvmap[0] = (int)dm_element[8*c+4];
    cellvmap[1] = (int)dm_element[8*c+5];
    cellvmap[2] = (int)dm_element[8*c+1];
    cellvmap[3] = (int)dm_element[8*c+0];
    
    cellvmap[4] = (int)dm_element[8*c+7];
    cellvmap[5] = (int)dm_element[8*c+6];
    cellvmap[6] = (int)dm_element[8*c+2];
    cellvmap[7] = (int)dm_element[8*c+3];
    fprintf(fp,"%d %d %d %d %d %d %d %d ",cellvmap[0],cellvmap[1],cellvmap[2],cellvmap[3],cellvmap[4],cellvmap[5],cellvmap[6],cellvmap[7]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  /* offsets */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  offset = 0;
  for (c=0; c<dm_nel; c++) {
    offset += 8;
    fprintf(fp,"%d ",offset);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  /* types */
  fprintf(fp,"      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (c=0; c<dm_nel; c++) {
    fprintf(fp,"%d ",12); /* VTK_HEXAHEDRON */
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Cells>\n");
  
  fprintf(fp,"  </Piece>\n");
  fprintf(fp,"</UnstructuredGrid>\n");
  fprintf(fp,"</VTKFile>\n");
  fclose(fp);
  
  ierr = VecRestoreArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Generate vtu
 faces,
 face normal vectors
 face_location
 face_type
*/
PetscErrorCode FVDAView_BFaceGeom_local(FVDA fv)
{
  PetscErrorCode  ierr;
  FILE            *fp = NULL;
  char            name[PETSC_MAX_PATH_LEN];
  PetscMPIInt     rank;
  Vec             coorl;
  const PetscReal *_coorl;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element;
  PetscInt        Nv,Nf,i,c,f;
  PetscInt        *indices = NULL;
  int             npoints,ncells,offset;

  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"fv-bfacegeom-r%d.vtu",(int)rank);CHKERRQ(ierr);
  if ((fp = fopen (name,"w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open new VTU file %s",name);
  }
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = VecGetSize(coorl,&Nv);CHKERRQ(ierr);
  Nv = Nv / 3;
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  
  Nf = fv->nfaces_boundary;
  indices = fv->face_idx_boundary;
  
  /* insert face */
  npoints = (int)Nv;
  ncells = (int)Nf;
  
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  
  fprintf(fp,"<UnstructuredGrid>\n");
  fprintf(fp,"  <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",npoints,ncells);
  
  fprintf(fp,"    <Points>\n");
  fprintf(fp,"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (i=0; i<Nv; i++) {
    fprintf(fp,"%+1.6e %+1.6e %+1.6e ",_coorl[3*i],_coorl[3*i+1],_coorl[3*i+2]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Points>\n");
  
  fprintf(fp,"    <PointData>\n");
  fprintf(fp,"    </PointData>\n");
  
  fprintf(fp,"    <CellData>\n");
  /* type */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"DACellFace\" format=\"ascii\">\n");
  for (f=0; f<Nf; f++) {
    PetscInt fid = indices[f];
    fprintf(fp,"%d ",(int)fv->face_type[fid]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  /* location */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"DACellFaceLocation\" format=\"ascii\">\n");
  for (f=0; f<Nf; f++) {
    PetscInt fid = indices[f];
    fprintf(fp,"%d ",(int)fv->face_location[fid]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  /* normal */
  fprintf(fp,"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"normal\" format=\"ascii\">\n");
  for (f=0; f<Nf; f++) {
    PetscInt fid = indices[f];
    fprintf(fp,"%+1.6e %+1.6e %+1.6e ",fv->face_normal[fv->dim*fid+0],fv->face_normal[fv->dim*fid+1],fv->face_normal[fv->dim*fid+2]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </CellData>\n");
  
  fprintf(fp,"    <Cells>\n");
  /* connectivity */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  for (f=0; f<Nf; f++) {
    PetscInt map[4];
    int      facevmap[4];
    PetscInt fid;
    
    fid = indices[f];
    ierr = DACellGeometry3d_GetFaceIndices(fv->dm_geometry,fv->face_type[fid],map);CHKERRQ(ierr);
    c = fv->face_element_map[2 * fid + 0];
    if (c == E_MINUS_OFF_RANK) {
      c = fv->face_element_map[2 * f + 1];
    }
    
    facevmap[0] = (int)dm_element[8*c+map[0]];
    facevmap[1] = (int)dm_element[8*c+map[1]];
    facevmap[2] = (int)dm_element[8*c+map[2]];
    facevmap[3] = (int)dm_element[8*c+map[3]];
    
    fprintf(fp,"%d %d %d %d ",facevmap[0],facevmap[1],facevmap[2],facevmap[3]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  /* offsets */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  offset = 0;
  for (c=0; c<Nf; c++) {
    offset += 4;
    fprintf(fp,"%d ",offset);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  /* types */
  fprintf(fp,"      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (c=0; c<Nf; c++) {
    fprintf(fp,"%d ",9); /* VTK_QUAD */
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Cells>\n");
  
  fprintf(fp,"  </Piece>\n");
  fprintf(fp,"</UnstructuredGrid>\n");
  fprintf(fp,"</VTKFile>\n");
  fclose(fp);
  
  ierr = VecRestoreArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAView_FaceGeom_local(FVDA fv)
{
  PetscErrorCode  ierr;
  FILE            *fp = NULL;
  char            name[PETSC_MAX_PATH_LEN];
  PetscMPIInt     rank;
  Vec             coorl;
  const PetscReal *_coorl;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element;
  PetscInt        Nv,Nf,i,c,f;
  int             npoints,ncells,offset;

  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"fv-facegeom-r%d.vtu",(int)rank);CHKERRQ(ierr);
  if ((fp = fopen (name,"w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open new VTU file %s",name);
  }
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = VecGetSize(coorl,&Nv);CHKERRQ(ierr);
  Nv = Nv / 3;
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  
  Nf = fv->nfaces;
  
  /* insert face */
  npoints = (int)Nv;
  ncells = (int)Nf;
  
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  
  fprintf(fp,"<UnstructuredGrid>\n");
  fprintf(fp,"  <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",npoints,ncells);
  
  fprintf(fp,"    <Points>\n");
  fprintf(fp,"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (i=0; i<Nv; i++) {
    fprintf(fp,"%+1.6e %+1.6e %+1.6e ",_coorl[3*i],_coorl[3*i+1],_coorl[3*i+2]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Points>\n");
  
  fprintf(fp,"    <PointData>\n");
  fprintf(fp,"    </PointData>\n");
  
  fprintf(fp,"    <CellData>\n");
  /* type */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"DACellFace\" format=\"ascii\">\n");
  for (f=0; f<Nf; f++) {
    fprintf(fp,"%d ",(int)fv->face_type[f]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  /* location */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"DACellFaceLocation\" format=\"ascii\">\n");
  for (f=0; f<Nf; f++) {
    fprintf(fp,"%d ",(int)fv->face_location[f]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  /* normal */
  fprintf(fp,"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"normal\" format=\"ascii\">\n");
  for (f=0; f<Nf; f++) {
    fprintf(fp,"%+1.6e %+1.6e %+1.6e ",fv->face_normal[fv->dim*f+0],fv->face_normal[fv->dim*f+1],fv->face_normal[fv->dim*f+2]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </CellData>\n");
  
  fprintf(fp,"    <Cells>\n");
  /* connectivity */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  for (f=0; f<Nf; f++) {
    PetscInt map[4];
    int      facevmap[4];
    
    ierr = DACellGeometry3d_GetFaceIndices(fv->dm_geometry,fv->face_type[f],map);CHKERRQ(ierr);
    c = fv->face_element_map[2 * f + 0];
    if (c == E_MINUS_OFF_RANK) {
      c = fv->face_element_map[2 * f + 1];
    }
    
    facevmap[0] = (int)dm_element[8*c+map[0]];
    facevmap[1] = (int)dm_element[8*c+map[1]];
    facevmap[2] = (int)dm_element[8*c+map[2]];
    facevmap[3] = (int)dm_element[8*c+map[3]];
    
    fprintf(fp,"%d %d %d %d ",facevmap[0],facevmap[1],facevmap[2],facevmap[3]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  /* offsets */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  offset = 0;
  for (c=0; c<Nf; c++) {
    offset += 4;
    fprintf(fp,"%d ",offset);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  /* types */
  fprintf(fp,"      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (c=0; c<Nf; c++) {
    fprintf(fp,"%d ",9); /* VTK_QUAD */
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Cells>\n");
  
  fprintf(fp,"  </Piece>\n");
  fprintf(fp,"</UnstructuredGrid>\n");
  fprintf(fp,"</VTKFile>\n");
  fclose(fp);
  
  ierr = VecRestoreArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAView_CellData_local(FVDA fv,Vec field,PetscBool view_cell_prop,const char prefix[])
{
  PetscErrorCode  ierr;
  FILE            *fp = NULL;
  char            name[PETSC_MAX_PATH_LEN];
  PetscMPIInt     rank;
  Vec             coorl;
  const PetscReal *_coorl,*_field;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element;
  PetscInt        Nv,i,c,p;
  int             npoints,ncells,offset;

  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"%s-r%d.vtu",prefix,(int)rank);CHKERRQ(ierr);
  if ((fp = fopen (name,"w")) == NULL)  {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open new VTU file %s",name);
  }
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = VecGetSize(coorl,&Nv);CHKERRQ(ierr);
  Nv = Nv / 3;
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_coorl);CHKERRQ(ierr);

  ierr = VecGetArrayRead(field,&_field);CHKERRQ(ierr);

  /* insert cells */
  npoints = (int)Nv;
  ncells = (int)dm_nel;
  
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  
  fprintf(fp,"<UnstructuredGrid>\n");
  fprintf(fp,"  <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",npoints,ncells);
  
  fprintf(fp,"    <Points>\n");
  fprintf(fp,"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (i=0; i<Nv; i++) {
    fprintf(fp,"%+1.6e %+1.6e %+1.6e ",_coorl[3*i],_coorl[3*i+1],_coorl[3*i+2]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Points>\n");
  
  fprintf(fp,"    <PointData>\n");
  fprintf(fp,"    </PointData>\n");
  
  
  fprintf(fp,"    <CellData>\n");
  
  fprintf(fp,"      <DataArray Name=\"Q\" type=\"Float64\" NumberOfComponents=\"1\" format=\"ascii\">\n");
  for (i=0; i<dm_nel; i++) {
    fprintf(fp,"%+1.6e ",_field[i]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");

  if (view_cell_prop) {
    for (p=0; p<fv->ncoeff_cell; p++) {
      PetscInt b,bs;
      
      ierr = FVDACellPropertyGetInfo(fv,fv->cell_coeff_name[p],NULL,NULL,&bs);CHKERRQ(ierr);
      if (bs == 1) {
        fprintf(fp,"      <DataArray Name=\"%s\" type=\"Float64\" NumberOfComponents=\"1\" format=\"ascii\">\n",fv->cell_coeff_name[p]);
        for (i=0; i<dm_nel; i++) {
          fprintf(fp,"%+1.6e ",fv->cell_coefficient[p][i]);
        }
        fprintf(fp,"\n");
        fprintf(fp,"      </DataArray>\n");
      } else {
        for (b=0; b<bs; b++) {
          fprintf(fp,"      <DataArray Name=\"%s_%d\" type=\"Float64\" NumberOfComponents=\"1\" format=\"ascii\">\n",fv->cell_coeff_name[p],b);
          for (i=0; i<dm_nel; i++) {
            fprintf(fp,"%+1.6e ",fv->cell_coefficient[p][bs*i+b]);
          }
          fprintf(fp,"\n");
          fprintf(fp,"      </DataArray>\n");
        }
      }
    }
  }
  
  fprintf(fp,"    </CellData>\n");
  
  fprintf(fp,"    <Cells>\n");
  /* connectivity */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  for (c=0; c<dm_nel; c++) {
    int cellvmap[8];
    
    cellvmap[0] = (int)dm_element[8*c+4];
    cellvmap[1] = (int)dm_element[8*c+5];
    cellvmap[2] = (int)dm_element[8*c+1];
    cellvmap[3] = (int)dm_element[8*c+0];
    
    cellvmap[4] = (int)dm_element[8*c+7];
    cellvmap[5] = (int)dm_element[8*c+6];
    cellvmap[6] = (int)dm_element[8*c+2];
    cellvmap[7] = (int)dm_element[8*c+3];
    fprintf(fp,"%d %d %d %d %d %d %d %d ",cellvmap[0],cellvmap[1],cellvmap[2],cellvmap[3],cellvmap[4],cellvmap[5],cellvmap[6],cellvmap[7]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  /* offsets */
  fprintf(fp,"      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  offset = 0;
  for (c=0; c<dm_nel; c++) {
    offset += 8;
    fprintf(fp,"%d ",offset);
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  
  /* types */
  fprintf(fp,"      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (c=0; c<dm_nel; c++) {
    fprintf(fp,"%d ",12); /* VTK_HEXAHEDRON */
  }
  fprintf(fp,"\n");
  fprintf(fp,"      </DataArray>\n");
  fprintf(fp,"    </Cells>\n");
  
  fprintf(fp,"  </Piece>\n");
  fprintf(fp,"</UnstructuredGrid>\n");
  fprintf(fp,"</VTKFile>\n");
  fclose(fp);
  
  ierr = VecRestoreArrayRead(field,&_field);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coorl,&_coorl);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAView_CellData(FVDA fv,Vec field,PetscBool view_cell_prop,const char prefix[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  char           pvtu_fname[PETSC_MAX_PATH_LEN];
  FILE           *vtk_fp = NULL;
  PetscInt       p;
  
  
  PetscFunctionBegin;
  /* write out sub-domain file */
  ierr = FVDAView_CellData_local(fv,field,view_cell_prop,prefix);CHKERRQ(ierr);
  
  ierr = PetscSNPrintf(pvtu_fname,PETSC_MAX_PATH_LEN-1,"%s.pvtu",prefix);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(fv->comm,&rank);CHKERRQ(ierr);
  if (rank == 0) {
    if ((vtk_fp = fopen(pvtu_fname,"w")) == NULL)  {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open PVTU file %s",pvtu_fname);
    }
  }
  
  if (vtk_fp) fprintf(vtk_fp,"<?xml version=\"1.0\"?>\n");
  
#ifdef WORDSIZE_BIGENDIAN
  if (vtk_fp) fprintf(vtk_fp,"<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  if (vtk_fp) fprintf(vtk_fp,"<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif
  
  if (vtk_fp) PetscFPrintf(PETSC_COMM_SELF,vtk_fp,"  <PUnstructuredGrid GhostLevel=\"0\">\n");
  
  if (vtk_fp) fprintf(vtk_fp,"    <PPoints>\n");
  if (vtk_fp) fprintf(vtk_fp,"      <PDataArray Name=\"Points\" NumberOfComponents=\"3\" type=\"Float64\"/>\n");
  if (vtk_fp) fprintf(vtk_fp,"    </PPoints>\n");
  
  if (vtk_fp) fprintf(vtk_fp,"    <PCellData>\n");
  if (vtk_fp) fprintf(vtk_fp,"      <PDataArray Name=\"Q\" NumberOfComponents=\"1\" type=\"Float64\"/>\n");
  if (view_cell_prop && vtk_fp) {
    for (p=0; p<fv->ncoeff_cell; p++) {
      PetscInt b,bs;
      
      ierr = FVDACellPropertyGetInfo(fv,fv->cell_coeff_name[p],NULL,NULL,&bs);CHKERRQ(ierr);
      if (bs == 1) {
        fprintf(vtk_fp,"      <PDataArray Name=\"%s\" NumberOfComponents=\"1\" type=\"Float64\"/>\n",fv->cell_coeff_name[p]);
      } else {
        for (b=0; b<bs; b++) {
          fprintf(vtk_fp,"      <PDataArray Name=\"%s_%d\" NumberOfComponents=\"1\" type=\"Float64\"/>\n",fv->cell_coeff_name[p],b);
        }
      }
    }
  }
  if (vtk_fp) fprintf(vtk_fp,"    </PCellData>\n");
  
  /* no point data */
  if (vtk_fp) fprintf(vtk_fp,"    <PPointData>\n");
  if (vtk_fp) fprintf(vtk_fp,"    </PPointData>\n");
  
  /* write out the parallel information */
  ierr = MPI_Comm_size(fv->comm,&size);CHKERRQ(ierr);
  if (vtk_fp) {
    size_t len;
    int    c;
    char   vtu_fname[PETSC_MAX_PATH_LEN];
    
    /* strip out the path */
    ierr = PetscStrlen(prefix,&len);CHKERRQ(ierr);
    for (c=len-1; c>=0; c--) {
      if (prefix[c] == '/') { break; }
    }
    c++;
    for (rank=0; rank<size; rank++) {
      ierr = PetscSNPrintf(vtu_fname,PETSC_MAX_PATH_LEN-1,"%s-r%d.vtu",&prefix[c],(int)rank);CHKERRQ(ierr);
      fprintf(vtk_fp,"    <Piece Source=\"%s\"/>\n",vtu_fname);
    }
  }
  
  if (vtk_fp) fprintf(vtk_fp,"  </PUnstructuredGrid>\n");
  if (vtk_fp) fprintf(vtk_fp,"</VTKFile>\n");
  if (vtk_fp) fclose(vtk_fp);

  ierr = MPI_Barrier(fv->comm);CHKERRQ(ierr); /* insert barrier to ensure all data is written before any rank returns */
  PetscFunctionReturn(0);
}

/*
 prefix_fv_cell_xdmf.json
 cell size
 cell_i
 cell_j
 cell_k
 cell_fields
 face_fields
 geometry_coor_file
 cell_field_file
 x_field_file
*/

#include <cjson_utils.h>
#include <dmda_checkpoint.h>

PetscErrorCode FVDAView_JSON(FVDA fv,const char path[],const char prefix[])
{
  MPI_Comm       comm;
  PetscMPIInt    commsize,commrank;
  PetscInt       ranks[] = {0,0,0};
  PetscInt       f;
  char           jprefix_fv[PETSC_MAX_PATH_LEN];
  char           jprefix_geom[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  {
    if (path) {
      if (prefix) { PetscSNPrintf(jprefix_fv,PETSC_MAX_PATH_LEN-1,"%s/%s_fvda_fvspace",path,prefix); }
      else { PetscSNPrintf(jprefix_fv,PETSC_MAX_PATH_LEN-1,"%s/fvda_fvspace",path); }
    } else {
      if (prefix) { PetscSNPrintf(jprefix_fv,PETSC_MAX_PATH_LEN-1,"%s_fvda_fvspace",prefix); }
      else { PetscSNPrintf(jprefix_fv,PETSC_MAX_PATH_LEN-1,"fvda_fvspace"); }
    }
    ierr = DMDACheckpointWrite(fv->dm_fv,jprefix_fv);CHKERRQ(ierr);
  }

  {
    if (path) {
      if (prefix) { PetscSNPrintf(jprefix_geom,PETSC_MAX_PATH_LEN-1,"%s/%s_fvda_geom",path,prefix); }
      else { PetscSNPrintf(jprefix_geom,PETSC_MAX_PATH_LEN-1,"%s/fvda_geom",path); }
    } else {
      if (prefix) { PetscSNPrintf(jprefix_geom,PETSC_MAX_PATH_LEN-1,"%s_fvda_geom",prefix); }
      else { PetscSNPrintf(jprefix_geom,PETSC_MAX_PATH_LEN-1,"fvda_geom"); }
    }
    ierr = DMDACheckpointWrite(fv->dm_geometry,jprefix_geom);CHKERRQ(ierr);
    
    /* geom coords */
    {
      char cfilename[PETSC_MAX_PATH_LEN];
      PetscSNPrintf(cfilename,PETSC_MAX_PATH_LEN-1,"%s_coords",jprefix_geom);
      ierr = PetscVecWriteJSON(fv->vertex_coor_geometry,0,cfilename);CHKERRQ(ierr);
    }
  }

  comm = fv->comm;
  ierr = MPI_Comm_size(comm,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&commrank);CHKERRQ(ierr);

  ierr = DMDAGetInfo(fv->dm_fv,NULL,NULL,NULL,NULL,&ranks[0],&ranks[1],&ranks[2],NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  if (commrank == 0) {
    cJSON *jso_file = NULL,*jso_dm = NULL,*jso_part,*content,*ja,*obj;
    
    /* create json meta data file */
    jso_file = cJSON_CreateObject();
    
    jso_dm = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_file,"FVDA",jso_dm);

    content = cJSON_CreateInt((int)fv->dim);    cJSON_AddItemToObject(jso_dm,"dim",content);
    content = cJSON_CreateInt((int)fv->ncells); cJSON_AddItemToObject(jso_dm,"nCells",content);
    content = cJSON_CreateInt((int)fv->nfaces); cJSON_AddItemToObject(jso_dm,"nFaces",content);
    {
      char filename[PETSC_MAX_PATH_LEN];
      
      //PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s_dmda.json",jprefix_fv); // abs path
      if (prefix) { PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s_fvda_fvspace_dmda.json",prefix); }
      else { PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"fvda_fvspace_dmda.json"); }
      content = cJSON_CreateString(filename); cJSON_AddItemToObject(jso_dm,"dm_fv_json",content);
      //PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s_dmda.json",jprefix_geom); // abs path
      if (prefix) { PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s_fvda_geom_dmda.json",prefix); }
      else { PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"fvda_geom_dmda.json"); }
      content = cJSON_CreateString(filename); cJSON_AddItemToObject(jso_dm,"dm_geometry_json",content);
      
      /* geom coords */
      //PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s_coords.json",jprefix_geom); // abs path
      if (prefix) { PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s_fvda_geom_coords.json",prefix); }
      else { PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"fvda_geom_coords.json"); }
      content = cJSON_CreateString(filename); cJSON_AddItemToObject(jso_dm,"dm_geometry_coords_json",content);
    }
    
    ja = cJSON_CreateArray();
    cJSON_AddItemToObject(jso_dm,"cellFields",ja);
    for (f=0; f<fv->ncoeff_cell; f++) {
      int bs = fv->cell_coeff_size[f] / fv->ncells;
      obj = cJSON_CreateObject();
      content = cJSON_CreateString(fv->cell_coeff_name[f]); cJSON_AddItemToObject(obj,"name",content);
      content = cJSON_CreateInt(bs);                        cJSON_AddItemToObject(obj,"blockSize",content);
      cJSON_AddItemToArray(ja,obj);
    }

    ja = cJSON_CreateArray();
    cJSON_AddItemToObject(jso_dm,"faceFields",ja);
    for (f=0; f<fv->ncoeff_face; f++) {
      int bs = fv->face_coeff_size[f] / fv->nfaces;
      obj = cJSON_CreateObject();
      content = cJSON_CreateString(fv->face_coeff_name[f]); cJSON_AddItemToObject(obj,"name",content);
      content = cJSON_CreateInt(bs);                        cJSON_AddItemToObject(obj,"blockSize",content);
      cJSON_AddItemToArray(ja,obj);
    }
    
    ja = cJSON_CreateArray();
    cJSON_AddItemToObject(jso_dm,"directions",ja);

    {
      obj = cJSON_CreateObject();
      content = cJSON_CreateInt((int)fv->Mi[0]); cJSON_AddItemToObject(obj,"M",content);
      content = cJSON_CreateInt((int)fv->mi[0]); cJSON_AddItemToObject(obj,"m",content);
      cJSON_AddItemToArray(ja,obj);
    }
    {
      obj = cJSON_CreateObject();
      content = cJSON_CreateInt((int)fv->Mi[1]); cJSON_AddItemToObject(obj,"M",content);
      content = cJSON_CreateInt((int)fv->mi[1]); cJSON_AddItemToObject(obj,"m",content);
      cJSON_AddItemToArray(ja,obj);
    }
    {
      obj = cJSON_CreateObject();
      content = cJSON_CreateInt((int)fv->Mi[2]); cJSON_AddItemToObject(obj,"M",content);
      content = cJSON_CreateInt((int)fv->mi[2]); cJSON_AddItemToObject(obj,"m",content);
      cJSON_AddItemToArray(ja,obj);
    }
    
    jso_part = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_dm,"partition",jso_part);
    content = cJSON_CreateInt((int)commsize); cJSON_AddItemToObject(jso_part,"commSize",content);

    ja = cJSON_CreateArray();
    cJSON_AddItemToObject(jso_part,"directions",ja);
    {
      obj = cJSON_CreateObject();
      content = cJSON_CreateInt((int)ranks[0]);                                       cJSON_AddItemToObject(obj,"ranks",content);
      content = cJSON_CreateIntArray((const int*)fv->cell_ownership_i,(int)ranks[0]); cJSON_AddItemToObject(obj,"pointsPerRank",content);
      cJSON_AddItemToArray(ja,obj);
    }
    {
      obj = cJSON_CreateObject();
      content = cJSON_CreateInt((int)ranks[1]);                                       cJSON_AddItemToObject(obj,"ranks",content);
      content = cJSON_CreateIntArray((const int*)fv->cell_ownership_j,(int)ranks[1]); cJSON_AddItemToObject(obj,"pointsPerRank",content);
      cJSON_AddItemToArray(ja,obj);
    }
    {
      obj = cJSON_CreateObject();
      content = cJSON_CreateInt((int)ranks[2]);                                       cJSON_AddItemToObject(obj,"ranks",content);
      content = cJSON_CreateIntArray((const int*)fv->cell_ownership_k,(int)ranks[2]); cJSON_AddItemToObject(obj,"pointsPerRank",content);
      cJSON_AddItemToArray(ja,obj);
    }

    /* write json meta data file */
    {
      FILE *fp;
      char jfilename[PETSC_MAX_PATH_LEN];
      char *jbuff = cJSON_Print(jso_file);
      
      if (path) {
        if (prefix) { PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s/%s_fvda.json",path,prefix); }
        else { PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s/fvda.json",path); }
      } else {
        if (prefix) { PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s_fvda.json",prefix); }
        else { PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"fvda.json"); }
      }

      fp = fopen(jfilename,"w");
      if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open file %s",jfilename);
      fprintf(fp,"%s\n",jbuff);
      fclose(fp);
      free(jbuff);
    }
    cJSON_Delete(jso_file);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscVecWriteJSON(Vec x,PetscInt format,const char suffix[])
{
  MPI_Comm       comm;
  PetscMPIInt    commrank;
  PetscViewer    viewer;
  char           x_filename[PETSC_MAX_PATH_LEN];
  /*size_t         len;
  int            c;
  const char     *suffix_tail;*/
  PetscErrorCode ierr;

  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&commrank);CHKERRQ(ierr);

  /* strip out the path */
  /*
  ierr = PetscStrlen(suffix,&len);CHKERRQ(ierr);
  for (c=len-1; c>=0; c--) {
    if (suffix[c] == '/') { break; }
  }
  c++;
  suffix_tail = &suffix[c];
   
  if      (format == 0) { PetscSNPrintf(x_filename,PETSC_MAX_PATH_LEN-1,"%s.pbvec",suffix_tail); }
  else if (format == 1) { PetscSNPrintf(x_filename,PETSC_MAX_PATH_LEN-1,"%s.pbvec",suffix_tail); }
  else if (format == 2) { PetscSNPrintf(x_filename,PETSC_MAX_PATH_LEN-1,"%s.h5",suffix_tail); }
  else if (format == 3) { PetscSNPrintf(x_filename,PETSC_MAX_PATH_LEN-1,"%s.pbvec.gz",suffix_tail); }
  else SETERRQ1(comm,PETSC_ERR_SUP,"Format %D is not recognized and not supported",format);
  */
   
  if      (format == 0) { PetscSNPrintf(x_filename,PETSC_MAX_PATH_LEN-1,"%s.pbvec",suffix); }
  else if (format == 1) { PetscSNPrintf(x_filename,PETSC_MAX_PATH_LEN-1,"%s.pbvec",suffix); }
  else if (format == 2) { PetscSNPrintf(x_filename,PETSC_MAX_PATH_LEN-1,"%s.h5",suffix); }
  else if (format == 3) { PetscSNPrintf(x_filename,PETSC_MAX_PATH_LEN-1,"%s.pbvec.gz",suffix); }
  else SETERRQ1(comm,PETSC_ERR_SUP,"Format %D is not recognized and not supported",format);
  
  if (format == 0 || format == 3) {
    ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,x_filename);CHKERRQ(ierr);
    
    ierr = VecView(x,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  if (format == 1) {
#if defined(PETSC_HAVE_MPIIO)
    ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerBinarySetUseMPIIO(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,x_filename);CHKERRQ(ierr);
    
    ierr = VecView(x,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#else
    SETERRQ(comm,PETSC_ERR_SUP,"Petsc is not configured to use MPI-IO");
#endif
  }
  
  if (format == 2) SETERRQ(comm,PETSC_ERR_SUP,"Format 2 (HDF5) not supported");
  
  if (commrank == 0) {
    cJSON *jso_file = NULL,*jso_vec = NULL,*content;
    
    /* create json meta data file */
    jso_file = cJSON_CreateObject();
    
    jso_vec = cJSON_CreateObject();
    cJSON_AddItemToObject(jso_file,"PETScVec",jso_vec);

    content = cJSON_CreateString(x_filename);  cJSON_AddItemToObject(jso_vec,"fileName",content);
    
    /* real, complex */
#if defined(PETSC_USE_COMPLEX)
    content = cJSON_CreateString("complex");  cJSON_AddItemToObject(jso_vec,"scalarType",content);
#else
    content = cJSON_CreateString("real");  cJSON_AddItemToObject(jso_vec,"scalarType",content);
#endif
    
    /* float16, float32, float64, float128 */
#if defined(PETSC_USE_REAL___FP16)
    content = cJSON_CreateString("float16");  cJSON_AddItemToObject(jso_vec,"numberType",content);
#elif defined(PETSC_USE_REAL_SINGLE)
    content = cJSON_CreateString("float32");  cJSON_AddItemToObject(jso_vec,"numberType",content);
#elif defined(PETSC_USE_REAL___FLOAT128)
    content = cJSON_CreateString("float128");  cJSON_AddItemToObject(jso_vec,"numberType",content);
#else
    content = cJSON_CreateString("float64");  cJSON_AddItemToObject(jso_vec,"numberType",content);
#endif
    
    if (format == 0) {
      PetscInt byte_offset = 2 * sizeof(PetscInt); /* CLASSID, length of vector */
      
      content = cJSON_CreateString("petsc-binary");  cJSON_AddItemToObject(jso_vec,"dataFormat",content);
      content = cJSON_CreateString("big");  cJSON_AddItemToObject(jso_vec,"endian",content);
      content = cJSON_CreateInt((int)byte_offset);  cJSON_AddItemToObject(jso_vec,"byteOffset",content);
    }
    
    if (format == 1) {
      PetscInt byte_offset = 2 * sizeof(PetscInt); /* CLASSID, length of vector */
      
      content = cJSON_CreateString("petsc-binary-mpiio");  cJSON_AddItemToObject(jso_vec,"dataFormat",content);
      content = cJSON_CreateString("big");  cJSON_AddItemToObject(jso_vec,"endian",content);
      content = cJSON_CreateInt((int)byte_offset);  cJSON_AddItemToObject(jso_vec,"byteOffset",content);
    }
    
    if (format == 2) {
    }

    if (format == 3) {
      PetscInt byte_offset = 2 * sizeof(PetscInt); /* CLASSID, length of vector */
      
      content = cJSON_CreateString("gzip");  cJSON_AddItemToObject(jso_vec,"compressionLibrary",content);
      content = cJSON_CreateString("petsc-binary");  cJSON_AddItemToObject(jso_vec,"dataFormat",content);
      content = cJSON_CreateString("big");  cJSON_AddItemToObject(jso_vec,"endian",content);
      content = cJSON_CreateInt((int)byte_offset);  cJSON_AddItemToObject(jso_vec,"byteOffset",content);
    }

    {
      FILE *fp;
      char jfilename[PETSC_MAX_PATH_LEN];
      char *jbuff = cJSON_Print(jso_file);
      
      PetscSNPrintf(jfilename,PETSC_MAX_PATH_LEN-1,"%s.json",suffix);
      fp = fopen(jfilename,"w");
      if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open file %s",jfilename);
      fprintf(fp,"%s\n",jbuff);
      fclose(fp);
      free(jbuff);
    }
    cJSON_Delete(jso_file);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAView_Heavy(FVDA fv,const char path[],const char suffix[])
{
  Vec            x_cell,xn_cell;
  PetscInt       m,f;
  PetscReal      *_x_cell;
  char           fname1[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  
  
  if (path) {
    if (suffix) { PetscSNPrintf(fname1,PETSC_MAX_PATH_LEN-1,"%s/%s_fvda_cellcoeff",path,suffix); }
    else { PetscSNPrintf(fname1,PETSC_MAX_PATH_LEN-1,"%s/fvda_cellcoeff",path); }
  } else {
    if (suffix) { PetscSNPrintf(fname1,PETSC_MAX_PATH_LEN-1,"%s_fvda_cellcoeff",suffix); }
    else { PetscSNPrintf(fname1,PETSC_MAX_PATH_LEN-1,"fvda_cellcoeff"); }
  }

  ierr = DMCreateGlobalVector(fv->dm_fv,&x_cell);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x_cell,&m);CHKERRQ(ierr);
  ierr = DMDACreateNaturalVector(fv->dm_fv,&xn_cell);CHKERRQ(ierr);
  
  /* pack, scatter, write */
  for (f=0; f<fv->ncoeff_cell; f++) {
    PetscInt        k,b,bs = fv->cell_coeff_size[f] / fv->ncells;
    const PetscReal *_cell_coefficient = fv->cell_coefficient[f];
    
    for (b=0; b<bs; b++) {
      ierr = VecGetArray(x_cell,&_x_cell);CHKERRQ(ierr);
      for (k=0; k<m; k++) { _x_cell[k] = _cell_coefficient[bs*k + b]; }
      ierr = VecRestoreArray(x_cell,&_x_cell);CHKERRQ(ierr);
      
      ierr = DMDAGlobalToNaturalBegin(fv->dm_fv,x_cell,INSERT_VALUES,xn_cell);CHKERRQ(ierr);
      ierr = DMDAGlobalToNaturalEnd(fv->dm_fv,x_cell,INSERT_VALUES,xn_cell);CHKERRQ(ierr);

      if (bs == 1) { PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_%s",fname1,fv->cell_coeff_name[f]); }
      else {         PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_%s_bs%D",fname1,fv->cell_coeff_name[f],b); }
      ierr = PetscVecWriteJSON(xn_cell,0,fname);CHKERRQ(ierr);
    }
  }
  
  ierr = VecDestroy(&xn_cell);CHKERRQ(ierr);
  ierr = VecDestroy(&x_cell);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
