
#ifndef __parse_h__
#define __parse_h__

#include <mpi.h>

typedef struct _p_CellPartition *CellPartition;
typedef struct _p_Mesh *Mesh;

struct _p_CellPartition {
  double cmin[3],cmax[3];
  int *cell_list;
  int ncell;
};

struct _p_Mesh {
  CellPartition *partition;
  double *vert;
  int *cell;
  int nvert,ncell,coor_dim,points_per_cell;
  int npartition;
};

void CellPartitionCreate(CellPartition *_c);
void CellPartitionDestroy(CellPartition *_c);
void MeshCreate(Mesh *_m);
void MeshDestroy(Mesh *_m);
void MeshView(Mesh m);
void parse_mesh(MPI_Comm comm,const char filename[],Mesh *m);
void parse_field(MPI_Comm comm,Mesh m,const char filename[],char ftypevoid,void **_data);


#endif
