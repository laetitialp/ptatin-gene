
#ifndef __MAP2D_H__
#define __MAP2D_H__

typedef enum { MAP2D_POINT_OUTSIDE=-1, MAP2D_POINT_INSIDE=1 } MapLocationIndicator;
typedef enum { MAP2D_INT=0, MAP2D_FLOAT=1, MAP2D_DOUBLE=2 } MapDataType;
typedef enum { MAP2D_ASCII=0, MAP2D_BINARY=1 } MapFileStorageType;


typedef struct _p_Map2d *Map2d;
struct _p_Map2d {
	double x0,y0,x1,y1;
	int mx,my;
	int ncomponents;
	MapFileStorageType storage;
	MapDataType data_type;
	double dx,dy;
	void *data;
};

void Map2dCreate(Map2d *map);
void Map2dDestroy(Map2d *map);
void Map2dGetIndex(Map2d pm,const int i,const int j, int *index);

void Map2dGetIndexFromCoordinate(Map2d phasemap,double xp[],int *index);
void Map2dGetValueFromCoordinate(Map2d map,double xp[],void **data);
void Map2dGetValueFromIndex(Map2d map,int index,void **data);

void Map2dCheckValidity(Map2d phasemap,int index,int *is_valid);
void Map2dLoadFromFile(const char filename[],Map2d *map);
void Map2dViewGnuplot(const char filename[],Map2d phasemap);


#endif