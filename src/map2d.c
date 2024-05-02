
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "ptatin3d.h"
#include "map2d.h"

void Map2dLoad(const char filename[],Map2d *map);
void Map2dParseHeader(Map2d phasemap,FILE *fp);
void Map2dReadDataFromFile_ASCII(Map2d map,FILE *fp);
void Map2dReadDataFromFile_BINARY(Map2d map,FILE *fp);

void Map2dCreate(Map2d *map)
{
	Map2d pm;
	pm = malloc(sizeof(struct _p_Map2d));
	memset(pm,0,sizeof(struct _p_Map2d));
	*map = pm;
}

void Map2dDestroy(Map2d *map)
{
	Map2d pm;

	if (map==NULL) { return; }
	pm = *map;
	
	if (pm->data!=NULL) {
		free(pm->data);
		pm->data = NULL;
	}
	*map = NULL;
}

void Map2dGetIndex(Map2d pm,const int i,const int j, int *index)
{
	if (i<0) { printf("ERROR(%s): i = %d  <0 \n", __func__, i ); exit(EXIT_FAILURE); }
	if (j<0) { printf("ERROR(%s): j = %d < 0 \n", __func__, j ); exit(EXIT_FAILURE); }
	if (i>=pm->mx) { printf("ERROR(%s): i = %d > %d\n", __func__, i, pm->mx ); exit(EXIT_FAILURE); }
	if (j>=pm->my) { printf("ERROR(%s): j = %d > %d\n", __func__, j, pm->my ); exit(EXIT_FAILURE); }
	
	*index = i + j * pm->mx;
}

void Map2dGetIndexFromCoordinate(Map2d phasemap,double xp[],int *index)
{
	int i,j;
	
	(*index) = (int)MAP2D_POINT_OUTSIDE;
	
	if (xp[0] < phasemap->x0) { return; }
	if (xp[0] > phasemap->x1) { return; }
	if (xp[1] < phasemap->y0) { return; }
	if (xp[1] > phasemap->y1) { return; }
	
	i = (xp[0] - phasemap->x0)/phasemap->dx;
	j = (xp[1] - phasemap->y0)/phasemap->dy;
	if (i==phasemap->mx) { i--; }
	if (j==phasemap->my) { j--; }
	
	Map2dGetIndex(phasemap,i,j,index);
}

void Map2dGetValueFromCoordinate(Map2d map,double xp[],void **data)
{
	int index;
	void *ref;
	int is_valid;

	Map2dGetIndexFromCoordinate(map,xp,&index);
	
	Map2dGetValueFromIndex(map,index,data);
}

void Map2dCheckValidity(Map2d phasemap,int index,int *is_valid)
{
	*is_valid = 0;
	
	if ( (index>=0) && (index<phasemap->mx*phasemap->my) ) {
		*is_valid = 1;
	} else if ( index==(int)MAP2D_POINT_OUTSIDE ) {
		*is_valid = 0;
	} else {
		printf("Unknown state of index (%d) \n",index);
		exit(0);
	}
}

void Map2dGetValueFromIndex(Map2d map,int index,void **data)
{
	void *ref;
	int is_valid;
	
	Map2dCheckValidity(map,index,&is_valid);
	if (is_valid == 0) {
		printf("ERROR(%s): index=%d is not valid \n", __func__, index ); exit(EXIT_FAILURE);
	}
	
	switch (map->data_type) {
		case MAP2D_INT:
		{
			ref = (void*)( (char*)map->data + index * map->ncomponents * sizeof(int) );
		}
			break;
			
		case MAP2D_FLOAT:
		{
			ref = (void*)( (char*)map->data + index * map->ncomponents * sizeof(float) );
		}
			break;
			
		case MAP2D_DOUBLE:
		{
			ref = (void*)( (char*)map->data + index * map->ncomponents * sizeof(double) );
		}
			break;
	}	
	
	*data = ref;
}


void Map2dLoadFromFile(const char filename[],Map2d *map)
{
	size_t len;
	int is_zipped;
	int matched_extension;
	
	is_zipped = 0;

	/* check extensions for common zipped file extensions */
	len = strlen(filename);
	matched_extension = strcmp(&filename[len-8],".tar.gz");
	if (matched_extension == 0) {
		printf("  Detected .tar.gz\n");
		is_zipped = 1;
	}
	matched_extension = strcmp(&filename[len-5],".tgz");
	if (matched_extension == 0) {
		printf("  Detected .tgz\n");
		is_zipped = 1;
	}
	matched_extension = strcmp(&filename[len-3],".Z");
	if (matched_extension == 0) {
		printf("  Detected .Z\n");
		is_zipped = 1;
	}

	if (is_zipped == 1) {
		printf("Zipped loading is not supported");
		exit(0);
	} else {
		Map2dLoad(filename,map);
	}
}

void Map2dParseHeader(Map2d phasemap,FILE *fp)
{
  char dummy[1000];
	int length,i_value;
	size_t data_size;
	
	/* read header information, mx,my,x0,y0,x1,y1 */
	//  fscanf(fp,"%s\n",dummy);
	//fgets(dummy,sizeof(dummy),fp);
  fscanf(fp,"%d\n",&phasemap->mx);
  fscanf(fp,"%d\n",&phasemap->my);
  fscanf(fp,"%d\n",&phasemap->ncomponents);

	fscanf(fp,"%d\n",&i_value);
	phasemap->data_type = (MapDataType)i_value;
  
  fscanf(fp,"%d\n",&i_value);
	phasemap->storage = (MapFileStorageType)i_value;
  
	fscanf(fp,"%lf %lf %lf %lf\n",&phasemap->x0,&phasemap->y0,&phasemap->x1,&phasemap->y1);
	
	/* set params */
	phasemap->dx = (phasemap->x1 - phasemap->x0)/(double)(phasemap->mx);
	phasemap->dy = (phasemap->y1 - phasemap->y0)/(double)(phasemap->my);
		
	/* allocate data */
	length = phasemap->mx * phasemap->my * phasemap->ncomponents;
	switch (phasemap->data_type) {
		case MAP2D_INT:
			data_size = sizeof(int);
			break;
		case MAP2D_FLOAT:
			data_size = sizeof(float);
			break;
		case MAP2D_DOUBLE:
			data_size = sizeof(double);
			break;
	}
	phasemap->data = malloc( data_size * length );		
}

void Map2dReadDataFromFile_ASCII(Map2d map,FILE *fp)
{
	int index,i,j,d;
	
	
  index = 0;
	switch (map->data_type) {
		case MAP2D_INT:
		{
			int i_value,*ref;
			
			for (j=0; j<map->my; j++) {
				for (i=0; i<map->mx; i++) {
					for (d=0; d<map->ncomponents; d++) {
						fscanf(fp,"%d",&i_value);
						ref = (int*)( (char*)map->data + index * sizeof(int) );
						*ref = i_value;
						index++; 
					}
				}
			}
		}
			break;
			
		case MAP2D_FLOAT:
		{
			float i_value,*ref;
			
			for (j=0; j<map->my; j++) {
				for (i=0; i<map->mx; i++) {
					for (d=0; d<map->ncomponents; d++) {
						fscanf(fp,"%f",&i_value);
						ref = (float*)( (char*)map->data + index * sizeof(float) );
						*ref = i_value;
						index++; 
					}
				}
			}
		}
			break;
			
		case MAP2D_DOUBLE:
		{
			double i_value,*ref;
			
			for (j=0; j<map->my; j++) {
				for (i=0; i<map->mx; i++) {
					for (d=0; d<map->ncomponents; d++) {
						fscanf(fp,"%lf",&i_value);
						ref = (double*)( (char*)map->data + index * sizeof(double) );
						*ref = i_value;
						index++; 
					}
				}
			}
		}
			break;
	}	
}

void Map2dReadDataFromFile_BINARY(Map2d map,FILE *fp)
{
	int length;
	
	length = map->mx * map->my * map->ncomponents;
	
	switch (map->data_type) {
		case MAP2D_INT:
		{
			int *ref;
			
			ref = (int*)( map->data );
			fread( ref, sizeof(int), length, fp);
		}
			break;
			
		case MAP2D_FLOAT:
		{
			float *ref;
			
			ref = (float*)( map->data );
			fread( ref, sizeof(float), length, fp);
		}
			break;
			
		case MAP2D_DOUBLE:
		{
			double *ref;
			
			ref = (double*)( map->data );
			fread( ref, sizeof(double), length, fp);
		}
			break;
	}	
}

void Map2dLoad(const char filename[],Map2d *map)
{
	FILE *fp = NULL;
	Map2d phasemap;
  
	/* open file to parse */
	fp = fopen(filename,"r");
	if (fp==NULL) {
		printf("Error(%s): Could not open file: %s \n",__func__, filename );
		exit(EXIT_FAILURE);
	}
	
	/* create data structure */
	Map2dCreate(&phasemap);
	
	Map2dParseHeader(phasemap,fp);

	/* parse data from file */
	switch (phasemap->storage) {
		
		case MAP2D_ASCII:
			Map2dReadDataFromFile_ASCII(phasemap,fp);
			break;
		
		case MAP2D_BINARY:
			Map2dReadDataFromFile_BINARY(phasemap,fp);
			break;
	}
	
	
	/* set pointer */
	*map = phasemap;
	fclose(fp);
}

void Map2dViewGnuplot(const char filename[],Map2d phasemap)
{
	FILE *fp = NULL;
	int i,j,d;
	
	/* open file to parse */
	fp = fopen(filename,"w");
	if (fp==NULL) {
		printf("Error(%s): Could not open file: %s \n",__func__, filename );
		exit(EXIT_FAILURE);
	}
	fprintf(fp,"# Map2d information \n");
	fprintf(fp,"# Map2d : (x0,y0) = (%1.4e,%1.4e) \n",phasemap->x0,phasemap->y0);
	fprintf(fp,"# Map2d : (x1,y1) = (%1.4e,%1.4e) \n",phasemap->x1,phasemap->y1);
	fprintf(fp,"# Map2d : (dx,dy) = (%1.4e,%1.4e) \n",phasemap->dx,phasemap->dy);
	fprintf(fp,"# Map2d : (mx,my) = (%d,%d) \n",phasemap->mx,phasemap->my);
	fprintf(fp,"# Map2d : ncomp = %d \n",phasemap->ncomponents);
	
	for (j=0; j<phasemap->my; j++) {
		for (i=0; i<phasemap->mx; i++) {
			double xp[2];
			int index;
			
			xp[0] = phasemap->x0 + i * phasemap->dx;
			xp[1] = phasemap->y0 + j * phasemap->dy;
			fprintf(fp,"%lf %lf ", xp[0],xp[1] );

			switch (phasemap->data_type) {
				case MAP2D_INT:
				{
					int *data;
					
					Map2dGetValueFromCoordinate(phasemap,xp,(void**)&data);
					for (d=0; d<phasemap->ncomponents; d++) {
						fprintf(fp,"%d ", data[d] );
					}
				}
					break;
					
				case MAP2D_FLOAT:
				{
					float *data;
					
					Map2dGetValueFromCoordinate(phasemap,xp,(void**)&data);
					for (d=0; d<phasemap->ncomponents; d++) {
						fprintf(fp,"%1.4e ", data[d] );
					}
				}
					break;
					
				case MAP2D_DOUBLE:
				{
					double *data;
					
					Map2dGetValueFromCoordinate(phasemap,xp,(void**)&data);
					for (d=0; d<phasemap->ncomponents; d++) {
						fprintf(fp,"%1.4e ", data[d] );
					}
				}
					break;
			}	
			
			
			
			fprintf(fp,"\n" );
		}fprintf(fp,"\n");
	}
	fclose(fp);
	
}
