


#error(<<REMOVED AUTOGENERATED TAG>> ================== FILE [QPntVolCoefEnergy_def.c] ==================)
/*
  Auto generated by version 0.0 of swarm_class_generator.py
  on otsu.local, at 2013-02-02 20:59:58.571203 by dmay
*/


#ifndef __QPntVolCoefEnergy_DEF_H__
#define __QPntVolCoefEnergy_DEF_H__

typedef struct {
  double diffusivity ;
  double heat_source ;
} QPntVolCoefEnergy ;


typedef enum {
  QPVCEgy_diffusivity = 0,
  QPVCEgy_heat_source
} QPntVolCoefEnergyTypeName ;


extern const char QPntVolCoefEnergy_classname[];

extern const int QPntVolCoefEnergy_nmembers;

extern const size_t QPntVolCoefEnergy_member_sizes[];

extern const char *QPntVolCoefEnergy_member_names[];

/* prototypes */
void QPntVolCoefEnergyGetField_diffusivity(QPntVolCoefEnergy *point,double *data);
void QPntVolCoefEnergyGetField_heat_source(QPntVolCoefEnergy *point,double *data);
void QPntVolCoefEnergySetField_diffusivity(QPntVolCoefEnergy *point,double data);
void QPntVolCoefEnergySetField_heat_source(QPntVolCoefEnergy *point,double data);
void QPntVolCoefEnergyView(QPntVolCoefEnergy *point);
void QPntVolCoefEnergyVTKWriteAsciiAllFields(FILE *vtk_fp,const int N,const QPntVolCoefEnergy points[]);
void QPntVolCoefEnergyPVTUWriteAllPPointDataFields(FILE *vtk_fp);
void QPntVolCoefEnergyVTKWriteBinaryAppendedHeaderAllFields(FILE *vtk_fp,int *offset,const int N,const QPntVolCoefEnergy points[]);
void QPntVolCoefEnergyVTKWriteBinaryAppendedDataAllFields(FILE *vtk_fp,const int N,const QPntVolCoefEnergy points[]);

#endif
