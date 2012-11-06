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
 **    Filename:      MaterialConst_PlasticMises_def.h
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
 **    $Id$
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@*/
/*
  Auto generated by version 0.0 of mat_prop_class_generator.py
  on otsu.local, at 2012-08-27 10:57:20.384973 by dmay
*/


#ifndef __MaterialConst_PlasticMises_DEF_H__
#define __MaterialConst_PlasticMises_DEF_H__

typedef struct {
  double tau_yield ;
  double tau_yield_inf ;
} MaterialConst_PlasticMises ;


typedef enum {
  PlasticMises_yield_stress = 0,
  PlasticMises_yield_stress_inf
} MaterialConst_PlasticMisesTypeName ;


extern const char MaterialConst_PlasticMises_classname[];

extern const char MaterialConst_PlasticMises_classname_short[];

extern const int MaterialConst_PlasticMises_nmembers;

extern const size_t MaterialConst_PlasticMises_member_sizes[];

extern const char *MaterialConst_PlasticMises_member_names_short[];

extern const char *MaterialConst_PlasticMises_member_names[];

extern const size_t MaterialConst_PlasticMises_member_byte_offset[];

/* prototypes */
void MaterialConst_PlasticMisesGetField_yield_stress(MaterialConst_PlasticMises *point,double *data);
void MaterialConst_PlasticMisesGetField_yield_stress_inf(MaterialConst_PlasticMises *point,double *data);
void MaterialConst_PlasticMisesSetField_yield_stress(MaterialConst_PlasticMises *point,double data);
void MaterialConst_PlasticMisesSetField_yield_stress_inf(MaterialConst_PlasticMises *point,double data);
void MaterialConst_PlasticMisesView(MaterialConst_PlasticMises *point);

#endif
