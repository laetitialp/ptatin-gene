
//
// fe_form_compiler.py version: 3d6089bab1f9f4b2826271e39b0d89d4acff6542
// sympy version: 1.13.3
// using common substring elimination: True
// form file: neumann.py version: 7ae8c5b5d1a0e57c18a8da341ee5de59af5212e2
//

#include <stdio.h>
#include <math.h>
// ---------------------------------------------------
//
// key: wu ==> NOTHING TO DO
//
// ---------------------------------------------------

// ---------------------------------------------------
//
// key: wp
//
// ---------------------------------------------------
// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_asmb_wp(
double wNt[],
double pN[],
double n[],  // parameter
double scale, double A[])
{
  int i,j;
  double __Aij[3];
  
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      double tce0 = pN[j]*wNt[i];
      __Aij[0] = n[0]*tce0;
      __Aij[1] = n[1]*tce0;
      __Aij[2] = n[2]*tce0;
      A[(3*i + 0)*4 + (1*j + 0)] += scale * ( __Aij[0] );
      A[(3*i + 1)*4 + (1*j + 0)] += scale * ( __Aij[1] );
      A[(3*i + 2)*4 + (1*j + 0)] += scale * ( __Aij[2] );
  }}
}

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_spmv_wp(
double wNt[], 
double pN[],
double p0[],
double n[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double p0j_pNj = 0.0;
  for (j=0; j<4; j++) { // p_nbasis_0
    p0j_pNj += p0[j]*pN[j];
  }
  
  for (i=0; i<27; i++) { // w_nbasis
    double tce0 = p0j_pNj*wNt[i];
    __Fi[0] = n[0]*tce0;
    __Fi[1] = n[1]*tce0;
    __Fi[2] = n[2]*tce0;
    F[3*i + 0] += scale * ( __Fi[0] );
    F[3*i + 1] += scale * ( __Fi[1] );
    F[3*i + 2] += scale * ( __Fi[2] );
  }
}

// ---------------------------------------------------
//
// key: qu NOTHING TO DO
//
// ---------------------------------------------------

// ---------------------------------------------------
//
// key: qp NOTHING TO DO
//
// ---------------------------------------------------

// ---------------------------------------------------
//
// key: w_up
//
// ---------------------------------------------------
// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// trial function[1] coeff:  [p0[j]]
// trial function[1]:        pN[j]
// trial function[1] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[1] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_spmv_w_up(
double wNt[],
double pN[],
double p0[],
double n[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double p0j_pNj = 0.0;
  for (j=0; j<4; j++) { // p_nbasis_1
    p0j_pNj += p0[j]*pN[j];
  }
  
  for (i=0; i<27; i++) { // w_nbasis
    double tce0 = p0j_pNj*wNt[i];
    __Fi[0] = n[0]*tce0;
    __Fi[1] = n[1]*tce0;
    __Fi[2] = n[2]*tce0;
    F[3*i + 0] += scale * ( __Fi[0] );
    F[3*i + 1] += scale * ( __Fi[1] );
    F[3*i + 2] += scale * ( __Fi[2] );
  }
}
// ---------------------------------------------------
//
// key: q_up NOTHING TO DO
//
// ---------------------------------------------------

//
// key: w
//

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
//
// trial function[1] coeff:  [p0[j]]
// trial function[1]:        pN[j]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[1] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_residual_w(
double wNt[],
double pN[],
double p0[],
double traction[],  // parameter
double n[],  // parameter
double t[],  // parameter
double s[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double p0j_pNj = 0.0;
  for (j=0; j<4; j++) { // p_nbasis_1
    p0j_pNj += p0[j]*pN[j];
  }
  
  for (i=0; i<27; i++) { // w_nbasis
    double tce0 = traction[2]*wNt[i];
    double tce1 = traction[1]*wNt[i];
    __Fi[0] = n[0]*p0j_pNj*wNt[i] - n[0]*traction[0]*wNt[i] - s[0]*tce0 - t[0]*tce1;
    __Fi[1] = n[1]*p0j_pNj*wNt[i] - n[1]*traction[0]*wNt[i] - s[1]*tce0 - t[1]*tce1;
    __Fi[2] = n[2]*p0j_pNj*wNt[i] - n[2]*traction[0]*wNt[i] - s[2]*tce0 - t[2]*tce1;
    F[3*i + 0] += scale * ( __Fi[0] );
    F[3*i + 1] += scale * ( __Fi[1] );
    F[3*i + 2] += scale * ( __Fi[2] );
  }
}

//
// key: q NOTHING TO DO
//
