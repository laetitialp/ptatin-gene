
//
// fe_form_compiler.py version: 3d6089bab1f9f4b2826271e39b0d89d4acff6542
// sympy version: 1.13.3
// using common substring elimination: True
// form file: neumann.py version: 7ae8c5b5d1a0e57c18a8da341ee5de59af5212e2
//

#include <stdio.h>
#include <math.h>

#if 1
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
#endif

#if 0
//
// key: wu
//

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_asmb_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double scale, double A[])
{
  int i,j;
  double __Aij[9];
  
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      
      __Aij[0] = 0;
      __Aij[1] = 0;
      __Aij[2] = 0;
      __Aij[3] = 0;
      __Aij[4] = 0;
      __Aij[5] = 0;
      __Aij[6] = 0;
      __Aij[7] = 0;
      __Aij[8] = 0;
      A[(3*i + 0)*81 + (3*j + 0)] += scale * ( __Aij[0] );
      A[(3*i + 0)*81 + (3*j + 1)] += scale * ( __Aij[1] );
      A[(3*i + 0)*81 + (3*j + 2)] += scale * ( __Aij[2] );
      A[(3*i + 1)*81 + (3*j + 0)] += scale * ( __Aij[3] );
      A[(3*i + 1)*81 + (3*j + 1)] += scale * ( __Aij[4] );
      A[(3*i + 1)*81 + (3*j + 2)] += scale * ( __Aij[5] );
      A[(3*i + 2)*81 + (3*j + 0)] += scale * ( __Aij[6] );
      A[(3*i + 2)*81 + (3*j + 1)] += scale * ( __Aij[7] );
      A[(3*i + 2)*81 + (3*j + 2)] += scale * ( __Aij[8] );
  }}
}

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_asmbdiag_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double scale, double F[])
{
  int i,j;
  double __Aij[3];
  
  for (i=0; i<27; i++) { // w_nbasis
    j = i;
    
    __Aij[0] = 0;
    __Aij[1] = 0;
    __Aij[2] = 0;
    F[3*i + 0] += scale * ( __Aij[0] );
    F[3*i + 1] += scale * ( __Aij[1] );
    F[3*i + 2] += scale * ( __Aij[2] );
  }
}

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_spmv_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double u0[], double u1[], double u2[],
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  
  for (i=0; i<27; i++) { // w_nbasis
    
    __Fi[0] = 0;
    __Fi[1] = 0;
    __Fi[2] = 0;
    F[3*i + 0] += scale * ( __Fi[0] );
    F[3*i + 1] += scale * ( __Fi[1] );
    F[3*i + 2] += scale * ( __Fi[2] );
  }
}

//
// key: wp
//

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
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
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
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
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

//
// key: qu
//

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_asmb_qu(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double scale, double A[])
{
  int i,j;
  double __Aij[3];
  
  for (i=0; i<4; i++) { // q_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      
      __Aij[0] = 0;
      __Aij[1] = 0;
      __Aij[2] = 0;
      A[(1*i + 0)*81 + (3*j + 0)] += scale * ( __Aij[0] );
      A[(1*i + 0)*81 + (3*j + 1)] += scale * ( __Aij[1] );
      A[(1*i + 0)*81 + (3*j + 2)] += scale * ( __Aij[2] );
  }}
}

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_spmv_qu(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double u0[], double u1[], double u2[],
double scale, double F[])
{
  int i,j;
  double __Fi[1];
  
  for (i=0; i<4; i++) { // q_nbasis
    
    __Fi[0] = 0;
    F[1*i + 0] += scale * ( __Fi[0] );
  }
}

//
// key: qp
//

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_asmb_qp(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double scale, double A[])
{
  int i,j;
  double __Aij[1];
  
  for (i=0; i<4; i++) { // q_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      
      __Aij[0] = 0;
      A[(1*i + 0)*4 + (1*j + 0)] += scale * ( __Aij[0] );
  }}
}

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_asmbdiag_qp(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double scale, double F[])
{
  int i,j;
  double __Aij[1];
  
  for (i=0; i<4; i++) { // q_nbasis
    j = i;
    
    __Aij[0] = 0;
    F[1*i + 0] += scale * ( __Aij[0] );
  }
}

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_spmv_qp(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double p0[],
double scale, double F[])
{
  int i,j;
  double __Fi[1];
  
  for (i=0; i<4; i++) { // q_nbasis
    
    __Fi[0] = 0;
    F[1*i + 0] += scale * ( __Fi[0] );
  }
}

//
// key: w_up
//

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
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
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

//
// key: q_up
//

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// trial function[1] coeff:  [p0[j]]
// trial function[1]:        pN[j]
// trial function[1] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[1] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_spmv_q_up(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double scale, double F[])
{
  int i,j;
  double __Fi[1];
  
  for (i=0; i<4; i++) { // q_nbasis
    
    __Fi[0] = 0;
    F[1*i + 0] += scale * ( __Fi[0] );
  }
}

//
// key: w
//

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
void neumann_deviatoric_nts_q2_3d_residual_w(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
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
// key: q
//

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// trial function[1] coeff:  [p0[j]]
// trial function[1]:        pN[j]
// trial function[1] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[1] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void neumann_deviatoric_nts_q2_3d_residual_q(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double scale, double F[])
{
  int i,j;
  double __Fi[1];
  
  for (i=0; i<4; i++) { // q_nbasis
    
    __Fi[0] = 0;
    F[1*i + 0] += scale * ( __Fi[0] );
  }
}
#endif