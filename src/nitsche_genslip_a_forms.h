
//
// fe_form_compiler.py version: ba26980b8db4ac4412a8ded988cf48aa986fbc80
// sympy version: 1.6.1
// using common substring elimination: True
// form file: nitsche-custom-h_IJ.py version: cd6c585d0922009ee6b85bfb39c0efd122a5046c
//

#include <stdio.h>
#include <math.h>

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
void nitsche_custom_h_a_q2_3d_asmb_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double gamma,  // parameter
double nhat[],  // parameter
double scale, double A[])
{
  int i,j;
  double __Aij[9];
  
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      
      __Aij[0] = gamma*pow(nhat[0], 2)*uN[j]*wNt[i];
      __Aij[1] = gamma*nhat[0]*nhat[1]*uN[j]*wNt[i];
      __Aij[2] = gamma*nhat[0]*nhat[2]*uN[j]*wNt[i];
      __Aij[3] = gamma*nhat[0]*nhat[1]*uN[j]*wNt[i];
      __Aij[4] = gamma*pow(nhat[1], 2)*uN[j]*wNt[i];
      __Aij[5] = gamma*nhat[1]*nhat[2]*uN[j]*wNt[i];
      __Aij[6] = gamma*nhat[0]*nhat[2]*uN[j]*wNt[i];
      __Aij[7] = gamma*nhat[1]*nhat[2]*uN[j]*wNt[i];
      __Aij[8] = gamma*pow(nhat[2], 2)*uN[j]*wNt[i];
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
void nitsche_custom_h_a_q2_3d_asmbdiag_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double gamma,  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Aij[3];
  
  for (i=0; i<27; i++) { // w_nbasis
    j = i;
    
    __Aij[0] = gamma*pow(nhat[0], 2)*uN[j]*wNt[i];
    __Aij[1] = gamma*pow(nhat[1], 2)*uN[j]*wNt[i];
    __Aij[2] = gamma*pow(nhat[2], 2)*uN[j]*wNt[i];
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
void nitsche_custom_h_a_q2_3d_spmv_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double u0[], double u1[], double u2[],
double gamma,  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  double aux0 = nhat[0]*nhat[1];
  for (i=0; i<27; i++) { // w_nbasis
    
    __Fi[0] = aux0*gamma*u1j_uNj*wNt[i] + gamma*pow(nhat[0], 2)*u0j_uNj*wNt[i] + gamma*nhat[0]*nhat[2]*u2j_uNj*wNt[i];
    __Fi[1] = aux0*gamma*u0j_uNj*wNt[i] + gamma*pow(nhat[1], 2)*u1j_uNj*wNt[i] + gamma*nhat[1]*nhat[2]*u2j_uNj*wNt[i];
    __Fi[2] = gamma*nhat[0]*nhat[2]*u0j_uNj*wNt[i] + gamma*nhat[1]*nhat[2]*u1j_uNj*wNt[i] + gamma*pow(nhat[2], 2)*u2j_uNj*wNt[i];
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
void nitsche_custom_h_a_q2_3d_asmb_wp(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double n[],  // parameter
double scale, double A[])
{
  int i,j;
  double __Aij[3];
  
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      
      __Aij[0] = n[0]*pN[j]*wNt[i];
      __Aij[1] = n[1]*pN[j]*wNt[i];
      __Aij[2] = n[2]*pN[j]*wNt[i];
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
void nitsche_custom_h_a_q2_3d_spmv_wp(
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
    
    __Fi[0] = n[0]*p0j_pNj*wNt[i];
    __Fi[1] = n[1]*p0j_pNj*wNt[i];
    __Fi[2] = n[2]*p0j_pNj*wNt[i];
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
void nitsche_custom_h_a_q2_3d_asmb_qu(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double n[],  // parameter
double nhat[],  // parameter
double scale, double A[])
{
  int i,j;
  double __Aij[3];
  double aux0 = nhat[0]*nhat[1];
  for (i=0; i<4; i++) { // q_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      
      __Aij[0] = aux0*n[1]*qNt[i]*uN[j] + n[0]*pow(nhat[0], 2)*qNt[i]*uN[j] + n[2]*nhat[0]*nhat[2]*qNt[i]*uN[j];
      __Aij[1] = aux0*n[0]*qNt[i]*uN[j] + n[1]*pow(nhat[1], 2)*qNt[i]*uN[j] + n[2]*nhat[1]*nhat[2]*qNt[i]*uN[j];
      __Aij[2] = n[0]*nhat[0]*nhat[2]*qNt[i]*uN[j] + n[1]*nhat[1]*nhat[2]*qNt[i]*uN[j] + n[2]*pow(nhat[2], 2)*qNt[i]*uN[j];
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
void nitsche_custom_h_a_q2_3d_spmv_qu(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double u0[], double u1[], double u2[],
double n[],  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[1];
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  double aux0 = nhat[1]*u1j_uNj;
  double aux1 = nhat[2]*u2j_uNj;
  double aux2 = nhat[0]*u0j_uNj;
  for (i=0; i<4; i++) { // q_nbasis
    
    __Fi[0] = aux0*n[0]*nhat[0]*qNt[i] + aux0*n[2]*nhat[2]*qNt[i] + aux1*n[0]*nhat[0]*qNt[i] + aux1*n[1]*nhat[1]*qNt[i] + aux2*n[1]*nhat[1]*qNt[i] + aux2*n[2]*nhat[2]*qNt[i] + n[0]*pow(nhat[0], 2)*qNt[i]*u0j_uNj + n[1]*pow(nhat[1], 2)*qNt[i]*u1j_uNj + n[2]*pow(nhat[2], 2)*qNt[i]*u2j_uNj;
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
void nitsche_custom_h_a_q2_3d_asmb_qp(
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
void nitsche_custom_h_a_q2_3d_asmbdiag_qp(
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
void nitsche_custom_h_a_q2_3d_spmv_qp(
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
void nitsche_custom_h_a_q2_3d_spmv_w_up(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double gamma,  // parameter
double n[],  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double p0j_pNj = 0.0;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  for (j=0; j<4; j++) { // p_nbasis_1
    p0j_pNj += p0[j]*pN[j];
  }
  double aux0 = nhat[0]*nhat[1];
  for (i=0; i<27; i++) { // w_nbasis
    
    __Fi[0] = aux0*gamma*u1j_uNj*wNt[i] + gamma*pow(nhat[0], 2)*u0j_uNj*wNt[i] + gamma*nhat[0]*nhat[2]*u2j_uNj*wNt[i] + n[0]*p0j_pNj*wNt[i];
    __Fi[1] = aux0*gamma*u0j_uNj*wNt[i] + gamma*pow(nhat[1], 2)*u1j_uNj*wNt[i] + gamma*nhat[1]*nhat[2]*u2j_uNj*wNt[i] + n[1]*p0j_pNj*wNt[i];
    __Fi[2] = gamma*nhat[0]*nhat[2]*u0j_uNj*wNt[i] + gamma*nhat[1]*nhat[2]*u1j_uNj*wNt[i] + gamma*pow(nhat[2], 2)*u2j_uNj*wNt[i] + n[2]*p0j_pNj*wNt[i];
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
void nitsche_custom_h_a_q2_3d_spmv_q_up(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double n[],  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[1];
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  double aux0 = nhat[1]*u1j_uNj;
  double aux1 = nhat[2]*u2j_uNj;
  double aux2 = nhat[0]*u0j_uNj;
  for (i=0; i<4; i++) { // q_nbasis
    
    __Fi[0] = aux0*n[0]*nhat[0]*qNt[i] + aux0*n[2]*nhat[2]*qNt[i] + aux1*n[0]*nhat[0]*qNt[i] + aux1*n[1]*nhat[1]*qNt[i] + aux2*n[1]*nhat[1]*qNt[i] + aux2*n[2]*nhat[2]*qNt[i] + n[0]*pow(nhat[0], 2)*qNt[i]*u0j_uNj + n[1]*pow(nhat[1], 2)*qNt[i]*u1j_uNj + n[2]*pow(nhat[2], 2)*qNt[i]*u2j_uNj;
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
void nitsche_custom_h_a_q2_3d_residual_w(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double gN,  // parameter
double gamma,  // parameter
double n[],  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double p0j_pNj = 0.0;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  for (j=0; j<4; j++) { // p_nbasis_1
    p0j_pNj += p0[j]*pN[j];
  }
  double aux0 = nhat[0]*nhat[1];
  for (i=0; i<27; i++) { // w_nbasis
    
    __Fi[0] = aux0*gamma*u1j_uNj*wNt[i] - gN*gamma*nhat[0]*wNt[i] + gamma*pow(nhat[0], 2)*u0j_uNj*wNt[i] + gamma*nhat[0]*nhat[2]*u2j_uNj*wNt[i] + n[0]*p0j_pNj*wNt[i];
    __Fi[1] = aux0*gamma*u0j_uNj*wNt[i] - gN*gamma*nhat[1]*wNt[i] + gamma*pow(nhat[1], 2)*u1j_uNj*wNt[i] + gamma*nhat[1]*nhat[2]*u2j_uNj*wNt[i] + n[1]*p0j_pNj*wNt[i];
    __Fi[2] = -gN*gamma*nhat[2]*wNt[i] + gamma*nhat[0]*nhat[2]*u0j_uNj*wNt[i] + gamma*nhat[1]*nhat[2]*u1j_uNj*wNt[i] + gamma*pow(nhat[2], 2)*u2j_uNj*wNt[i] + n[2]*p0j_pNj*wNt[i];
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
void nitsche_custom_h_a_q2_3d_residual_q(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double gN,  // parameter
double n[],  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[1];
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  double aux0 = nhat[1]*u1j_uNj;
  double aux1 = nhat[2]*u2j_uNj;
  double aux2 = nhat[0]*u0j_uNj;
  for (i=0; i<4; i++) { // q_nbasis
    
    __Fi[0] = aux0*n[0]*nhat[0]*qNt[i] + aux0*n[2]*nhat[2]*qNt[i] + aux1*n[0]*nhat[0]*qNt[i] + aux1*n[1]*nhat[1]*qNt[i] + aux2*n[1]*nhat[1]*qNt[i] + aux2*n[2]*nhat[2]*qNt[i] - gN*qNt[i] + n[0]*pow(nhat[0], 2)*qNt[i]*u0j_uNj + n[1]*pow(nhat[1], 2)*qNt[i]*u1j_uNj + n[2]*pow(nhat[2], 2)*qNt[i]*u2j_uNj;
    F[1*i + 0] += scale * ( __Fi[0] );
  }
}
