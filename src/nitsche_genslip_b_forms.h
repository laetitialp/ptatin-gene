
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
void nitsche_custom_h_b_q2_3d_asmb_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double eta,  // parameter
double n[],  // parameter
double L_I[],  // parameter
double L_J[],  // parameter
double H_IJ,  // parameter
double scale, double A[])
{
  int i,j;
  double __Aij[9];
  double aux0 = L_J[1]*n[1];
  double aux1 = L_J[2]*n[2];
  double aux2 = pow(L_J[0], 2)*n[0];
  double aux3 = L_J[0]*n[0];
  double aux4 = pow(L_I[0], 2);
  double aux5 = pow(L_J[1], 2);
  double aux6 = pow(L_J[2], 2)*n[2];
  double aux7 = aux5*n[1];
  double aux8 = L_J[0]*aux1;
  double aux9 = pow(L_I[1], 2);
  double aux10 = L_J[0]*aux0;
  double aux11 = pow(L_I[2], 2);
  double aux12 = L_J[2]*aux3;
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      
      __Aij[0] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux0*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux1*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux2*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux0*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux1*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux2*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_J[0]*aux0*aux4*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_J[0]*aux1*aux4*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux1*aux4*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux3*aux4*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux0*aux4*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux3*aux4*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*aux2*aux4*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*aux4*aux5*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*aux4*aux6*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux0*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux1*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux2*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux0*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux1*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux2*eta*udNx2[j]*wNt[i] - 2.0*L_J[0]*aux0*aux4*eta*udNx0[j]*wNt[i] - 2.0*L_J[0]*aux1*aux4*eta*udNx0[j]*wNt[i] - 1.0*L_J[1]*aux1*aux4*eta*udNx1[j]*wNt[i] - 1.0*L_J[1]*aux3*aux4*eta*udNx1[j]*wNt[i] - 1.0*L_J[2]*aux0*aux4*eta*udNx2[j]*wNt[i] - 1.0*L_J[2]*aux3*aux4*eta*udNx2[j]*wNt[i] - 2.0*aux2*aux4*eta*udNx0[j]*wNt[i] - 1.0*aux4*aux5*eta*n[1]*udNx1[j]*wNt[i] - 1.0*aux4*aux6*eta*udNx2[j]*wNt[i];
      __Aij[1] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux0*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux1*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[1]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[1]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux2*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[1]*aux5*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[1]*aux1*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[1]*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux1*aux4*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux3*aux4*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*aux4*aux7*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux0*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux1*eta*udNx0[j]*wNt[i] - 2.0*L_I[0]*L_I[1]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] - 2.0*L_I[0]*L_I[1]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux2*eta*udNx0[j]*wNt[i] - 2.0*L_I[0]*L_I[1]*aux5*eta*n[1]*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[1]*aux1*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[1]*aux3*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*udNx2[j]*wNt[i] - 1.0*L_J[1]*aux1*aux4*eta*udNx0[j]*wNt[i] - 1.0*L_J[1]*aux3*aux4*eta*udNx0[j]*wNt[i] - 1.0*aux4*aux7*eta*udNx0[j]*wNt[i];
      __Aij[2] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[2]*aux0*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[2]*aux3*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux0*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[2]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[2]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux2*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux5*eta*n[1]*udNx1[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[2]*aux6*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux8*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux0*aux4*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux3*aux4*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*aux4*aux6*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[2]*aux0*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[2]*aux3*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux0*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] - 2.0*L_I[0]*L_I[2]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] - 2.0*L_I[0]*L_I[2]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux2*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux5*eta*n[1]*udNx1[j]*wNt[i] - 2.0*L_I[0]*L_I[2]*aux6*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux8*eta*udNx0[j]*wNt[i] - 1.0*L_J[2]*aux0*aux4*eta*udNx0[j]*wNt[i] - 1.0*L_J[2]*aux3*aux4*eta*udNx0[j]*wNt[i] - 1.0*aux4*aux6*eta*udNx0[j]*wNt[i];
      __Aij[3] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[1]*aux10*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[1]*aux2*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux5*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[1]*aux8*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux10*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux2*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux8*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[0]*L_J[1]*aux9*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux1*aux9*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*aux2*aux9*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] - 2.0*L_I[0]*L_I[1]*aux10*eta*udNx0[j]*wNt[i] - 2.0*L_I[0]*L_I[1]*aux2*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux5*eta*n[1]*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*udNx2[j]*wNt[i] - 2.0*L_I[0]*L_I[1]*aux8*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux10*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux2*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux8*eta*udNx2[j]*wNt[i] - 1.0*L_J[0]*L_J[1]*aux9*eta*n[1]*udNx1[j]*wNt[i] - 1.0*L_J[0]*aux1*aux9*eta*udNx1[j]*wNt[i] - 1.0*aux2*aux9*eta*udNx1[j]*wNt[i];
      __Aij[4] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[1]*aux1*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[1]*aux3*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[1]*aux1*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[1]*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux7*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux0*aux9*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux1*aux9*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_J[1]*aux1*aux9*eta*udNx1[j]*wNt[i] + 2.0*H_IJ*L_J[1]*aux3*aux9*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux0*aux9*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux3*aux9*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*aux2*aux9*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*aux5*aux9*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*aux6*aux9*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[1]*aux1*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[1]*aux3*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[1]*aux1*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[1]*aux3*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux7*eta*udNx2[j]*wNt[i] - 1.0*L_J[0]*aux0*aux9*eta*udNx0[j]*wNt[i] - 1.0*L_J[0]*aux1*aux9*eta*udNx0[j]*wNt[i] - 2.0*L_J[1]*aux1*aux9*eta*udNx1[j]*wNt[i] - 2.0*L_J[1]*aux3*aux9*eta*udNx1[j]*wNt[i] - 1.0*L_J[2]*aux0*aux9*eta*udNx2[j]*wNt[i] - 1.0*L_J[2]*aux3*aux9*eta*udNx2[j]*wNt[i] - 1.0*aux2*aux9*eta*udNx0[j]*wNt[i] - 2.0*aux5*aux9*eta*n[1]*udNx1[j]*wNt[i] - 1.0*aux6*aux9*eta*udNx2[j]*wNt[i];
      __Aij[5] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[2]*aux0*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[2]*aux3*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux0*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux1*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] + 2.0*H_IJ*L_I[1]*L_I[2]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_I[1]*L_I[2]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux2*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux5*eta*n[1]*udNx1[j]*wNt[i] + 2.0*H_IJ*L_I[1]*L_I[2]*aux6*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[1]*L_J[2]*aux9*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux3*aux9*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*aux6*aux9*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[2]*aux0*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[2]*aux3*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux0*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux1*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] - 2.0*L_I[1]*L_I[2]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] - 2.0*L_I[1]*L_I[2]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux2*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux5*eta*n[1]*udNx1[j]*wNt[i] - 2.0*L_I[1]*L_I[2]*aux6*eta*udNx2[j]*wNt[i] - 1.0*L_J[1]*L_J[2]*aux9*eta*n[1]*udNx1[j]*wNt[i] - 1.0*L_J[2]*aux3*aux9*eta*udNx1[j]*wNt[i] - 1.0*aux6*aux9*eta*udNx1[j]*wNt[i];
      __Aij[6] = 1.0*H_IJ*L_I[0]*L_I[2]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[2]*aux10*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[2]*aux2*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux5*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux6*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_I[0]*L_I[2]*aux8*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux10*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux2*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux8*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux0*aux11*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux1*aux11*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*aux11*aux2*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] - 2.0*L_I[0]*L_I[2]*aux10*eta*udNx0[j]*wNt[i] - 2.0*L_I[0]*L_I[2]*aux2*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux5*eta*n[1]*udNx1[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux6*eta*udNx2[j]*wNt[i] - 2.0*L_I[0]*L_I[2]*aux8*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux10*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux2*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux8*eta*udNx1[j]*wNt[i] - 1.0*L_J[0]*aux0*aux11*eta*udNx2[j]*wNt[i] - 1.0*L_J[0]*aux1*aux11*eta*udNx2[j]*wNt[i] - 1.0*aux11*aux2*eta*udNx2[j]*wNt[i];
      __Aij[7] = 1.0*H_IJ*L_I[0]*L_I[2]*L_J[1]*aux1*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[1]*aux3*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux0*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux1*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_I[1]*L_I[2]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] + 2.0*H_IJ*L_I[1]*L_I[2]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux2*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_I[1]*L_I[2]*aux5*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux6*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux1*aux11*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux11*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*aux11*aux7*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[1]*aux1*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[1]*aux3*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux0*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux1*eta*udNx0[j]*wNt[i] - 2.0*L_I[1]*L_I[2]*L_J[1]*aux1*eta*udNx1[j]*wNt[i] - 2.0*L_I[1]*L_I[2]*L_J[1]*aux3*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[2]*aux0*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[2]*aux3*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux2*eta*udNx0[j]*wNt[i] - 2.0*L_I[1]*L_I[2]*aux5*eta*n[1]*udNx1[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux6*eta*udNx2[j]*wNt[i] - 1.0*L_J[1]*aux1*aux11*eta*udNx2[j]*wNt[i] - 1.0*L_J[1]*aux11*aux3*eta*udNx2[j]*wNt[i] - 1.0*aux11*aux7*eta*udNx2[j]*wNt[i];
      __Aij[8] = 1.0*H_IJ*L_I[0]*L_I[2]*L_J[2]*aux0*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux6*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[2]*aux0*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux12*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux6*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux0*aux11*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux1*aux11*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux1*aux11*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux11*aux3*eta*udNx1[j]*wNt[i] + 2.0*H_IJ*L_J[2]*aux0*aux11*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_J[2]*aux11*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*aux11*aux2*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*aux11*aux5*eta*n[1]*udNx1[j]*wNt[i] + 2.0*H_IJ*aux11*aux6*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[2]*aux0*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[2]*aux6*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[2]*aux0*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux12*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux6*eta*udNx1[j]*wNt[i] - 1.0*L_J[0]*aux0*aux11*eta*udNx0[j]*wNt[i] - 1.0*L_J[0]*aux1*aux11*eta*udNx0[j]*wNt[i] - 1.0*L_J[1]*aux1*aux11*eta*udNx1[j]*wNt[i] - 1.0*L_J[1]*aux11*aux3*eta*udNx1[j]*wNt[i] - 2.0*L_J[2]*aux0*aux11*eta*udNx2[j]*wNt[i] - 2.0*L_J[2]*aux11*aux3*eta*udNx2[j]*wNt[i] - 1.0*aux11*aux2*eta*udNx0[j]*wNt[i] - 1.0*aux11*aux5*eta*n[1]*udNx1[j]*wNt[i] - 2.0*aux11*aux6*eta*udNx2[j]*wNt[i];
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
void nitsche_custom_h_b_q2_3d_asmbdiag_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double eta,  // parameter
double n[],  // parameter
double L_I[],  // parameter
double L_J[],  // parameter
double H_IJ,  // parameter
double scale, double F[])
{
  int i,j;
  double __Aij[3];
  double aux0 = L_I[0]*L_J[0];
  double aux1 = L_J[1]*n[1];
  double aux2 = L_J[2]*n[2];
  double aux3 = aux0*aux2;
  double aux4 = pow(L_J[0], 2)*n[0];
  double aux5 = L_I[0]*aux4;
  double aux6 = L_J[0]*n[0];
  double aux7 = pow(L_I[0], 2);
  double aux8 = pow(L_J[1], 2);
  double aux9 = pow(L_J[2], 2)*n[2];
  double aux10 = pow(L_I[1], 2);
  double aux11 = L_I[2]*L_J[2];
  double aux12 = L_I[2]*aux9;
  double aux13 = pow(L_I[2], 2);
  for (i=0; i<27; i++) { // w_nbasis
    j = i;
    
    __Aij[0] = 1.0*H_IJ*L_I[1]*aux0*aux1*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*aux3*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*aux5*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[2]*aux0*aux1*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[2]*aux3*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[2]*aux5*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_J[0]*aux1*aux7*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_J[0]*aux2*aux7*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux2*aux7*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux6*aux7*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux1*aux7*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux6*aux7*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*aux4*aux7*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*aux7*aux8*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*aux7*aux9*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*aux0*aux1*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*aux3*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*aux5*eta*udNx1[j]*wNt[i] - 1.0*L_I[2]*aux0*aux1*eta*udNx2[j]*wNt[i] - 1.0*L_I[2]*aux3*eta*udNx2[j]*wNt[i] - 1.0*L_I[2]*aux5*eta*udNx2[j]*wNt[i] - 2.0*L_J[0]*aux1*aux7*eta*udNx0[j]*wNt[i] - 2.0*L_J[0]*aux2*aux7*eta*udNx0[j]*wNt[i] - 1.0*L_J[1]*aux2*aux7*eta*udNx1[j]*wNt[i] - 1.0*L_J[1]*aux6*aux7*eta*udNx1[j]*wNt[i] - 1.0*L_J[2]*aux1*aux7*eta*udNx2[j]*wNt[i] - 1.0*L_J[2]*aux6*aux7*eta*udNx2[j]*wNt[i] - 2.0*aux4*aux7*eta*udNx0[j]*wNt[i] - 1.0*aux7*aux8*eta*n[1]*udNx1[j]*wNt[i] - 1.0*aux7*aux9*eta*udNx2[j]*wNt[i];
    __Aij[1] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[1]*aux2*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[1]*aux6*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux8*eta*n[1]*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[1]*aux2*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[1]*aux6*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux8*eta*n[1]*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux1*aux10*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux10*aux2*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*L_J[1]*aux10*aux2*eta*udNx1[j]*wNt[i] + 2.0*H_IJ*L_J[1]*aux10*aux6*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux1*aux10*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*L_J[2]*aux10*aux6*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*aux10*aux4*eta*udNx0[j]*wNt[i] + 2.0*H_IJ*aux10*aux8*eta*n[1]*udNx1[j]*wNt[i] + 1.0*H_IJ*aux10*aux9*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[1]*aux2*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[1]*aux6*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*L_I[1]*aux8*eta*n[1]*udNx0[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[1]*aux2*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[1]*aux6*eta*udNx2[j]*wNt[i] - 1.0*L_I[1]*L_I[2]*aux8*eta*n[1]*udNx2[j]*wNt[i] - 1.0*L_J[0]*aux1*aux10*eta*udNx0[j]*wNt[i] - 1.0*L_J[0]*aux10*aux2*eta*udNx0[j]*wNt[i] - 2.0*L_J[1]*aux10*aux2*eta*udNx1[j]*wNt[i] - 2.0*L_J[1]*aux10*aux6*eta*udNx1[j]*wNt[i] - 1.0*L_J[2]*aux1*aux10*eta*udNx2[j]*wNt[i] - 1.0*L_J[2]*aux10*aux6*eta*udNx2[j]*wNt[i] - 1.0*aux10*aux4*eta*udNx0[j]*wNt[i] - 2.0*aux10*aux8*eta*n[1]*udNx1[j]*wNt[i] - 1.0*aux10*aux9*eta*udNx2[j]*wNt[i];
    __Aij[2] = 1.0*H_IJ*L_I[0]*aux1*aux11*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*aux11*aux6*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[0]*aux12*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_I[1]*aux1*aux11*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*aux11*aux6*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_I[1]*aux12*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux1*aux13*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[0]*aux13*aux2*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux13*aux2*eta*udNx1[j]*wNt[i] + 1.0*H_IJ*L_J[1]*aux13*aux6*eta*udNx1[j]*wNt[i] + 2.0*H_IJ*L_J[2]*aux1*aux13*eta*udNx2[j]*wNt[i] + 2.0*H_IJ*L_J[2]*aux13*aux6*eta*udNx2[j]*wNt[i] + 1.0*H_IJ*aux13*aux4*eta*udNx0[j]*wNt[i] + 1.0*H_IJ*aux13*aux8*eta*n[1]*udNx1[j]*wNt[i] + 2.0*H_IJ*aux13*aux9*eta*udNx2[j]*wNt[i] - 1.0*L_I[0]*aux1*aux11*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*aux11*aux6*eta*udNx0[j]*wNt[i] - 1.0*L_I[0]*aux12*eta*udNx0[j]*wNt[i] - 1.0*L_I[1]*aux1*aux11*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*aux11*aux6*eta*udNx1[j]*wNt[i] - 1.0*L_I[1]*aux12*eta*udNx1[j]*wNt[i] - 1.0*L_J[0]*aux1*aux13*eta*udNx0[j]*wNt[i] - 1.0*L_J[0]*aux13*aux2*eta*udNx0[j]*wNt[i] - 1.0*L_J[1]*aux13*aux2*eta*udNx1[j]*wNt[i] - 1.0*L_J[1]*aux13*aux6*eta*udNx1[j]*wNt[i] - 2.0*L_J[2]*aux1*aux13*eta*udNx2[j]*wNt[i] - 2.0*L_J[2]*aux13*aux6*eta*udNx2[j]*wNt[i] - 1.0*aux13*aux4*eta*udNx0[j]*wNt[i] - 1.0*aux13*aux8*eta*n[1]*udNx1[j]*wNt[i] - 2.0*aux13*aux9*eta*udNx2[j]*wNt[i];
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
void nitsche_custom_h_b_q2_3d_spmv_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double u0[], double u1[], double u2[],
double eta,  // parameter
double n[],  // parameter
double L_I[],  // parameter
double L_J[],  // parameter
double H_IJ,  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double u0j_udNx0j = 0.0;
  double u0j_udNx1j = 0.0;
  double u0j_udNx2j = 0.0;
  double u1j_udNx0j = 0.0;
  double u1j_udNx1j = 0.0;
  double u1j_udNx2j = 0.0;
  double u2j_udNx0j = 0.0;
  double u2j_udNx1j = 0.0;
  double u2j_udNx2j = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_udNx0j += u0[j]*udNx0[j];
    u0j_udNx1j += u0[j]*udNx1[j];
    u0j_udNx2j += u0[j]*udNx2[j];
    u1j_udNx0j += u1[j]*udNx0[j];
    u1j_udNx1j += u1[j]*udNx1[j];
    u1j_udNx2j += u1[j]*udNx2[j];
    u2j_udNx0j += u2[j]*udNx0[j];
    u2j_udNx1j += u2[j]*udNx1[j];
    u2j_udNx2j += u2[j]*udNx2[j];
  }
  double aux0 = L_J[0]*n[0];
  double aux1 = L_J[1]*aux0;
  double aux2 = 2.0*u1j_udNx1j;
  double aux3 = L_J[1]*n[1];
  double aux4 = L_J[2]*aux0;
  double aux5 = L_J[2]*n[2];
  double aux6 = L_J[2]*aux3;
  double aux7 = L_J[1]*aux5;
  double aux8 = 2.0*u2j_udNx2j;
  double aux9 = aux4*aux8;
  double aux10 = aux6*aux8;
  double aux11 = pow(L_J[0], 2)*n[0];
  double aux12 = pow(L_J[1], 2)*n[1];
  double aux13 = aux12*aux2;
  double aux14 = pow(L_J[2], 2)*n[2];
  double aux15 = aux14*aux8;
  double aux16 = L_I[2]*aux15;
  double aux17 = pow(L_I[0], 2);
  double aux18 = 2.0*u0j_udNx0j;
  double aux19 = L_I[2]*aux11;
  double aux20 = pow(L_I[1], 2);
  double aux21 = pow(L_I[2], 2);
  for (i=0; i<27; i++) { // w_nbasis
    
    __Fi[0] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux3*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux3*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux5*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux5*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux1*aux2*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux11*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux11*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux13*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux14*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux14*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux2*aux7*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux3*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux5*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux10*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux11*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux11*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux9*eta*wNt[i] + H_IJ*L_I[0]*aux16*eta*wNt[i] + H_IJ*L_J[0]*aux17*aux18*aux3*eta*wNt[i] + H_IJ*L_J[0]*aux17*aux18*aux5*eta*wNt[i] + 1.0*H_IJ*L_J[1]*aux0*aux17*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_J[1]*aux0*aux17*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[1]*aux17*aux5*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_J[1]*aux17*aux5*eta*u1j_udNx0j*wNt[i] + H_IJ*aux11*aux17*aux18*eta*wNt[i] + 1.0*H_IJ*aux12*aux17*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*aux12*aux17*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*aux14*aux17*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux14*aux17*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*aux17*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux17*aux4*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*aux17*aux6*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux17*aux6*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux3*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux3*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux5*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux5*eta*u1j_udNx0j*wNt[i] - L_I[0]*L_I[1]*aux1*aux2*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux11*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux11*eta*u1j_udNx0j*wNt[i] - L_I[0]*L_I[1]*aux13*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux14*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux14*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[1]*aux2*aux7*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux3*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[2]*aux10*eta*wNt[i] - 1.0*L_I[0]*L_I[2]*aux11*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux11*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[2]*aux9*eta*wNt[i] - L_I[0]*aux16*eta*wNt[i] - L_J[0]*aux17*aux18*aux3*eta*wNt[i] - L_J[0]*aux17*aux18*aux5*eta*wNt[i] - 1.0*L_J[1]*aux0*aux17*eta*u0j_udNx1j*wNt[i] - 1.0*L_J[1]*aux0*aux17*eta*u1j_udNx0j*wNt[i] - 1.0*L_J[1]*aux17*aux5*eta*u0j_udNx1j*wNt[i] - 1.0*L_J[1]*aux17*aux5*eta*u1j_udNx0j*wNt[i] - aux11*aux17*aux18*eta*wNt[i] - 1.0*aux12*aux17*eta*u0j_udNx1j*wNt[i] - 1.0*aux12*aux17*eta*u1j_udNx0j*wNt[i] - 1.0*aux14*aux17*eta*u0j_udNx2j*wNt[i] - 1.0*aux14*aux17*eta*u2j_udNx0j*wNt[i] - 1.0*aux17*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*aux17*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*aux17*aux6*eta*u0j_udNx2j*wNt[i] - 1.0*aux17*aux6*eta*u2j_udNx0j*wNt[i];
    __Fi[1] = H_IJ*L_I[0]*L_I[1]*L_J[0]*aux18*aux3*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*L_J[0]*aux18*aux5*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux1*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux1*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux11*aux18*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux12*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux12*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux14*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux14*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux10*eta*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux12*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux12*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux7*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux9*eta*wNt[i] + H_IJ*L_I[1]*aux16*eta*wNt[i] + 1.0*H_IJ*L_I[1]*aux19*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*aux19*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[0]*aux20*aux3*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_J[0]*aux20*aux3*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[0]*aux20*aux5*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_J[0]*aux20*aux5*eta*u1j_udNx0j*wNt[i] + H_IJ*aux1*aux2*aux20*eta*wNt[i] + 1.0*H_IJ*aux11*aux20*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*aux11*aux20*eta*u1j_udNx0j*wNt[i] + H_IJ*aux13*aux20*eta*wNt[i] + 1.0*H_IJ*aux14*aux20*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux14*aux20*eta*u2j_udNx1j*wNt[i] + H_IJ*aux2*aux20*aux7*eta*wNt[i] + 1.0*H_IJ*aux20*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux20*aux4*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*aux20*aux6*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux20*aux6*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[1]*L_J[0]*aux18*aux3*eta*wNt[i] - L_I[0]*L_I[1]*L_J[0]*aux18*aux5*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux1*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux1*eta*u1j_udNx0j*wNt[i] - L_I[0]*L_I[1]*aux11*aux18*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux12*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux12*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux14*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux14*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] - L_I[1]*L_I[2]*aux10*eta*wNt[i] - 1.0*L_I[1]*L_I[2]*aux12*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux12*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux7*eta*u2j_udNx1j*wNt[i] - L_I[1]*L_I[2]*aux9*eta*wNt[i] - L_I[1]*aux16*eta*wNt[i] - 1.0*L_I[1]*aux19*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[1]*aux19*eta*u2j_udNx0j*wNt[i] - 1.0*L_J[0]*aux20*aux3*eta*u0j_udNx1j*wNt[i] - 1.0*L_J[0]*aux20*aux3*eta*u1j_udNx0j*wNt[i] - 1.0*L_J[0]*aux20*aux5*eta*u0j_udNx1j*wNt[i] - 1.0*L_J[0]*aux20*aux5*eta*u1j_udNx0j*wNt[i] - aux1*aux2*aux20*eta*wNt[i] - 1.0*aux11*aux20*eta*u0j_udNx1j*wNt[i] - 1.0*aux11*aux20*eta*u1j_udNx0j*wNt[i] - aux13*aux20*eta*wNt[i] - 1.0*aux14*aux20*eta*u1j_udNx2j*wNt[i] - 1.0*aux14*aux20*eta*u2j_udNx1j*wNt[i] - aux2*aux20*aux7*eta*wNt[i] - 1.0*aux20*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*aux20*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*aux20*aux6*eta*u1j_udNx2j*wNt[i] - 1.0*aux20*aux6*eta*u2j_udNx1j*wNt[i];
    __Fi[2] = H_IJ*L_I[0]*L_I[2]*L_J[0]*aux18*aux3*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*L_J[0]*aux18*aux5*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux14*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux14*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux4*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux6*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux6*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[0]*aux18*aux19*eta*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux1*aux2*eta*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux13*eta*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux14*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux14*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux2*aux7*eta*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux4*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux6*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux6*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*aux19*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*aux19*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[0]*aux21*aux3*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_J[0]*aux21*aux3*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[0]*aux21*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_J[0]*aux21*aux5*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*aux1*aux21*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux1*aux21*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*aux11*aux21*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux11*aux21*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*aux12*aux21*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux12*aux21*eta*u2j_udNx1j*wNt[i] + H_IJ*aux15*aux21*eta*wNt[i] + H_IJ*aux21*aux4*aux8*eta*wNt[i] + H_IJ*aux21*aux6*aux8*eta*wNt[i] + 1.0*H_IJ*aux21*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux21*aux7*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[2]*L_J[0]*aux18*aux3*eta*wNt[i] - L_I[0]*L_I[2]*L_J[0]*aux18*aux5*eta*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux14*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux14*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux6*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux6*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u1j_udNx0j*wNt[i] - L_I[0]*aux18*aux19*eta*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u1j_udNx0j*wNt[i] - L_I[1]*L_I[2]*aux1*aux2*eta*wNt[i] - L_I[1]*L_I[2]*aux13*eta*wNt[i] - 1.0*L_I[1]*L_I[2]*aux14*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux14*eta*u2j_udNx1j*wNt[i] - L_I[1]*L_I[2]*aux2*aux7*eta*wNt[i] - 1.0*L_I[1]*L_I[2]*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux6*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux6*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[1]*aux19*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[1]*aux19*eta*u1j_udNx0j*wNt[i] - 1.0*L_J[0]*aux21*aux3*eta*u0j_udNx2j*wNt[i] - 1.0*L_J[0]*aux21*aux3*eta*u2j_udNx0j*wNt[i] - 1.0*L_J[0]*aux21*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*L_J[0]*aux21*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*aux1*aux21*eta*u1j_udNx2j*wNt[i] - 1.0*aux1*aux21*eta*u2j_udNx1j*wNt[i] - 1.0*aux11*aux21*eta*u0j_udNx2j*wNt[i] - 1.0*aux11*aux21*eta*u2j_udNx0j*wNt[i] - 1.0*aux12*aux21*eta*u1j_udNx2j*wNt[i] - 1.0*aux12*aux21*eta*u2j_udNx1j*wNt[i] - aux15*aux21*eta*wNt[i] - aux21*aux4*aux8*eta*wNt[i] - aux21*aux6*aux8*eta*wNt[i] - 1.0*aux21*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*aux21*aux7*eta*u2j_udNx1j*wNt[i];
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
void nitsche_custom_h_b_q2_3d_asmb_wp(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double scale, double A[])
{
  int i,j;
  double __Aij[3];
  
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      
      __Aij[0] = 0;
      __Aij[1] = 0;
      __Aij[2] = 0;
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
void nitsche_custom_h_b_q2_3d_spmv_wp(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double p0[],
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
void nitsche_custom_h_b_q2_3d_asmb_qu(
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
void nitsche_custom_h_b_q2_3d_spmv_qu(
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
void nitsche_custom_h_b_q2_3d_asmb_qp(
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
void nitsche_custom_h_b_q2_3d_asmbdiag_qp(
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
void nitsche_custom_h_b_q2_3d_spmv_qp(
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
void nitsche_custom_h_b_q2_3d_spmv_w_up(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double eta,  // parameter
double n[],  // parameter
double L_I[],  // parameter
double L_J[],  // parameter
double H_IJ,  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double u0j_udNx0j = 0.0;
  double u0j_udNx1j = 0.0;
  double u0j_udNx2j = 0.0;
  double u1j_udNx0j = 0.0;
  double u1j_udNx1j = 0.0;
  double u1j_udNx2j = 0.0;
  double u2j_udNx0j = 0.0;
  double u2j_udNx1j = 0.0;
  double u2j_udNx2j = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_udNx0j += u0[j]*udNx0[j];
    u0j_udNx1j += u0[j]*udNx1[j];
    u0j_udNx2j += u0[j]*udNx2[j];
    u1j_udNx0j += u1[j]*udNx0[j];
    u1j_udNx1j += u1[j]*udNx1[j];
    u1j_udNx2j += u1[j]*udNx2[j];
    u2j_udNx0j += u2[j]*udNx0[j];
    u2j_udNx1j += u2[j]*udNx1[j];
    u2j_udNx2j += u2[j]*udNx2[j];
  }
  double aux0 = L_J[0]*n[0];
  double aux1 = L_J[1]*aux0;
  double aux2 = 2.0*u1j_udNx1j;
  double aux3 = L_J[1]*n[1];
  double aux4 = L_J[2]*aux0;
  double aux5 = L_J[2]*n[2];
  double aux6 = L_J[2]*aux3;
  double aux7 = L_J[1]*aux5;
  double aux8 = 2.0*u2j_udNx2j;
  double aux9 = aux4*aux8;
  double aux10 = aux6*aux8;
  double aux11 = pow(L_J[0], 2)*n[0];
  double aux12 = pow(L_J[1], 2)*n[1];
  double aux13 = aux12*aux2;
  double aux14 = pow(L_J[2], 2)*n[2];
  double aux15 = aux14*aux8;
  double aux16 = L_I[2]*aux15;
  double aux17 = pow(L_I[0], 2);
  double aux18 = 2.0*u0j_udNx0j;
  double aux19 = L_I[2]*aux11;
  double aux20 = pow(L_I[1], 2);
  double aux21 = pow(L_I[2], 2);
  for (i=0; i<27; i++) { // w_nbasis
    
    __Fi[0] = 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux3*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux3*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux5*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*L_J[0]*aux5*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux1*aux2*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux11*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux11*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux13*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux14*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux14*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux2*aux7*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux3*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*L_J[0]*aux5*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux10*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux11*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux11*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux9*eta*wNt[i] + H_IJ*L_I[0]*aux16*eta*wNt[i] + H_IJ*L_J[0]*aux17*aux18*aux3*eta*wNt[i] + H_IJ*L_J[0]*aux17*aux18*aux5*eta*wNt[i] + 1.0*H_IJ*L_J[1]*aux0*aux17*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_J[1]*aux0*aux17*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[1]*aux17*aux5*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_J[1]*aux17*aux5*eta*u1j_udNx0j*wNt[i] + H_IJ*aux11*aux17*aux18*eta*wNt[i] + 1.0*H_IJ*aux12*aux17*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*aux12*aux17*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*aux14*aux17*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux14*aux17*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*aux17*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux17*aux4*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*aux17*aux6*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux17*aux6*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux3*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux3*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux5*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*L_J[0]*aux5*eta*u1j_udNx0j*wNt[i] - L_I[0]*L_I[1]*aux1*aux2*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux11*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux11*eta*u1j_udNx0j*wNt[i] - L_I[0]*L_I[1]*aux13*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux14*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux14*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[1]*aux2*aux7*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux3*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*L_J[0]*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[2]*aux10*eta*wNt[i] - 1.0*L_I[0]*L_I[2]*aux11*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux11*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[2]*aux9*eta*wNt[i] - L_I[0]*aux16*eta*wNt[i] - L_J[0]*aux17*aux18*aux3*eta*wNt[i] - L_J[0]*aux17*aux18*aux5*eta*wNt[i] - 1.0*L_J[1]*aux0*aux17*eta*u0j_udNx1j*wNt[i] - 1.0*L_J[1]*aux0*aux17*eta*u1j_udNx0j*wNt[i] - 1.0*L_J[1]*aux17*aux5*eta*u0j_udNx1j*wNt[i] - 1.0*L_J[1]*aux17*aux5*eta*u1j_udNx0j*wNt[i] - aux11*aux17*aux18*eta*wNt[i] - 1.0*aux12*aux17*eta*u0j_udNx1j*wNt[i] - 1.0*aux12*aux17*eta*u1j_udNx0j*wNt[i] - 1.0*aux14*aux17*eta*u0j_udNx2j*wNt[i] - 1.0*aux14*aux17*eta*u2j_udNx0j*wNt[i] - 1.0*aux17*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*aux17*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*aux17*aux6*eta*u0j_udNx2j*wNt[i] - 1.0*aux17*aux6*eta*u2j_udNx0j*wNt[i];
    __Fi[1] = H_IJ*L_I[0]*L_I[1]*L_J[0]*aux18*aux3*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*L_J[0]*aux18*aux5*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux1*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux1*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux11*aux18*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux12*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux12*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux14*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux14*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux6*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux10*eta*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux12*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux12*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux7*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux9*eta*wNt[i] + H_IJ*L_I[1]*aux16*eta*wNt[i] + 1.0*H_IJ*L_I[1]*aux19*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*aux19*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[0]*aux20*aux3*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_J[0]*aux20*aux3*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[0]*aux20*aux5*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_J[0]*aux20*aux5*eta*u1j_udNx0j*wNt[i] + H_IJ*aux1*aux2*aux20*eta*wNt[i] + 1.0*H_IJ*aux11*aux20*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*aux11*aux20*eta*u1j_udNx0j*wNt[i] + H_IJ*aux13*aux20*eta*wNt[i] + 1.0*H_IJ*aux14*aux20*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux14*aux20*eta*u2j_udNx1j*wNt[i] + H_IJ*aux2*aux20*aux7*eta*wNt[i] + 1.0*H_IJ*aux20*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux20*aux4*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*aux20*aux6*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux20*aux6*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[1]*L_J[0]*aux18*aux3*eta*wNt[i] - L_I[0]*L_I[1]*L_J[0]*aux18*aux5*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux1*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux1*eta*u1j_udNx0j*wNt[i] - L_I[0]*L_I[1]*aux11*aux18*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux12*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux12*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux14*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux14*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux6*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] - L_I[1]*L_I[2]*aux10*eta*wNt[i] - 1.0*L_I[1]*L_I[2]*aux12*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux12*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux7*eta*u2j_udNx1j*wNt[i] - L_I[1]*L_I[2]*aux9*eta*wNt[i] - L_I[1]*aux16*eta*wNt[i] - 1.0*L_I[1]*aux19*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[1]*aux19*eta*u2j_udNx0j*wNt[i] - 1.0*L_J[0]*aux20*aux3*eta*u0j_udNx1j*wNt[i] - 1.0*L_J[0]*aux20*aux3*eta*u1j_udNx0j*wNt[i] - 1.0*L_J[0]*aux20*aux5*eta*u0j_udNx1j*wNt[i] - 1.0*L_J[0]*aux20*aux5*eta*u1j_udNx0j*wNt[i] - aux1*aux2*aux20*eta*wNt[i] - 1.0*aux11*aux20*eta*u0j_udNx1j*wNt[i] - 1.0*aux11*aux20*eta*u1j_udNx0j*wNt[i] - aux13*aux20*eta*wNt[i] - 1.0*aux14*aux20*eta*u1j_udNx2j*wNt[i] - 1.0*aux14*aux20*eta*u2j_udNx1j*wNt[i] - aux2*aux20*aux7*eta*wNt[i] - 1.0*aux20*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*aux20*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*aux20*aux6*eta*u1j_udNx2j*wNt[i] - 1.0*aux20*aux6*eta*u2j_udNx1j*wNt[i];
    __Fi[2] = H_IJ*L_I[0]*L_I[2]*L_J[0]*aux18*aux3*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*L_J[0]*aux18*aux5*eta*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux12*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux14*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux14*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux4*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux6*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux6*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[0]*aux18*aux19*eta*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u1j_udNx0j*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux1*aux2*eta*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux13*eta*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux14*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux14*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[1]*L_I[2]*aux2*aux7*eta*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux4*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux6*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[1]*L_I[2]*aux6*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*aux19*eta*u0j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[1]*aux19*eta*u1j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[0]*aux21*aux3*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_J[0]*aux21*aux3*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*L_J[0]*aux21*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_J[0]*aux21*aux5*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*aux1*aux21*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux1*aux21*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*aux11*aux21*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux11*aux21*eta*u2j_udNx0j*wNt[i] + 1.0*H_IJ*aux12*aux21*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux12*aux21*eta*u2j_udNx1j*wNt[i] + H_IJ*aux15*aux21*eta*wNt[i] + H_IJ*aux21*aux4*aux8*eta*wNt[i] + H_IJ*aux21*aux6*aux8*eta*wNt[i] + 1.0*H_IJ*aux21*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux21*aux7*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[2]*L_J[0]*aux18*aux3*eta*wNt[i] - L_I[0]*L_I[2]*L_J[0]*aux18*aux5*eta*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux12*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux14*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux14*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux6*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux6*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u1j_udNx0j*wNt[i] - L_I[0]*aux18*aux19*eta*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux3*eta*u1j_udNx0j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[1]*L_I[2]*L_J[0]*aux5*eta*u1j_udNx0j*wNt[i] - L_I[1]*L_I[2]*aux1*aux2*eta*wNt[i] - L_I[1]*L_I[2]*aux13*eta*wNt[i] - 1.0*L_I[1]*L_I[2]*aux14*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux14*eta*u2j_udNx1j*wNt[i] - L_I[1]*L_I[2]*aux2*aux7*eta*wNt[i] - 1.0*L_I[1]*L_I[2]*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux6*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[1]*L_I[2]*aux6*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[1]*aux19*eta*u0j_udNx1j*wNt[i] - 1.0*L_I[1]*aux19*eta*u1j_udNx0j*wNt[i] - 1.0*L_J[0]*aux21*aux3*eta*u0j_udNx2j*wNt[i] - 1.0*L_J[0]*aux21*aux3*eta*u2j_udNx0j*wNt[i] - 1.0*L_J[0]*aux21*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*L_J[0]*aux21*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*aux1*aux21*eta*u1j_udNx2j*wNt[i] - 1.0*aux1*aux21*eta*u2j_udNx1j*wNt[i] - 1.0*aux11*aux21*eta*u0j_udNx2j*wNt[i] - 1.0*aux11*aux21*eta*u2j_udNx0j*wNt[i] - 1.0*aux12*aux21*eta*u1j_udNx2j*wNt[i] - 1.0*aux12*aux21*eta*u2j_udNx1j*wNt[i] - aux15*aux21*eta*wNt[i] - aux21*aux4*aux8*eta*wNt[i] - aux21*aux6*aux8*eta*wNt[i] - 1.0*aux21*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*aux21*aux7*eta*u2j_udNx1j*wNt[i];
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
void nitsche_custom_h_b_q2_3d_spmv_q_up(
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
void nitsche_custom_h_b_q2_3d_residual_w(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double eta,  // parameter
double n[],  // parameter
double L_I[],  // parameter
double L_J[],  // parameter
double H_IJ,  // parameter
double tau_S[],  // parameter
double scale, double F[])
{
  int i,j;
  double __Fi[3];
  double u0j_udNx0j = 0.0;
  double u0j_udNx1j = 0.0;
  double u0j_udNx2j = 0.0;
  double u1j_udNx0j = 0.0;
  double u1j_udNx1j = 0.0;
  double u1j_udNx2j = 0.0;
  double u2j_udNx0j = 0.0;
  double u2j_udNx1j = 0.0;
  double u2j_udNx2j = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_udNx0j += u0[j]*udNx0[j];
    u0j_udNx1j += u0[j]*udNx1[j];
    u0j_udNx2j += u0[j]*udNx2[j];
    u1j_udNx0j += u1[j]*udNx0[j];
    u1j_udNx1j += u1[j]*udNx1[j];
    u1j_udNx2j += u1[j]*udNx2[j];
    u2j_udNx0j += u2[j]*udNx0[j];
    u2j_udNx1j += u2[j]*udNx1[j];
    u2j_udNx2j += u2[j]*udNx2[j];
  }
  double aux0 = L_J[0]*L_J[1];
  double aux1 = aux0*n[0];
  double aux2 = aux0*n[1];
  double aux3 = L_J[0]*L_J[2];
  double aux4 = aux3*n[0];
  double aux5 = aux3*n[2];
  double aux6 = L_J[1]*L_J[2];
  double aux7 = aux6*n[1];
  double aux8 = aux6*n[2];
  double aux9 = pow(L_J[0], 2)*n[0];
  double aux10 = aux9*tau_S[3];
  double aux11 = pow(L_J[1], 2)*n[1];
  double aux12 = aux11*tau_S[1];
  double aux13 = pow(L_J[2], 2)*n[2];
  double aux14 = aux13*tau_S[5];
  double aux15 = aux13*tau_S[2];
  double aux16 = pow(L_I[0], 2);
  double aux17 = 2.0*u1j_udNx1j;
  double aux18 = 1.0*u0j_udNx1j;
  double aux19 = 1.0*u1j_udNx0j;
  double aux20 = 2.0*u2j_udNx2j;
  double aux21 = aux18*aux9;
  double aux22 = 1.0*aux9;
  double aux23 = aux22*u1j_udNx0j;
  double aux24 = aux11*aux17;
  double aux25 = 1.0*aux13;
  double aux26 = aux22*u0j_udNx2j;
  double aux27 = aux22*u2j_udNx0j;
  double aux28 = 1.0*aux11;
  double aux29 = aux13*aux20;
  double aux30 = 2.0*u0j_udNx0j;
  double aux31 = aux30*aux9;
  double aux32 = aux28*u0j_udNx1j;
  double aux33 = aux28*u1j_udNx0j;
  double aux34 = L_I[1]*L_I[2];
  double aux35 = aux9*tau_S[0];
  double aux36 = aux11*tau_S[3];
  double aux37 = pow(L_I[1], 2);
  double aux38 = pow(L_I[2], 2);
  for (i=0; i<27; i++) { // w_nbasis
    
    __Fi[0] = H_IJ*L_I[0]*L_I[1]*aux1*aux17*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux1*tau_S[1]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux10*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux12*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux14*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux17*aux8*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux18*aux2*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux18*aux5*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux19*aux2*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux19*aux5*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux2*tau_S[3]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux21*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux23*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux24*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux25*eta*u1j_udNx2j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux25*eta*u2j_udNx1j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux4*tau_S[5]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux5*tau_S[3]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux7*tau_S[5]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux8*tau_S[1]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux1*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux11*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux15*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux2*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux2*eta*u2j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux2*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux20*aux4*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux20*aux7*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux26*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux27*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux28*eta*u1j_udNx2j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux28*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux29*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux4*tau_S[2]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux5*eta*u2j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux5*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux7*tau_S[2]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux8*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux8*eta*u2j_udNx1j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux8*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux9*tau_S[4]*wNt[i] + H_IJ*aux1*aux16*aux18*eta*wNt[i] + H_IJ*aux1*aux16*aux19*eta*wNt[i] + H_IJ*aux1*aux16*tau_S[3]*wNt[i] + H_IJ*aux11*aux16*tau_S[3]*wNt[i] + H_IJ*aux13*aux16*tau_S[4]*wNt[i] + H_IJ*aux16*aux18*aux8*eta*wNt[i] + H_IJ*aux16*aux19*aux8*eta*wNt[i] + H_IJ*aux16*aux2*aux30*eta*wNt[i] + H_IJ*aux16*aux2*tau_S[0]*wNt[i] + H_IJ*aux16*aux25*eta*u0j_udNx2j*wNt[i] + H_IJ*aux16*aux25*eta*u2j_udNx0j*wNt[i] + H_IJ*aux16*aux30*aux5*eta*wNt[i] + H_IJ*aux16*aux31*eta*wNt[i] + H_IJ*aux16*aux32*eta*wNt[i] + H_IJ*aux16*aux33*eta*wNt[i] + 1.0*H_IJ*aux16*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux16*aux4*eta*u2j_udNx0j*wNt[i] + H_IJ*aux16*aux4*tau_S[4]*wNt[i] + H_IJ*aux16*aux5*tau_S[0]*wNt[i] + 1.0*H_IJ*aux16*aux7*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux16*aux7*eta*u2j_udNx0j*wNt[i] + H_IJ*aux16*aux7*tau_S[4]*wNt[i] + H_IJ*aux16*aux8*tau_S[3]*wNt[i] + H_IJ*aux16*aux9*tau_S[0]*wNt[i] - L_I[0]*L_I[1]*aux1*aux17*eta*wNt[i] - L_I[0]*L_I[1]*aux17*aux8*eta*wNt[i] - L_I[0]*L_I[1]*aux18*aux2*eta*wNt[i] - L_I[0]*L_I[1]*aux18*aux5*eta*wNt[i] - L_I[0]*L_I[1]*aux19*aux2*eta*wNt[i] - L_I[0]*L_I[1]*aux19*aux5*eta*wNt[i] - L_I[0]*L_I[1]*aux21*eta*wNt[i] - L_I[0]*L_I[1]*aux23*eta*wNt[i] - L_I[0]*L_I[1]*aux24*eta*wNt[i] - L_I[0]*L_I[1]*aux25*eta*u1j_udNx2j*wNt[i] - L_I[0]*L_I[1]*aux25*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux1*eta*u2j_udNx1j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux2*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux2*eta*u2j_udNx0j*wNt[i] - L_I[0]*L_I[2]*aux20*aux4*eta*wNt[i] - L_I[0]*L_I[2]*aux20*aux7*eta*wNt[i] - L_I[0]*L_I[2]*aux26*eta*wNt[i] - L_I[0]*L_I[2]*aux27*eta*wNt[i] - L_I[0]*L_I[2]*aux28*eta*u1j_udNx2j*wNt[i] - L_I[0]*L_I[2]*aux28*eta*u2j_udNx1j*wNt[i] - L_I[0]*L_I[2]*aux29*eta*wNt[i] - 1.0*L_I[0]*L_I[2]*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux8*eta*u1j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux8*eta*u2j_udNx1j*wNt[i] - aux1*aux16*aux18*eta*wNt[i] - aux1*aux16*aux19*eta*wNt[i] - aux16*aux18*aux8*eta*wNt[i] - aux16*aux19*aux8*eta*wNt[i] - aux16*aux2*aux30*eta*wNt[i] - aux16*aux25*eta*u0j_udNx2j*wNt[i] - aux16*aux25*eta*u2j_udNx0j*wNt[i] - aux16*aux30*aux5*eta*wNt[i] - aux16*aux31*eta*wNt[i] - aux16*aux32*eta*wNt[i] - aux16*aux33*eta*wNt[i] - 1.0*aux16*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*aux16*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*aux16*aux7*eta*u0j_udNx2j*wNt[i] - 1.0*aux16*aux7*eta*u2j_udNx0j*wNt[i];
    __Fi[1] = H_IJ*L_I[0]*L_I[1]*aux1*aux18*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux1*aux19*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux1*tau_S[3]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux13*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux18*aux8*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux19*aux8*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux2*aux30*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux2*tau_S[0]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux25*eta*u0j_udNx2j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux25*eta*u2j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux30*aux5*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux31*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux32*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux33*eta*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux35*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux36*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux4*eta*u2j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux4*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux5*tau_S[0]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[1]*aux7*eta*u2j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux7*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[1]*aux8*tau_S[3]*wNt[i] + H_IJ*aux1*aux17*aux37*eta*wNt[i] + 1.0*H_IJ*aux1*aux34*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux1*aux34*eta*u2j_udNx1j*wNt[i] + H_IJ*aux1*aux34*tau_S[4]*wNt[i] + H_IJ*aux1*aux37*tau_S[1]*wNt[i] + H_IJ*aux10*aux37*wNt[i] + H_IJ*aux11*aux34*tau_S[4]*wNt[i] + H_IJ*aux12*aux37*wNt[i] + H_IJ*aux14*aux37*wNt[i] + H_IJ*aux15*aux34*wNt[i] + H_IJ*aux17*aux37*aux8*eta*wNt[i] + H_IJ*aux18*aux2*aux37*eta*wNt[i] + H_IJ*aux18*aux37*aux5*eta*wNt[i] + H_IJ*aux19*aux2*aux37*eta*wNt[i] + H_IJ*aux19*aux37*aux5*eta*wNt[i] + 1.0*H_IJ*aux2*aux34*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux2*aux34*eta*u2j_udNx0j*wNt[i] + H_IJ*aux2*aux34*tau_S[4]*wNt[i] + H_IJ*aux2*aux37*tau_S[3]*wNt[i] + H_IJ*aux20*aux34*aux4*eta*wNt[i] + H_IJ*aux20*aux34*aux7*eta*wNt[i] + H_IJ*aux21*aux37*eta*wNt[i] + H_IJ*aux23*aux37*eta*wNt[i] + H_IJ*aux24*aux37*eta*wNt[i] + H_IJ*aux25*aux37*eta*u1j_udNx2j*wNt[i] + H_IJ*aux25*aux37*eta*u2j_udNx1j*wNt[i] + H_IJ*aux26*aux34*eta*wNt[i] + H_IJ*aux27*aux34*eta*wNt[i] + H_IJ*aux28*aux34*eta*u1j_udNx2j*wNt[i] + H_IJ*aux28*aux34*eta*u2j_udNx1j*wNt[i] + H_IJ*aux29*aux34*eta*wNt[i] + H_IJ*aux34*aux4*tau_S[2]*wNt[i] + 1.0*H_IJ*aux34*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux34*aux5*eta*u2j_udNx0j*wNt[i] + H_IJ*aux34*aux5*tau_S[4]*wNt[i] + H_IJ*aux34*aux7*tau_S[2]*wNt[i] + 1.0*H_IJ*aux34*aux8*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux34*aux8*eta*u2j_udNx1j*wNt[i] + H_IJ*aux34*aux8*tau_S[4]*wNt[i] + H_IJ*aux34*aux9*tau_S[4]*wNt[i] + 1.0*H_IJ*aux37*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux37*aux4*eta*u2j_udNx1j*wNt[i] + H_IJ*aux37*aux4*tau_S[5]*wNt[i] + H_IJ*aux37*aux5*tau_S[3]*wNt[i] + 1.0*H_IJ*aux37*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux37*aux7*eta*u2j_udNx1j*wNt[i] + H_IJ*aux37*aux7*tau_S[5]*wNt[i] + H_IJ*aux37*aux8*tau_S[1]*wNt[i] - L_I[0]*L_I[1]*aux1*aux18*eta*wNt[i] - L_I[0]*L_I[1]*aux1*aux19*eta*wNt[i] - L_I[0]*L_I[1]*aux18*aux8*eta*wNt[i] - L_I[0]*L_I[1]*aux19*aux8*eta*wNt[i] - L_I[0]*L_I[1]*aux2*aux30*eta*wNt[i] - L_I[0]*L_I[1]*aux25*eta*u0j_udNx2j*wNt[i] - L_I[0]*L_I[1]*aux25*eta*u2j_udNx0j*wNt[i] - L_I[0]*L_I[1]*aux30*aux5*eta*wNt[i] - L_I[0]*L_I[1]*aux31*eta*wNt[i] - L_I[0]*L_I[1]*aux32*eta*wNt[i] - L_I[0]*L_I[1]*aux33*eta*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[1]*aux7*eta*u2j_udNx0j*wNt[i] - aux1*aux17*aux37*eta*wNt[i] - 1.0*aux1*aux34*eta*u1j_udNx2j*wNt[i] - 1.0*aux1*aux34*eta*u2j_udNx1j*wNt[i] - aux17*aux37*aux8*eta*wNt[i] - aux18*aux2*aux37*eta*wNt[i] - aux18*aux37*aux5*eta*wNt[i] - aux19*aux2*aux37*eta*wNt[i] - aux19*aux37*aux5*eta*wNt[i] - 1.0*aux2*aux34*eta*u0j_udNx2j*wNt[i] - 1.0*aux2*aux34*eta*u2j_udNx0j*wNt[i] - aux20*aux34*aux4*eta*wNt[i] - aux20*aux34*aux7*eta*wNt[i] - aux21*aux37*eta*wNt[i] - aux23*aux37*eta*wNt[i] - aux24*aux37*eta*wNt[i] - aux25*aux37*eta*u1j_udNx2j*wNt[i] - aux25*aux37*eta*u2j_udNx1j*wNt[i] - aux26*aux34*eta*wNt[i] - aux27*aux34*eta*wNt[i] - aux28*aux34*eta*u1j_udNx2j*wNt[i] - aux28*aux34*eta*u2j_udNx1j*wNt[i] - aux29*aux34*eta*wNt[i] - 1.0*aux34*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*aux34*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*aux34*aux8*eta*u1j_udNx2j*wNt[i] - 1.0*aux34*aux8*eta*u2j_udNx1j*wNt[i] - 1.0*aux37*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*aux37*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*aux37*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*aux37*aux7*eta*u2j_udNx1j*wNt[i];
    __Fi[2] = H_IJ*L_I[0]*L_I[2]*aux1*aux18*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux1*aux19*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux1*tau_S[3]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux13*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux18*aux8*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux19*aux8*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux2*aux30*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux2*tau_S[0]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux25*eta*u0j_udNx2j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux25*eta*u2j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux30*aux5*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux31*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux32*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux33*eta*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux35*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux36*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux4*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux4*eta*u2j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux4*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux5*tau_S[0]*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*L_I[0]*L_I[2]*aux7*eta*u2j_udNx0j*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux7*tau_S[4]*wNt[i] + H_IJ*L_I[0]*L_I[2]*aux8*tau_S[3]*wNt[i] + H_IJ*aux1*aux17*aux34*eta*wNt[i] + H_IJ*aux1*aux34*tau_S[1]*wNt[i] + 1.0*H_IJ*aux1*aux38*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux1*aux38*eta*u2j_udNx1j*wNt[i] + H_IJ*aux1*aux38*tau_S[4]*wNt[i] + H_IJ*aux10*aux34*wNt[i] + H_IJ*aux11*aux38*tau_S[4]*wNt[i] + H_IJ*aux12*aux34*wNt[i] + H_IJ*aux14*aux34*wNt[i] + H_IJ*aux15*aux38*wNt[i] + H_IJ*aux17*aux34*aux8*eta*wNt[i] + H_IJ*aux18*aux2*aux34*eta*wNt[i] + H_IJ*aux18*aux34*aux5*eta*wNt[i] + H_IJ*aux19*aux2*aux34*eta*wNt[i] + H_IJ*aux19*aux34*aux5*eta*wNt[i] + H_IJ*aux2*aux34*tau_S[3]*wNt[i] + 1.0*H_IJ*aux2*aux38*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux2*aux38*eta*u2j_udNx0j*wNt[i] + H_IJ*aux2*aux38*tau_S[4]*wNt[i] + H_IJ*aux20*aux38*aux4*eta*wNt[i] + H_IJ*aux20*aux38*aux7*eta*wNt[i] + H_IJ*aux21*aux34*eta*wNt[i] + H_IJ*aux23*aux34*eta*wNt[i] + H_IJ*aux24*aux34*eta*wNt[i] + H_IJ*aux25*aux34*eta*u1j_udNx2j*wNt[i] + H_IJ*aux25*aux34*eta*u2j_udNx1j*wNt[i] + H_IJ*aux26*aux38*eta*wNt[i] + H_IJ*aux27*aux38*eta*wNt[i] + H_IJ*aux28*aux38*eta*u1j_udNx2j*wNt[i] + H_IJ*aux28*aux38*eta*u2j_udNx1j*wNt[i] + H_IJ*aux29*aux38*eta*wNt[i] + 1.0*H_IJ*aux34*aux4*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux34*aux4*eta*u2j_udNx1j*wNt[i] + H_IJ*aux34*aux4*tau_S[5]*wNt[i] + H_IJ*aux34*aux5*tau_S[3]*wNt[i] + 1.0*H_IJ*aux34*aux7*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux34*aux7*eta*u2j_udNx1j*wNt[i] + H_IJ*aux34*aux7*tau_S[5]*wNt[i] + H_IJ*aux34*aux8*tau_S[1]*wNt[i] + H_IJ*aux38*aux4*tau_S[2]*wNt[i] + 1.0*H_IJ*aux38*aux5*eta*u0j_udNx2j*wNt[i] + 1.0*H_IJ*aux38*aux5*eta*u2j_udNx0j*wNt[i] + H_IJ*aux38*aux5*tau_S[4]*wNt[i] + H_IJ*aux38*aux7*tau_S[2]*wNt[i] + 1.0*H_IJ*aux38*aux8*eta*u1j_udNx2j*wNt[i] + 1.0*H_IJ*aux38*aux8*eta*u2j_udNx1j*wNt[i] + H_IJ*aux38*aux8*tau_S[4]*wNt[i] + H_IJ*aux38*aux9*tau_S[4]*wNt[i] - L_I[0]*L_I[2]*aux1*aux18*eta*wNt[i] - L_I[0]*L_I[2]*aux1*aux19*eta*wNt[i] - L_I[0]*L_I[2]*aux18*aux8*eta*wNt[i] - L_I[0]*L_I[2]*aux19*aux8*eta*wNt[i] - L_I[0]*L_I[2]*aux2*aux30*eta*wNt[i] - L_I[0]*L_I[2]*aux25*eta*u0j_udNx2j*wNt[i] - L_I[0]*L_I[2]*aux25*eta*u2j_udNx0j*wNt[i] - L_I[0]*L_I[2]*aux30*aux5*eta*wNt[i] - L_I[0]*L_I[2]*aux31*eta*wNt[i] - L_I[0]*L_I[2]*aux32*eta*wNt[i] - L_I[0]*L_I[2]*aux33*eta*wNt[i] - 1.0*L_I[0]*L_I[2]*aux4*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux4*eta*u2j_udNx0j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u0j_udNx2j*wNt[i] - 1.0*L_I[0]*L_I[2]*aux7*eta*u2j_udNx0j*wNt[i] - aux1*aux17*aux34*eta*wNt[i] - 1.0*aux1*aux38*eta*u1j_udNx2j*wNt[i] - 1.0*aux1*aux38*eta*u2j_udNx1j*wNt[i] - aux17*aux34*aux8*eta*wNt[i] - aux18*aux2*aux34*eta*wNt[i] - aux18*aux34*aux5*eta*wNt[i] - aux19*aux2*aux34*eta*wNt[i] - aux19*aux34*aux5*eta*wNt[i] - 1.0*aux2*aux38*eta*u0j_udNx2j*wNt[i] - 1.0*aux2*aux38*eta*u2j_udNx0j*wNt[i] - aux20*aux38*aux4*eta*wNt[i] - aux20*aux38*aux7*eta*wNt[i] - aux21*aux34*eta*wNt[i] - aux23*aux34*eta*wNt[i] - aux24*aux34*eta*wNt[i] - aux25*aux34*eta*u1j_udNx2j*wNt[i] - aux25*aux34*eta*u2j_udNx1j*wNt[i] - aux26*aux38*eta*wNt[i] - aux27*aux38*eta*wNt[i] - aux28*aux38*eta*u1j_udNx2j*wNt[i] - aux28*aux38*eta*u2j_udNx1j*wNt[i] - aux29*aux38*eta*wNt[i] - 1.0*aux34*aux4*eta*u1j_udNx2j*wNt[i] - 1.0*aux34*aux4*eta*u2j_udNx1j*wNt[i] - 1.0*aux34*aux7*eta*u1j_udNx2j*wNt[i] - 1.0*aux34*aux7*eta*u2j_udNx1j*wNt[i] - 1.0*aux38*aux5*eta*u0j_udNx2j*wNt[i] - 1.0*aux38*aux5*eta*u2j_udNx0j*wNt[i] - 1.0*aux38*aux8*eta*u1j_udNx2j*wNt[i] - 1.0*aux38*aux8*eta*u2j_udNx1j*wNt[i];
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
void nitsche_custom_h_b_q2_3d_residual_q(
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
