
//
// fe_form_compiler.py version: 8d4b0b5b8d2e57803682a919e42ac439d4c64103
// sympy version: 1.6.1
// using common substring elimination: True
// form file: nitsche-custom-h_IJ.py version: 4543cb74a6c7824e8f0779441abf7da668e0d359
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
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31;
      tce0 = L_J[0]*n[0];
      tce1 = 1.0*L_I[0];
      tce2 = tce1*udNx1[j];
      tce3 = L_J[1]*tce2;
      tce4 = tce0*tce3;
      tce5 = L_J[1]*n[1];
      tce6 = 2.0*L_I[0]*udNx0[j];
      tce7 = L_J[0]*tce6;
      tce8 = tce5*tce7;
      tce9 = tce1*udNx2[j];
      tce10 = L_J[2]*tce9;
      tce11 = tce0*tce10;
      tce12 = L_J[2]*n[2];
      tce13 = tce12*tce7;
      tce14 = tce10*tce5;
      tce15 = tce12*tce3;
      tce16 = L_I[1]*udNx1[j];
      tce17 = 1.0*L_J[0];
      tce18 = tce16*tce17;
      tce19 = tce18*tce5;
      tce20 = tce12*tce18;
      tce21 = L_I[2]*udNx2[j];
      tce22 = tce17*tce21;
      tce23 = tce22*tce5;
      tce24 = tce12*tce22;
      tce25 = pow(L_J[0], 2)*n[0];
      tce26 = tce25*tce6;
      tce27 = pow(L_J[1], 2)*n[1]*tce2;
      tce28 = pow(L_J[2], 2)*n[2]*tce9;
      tce29 = 1.0*tce25;
      tce30 = tce16*tce29;
      tce31 = tce21*tce29;
      A[(3*i + 0)*81 + (3*j + 0)] += scale * (L_I[0]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce13 + H_IJ*tce14 + H_IJ*tce15 + H_IJ*tce19 + H_IJ*tce20 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce30 + H_IJ*tce31 + H_IJ*tce4 + H_IJ*tce8 - tce11 - tce13 - tce14 - tce15 - tce19 - tce20 - tce23 - tce24 - tce26 - tce27 - tce28 - tce30 - tce31 - tce4 - tce8));
      }
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30;
      tce0 = 1.0*udNx0[j];
      tce1 = L_I[0]*tce0;
      tce2 = L_J[1]*tce1;
      tce3 = L_J[0]*n[0];
      tce4 = tce2*tce3;
      tce5 = L_J[2]*n[2];
      tce6 = tce2*tce5;
      tce7 = 2.0*udNx1[j];
      tce8 = L_I[1]*tce7;
      tce9 = L_J[1]*tce3*tce8;
      tce10 = L_I[1]*tce0;
      tce11 = L_J[0]*tce10;
      tce12 = L_J[1]*n[1];
      tce13 = tce11*tce12;
      tce14 = 1.0*udNx2[j];
      tce15 = L_I[1]*L_J[2]*tce14;
      tce16 = tce15*tce3;
      tce17 = tce11*tce5;
      tce18 = tce12*tce15;
      tce19 = L_I[1]*n[2];
      tce20 = L_J[1]*L_J[2]*tce19*tce7;
      tce21 = L_I[2]*tce14;
      tce22 = L_J[1]*tce21;
      tce23 = tce22*tce3;
      tce24 = tce22*tce5;
      tce25 = pow(L_J[1], 2)*n[1];
      tce26 = tce1*tce25;
      tce27 = pow(L_J[0], 2)*n[0]*tce10;
      tce28 = tce25*tce8;
      tce29 = pow(L_J[2], 2)*tce14*tce19;
      tce30 = tce21*tce25;
      A[(3*i + 0)*81 + (3*j + 1)] += scale * (L_I[0]*eta*wNt[i]*(H_IJ*tce13 + H_IJ*tce16 + H_IJ*tce17 + H_IJ*tce18 + H_IJ*tce20 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce29 + H_IJ*tce30 + H_IJ*tce4 + H_IJ*tce6 + H_IJ*tce9 - tce13 - tce16 - tce17 - tce18 - tce20 - tce23 - tce24 - tce26 - tce27 - tce28 - tce29 - tce30 - tce4 - tce6 - tce9));
      }
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30;
      tce0 = 1.0*udNx0[j];
      tce1 = L_I[0]*tce0;
      tce2 = L_J[2]*tce1;
      tce3 = L_J[0]*n[0];
      tce4 = tce2*tce3;
      tce5 = L_J[1]*n[1];
      tce6 = tce2*tce5;
      tce7 = 1.0*udNx1[j];
      tce8 = L_I[1]*tce7;
      tce9 = L_J[2]*tce8;
      tce10 = tce3*tce9;
      tce11 = tce5*tce9;
      tce12 = L_I[2]*n[0];
      tce13 = L_J[0]*L_J[1];
      tce14 = tce12*tce13*tce7;
      tce15 = L_I[2]*n[1];
      tce16 = tce0*tce13*tce15;
      tce17 = 2.0*udNx2[j];
      tce18 = L_J[0]*L_J[2];
      tce19 = tce12*tce17*tce18;
      tce20 = L_I[2]*n[2];
      tce21 = tce0*tce18*tce20;
      tce22 = L_J[1]*L_J[2];
      tce23 = tce15*tce17*tce22;
      tce24 = tce20*tce22*tce7;
      tce25 = pow(L_J[2], 2)*n[2];
      tce26 = tce1*tce25;
      tce27 = tce25*tce8;
      tce28 = pow(L_J[0], 2)*tce0*tce12;
      tce29 = pow(L_J[1], 2)*tce15*tce7;
      tce30 = L_I[2]*tce17*tce25;
      A[(3*i + 0)*81 + (3*j + 2)] += scale * (L_I[0]*eta*wNt[i]*(H_IJ*tce10 + H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce29 + H_IJ*tce30 + H_IJ*tce4 + H_IJ*tce6 - tce10 - tce11 - tce14 - tce16 - tce19 - tce21 - tce23 - tce24 - tce26 - tce27 - tce28 - tce29 - tce30 - tce4 - tce6));
      }
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31;
      tce0 = L_J[0]*n[0];
      tce1 = 1.0*L_I[0];
      tce2 = tce1*udNx1[j];
      tce3 = L_J[1]*tce2;
      tce4 = tce0*tce3;
      tce5 = L_J[1]*n[1];
      tce6 = 2.0*L_I[0]*udNx0[j];
      tce7 = L_J[0]*tce6;
      tce8 = tce5*tce7;
      tce9 = tce1*udNx2[j];
      tce10 = L_J[2]*tce9;
      tce11 = tce0*tce10;
      tce12 = L_J[2]*n[2];
      tce13 = tce12*tce7;
      tce14 = tce10*tce5;
      tce15 = tce12*tce3;
      tce16 = L_I[1]*udNx1[j];
      tce17 = 1.0*L_J[0];
      tce18 = tce16*tce17;
      tce19 = tce18*tce5;
      tce20 = tce12*tce18;
      tce21 = L_I[2]*udNx2[j];
      tce22 = tce17*tce21;
      tce23 = tce22*tce5;
      tce24 = tce12*tce22;
      tce25 = pow(L_J[0], 2)*n[0];
      tce26 = tce25*tce6;
      tce27 = pow(L_J[1], 2)*n[1]*tce2;
      tce28 = pow(L_J[2], 2)*n[2]*tce9;
      tce29 = 1.0*tce25;
      tce30 = tce16*tce29;
      tce31 = tce21*tce29;
      A[(3*i + 1)*81 + (3*j + 0)] += scale * (L_I[1]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce13 + H_IJ*tce14 + H_IJ*tce15 + H_IJ*tce19 + H_IJ*tce20 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce30 + H_IJ*tce31 + H_IJ*tce4 + H_IJ*tce8 - tce11 - tce13 - tce14 - tce15 - tce19 - tce20 - tce23 - tce24 - tce26 - tce27 - tce28 - tce30 - tce31 - tce4 - tce8));
      }
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30;
      tce0 = 1.0*udNx0[j];
      tce1 = L_I[0]*tce0;
      tce2 = L_J[1]*tce1;
      tce3 = L_J[0]*n[0];
      tce4 = tce2*tce3;
      tce5 = L_J[2]*n[2];
      tce6 = tce2*tce5;
      tce7 = 2.0*udNx1[j];
      tce8 = L_I[1]*tce7;
      tce9 = L_J[1]*tce3*tce8;
      tce10 = L_I[1]*tce0;
      tce11 = L_J[0]*tce10;
      tce12 = L_J[1]*n[1];
      tce13 = tce11*tce12;
      tce14 = 1.0*udNx2[j];
      tce15 = L_I[1]*L_J[2]*tce14;
      tce16 = tce15*tce3;
      tce17 = tce11*tce5;
      tce18 = tce12*tce15;
      tce19 = L_I[1]*n[2];
      tce20 = L_J[1]*L_J[2]*tce19*tce7;
      tce21 = L_I[2]*tce14;
      tce22 = L_J[1]*tce21;
      tce23 = tce22*tce3;
      tce24 = tce22*tce5;
      tce25 = pow(L_J[1], 2)*n[1];
      tce26 = tce1*tce25;
      tce27 = pow(L_J[0], 2)*n[0]*tce10;
      tce28 = tce25*tce8;
      tce29 = pow(L_J[2], 2)*tce14*tce19;
      tce30 = tce21*tce25;
      A[(3*i + 1)*81 + (3*j + 1)] += scale * (L_I[1]*eta*wNt[i]*(H_IJ*tce13 + H_IJ*tce16 + H_IJ*tce17 + H_IJ*tce18 + H_IJ*tce20 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce29 + H_IJ*tce30 + H_IJ*tce4 + H_IJ*tce6 + H_IJ*tce9 - tce13 - tce16 - tce17 - tce18 - tce20 - tce23 - tce24 - tce26 - tce27 - tce28 - tce29 - tce30 - tce4 - tce6 - tce9));
      }
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30;
      tce0 = 1.0*udNx0[j];
      tce1 = L_I[0]*tce0;
      tce2 = L_J[2]*tce1;
      tce3 = L_J[0]*n[0];
      tce4 = tce2*tce3;
      tce5 = L_J[1]*n[1];
      tce6 = tce2*tce5;
      tce7 = 1.0*udNx1[j];
      tce8 = L_I[1]*tce7;
      tce9 = L_J[2]*tce8;
      tce10 = tce3*tce9;
      tce11 = tce5*tce9;
      tce12 = L_I[2]*n[0];
      tce13 = L_J[0]*L_J[1];
      tce14 = tce12*tce13*tce7;
      tce15 = L_I[2]*n[1];
      tce16 = tce0*tce13*tce15;
      tce17 = 2.0*udNx2[j];
      tce18 = L_J[0]*L_J[2];
      tce19 = tce12*tce17*tce18;
      tce20 = L_I[2]*n[2];
      tce21 = tce0*tce18*tce20;
      tce22 = L_J[1]*L_J[2];
      tce23 = tce15*tce17*tce22;
      tce24 = tce20*tce22*tce7;
      tce25 = pow(L_J[2], 2)*n[2];
      tce26 = tce1*tce25;
      tce27 = tce25*tce8;
      tce28 = pow(L_J[0], 2)*tce0*tce12;
      tce29 = pow(L_J[1], 2)*tce15*tce7;
      tce30 = L_I[2]*tce17*tce25;
      A[(3*i + 1)*81 + (3*j + 2)] += scale * (L_I[1]*eta*wNt[i]*(H_IJ*tce10 + H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce29 + H_IJ*tce30 + H_IJ*tce4 + H_IJ*tce6 - tce10 - tce11 - tce14 - tce16 - tce19 - tce21 - tce23 - tce24 - tce26 - tce27 - tce28 - tce29 - tce30 - tce4 - tce6));
      }
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31;
      tce0 = L_J[0]*n[0];
      tce1 = 1.0*L_I[0];
      tce2 = tce1*udNx1[j];
      tce3 = L_J[1]*tce2;
      tce4 = tce0*tce3;
      tce5 = L_J[1]*n[1];
      tce6 = 2.0*L_I[0]*udNx0[j];
      tce7 = L_J[0]*tce6;
      tce8 = tce5*tce7;
      tce9 = tce1*udNx2[j];
      tce10 = L_J[2]*tce9;
      tce11 = tce0*tce10;
      tce12 = L_J[2]*n[2];
      tce13 = tce12*tce7;
      tce14 = tce10*tce5;
      tce15 = tce12*tce3;
      tce16 = L_I[1]*udNx1[j];
      tce17 = 1.0*L_J[0];
      tce18 = tce16*tce17;
      tce19 = tce18*tce5;
      tce20 = tce12*tce18;
      tce21 = L_I[2]*udNx2[j];
      tce22 = tce17*tce21;
      tce23 = tce22*tce5;
      tce24 = tce12*tce22;
      tce25 = pow(L_J[0], 2)*n[0];
      tce26 = tce25*tce6;
      tce27 = pow(L_J[1], 2)*n[1]*tce2;
      tce28 = pow(L_J[2], 2)*n[2]*tce9;
      tce29 = 1.0*tce25;
      tce30 = tce16*tce29;
      tce31 = tce21*tce29;
      A[(3*i + 2)*81 + (3*j + 0)] += scale * (L_I[2]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce13 + H_IJ*tce14 + H_IJ*tce15 + H_IJ*tce19 + H_IJ*tce20 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce30 + H_IJ*tce31 + H_IJ*tce4 + H_IJ*tce8 - tce11 - tce13 - tce14 - tce15 - tce19 - tce20 - tce23 - tce24 - tce26 - tce27 - tce28 - tce30 - tce31 - tce4 - tce8));
      }
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30;
      tce0 = 1.0*udNx0[j];
      tce1 = L_I[0]*tce0;
      tce2 = L_J[1]*tce1;
      tce3 = L_J[0]*n[0];
      tce4 = tce2*tce3;
      tce5 = L_J[2]*n[2];
      tce6 = tce2*tce5;
      tce7 = 2.0*udNx1[j];
      tce8 = L_I[1]*tce7;
      tce9 = L_J[1]*tce3*tce8;
      tce10 = L_I[1]*tce0;
      tce11 = L_J[0]*tce10;
      tce12 = L_J[1]*n[1];
      tce13 = tce11*tce12;
      tce14 = 1.0*udNx2[j];
      tce15 = L_I[1]*L_J[2]*tce14;
      tce16 = tce15*tce3;
      tce17 = tce11*tce5;
      tce18 = tce12*tce15;
      tce19 = L_I[1]*n[2];
      tce20 = L_J[1]*L_J[2]*tce19*tce7;
      tce21 = L_I[2]*tce14;
      tce22 = L_J[1]*tce21;
      tce23 = tce22*tce3;
      tce24 = tce22*tce5;
      tce25 = pow(L_J[1], 2)*n[1];
      tce26 = tce1*tce25;
      tce27 = pow(L_J[0], 2)*n[0]*tce10;
      tce28 = tce25*tce8;
      tce29 = pow(L_J[2], 2)*tce14*tce19;
      tce30 = tce21*tce25;
      A[(3*i + 2)*81 + (3*j + 1)] += scale * (L_I[2]*eta*wNt[i]*(H_IJ*tce13 + H_IJ*tce16 + H_IJ*tce17 + H_IJ*tce18 + H_IJ*tce20 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce29 + H_IJ*tce30 + H_IJ*tce4 + H_IJ*tce6 + H_IJ*tce9 - tce13 - tce16 - tce17 - tce18 - tce20 - tce23 - tce24 - tce26 - tce27 - tce28 - tce29 - tce30 - tce4 - tce6 - tce9));
      }
      {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30;
      tce0 = 1.0*udNx0[j];
      tce1 = L_I[0]*tce0;
      tce2 = L_J[2]*tce1;
      tce3 = L_J[0]*n[0];
      tce4 = tce2*tce3;
      tce5 = L_J[1]*n[1];
      tce6 = tce2*tce5;
      tce7 = 1.0*udNx1[j];
      tce8 = L_I[1]*tce7;
      tce9 = L_J[2]*tce8;
      tce10 = tce3*tce9;
      tce11 = tce5*tce9;
      tce12 = L_I[2]*n[0];
      tce13 = L_J[0]*L_J[1];
      tce14 = tce12*tce13*tce7;
      tce15 = L_I[2]*n[1];
      tce16 = tce0*tce13*tce15;
      tce17 = 2.0*udNx2[j];
      tce18 = L_J[0]*L_J[2];
      tce19 = tce12*tce17*tce18;
      tce20 = L_I[2]*n[2];
      tce21 = tce0*tce18*tce20;
      tce22 = L_J[1]*L_J[2];
      tce23 = tce15*tce17*tce22;
      tce24 = tce20*tce22*tce7;
      tce25 = pow(L_J[2], 2)*n[2];
      tce26 = tce1*tce25;
      tce27 = tce25*tce8;
      tce28 = pow(L_J[0], 2)*tce0*tce12;
      tce29 = pow(L_J[1], 2)*tce15*tce7;
      tce30 = L_I[2]*tce17*tce25;
      A[(3*i + 2)*81 + (3*j + 2)] += scale * (L_I[2]*eta*wNt[i]*(H_IJ*tce10 + H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce29 + H_IJ*tce30 + H_IJ*tce4 + H_IJ*tce6 - tce10 - tce11 - tce14 - tce16 - tce19 - tce21 - tce23 - tce24 - tce26 - tce27 - tce28 - tce29 - tce30 - tce4 - tce6));
      }
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
  for (i=0; i<27; i++) { // w_nbasis
    j = i;
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31;
    tce0 = L_J[0]*n[0];
    tce1 = 1.0*L_I[0];
    tce2 = tce1*udNx1[j];
    tce3 = L_J[1]*tce2;
    tce4 = tce0*tce3;
    tce5 = L_J[1]*n[1];
    tce6 = 2.0*L_I[0]*udNx0[j];
    tce7 = L_J[0]*tce6;
    tce8 = tce5*tce7;
    tce9 = tce1*udNx2[j];
    tce10 = L_J[2]*tce9;
    tce11 = tce0*tce10;
    tce12 = L_J[2]*n[2];
    tce13 = tce12*tce7;
    tce14 = tce10*tce5;
    tce15 = tce12*tce3;
    tce16 = L_I[1]*udNx1[j];
    tce17 = 1.0*L_J[0];
    tce18 = tce16*tce17;
    tce19 = tce18*tce5;
    tce20 = tce12*tce18;
    tce21 = L_I[2]*udNx2[j];
    tce22 = tce17*tce21;
    tce23 = tce22*tce5;
    tce24 = tce12*tce22;
    tce25 = pow(L_J[0], 2)*n[0];
    tce26 = tce25*tce6;
    tce27 = pow(L_J[1], 2)*n[1]*tce2;
    tce28 = pow(L_J[2], 2)*n[2]*tce9;
    tce29 = 1.0*tce25;
    tce30 = tce16*tce29;
    tce31 = tce21*tce29;
    F[3*i + 0] += scale * (L_I[0]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce13 + H_IJ*tce14 + H_IJ*tce15 + H_IJ*tce19 + H_IJ*tce20 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce30 + H_IJ*tce31 + H_IJ*tce4 + H_IJ*tce8 - tce11 - tce13 - tce14 - tce15 - tce19 - tce20 - tce23 - tce24 - tce26 - tce27 - tce28 - tce30 - tce31 - tce4 - tce8));
    }
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30;
    tce0 = 1.0*udNx0[j];
    tce1 = L_I[0]*tce0;
    tce2 = L_J[1]*tce1;
    tce3 = L_J[0]*n[0];
    tce4 = tce2*tce3;
    tce5 = L_J[2]*n[2];
    tce6 = tce2*tce5;
    tce7 = 2.0*udNx1[j];
    tce8 = L_I[1]*tce7;
    tce9 = L_J[1]*tce3*tce8;
    tce10 = L_I[1]*tce0;
    tce11 = L_J[0]*tce10;
    tce12 = L_J[1]*n[1];
    tce13 = tce11*tce12;
    tce14 = 1.0*udNx2[j];
    tce15 = L_I[1]*L_J[2]*tce14;
    tce16 = tce15*tce3;
    tce17 = tce11*tce5;
    tce18 = tce12*tce15;
    tce19 = L_I[1]*n[2];
    tce20 = L_J[1]*L_J[2]*tce19*tce7;
    tce21 = L_I[2]*tce14;
    tce22 = L_J[1]*tce21;
    tce23 = tce22*tce3;
    tce24 = tce22*tce5;
    tce25 = pow(L_J[1], 2)*n[1];
    tce26 = tce1*tce25;
    tce27 = pow(L_J[0], 2)*n[0]*tce10;
    tce28 = tce25*tce8;
    tce29 = pow(L_J[2], 2)*tce14*tce19;
    tce30 = tce21*tce25;
    F[3*i + 1] += scale * (L_I[1]*eta*wNt[i]*(H_IJ*tce13 + H_IJ*tce16 + H_IJ*tce17 + H_IJ*tce18 + H_IJ*tce20 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce29 + H_IJ*tce30 + H_IJ*tce4 + H_IJ*tce6 + H_IJ*tce9 - tce13 - tce16 - tce17 - tce18 - tce20 - tce23 - tce24 - tce26 - tce27 - tce28 - tce29 - tce30 - tce4 - tce6 - tce9));
    }
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30;
    tce0 = 1.0*udNx0[j];
    tce1 = L_I[0]*tce0;
    tce2 = L_J[2]*tce1;
    tce3 = L_J[0]*n[0];
    tce4 = tce2*tce3;
    tce5 = L_J[1]*n[1];
    tce6 = tce2*tce5;
    tce7 = 1.0*udNx1[j];
    tce8 = L_I[1]*tce7;
    tce9 = L_J[2]*tce8;
    tce10 = tce3*tce9;
    tce11 = tce5*tce9;
    tce12 = L_I[2]*n[0];
    tce13 = L_J[0]*L_J[1];
    tce14 = tce12*tce13*tce7;
    tce15 = L_I[2]*n[1];
    tce16 = tce0*tce13*tce15;
    tce17 = 2.0*udNx2[j];
    tce18 = L_J[0]*L_J[2];
    tce19 = tce12*tce17*tce18;
    tce20 = L_I[2]*n[2];
    tce21 = tce0*tce18*tce20;
    tce22 = L_J[1]*L_J[2];
    tce23 = tce15*tce17*tce22;
    tce24 = tce20*tce22*tce7;
    tce25 = pow(L_J[2], 2)*n[2];
    tce26 = tce1*tce25;
    tce27 = tce25*tce8;
    tce28 = pow(L_J[0], 2)*tce0*tce12;
    tce29 = pow(L_J[1], 2)*tce15*tce7;
    tce30 = L_I[2]*tce17*tce25;
    F[3*i + 2] += scale * (L_I[2]*eta*wNt[i]*(H_IJ*tce10 + H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce26 + H_IJ*tce27 + H_IJ*tce28 + H_IJ*tce29 + H_IJ*tce30 + H_IJ*tce4 + H_IJ*tce6 - tce10 - tce11 - tce14 - tce16 - tce19 - tce21 - tce23 - tce24 - tce26 - tce27 - tce28 - tce29 - tce30 - tce4 - tce6));
    }
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
  for (i=0; i<27; i++) { // w_nbasis
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78, tce79;
    tce0 = L_J[0]*n[0];
    tce1 = 1.0*L_I[0];
    tce2 = tce1*u0j_udNx1j;
    tce3 = L_J[1]*tce2;
    tce4 = tce0*tce3;
    tce5 = tce1*u1j_udNx0j;
    tce6 = L_J[1]*tce5;
    tce7 = tce0*tce6;
    tce8 = 2.0*L_I[0]*u0j_udNx0j;
    tce9 = L_J[1]*n[1];
    tce10 = L_J[0]*tce9;
    tce11 = tce10*tce8;
    tce12 = tce1*u0j_udNx2j;
    tce13 = L_J[2]*tce0;
    tce14 = tce12*tce13;
    tce15 = tce1*u2j_udNx0j;
    tce16 = tce13*tce15;
    tce17 = L_J[2]*n[2];
    tce18 = L_J[0]*tce17;
    tce19 = tce18*tce8;
    tce20 = L_J[2]*tce9;
    tce21 = tce12*tce20;
    tce22 = tce15*tce20;
    tce23 = tce17*tce3;
    tce24 = tce17*tce6;
    tce25 = 2.0*L_I[1]*u1j_udNx1j;
    tce26 = L_J[1]*tce25;
    tce27 = tce0*tce26;
    tce28 = 1.0*L_I[1];
    tce29 = tce28*u0j_udNx1j;
    tce30 = tce10*tce29;
    tce31 = tce28*u1j_udNx0j;
    tce32 = tce10*tce31;
    tce33 = tce13*tce28;
    tce34 = tce33*u1j_udNx2j;
    tce35 = tce33*u2j_udNx1j;
    tce36 = tce18*tce29;
    tce37 = tce18*tce31;
    tce38 = tce20*tce28;
    tce39 = tce38*u1j_udNx2j;
    tce40 = tce38*u2j_udNx1j;
    tce41 = tce17*tce26;
    tce42 = 1.0*L_I[2];
    tce43 = L_J[1]*tce42;
    tce44 = tce0*tce43;
    tce45 = tce44*u1j_udNx2j;
    tce46 = tce44*u2j_udNx1j;
    tce47 = tce10*tce42;
    tce48 = tce47*u0j_udNx2j;
    tce49 = tce47*u2j_udNx0j;
    tce50 = 2.0*L_I[2]*u2j_udNx2j;
    tce51 = tce13*tce50;
    tce52 = tce18*tce42;
    tce53 = tce52*u0j_udNx2j;
    tce54 = tce52*u2j_udNx0j;
    tce55 = tce20*tce50;
    tce56 = tce17*tce43;
    tce57 = tce56*u1j_udNx2j;
    tce58 = tce56*u2j_udNx1j;
    tce59 = pow(L_J[0], 2)*n[0];
    tce60 = tce59*tce8;
    tce61 = pow(L_J[1], 2)*n[1];
    tce62 = tce2*tce61;
    tce63 = tce5*tce61;
    tce64 = pow(L_J[2], 2)*n[2];
    tce65 = tce12*tce64;
    tce66 = tce15*tce64;
    tce67 = tce29*tce59;
    tce68 = tce31*tce59;
    tce69 = tce25*tce61;
    tce70 = tce28*tce64;
    tce71 = tce70*u1j_udNx2j;
    tce72 = tce70*u2j_udNx1j;
    tce73 = tce42*tce59;
    tce74 = tce73*u0j_udNx2j;
    tce75 = tce73*u2j_udNx0j;
    tce76 = tce42*tce61;
    tce77 = tce76*u1j_udNx2j;
    tce78 = tce76*u2j_udNx1j;
    tce79 = tce50*tce64;
    F[3*i + 0] += scale * (L_I[0]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce22 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce27 + H_IJ*tce30 + H_IJ*tce32 + H_IJ*tce34 + H_IJ*tce35 + H_IJ*tce36 + H_IJ*tce37 + H_IJ*tce39 + H_IJ*tce4 + H_IJ*tce40 + H_IJ*tce41 + H_IJ*tce45 + H_IJ*tce46 + H_IJ*tce48 + H_IJ*tce49 + H_IJ*tce51 + H_IJ*tce53 + H_IJ*tce54 + H_IJ*tce55 + H_IJ*tce57 + H_IJ*tce58 + H_IJ*tce60 + H_IJ*tce62 + H_IJ*tce63 + H_IJ*tce65 + H_IJ*tce66 + H_IJ*tce67 + H_IJ*tce68 + H_IJ*tce69 + H_IJ*tce7 + H_IJ*tce71 + H_IJ*tce72 + H_IJ*tce74 + H_IJ*tce75 + H_IJ*tce77 + H_IJ*tce78 + H_IJ*tce79 - tce11 - tce14 - tce16 - tce19 - tce21 - tce22 - tce23 - tce24 - tce27 - tce30 - tce32 - tce34 - tce35 - tce36 - tce37 - tce39 - tce4 - tce40 - tce41 - tce45 - tce46 - tce48 - tce49 - tce51 - tce53 - tce54 - tce55 - tce57 - tce58 - tce60 - tce62 - tce63 - tce65 - tce66 - tce67 - tce68 - tce69 - tce7 - tce71 - tce72 - tce74 - tce75 - tce77 - tce78 - tce79));
    }
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78, tce79;
    tce0 = L_J[0]*n[0];
    tce1 = 1.0*L_I[0];
    tce2 = tce1*u0j_udNx1j;
    tce3 = L_J[1]*tce2;
    tce4 = tce0*tce3;
    tce5 = tce1*u1j_udNx0j;
    tce6 = L_J[1]*tce5;
    tce7 = tce0*tce6;
    tce8 = 2.0*L_I[0]*u0j_udNx0j;
    tce9 = L_J[1]*n[1];
    tce10 = L_J[0]*tce9;
    tce11 = tce10*tce8;
    tce12 = tce1*u0j_udNx2j;
    tce13 = L_J[2]*tce0;
    tce14 = tce12*tce13;
    tce15 = tce1*u2j_udNx0j;
    tce16 = tce13*tce15;
    tce17 = L_J[2]*n[2];
    tce18 = L_J[0]*tce17;
    tce19 = tce18*tce8;
    tce20 = L_J[2]*tce9;
    tce21 = tce12*tce20;
    tce22 = tce15*tce20;
    tce23 = tce17*tce3;
    tce24 = tce17*tce6;
    tce25 = 2.0*L_I[1]*u1j_udNx1j;
    tce26 = L_J[1]*tce25;
    tce27 = tce0*tce26;
    tce28 = 1.0*L_I[1];
    tce29 = tce28*u0j_udNx1j;
    tce30 = tce10*tce29;
    tce31 = tce28*u1j_udNx0j;
    tce32 = tce10*tce31;
    tce33 = tce13*tce28;
    tce34 = tce33*u1j_udNx2j;
    tce35 = tce33*u2j_udNx1j;
    tce36 = tce18*tce29;
    tce37 = tce18*tce31;
    tce38 = tce20*tce28;
    tce39 = tce38*u1j_udNx2j;
    tce40 = tce38*u2j_udNx1j;
    tce41 = tce17*tce26;
    tce42 = 1.0*L_I[2];
    tce43 = L_J[1]*tce42;
    tce44 = tce0*tce43;
    tce45 = tce44*u1j_udNx2j;
    tce46 = tce44*u2j_udNx1j;
    tce47 = tce10*tce42;
    tce48 = tce47*u0j_udNx2j;
    tce49 = tce47*u2j_udNx0j;
    tce50 = 2.0*L_I[2]*u2j_udNx2j;
    tce51 = tce13*tce50;
    tce52 = tce18*tce42;
    tce53 = tce52*u0j_udNx2j;
    tce54 = tce52*u2j_udNx0j;
    tce55 = tce20*tce50;
    tce56 = tce17*tce43;
    tce57 = tce56*u1j_udNx2j;
    tce58 = tce56*u2j_udNx1j;
    tce59 = pow(L_J[0], 2)*n[0];
    tce60 = tce59*tce8;
    tce61 = pow(L_J[1], 2)*n[1];
    tce62 = tce2*tce61;
    tce63 = tce5*tce61;
    tce64 = pow(L_J[2], 2)*n[2];
    tce65 = tce12*tce64;
    tce66 = tce15*tce64;
    tce67 = tce29*tce59;
    tce68 = tce31*tce59;
    tce69 = tce25*tce61;
    tce70 = tce28*tce64;
    tce71 = tce70*u1j_udNx2j;
    tce72 = tce70*u2j_udNx1j;
    tce73 = tce42*tce59;
    tce74 = tce73*u0j_udNx2j;
    tce75 = tce73*u2j_udNx0j;
    tce76 = tce42*tce61;
    tce77 = tce76*u1j_udNx2j;
    tce78 = tce76*u2j_udNx1j;
    tce79 = tce50*tce64;
    F[3*i + 1] += scale * (L_I[1]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce22 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce27 + H_IJ*tce30 + H_IJ*tce32 + H_IJ*tce34 + H_IJ*tce35 + H_IJ*tce36 + H_IJ*tce37 + H_IJ*tce39 + H_IJ*tce4 + H_IJ*tce40 + H_IJ*tce41 + H_IJ*tce45 + H_IJ*tce46 + H_IJ*tce48 + H_IJ*tce49 + H_IJ*tce51 + H_IJ*tce53 + H_IJ*tce54 + H_IJ*tce55 + H_IJ*tce57 + H_IJ*tce58 + H_IJ*tce60 + H_IJ*tce62 + H_IJ*tce63 + H_IJ*tce65 + H_IJ*tce66 + H_IJ*tce67 + H_IJ*tce68 + H_IJ*tce69 + H_IJ*tce7 + H_IJ*tce71 + H_IJ*tce72 + H_IJ*tce74 + H_IJ*tce75 + H_IJ*tce77 + H_IJ*tce78 + H_IJ*tce79 - tce11 - tce14 - tce16 - tce19 - tce21 - tce22 - tce23 - tce24 - tce27 - tce30 - tce32 - tce34 - tce35 - tce36 - tce37 - tce39 - tce4 - tce40 - tce41 - tce45 - tce46 - tce48 - tce49 - tce51 - tce53 - tce54 - tce55 - tce57 - tce58 - tce60 - tce62 - tce63 - tce65 - tce66 - tce67 - tce68 - tce69 - tce7 - tce71 - tce72 - tce74 - tce75 - tce77 - tce78 - tce79));
    }
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78, tce79;
    tce0 = L_J[0]*n[0];
    tce1 = 1.0*L_I[0];
    tce2 = tce1*u0j_udNx1j;
    tce3 = L_J[1]*tce2;
    tce4 = tce0*tce3;
    tce5 = tce1*u1j_udNx0j;
    tce6 = L_J[1]*tce5;
    tce7 = tce0*tce6;
    tce8 = 2.0*L_I[0]*u0j_udNx0j;
    tce9 = L_J[1]*n[1];
    tce10 = L_J[0]*tce9;
    tce11 = tce10*tce8;
    tce12 = tce1*u0j_udNx2j;
    tce13 = L_J[2]*tce0;
    tce14 = tce12*tce13;
    tce15 = tce1*u2j_udNx0j;
    tce16 = tce13*tce15;
    tce17 = L_J[2]*n[2];
    tce18 = L_J[0]*tce17;
    tce19 = tce18*tce8;
    tce20 = L_J[2]*tce9;
    tce21 = tce12*tce20;
    tce22 = tce15*tce20;
    tce23 = tce17*tce3;
    tce24 = tce17*tce6;
    tce25 = 2.0*L_I[1]*u1j_udNx1j;
    tce26 = L_J[1]*tce25;
    tce27 = tce0*tce26;
    tce28 = 1.0*L_I[1];
    tce29 = tce28*u0j_udNx1j;
    tce30 = tce10*tce29;
    tce31 = tce28*u1j_udNx0j;
    tce32 = tce10*tce31;
    tce33 = tce13*tce28;
    tce34 = tce33*u1j_udNx2j;
    tce35 = tce33*u2j_udNx1j;
    tce36 = tce18*tce29;
    tce37 = tce18*tce31;
    tce38 = tce20*tce28;
    tce39 = tce38*u1j_udNx2j;
    tce40 = tce38*u2j_udNx1j;
    tce41 = tce17*tce26;
    tce42 = 1.0*L_I[2];
    tce43 = L_J[1]*tce42;
    tce44 = tce0*tce43;
    tce45 = tce44*u1j_udNx2j;
    tce46 = tce44*u2j_udNx1j;
    tce47 = tce10*tce42;
    tce48 = tce47*u0j_udNx2j;
    tce49 = tce47*u2j_udNx0j;
    tce50 = 2.0*L_I[2]*u2j_udNx2j;
    tce51 = tce13*tce50;
    tce52 = tce18*tce42;
    tce53 = tce52*u0j_udNx2j;
    tce54 = tce52*u2j_udNx0j;
    tce55 = tce20*tce50;
    tce56 = tce17*tce43;
    tce57 = tce56*u1j_udNx2j;
    tce58 = tce56*u2j_udNx1j;
    tce59 = pow(L_J[0], 2)*n[0];
    tce60 = tce59*tce8;
    tce61 = pow(L_J[1], 2)*n[1];
    tce62 = tce2*tce61;
    tce63 = tce5*tce61;
    tce64 = pow(L_J[2], 2)*n[2];
    tce65 = tce12*tce64;
    tce66 = tce15*tce64;
    tce67 = tce29*tce59;
    tce68 = tce31*tce59;
    tce69 = tce25*tce61;
    tce70 = tce28*tce64;
    tce71 = tce70*u1j_udNx2j;
    tce72 = tce70*u2j_udNx1j;
    tce73 = tce42*tce59;
    tce74 = tce73*u0j_udNx2j;
    tce75 = tce73*u2j_udNx0j;
    tce76 = tce42*tce61;
    tce77 = tce76*u1j_udNx2j;
    tce78 = tce76*u2j_udNx1j;
    tce79 = tce50*tce64;
    F[3*i + 2] += scale * (L_I[2]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce22 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce27 + H_IJ*tce30 + H_IJ*tce32 + H_IJ*tce34 + H_IJ*tce35 + H_IJ*tce36 + H_IJ*tce37 + H_IJ*tce39 + H_IJ*tce4 + H_IJ*tce40 + H_IJ*tce41 + H_IJ*tce45 + H_IJ*tce46 + H_IJ*tce48 + H_IJ*tce49 + H_IJ*tce51 + H_IJ*tce53 + H_IJ*tce54 + H_IJ*tce55 + H_IJ*tce57 + H_IJ*tce58 + H_IJ*tce60 + H_IJ*tce62 + H_IJ*tce63 + H_IJ*tce65 + H_IJ*tce66 + H_IJ*tce67 + H_IJ*tce68 + H_IJ*tce69 + H_IJ*tce7 + H_IJ*tce71 + H_IJ*tce72 + H_IJ*tce74 + H_IJ*tce75 + H_IJ*tce77 + H_IJ*tce78 + H_IJ*tce79 - tce11 - tce14 - tce16 - tce19 - tce21 - tce22 - tce23 - tce24 - tce27 - tce30 - tce32 - tce34 - tce35 - tce36 - tce37 - tce39 - tce4 - tce40 - tce41 - tce45 - tce46 - tce48 - tce49 - tce51 - tce53 - tce54 - tce55 - tce57 - tce58 - tce60 - tce62 - tce63 - tce65 - tce66 - tce67 - tce68 - tce69 - tce7 - tce71 - tce72 - tce74 - tce75 - tce77 - tce78 - tce79));
    }
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
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      A[(3*i + 0)*4 + (1*j + 0)] += scale * (0);
      A[(3*i + 1)*4 + (1*j + 0)] += scale * (0);
      A[(3*i + 2)*4 + (1*j + 0)] += scale * (0);
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
  for (i=0; i<27; i++) { // w_nbasis
    F[3*i + 0] += scale * (0);
    F[3*i + 1] += scale * (0);
    F[3*i + 2] += scale * (0);
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
  for (i=0; i<4; i++) { // q_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      A[(1*i + 0)*81 + (3*j + 0)] += scale * (0);
      A[(1*i + 0)*81 + (3*j + 1)] += scale * (0);
      A[(1*i + 0)*81 + (3*j + 2)] += scale * (0);
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
  for (i=0; i<4; i++) { // q_nbasis
    F[1*i + 0] += scale * (0);
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
  for (i=0; i<4; i++) { // q_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      A[(1*i + 0)*4 + (1*j + 0)] += scale * (0);
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
  for (i=0; i<4; i++) { // q_nbasis
    j = i;
    F[1*i + 0] += scale * (0);
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
  for (i=0; i<4; i++) { // q_nbasis
    F[1*i + 0] += scale * (0);
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
  for (i=0; i<27; i++) { // w_nbasis
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78, tce79;
    tce0 = L_J[0]*n[0];
    tce1 = 1.0*L_I[0];
    tce2 = tce1*u0j_udNx1j;
    tce3 = L_J[1]*tce2;
    tce4 = tce0*tce3;
    tce5 = tce1*u1j_udNx0j;
    tce6 = L_J[1]*tce5;
    tce7 = tce0*tce6;
    tce8 = 2.0*L_I[0]*u0j_udNx0j;
    tce9 = L_J[1]*n[1];
    tce10 = L_J[0]*tce9;
    tce11 = tce10*tce8;
    tce12 = tce1*u0j_udNx2j;
    tce13 = L_J[2]*tce0;
    tce14 = tce12*tce13;
    tce15 = tce1*u2j_udNx0j;
    tce16 = tce13*tce15;
    tce17 = L_J[2]*n[2];
    tce18 = L_J[0]*tce17;
    tce19 = tce18*tce8;
    tce20 = L_J[2]*tce9;
    tce21 = tce12*tce20;
    tce22 = tce15*tce20;
    tce23 = tce17*tce3;
    tce24 = tce17*tce6;
    tce25 = 2.0*L_I[1]*u1j_udNx1j;
    tce26 = L_J[1]*tce25;
    tce27 = tce0*tce26;
    tce28 = 1.0*L_I[1];
    tce29 = tce28*u0j_udNx1j;
    tce30 = tce10*tce29;
    tce31 = tce28*u1j_udNx0j;
    tce32 = tce10*tce31;
    tce33 = tce13*tce28;
    tce34 = tce33*u1j_udNx2j;
    tce35 = tce33*u2j_udNx1j;
    tce36 = tce18*tce29;
    tce37 = tce18*tce31;
    tce38 = tce20*tce28;
    tce39 = tce38*u1j_udNx2j;
    tce40 = tce38*u2j_udNx1j;
    tce41 = tce17*tce26;
    tce42 = 1.0*L_I[2];
    tce43 = L_J[1]*tce42;
    tce44 = tce0*tce43;
    tce45 = tce44*u1j_udNx2j;
    tce46 = tce44*u2j_udNx1j;
    tce47 = tce10*tce42;
    tce48 = tce47*u0j_udNx2j;
    tce49 = tce47*u2j_udNx0j;
    tce50 = 2.0*L_I[2]*u2j_udNx2j;
    tce51 = tce13*tce50;
    tce52 = tce18*tce42;
    tce53 = tce52*u0j_udNx2j;
    tce54 = tce52*u2j_udNx0j;
    tce55 = tce20*tce50;
    tce56 = tce17*tce43;
    tce57 = tce56*u1j_udNx2j;
    tce58 = tce56*u2j_udNx1j;
    tce59 = pow(L_J[0], 2)*n[0];
    tce60 = tce59*tce8;
    tce61 = pow(L_J[1], 2)*n[1];
    tce62 = tce2*tce61;
    tce63 = tce5*tce61;
    tce64 = pow(L_J[2], 2)*n[2];
    tce65 = tce12*tce64;
    tce66 = tce15*tce64;
    tce67 = tce29*tce59;
    tce68 = tce31*tce59;
    tce69 = tce25*tce61;
    tce70 = tce28*tce64;
    tce71 = tce70*u1j_udNx2j;
    tce72 = tce70*u2j_udNx1j;
    tce73 = tce42*tce59;
    tce74 = tce73*u0j_udNx2j;
    tce75 = tce73*u2j_udNx0j;
    tce76 = tce42*tce61;
    tce77 = tce76*u1j_udNx2j;
    tce78 = tce76*u2j_udNx1j;
    tce79 = tce50*tce64;
    F[3*i + 0] += scale * (L_I[0]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce22 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce27 + H_IJ*tce30 + H_IJ*tce32 + H_IJ*tce34 + H_IJ*tce35 + H_IJ*tce36 + H_IJ*tce37 + H_IJ*tce39 + H_IJ*tce4 + H_IJ*tce40 + H_IJ*tce41 + H_IJ*tce45 + H_IJ*tce46 + H_IJ*tce48 + H_IJ*tce49 + H_IJ*tce51 + H_IJ*tce53 + H_IJ*tce54 + H_IJ*tce55 + H_IJ*tce57 + H_IJ*tce58 + H_IJ*tce60 + H_IJ*tce62 + H_IJ*tce63 + H_IJ*tce65 + H_IJ*tce66 + H_IJ*tce67 + H_IJ*tce68 + H_IJ*tce69 + H_IJ*tce7 + H_IJ*tce71 + H_IJ*tce72 + H_IJ*tce74 + H_IJ*tce75 + H_IJ*tce77 + H_IJ*tce78 + H_IJ*tce79 - tce11 - tce14 - tce16 - tce19 - tce21 - tce22 - tce23 - tce24 - tce27 - tce30 - tce32 - tce34 - tce35 - tce36 - tce37 - tce39 - tce4 - tce40 - tce41 - tce45 - tce46 - tce48 - tce49 - tce51 - tce53 - tce54 - tce55 - tce57 - tce58 - tce60 - tce62 - tce63 - tce65 - tce66 - tce67 - tce68 - tce69 - tce7 - tce71 - tce72 - tce74 - tce75 - tce77 - tce78 - tce79));
    }
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78, tce79;
    tce0 = L_J[0]*n[0];
    tce1 = 1.0*L_I[0];
    tce2 = tce1*u0j_udNx1j;
    tce3 = L_J[1]*tce2;
    tce4 = tce0*tce3;
    tce5 = tce1*u1j_udNx0j;
    tce6 = L_J[1]*tce5;
    tce7 = tce0*tce6;
    tce8 = 2.0*L_I[0]*u0j_udNx0j;
    tce9 = L_J[1]*n[1];
    tce10 = L_J[0]*tce9;
    tce11 = tce10*tce8;
    tce12 = tce1*u0j_udNx2j;
    tce13 = L_J[2]*tce0;
    tce14 = tce12*tce13;
    tce15 = tce1*u2j_udNx0j;
    tce16 = tce13*tce15;
    tce17 = L_J[2]*n[2];
    tce18 = L_J[0]*tce17;
    tce19 = tce18*tce8;
    tce20 = L_J[2]*tce9;
    tce21 = tce12*tce20;
    tce22 = tce15*tce20;
    tce23 = tce17*tce3;
    tce24 = tce17*tce6;
    tce25 = 2.0*L_I[1]*u1j_udNx1j;
    tce26 = L_J[1]*tce25;
    tce27 = tce0*tce26;
    tce28 = 1.0*L_I[1];
    tce29 = tce28*u0j_udNx1j;
    tce30 = tce10*tce29;
    tce31 = tce28*u1j_udNx0j;
    tce32 = tce10*tce31;
    tce33 = tce13*tce28;
    tce34 = tce33*u1j_udNx2j;
    tce35 = tce33*u2j_udNx1j;
    tce36 = tce18*tce29;
    tce37 = tce18*tce31;
    tce38 = tce20*tce28;
    tce39 = tce38*u1j_udNx2j;
    tce40 = tce38*u2j_udNx1j;
    tce41 = tce17*tce26;
    tce42 = 1.0*L_I[2];
    tce43 = L_J[1]*tce42;
    tce44 = tce0*tce43;
    tce45 = tce44*u1j_udNx2j;
    tce46 = tce44*u2j_udNx1j;
    tce47 = tce10*tce42;
    tce48 = tce47*u0j_udNx2j;
    tce49 = tce47*u2j_udNx0j;
    tce50 = 2.0*L_I[2]*u2j_udNx2j;
    tce51 = tce13*tce50;
    tce52 = tce18*tce42;
    tce53 = tce52*u0j_udNx2j;
    tce54 = tce52*u2j_udNx0j;
    tce55 = tce20*tce50;
    tce56 = tce17*tce43;
    tce57 = tce56*u1j_udNx2j;
    tce58 = tce56*u2j_udNx1j;
    tce59 = pow(L_J[0], 2)*n[0];
    tce60 = tce59*tce8;
    tce61 = pow(L_J[1], 2)*n[1];
    tce62 = tce2*tce61;
    tce63 = tce5*tce61;
    tce64 = pow(L_J[2], 2)*n[2];
    tce65 = tce12*tce64;
    tce66 = tce15*tce64;
    tce67 = tce29*tce59;
    tce68 = tce31*tce59;
    tce69 = tce25*tce61;
    tce70 = tce28*tce64;
    tce71 = tce70*u1j_udNx2j;
    tce72 = tce70*u2j_udNx1j;
    tce73 = tce42*tce59;
    tce74 = tce73*u0j_udNx2j;
    tce75 = tce73*u2j_udNx0j;
    tce76 = tce42*tce61;
    tce77 = tce76*u1j_udNx2j;
    tce78 = tce76*u2j_udNx1j;
    tce79 = tce50*tce64;
    F[3*i + 1] += scale * (L_I[1]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce22 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce27 + H_IJ*tce30 + H_IJ*tce32 + H_IJ*tce34 + H_IJ*tce35 + H_IJ*tce36 + H_IJ*tce37 + H_IJ*tce39 + H_IJ*tce4 + H_IJ*tce40 + H_IJ*tce41 + H_IJ*tce45 + H_IJ*tce46 + H_IJ*tce48 + H_IJ*tce49 + H_IJ*tce51 + H_IJ*tce53 + H_IJ*tce54 + H_IJ*tce55 + H_IJ*tce57 + H_IJ*tce58 + H_IJ*tce60 + H_IJ*tce62 + H_IJ*tce63 + H_IJ*tce65 + H_IJ*tce66 + H_IJ*tce67 + H_IJ*tce68 + H_IJ*tce69 + H_IJ*tce7 + H_IJ*tce71 + H_IJ*tce72 + H_IJ*tce74 + H_IJ*tce75 + H_IJ*tce77 + H_IJ*tce78 + H_IJ*tce79 - tce11 - tce14 - tce16 - tce19 - tce21 - tce22 - tce23 - tce24 - tce27 - tce30 - tce32 - tce34 - tce35 - tce36 - tce37 - tce39 - tce4 - tce40 - tce41 - tce45 - tce46 - tce48 - tce49 - tce51 - tce53 - tce54 - tce55 - tce57 - tce58 - tce60 - tce62 - tce63 - tce65 - tce66 - tce67 - tce68 - tce69 - tce7 - tce71 - tce72 - tce74 - tce75 - tce77 - tce78 - tce79));
    }
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78, tce79;
    tce0 = L_J[0]*n[0];
    tce1 = 1.0*L_I[0];
    tce2 = tce1*u0j_udNx1j;
    tce3 = L_J[1]*tce2;
    tce4 = tce0*tce3;
    tce5 = tce1*u1j_udNx0j;
    tce6 = L_J[1]*tce5;
    tce7 = tce0*tce6;
    tce8 = 2.0*L_I[0]*u0j_udNx0j;
    tce9 = L_J[1]*n[1];
    tce10 = L_J[0]*tce9;
    tce11 = tce10*tce8;
    tce12 = tce1*u0j_udNx2j;
    tce13 = L_J[2]*tce0;
    tce14 = tce12*tce13;
    tce15 = tce1*u2j_udNx0j;
    tce16 = tce13*tce15;
    tce17 = L_J[2]*n[2];
    tce18 = L_J[0]*tce17;
    tce19 = tce18*tce8;
    tce20 = L_J[2]*tce9;
    tce21 = tce12*tce20;
    tce22 = tce15*tce20;
    tce23 = tce17*tce3;
    tce24 = tce17*tce6;
    tce25 = 2.0*L_I[1]*u1j_udNx1j;
    tce26 = L_J[1]*tce25;
    tce27 = tce0*tce26;
    tce28 = 1.0*L_I[1];
    tce29 = tce28*u0j_udNx1j;
    tce30 = tce10*tce29;
    tce31 = tce28*u1j_udNx0j;
    tce32 = tce10*tce31;
    tce33 = tce13*tce28;
    tce34 = tce33*u1j_udNx2j;
    tce35 = tce33*u2j_udNx1j;
    tce36 = tce18*tce29;
    tce37 = tce18*tce31;
    tce38 = tce20*tce28;
    tce39 = tce38*u1j_udNx2j;
    tce40 = tce38*u2j_udNx1j;
    tce41 = tce17*tce26;
    tce42 = 1.0*L_I[2];
    tce43 = L_J[1]*tce42;
    tce44 = tce0*tce43;
    tce45 = tce44*u1j_udNx2j;
    tce46 = tce44*u2j_udNx1j;
    tce47 = tce10*tce42;
    tce48 = tce47*u0j_udNx2j;
    tce49 = tce47*u2j_udNx0j;
    tce50 = 2.0*L_I[2]*u2j_udNx2j;
    tce51 = tce13*tce50;
    tce52 = tce18*tce42;
    tce53 = tce52*u0j_udNx2j;
    tce54 = tce52*u2j_udNx0j;
    tce55 = tce20*tce50;
    tce56 = tce17*tce43;
    tce57 = tce56*u1j_udNx2j;
    tce58 = tce56*u2j_udNx1j;
    tce59 = pow(L_J[0], 2)*n[0];
    tce60 = tce59*tce8;
    tce61 = pow(L_J[1], 2)*n[1];
    tce62 = tce2*tce61;
    tce63 = tce5*tce61;
    tce64 = pow(L_J[2], 2)*n[2];
    tce65 = tce12*tce64;
    tce66 = tce15*tce64;
    tce67 = tce29*tce59;
    tce68 = tce31*tce59;
    tce69 = tce25*tce61;
    tce70 = tce28*tce64;
    tce71 = tce70*u1j_udNx2j;
    tce72 = tce70*u2j_udNx1j;
    tce73 = tce42*tce59;
    tce74 = tce73*u0j_udNx2j;
    tce75 = tce73*u2j_udNx0j;
    tce76 = tce42*tce61;
    tce77 = tce76*u1j_udNx2j;
    tce78 = tce76*u2j_udNx1j;
    tce79 = tce50*tce64;
    F[3*i + 2] += scale * (L_I[2]*eta*wNt[i]*(H_IJ*tce11 + H_IJ*tce14 + H_IJ*tce16 + H_IJ*tce19 + H_IJ*tce21 + H_IJ*tce22 + H_IJ*tce23 + H_IJ*tce24 + H_IJ*tce27 + H_IJ*tce30 + H_IJ*tce32 + H_IJ*tce34 + H_IJ*tce35 + H_IJ*tce36 + H_IJ*tce37 + H_IJ*tce39 + H_IJ*tce4 + H_IJ*tce40 + H_IJ*tce41 + H_IJ*tce45 + H_IJ*tce46 + H_IJ*tce48 + H_IJ*tce49 + H_IJ*tce51 + H_IJ*tce53 + H_IJ*tce54 + H_IJ*tce55 + H_IJ*tce57 + H_IJ*tce58 + H_IJ*tce60 + H_IJ*tce62 + H_IJ*tce63 + H_IJ*tce65 + H_IJ*tce66 + H_IJ*tce67 + H_IJ*tce68 + H_IJ*tce69 + H_IJ*tce7 + H_IJ*tce71 + H_IJ*tce72 + H_IJ*tce74 + H_IJ*tce75 + H_IJ*tce77 + H_IJ*tce78 + H_IJ*tce79 - tce11 - tce14 - tce16 - tce19 - tce21 - tce22 - tce23 - tce24 - tce27 - tce30 - tce32 - tce34 - tce35 - tce36 - tce37 - tce39 - tce4 - tce40 - tce41 - tce45 - tce46 - tce48 - tce49 - tce51 - tce53 - tce54 - tce55 - tce57 - tce58 - tce60 - tce62 - tce63 - tce65 - tce66 - tce67 - tce68 - tce69 - tce7 - tce71 - tce72 - tce74 - tce75 - tce77 - tce78 - tce79));
    }
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
  for (i=0; i<4; i++) { // q_nbasis
    F[1*i + 0] += scale * (0);
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
  for (i=0; i<27; i++) { // w_nbasis
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78;
    tce0 = H_IJ*L_I[0];
    tce1 = tau_S[3]*tce0;
    tce2 = L_J[0]*L_J[1];
    tce3 = n[0]*tce2;
    tce4 = tau_S[0]*tce0;
    tce5 = n[1]*tce2;
    tce6 = tau_S[4]*tce0;
    tce7 = L_J[0]*L_J[2];
    tce8 = n[0]*tce7;
    tce9 = n[2]*tce7;
    tce10 = L_J[1]*L_J[2];
    tce11 = n[1]*tce10;
    tce12 = n[2]*tce10;
    tce13 = H_IJ*L_I[1];
    tce14 = tau_S[1]*tce13;
    tce15 = tau_S[3]*tce13;
    tce16 = tau_S[5]*tce13;
    tce17 = H_IJ*L_I[2];
    tce18 = tau_S[4]*tce17;
    tce19 = tau_S[2]*tce17;
    tce20 = pow(L_J[0], 2)*n[0];
    tce21 = pow(L_J[1], 2)*n[1];
    tce22 = pow(L_J[2], 2)*n[2];
    tce23 = 1.0*eta;
    tce24 = L_I[0]*tce23;
    tce25 = tce24*tce3;
    tce26 = 2.0*eta;
    tce27 = tce26*u0j_udNx0j;
    tce28 = L_I[0]*tce27;
    tce29 = tce24*tce8;
    tce30 = tce11*tce24;
    tce31 = tce12*tce24;
    tce32 = tce26*u1j_udNx1j;
    tce33 = L_I[1]*tce32;
    tce34 = L_I[1]*tce23;
    tce35 = tce34*tce5;
    tce36 = tce34*tce8;
    tce37 = tce34*tce9;
    tce38 = tce11*tce34;
    tce39 = L_I[2]*tce23;
    tce40 = tce3*tce39;
    tce41 = tce39*tce5;
    tce42 = tce26*u2j_udNx2j;
    tce43 = L_I[2]*tce42;
    tce44 = tce39*tce9;
    tce45 = tce12*tce39;
    tce46 = tce20*tce27;
    tce47 = tce21*tce24;
    tce48 = tce22*tce24;
    tce49 = tce20*tce34;
    tce50 = tce21*tce32;
    tce51 = tce22*tce34;
    tce52 = tce20*tce39;
    tce53 = tce21*tce39;
    tce54 = tce22*tce42;
    tce55 = tce0*tce23;
    tce56 = tce3*tce55;
    tce57 = tce0*tce27;
    tce58 = tce55*tce8;
    tce59 = tce11*tce55;
    tce60 = tce12*tce55;
    tce61 = tce13*tce32;
    tce62 = tce13*tce23;
    tce63 = tce5*tce62;
    tce64 = tce62*tce8;
    tce65 = tce62*tce9;
    tce66 = tce11*tce62;
    tce67 = tce17*tce23;
    tce68 = tce3*tce67;
    tce69 = tce5*tce67;
    tce70 = tce17*tce42;
    tce71 = tce67*tce9;
    tce72 = tce12*tce67;
    tce73 = tce21*tce55;
    tce74 = tce22*tce55;
    tce75 = tce20*tce62;
    tce76 = tce22*tce62;
    tce77 = tce20*tce67;
    tce78 = tce21*tce67;
    F[3*i + 0] += scale * (L_I[0]*wNt[i]*(-L_I[0]*tce46 - L_I[1]*tce50 - L_I[2]*tce54 + tce0*tce46 + tce1*tce12 + tce1*tce21 + tce1*tce3 + tce11*tce16 + tce11*tce19 - tce11*tce43 + tce11*tce6 + tce11*tce70 + tce12*tce14 + tce12*tce18 - tce12*tce33 + tce12*tce61 + tce13*tce50 + tce14*tce21 + tce14*tce3 + tce15*tce20 + tce15*tce5 + tce15*tce9 + tce16*tce22 + tce16*tce8 + tce17*tce54 + tce18*tce20 + tce18*tce21 + tce18*tce3 + tce18*tce5 + tce18*tce9 + tce19*tce22 + tce19*tce8 + tce20*tce4 + tce22*tce6 - tce25*u0j_udNx1j - tce25*u1j_udNx0j - tce28*tce5 - tce28*tce9 - tce29*u0j_udNx2j - tce29*u2j_udNx0j - tce3*tce33 + tce3*tce61 - tce30*u0j_udNx2j - tce30*u2j_udNx0j - tce31*u0j_udNx1j - tce31*u1j_udNx0j - tce35*u0j_udNx1j - tce35*u1j_udNx0j - tce36*u1j_udNx2j - tce36*u2j_udNx1j - tce37*u0j_udNx1j - tce37*u1j_udNx0j - tce38*u1j_udNx2j - tce38*u2j_udNx1j + tce4*tce5 + tce4*tce9 - tce40*u1j_udNx2j - tce40*u2j_udNx1j - tce41*u0j_udNx2j - tce41*u2j_udNx0j - tce43*tce8 - tce44*u0j_udNx2j - tce44*u2j_udNx0j - tce45*u1j_udNx2j - tce45*u2j_udNx1j - tce47*u0j_udNx1j - tce47*u1j_udNx0j - tce48*u0j_udNx2j - tce48*u2j_udNx0j - tce49*u0j_udNx1j - tce49*u1j_udNx0j + tce5*tce57 - tce51*u1j_udNx2j - tce51*u2j_udNx1j - tce52*u0j_udNx2j - tce52*u2j_udNx0j - tce53*u1j_udNx2j - tce53*u2j_udNx1j + tce56*u0j_udNx1j + tce56*u1j_udNx0j + tce57*tce9 + tce58*u0j_udNx2j + tce58*u2j_udNx0j + tce59*u0j_udNx2j + tce59*u2j_udNx0j + tce6*tce8 + tce60*u0j_udNx1j + tce60*u1j_udNx0j + tce63*u0j_udNx1j + tce63*u1j_udNx0j + tce64*u1j_udNx2j + tce64*u2j_udNx1j + tce65*u0j_udNx1j + tce65*u1j_udNx0j + tce66*u1j_udNx2j + tce66*u2j_udNx1j + tce68*u1j_udNx2j + tce68*u2j_udNx1j + tce69*u0j_udNx2j + tce69*u2j_udNx0j + tce70*tce8 + tce71*u0j_udNx2j + tce71*u2j_udNx0j + tce72*u1j_udNx2j + tce72*u2j_udNx1j + tce73*u0j_udNx1j + tce73*u1j_udNx0j + tce74*u0j_udNx2j + tce74*u2j_udNx0j + tce75*u0j_udNx1j + tce75*u1j_udNx0j + tce76*u1j_udNx2j + tce76*u2j_udNx1j + tce77*u0j_udNx2j + tce77*u2j_udNx0j + tce78*u1j_udNx2j + tce78*u2j_udNx1j));
    }
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78;
    tce0 = H_IJ*L_I[0];
    tce1 = tau_S[3]*tce0;
    tce2 = L_J[0]*L_J[1];
    tce3 = n[0]*tce2;
    tce4 = tau_S[0]*tce0;
    tce5 = n[1]*tce2;
    tce6 = tau_S[4]*tce0;
    tce7 = L_J[0]*L_J[2];
    tce8 = n[0]*tce7;
    tce9 = n[2]*tce7;
    tce10 = L_J[1]*L_J[2];
    tce11 = n[1]*tce10;
    tce12 = n[2]*tce10;
    tce13 = H_IJ*L_I[1];
    tce14 = tau_S[1]*tce13;
    tce15 = tau_S[3]*tce13;
    tce16 = tau_S[5]*tce13;
    tce17 = H_IJ*L_I[2];
    tce18 = tau_S[4]*tce17;
    tce19 = tau_S[2]*tce17;
    tce20 = pow(L_J[0], 2)*n[0];
    tce21 = pow(L_J[1], 2)*n[1];
    tce22 = pow(L_J[2], 2)*n[2];
    tce23 = 1.0*eta;
    tce24 = L_I[0]*tce23;
    tce25 = tce24*tce3;
    tce26 = 2.0*eta;
    tce27 = tce26*u0j_udNx0j;
    tce28 = L_I[0]*tce27;
    tce29 = tce24*tce8;
    tce30 = tce11*tce24;
    tce31 = tce12*tce24;
    tce32 = tce26*u1j_udNx1j;
    tce33 = L_I[1]*tce32;
    tce34 = L_I[1]*tce23;
    tce35 = tce34*tce5;
    tce36 = tce34*tce8;
    tce37 = tce34*tce9;
    tce38 = tce11*tce34;
    tce39 = L_I[2]*tce23;
    tce40 = tce3*tce39;
    tce41 = tce39*tce5;
    tce42 = tce26*u2j_udNx2j;
    tce43 = L_I[2]*tce42;
    tce44 = tce39*tce9;
    tce45 = tce12*tce39;
    tce46 = tce20*tce27;
    tce47 = tce21*tce24;
    tce48 = tce22*tce24;
    tce49 = tce20*tce34;
    tce50 = tce21*tce32;
    tce51 = tce22*tce34;
    tce52 = tce20*tce39;
    tce53 = tce21*tce39;
    tce54 = tce22*tce42;
    tce55 = tce0*tce23;
    tce56 = tce3*tce55;
    tce57 = tce0*tce27;
    tce58 = tce55*tce8;
    tce59 = tce11*tce55;
    tce60 = tce12*tce55;
    tce61 = tce13*tce32;
    tce62 = tce13*tce23;
    tce63 = tce5*tce62;
    tce64 = tce62*tce8;
    tce65 = tce62*tce9;
    tce66 = tce11*tce62;
    tce67 = tce17*tce23;
    tce68 = tce3*tce67;
    tce69 = tce5*tce67;
    tce70 = tce17*tce42;
    tce71 = tce67*tce9;
    tce72 = tce12*tce67;
    tce73 = tce21*tce55;
    tce74 = tce22*tce55;
    tce75 = tce20*tce62;
    tce76 = tce22*tce62;
    tce77 = tce20*tce67;
    tce78 = tce21*tce67;
    F[3*i + 1] += scale * (L_I[1]*wNt[i]*(-L_I[0]*tce46 - L_I[1]*tce50 - L_I[2]*tce54 + tce0*tce46 + tce1*tce12 + tce1*tce21 + tce1*tce3 + tce11*tce16 + tce11*tce19 - tce11*tce43 + tce11*tce6 + tce11*tce70 + tce12*tce14 + tce12*tce18 - tce12*tce33 + tce12*tce61 + tce13*tce50 + tce14*tce21 + tce14*tce3 + tce15*tce20 + tce15*tce5 + tce15*tce9 + tce16*tce22 + tce16*tce8 + tce17*tce54 + tce18*tce20 + tce18*tce21 + tce18*tce3 + tce18*tce5 + tce18*tce9 + tce19*tce22 + tce19*tce8 + tce20*tce4 + tce22*tce6 - tce25*u0j_udNx1j - tce25*u1j_udNx0j - tce28*tce5 - tce28*tce9 - tce29*u0j_udNx2j - tce29*u2j_udNx0j - tce3*tce33 + tce3*tce61 - tce30*u0j_udNx2j - tce30*u2j_udNx0j - tce31*u0j_udNx1j - tce31*u1j_udNx0j - tce35*u0j_udNx1j - tce35*u1j_udNx0j - tce36*u1j_udNx2j - tce36*u2j_udNx1j - tce37*u0j_udNx1j - tce37*u1j_udNx0j - tce38*u1j_udNx2j - tce38*u2j_udNx1j + tce4*tce5 + tce4*tce9 - tce40*u1j_udNx2j - tce40*u2j_udNx1j - tce41*u0j_udNx2j - tce41*u2j_udNx0j - tce43*tce8 - tce44*u0j_udNx2j - tce44*u2j_udNx0j - tce45*u1j_udNx2j - tce45*u2j_udNx1j - tce47*u0j_udNx1j - tce47*u1j_udNx0j - tce48*u0j_udNx2j - tce48*u2j_udNx0j - tce49*u0j_udNx1j - tce49*u1j_udNx0j + tce5*tce57 - tce51*u1j_udNx2j - tce51*u2j_udNx1j - tce52*u0j_udNx2j - tce52*u2j_udNx0j - tce53*u1j_udNx2j - tce53*u2j_udNx1j + tce56*u0j_udNx1j + tce56*u1j_udNx0j + tce57*tce9 + tce58*u0j_udNx2j + tce58*u2j_udNx0j + tce59*u0j_udNx2j + tce59*u2j_udNx0j + tce6*tce8 + tce60*u0j_udNx1j + tce60*u1j_udNx0j + tce63*u0j_udNx1j + tce63*u1j_udNx0j + tce64*u1j_udNx2j + tce64*u2j_udNx1j + tce65*u0j_udNx1j + tce65*u1j_udNx0j + tce66*u1j_udNx2j + tce66*u2j_udNx1j + tce68*u1j_udNx2j + tce68*u2j_udNx1j + tce69*u0j_udNx2j + tce69*u2j_udNx0j + tce70*tce8 + tce71*u0j_udNx2j + tce71*u2j_udNx0j + tce72*u1j_udNx2j + tce72*u2j_udNx1j + tce73*u0j_udNx1j + tce73*u1j_udNx0j + tce74*u0j_udNx2j + tce74*u2j_udNx0j + tce75*u0j_udNx1j + tce75*u1j_udNx0j + tce76*u1j_udNx2j + tce76*u2j_udNx1j + tce77*u0j_udNx2j + tce77*u2j_udNx0j + tce78*u1j_udNx2j + tce78*u2j_udNx1j));
    }
    {
    double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22, tce23, tce24, tce25, tce26, tce27, tce28, tce29, tce30, tce31, tce32, tce33, tce34, tce35, tce36, tce37, tce38, tce39, tce40, tce41, tce42, tce43, tce44, tce45, tce46, tce47, tce48, tce49, tce50, tce51, tce52, tce53, tce54, tce55, tce56, tce57, tce58, tce59, tce60, tce61, tce62, tce63, tce64, tce65, tce66, tce67, tce68, tce69, tce70, tce71, tce72, tce73, tce74, tce75, tce76, tce77, tce78;
    tce0 = H_IJ*L_I[0];
    tce1 = tau_S[3]*tce0;
    tce2 = L_J[0]*L_J[1];
    tce3 = n[0]*tce2;
    tce4 = tau_S[0]*tce0;
    tce5 = n[1]*tce2;
    tce6 = tau_S[4]*tce0;
    tce7 = L_J[0]*L_J[2];
    tce8 = n[0]*tce7;
    tce9 = n[2]*tce7;
    tce10 = L_J[1]*L_J[2];
    tce11 = n[1]*tce10;
    tce12 = n[2]*tce10;
    tce13 = H_IJ*L_I[1];
    tce14 = tau_S[1]*tce13;
    tce15 = tau_S[3]*tce13;
    tce16 = tau_S[5]*tce13;
    tce17 = H_IJ*L_I[2];
    tce18 = tau_S[4]*tce17;
    tce19 = tau_S[2]*tce17;
    tce20 = pow(L_J[0], 2)*n[0];
    tce21 = pow(L_J[1], 2)*n[1];
    tce22 = pow(L_J[2], 2)*n[2];
    tce23 = 1.0*eta;
    tce24 = L_I[0]*tce23;
    tce25 = tce24*tce3;
    tce26 = 2.0*eta;
    tce27 = tce26*u0j_udNx0j;
    tce28 = L_I[0]*tce27;
    tce29 = tce24*tce8;
    tce30 = tce11*tce24;
    tce31 = tce12*tce24;
    tce32 = tce26*u1j_udNx1j;
    tce33 = L_I[1]*tce32;
    tce34 = L_I[1]*tce23;
    tce35 = tce34*tce5;
    tce36 = tce34*tce8;
    tce37 = tce34*tce9;
    tce38 = tce11*tce34;
    tce39 = L_I[2]*tce23;
    tce40 = tce3*tce39;
    tce41 = tce39*tce5;
    tce42 = tce26*u2j_udNx2j;
    tce43 = L_I[2]*tce42;
    tce44 = tce39*tce9;
    tce45 = tce12*tce39;
    tce46 = tce20*tce27;
    tce47 = tce21*tce24;
    tce48 = tce22*tce24;
    tce49 = tce20*tce34;
    tce50 = tce21*tce32;
    tce51 = tce22*tce34;
    tce52 = tce20*tce39;
    tce53 = tce21*tce39;
    tce54 = tce22*tce42;
    tce55 = tce0*tce23;
    tce56 = tce3*tce55;
    tce57 = tce0*tce27;
    tce58 = tce55*tce8;
    tce59 = tce11*tce55;
    tce60 = tce12*tce55;
    tce61 = tce13*tce32;
    tce62 = tce13*tce23;
    tce63 = tce5*tce62;
    tce64 = tce62*tce8;
    tce65 = tce62*tce9;
    tce66 = tce11*tce62;
    tce67 = tce17*tce23;
    tce68 = tce3*tce67;
    tce69 = tce5*tce67;
    tce70 = tce17*tce42;
    tce71 = tce67*tce9;
    tce72 = tce12*tce67;
    tce73 = tce21*tce55;
    tce74 = tce22*tce55;
    tce75 = tce20*tce62;
    tce76 = tce22*tce62;
    tce77 = tce20*tce67;
    tce78 = tce21*tce67;
    F[3*i + 2] += scale * (L_I[2]*wNt[i]*(-L_I[0]*tce46 - L_I[1]*tce50 - L_I[2]*tce54 + tce0*tce46 + tce1*tce12 + tce1*tce21 + tce1*tce3 + tce11*tce16 + tce11*tce19 - tce11*tce43 + tce11*tce6 + tce11*tce70 + tce12*tce14 + tce12*tce18 - tce12*tce33 + tce12*tce61 + tce13*tce50 + tce14*tce21 + tce14*tce3 + tce15*tce20 + tce15*tce5 + tce15*tce9 + tce16*tce22 + tce16*tce8 + tce17*tce54 + tce18*tce20 + tce18*tce21 + tce18*tce3 + tce18*tce5 + tce18*tce9 + tce19*tce22 + tce19*tce8 + tce20*tce4 + tce22*tce6 - tce25*u0j_udNx1j - tce25*u1j_udNx0j - tce28*tce5 - tce28*tce9 - tce29*u0j_udNx2j - tce29*u2j_udNx0j - tce3*tce33 + tce3*tce61 - tce30*u0j_udNx2j - tce30*u2j_udNx0j - tce31*u0j_udNx1j - tce31*u1j_udNx0j - tce35*u0j_udNx1j - tce35*u1j_udNx0j - tce36*u1j_udNx2j - tce36*u2j_udNx1j - tce37*u0j_udNx1j - tce37*u1j_udNx0j - tce38*u1j_udNx2j - tce38*u2j_udNx1j + tce4*tce5 + tce4*tce9 - tce40*u1j_udNx2j - tce40*u2j_udNx1j - tce41*u0j_udNx2j - tce41*u2j_udNx0j - tce43*tce8 - tce44*u0j_udNx2j - tce44*u2j_udNx0j - tce45*u1j_udNx2j - tce45*u2j_udNx1j - tce47*u0j_udNx1j - tce47*u1j_udNx0j - tce48*u0j_udNx2j - tce48*u2j_udNx0j - tce49*u0j_udNx1j - tce49*u1j_udNx0j + tce5*tce57 - tce51*u1j_udNx2j - tce51*u2j_udNx1j - tce52*u0j_udNx2j - tce52*u2j_udNx0j - tce53*u1j_udNx2j - tce53*u2j_udNx1j + tce56*u0j_udNx1j + tce56*u1j_udNx0j + tce57*tce9 + tce58*u0j_udNx2j + tce58*u2j_udNx0j + tce59*u0j_udNx2j + tce59*u2j_udNx0j + tce6*tce8 + tce60*u0j_udNx1j + tce60*u1j_udNx0j + tce63*u0j_udNx1j + tce63*u1j_udNx0j + tce64*u1j_udNx2j + tce64*u2j_udNx1j + tce65*u0j_udNx1j + tce65*u1j_udNx0j + tce66*u1j_udNx2j + tce66*u2j_udNx1j + tce68*u1j_udNx2j + tce68*u2j_udNx1j + tce69*u0j_udNx2j + tce69*u2j_udNx0j + tce70*tce8 + tce71*u0j_udNx2j + tce71*u2j_udNx0j + tce72*u1j_udNx2j + tce72*u2j_udNx1j + tce73*u0j_udNx1j + tce73*u1j_udNx0j + tce74*u0j_udNx2j + tce74*u2j_udNx0j + tce75*u0j_udNx1j + tce75*u1j_udNx0j + tce76*u1j_udNx2j + tce76*u2j_udNx1j + tce77*u0j_udNx2j + tce77*u2j_udNx0j + tce78*u1j_udNx2j + tce78*u2j_udNx1j));
    }
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
  for (i=0; i<4; i++) { // q_nbasis
    F[1*i + 0] += scale * (0);
  }
}
