//
// Created by SENSETIME\luxiao on 19-4-2.
//

#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include "gpc.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_poly.h>

using std::cout;
using std::endl;

int gpc_solve(int K, const struct gpc_corr*c, double *x) {
    return gpc_solve_valid(K,c,0,x);
}

int gpc_solve_valid(int K, const struct gpc_corr*c, int*valid, double *x_out) {
    Eigen::Matrix4d bigM = Eigen::Matrix4d::Zero();
    Eigen::Vector4d g = Eigen::Vector4d::Zero();

    int k;
    for(k=0;k<K;k++) {
        if(valid && !valid[k]) continue;

        Eigen::Matrix<double, 2, 4> Mi = Eigen::Matrix<double, 2, 4>::Identity();
        Mi(0, 2) = c[k].p[0];
        Mi(0, 3) = -c[k].p[1];
        Mi(1, 2) = c[k].p[1];
        Mi(1, 3) = c[k].p[0];

        Eigen::Matrix2d Ci = c[k].normal * c[k].normal.transpose();
//        Ci(0, 0) = c[k].C[0][0];
//        Ci(0, 1) = c[k].C[0][1];
//        Ci(1, 0) = c[k].C[1][0];
//        Ci(1, 1) = c[k].C[1][1];

        Eigen::Vector2d qi = Eigen::Vector2d::Zero();
        qi(0) = c[k].q[0];
        qi(1) = c[k].q[1];

        Eigen::Matrix4d M = 2 * Mi.transpose() * Ci * Mi;
        bigM += M;

        Eigen::Vector4d gi = -2 * qi.transpose() * Ci * Mi;
        g += gi;

        if(1) {
            cout << "bigM_k =\n" << Mi << endl;
            cout << "q_k =\n" << qi << endl;
            cout << "C_k =\n" << Ci << endl << endl;
//            m_display("bigM_k",bigM_k);
//            m_display("q_k",q_k);
//            m_display("C_k",C_k);
//            m_display("now M is ",bigM);
//            m_display("now g is ",g);
        }
    }

    if(1) {
        cout << "bigM =\n" << bigM << endl;
        cout << "g =\n" << g << endl;
    }

    Eigen::Matrix2d A = bigM.topLeftCorner(2, 2);
    Eigen::Matrix2d B = bigM.topRightCorner(2, 2);
    Eigen::Matrix2d D = bigM.bottomRightCorner(2, 2);
    Eigen::Matrix2d S = D - B.transpose() * A.inverse() * B;
    Eigen::Matrix2d Sa = S.inverse() * S.determinant();



    if(1) {
        cout << "A =\n" << A << endl;
        cout << "B =\n" << B << endl;
        cout << "D =\n" << D << endl;
        cout << "S =\n" << S << endl;
        cout << "Sa =\n" << Sa << endl;
    }

    Eigen::Vector2d g1 = g.topRows(2);
    Eigen::Vector2d g2 = g.bottomRows(2);

    Eigen::Vector2d m1 = g1.transpose() * A.inverse() * B;
    Eigen::Vector2d m2 = m1.transpose() * Sa;
    Eigen::Vector2d m3 = g2.transpose() * Sa;

    Eigen::Vector3d p = Eigen::Vector3d(m2.dot(m2) - 2 * m2.dot(m3) + m3.dot(m3),
                                        4 * m2.dot(m1) - 8 * m2.dot(g2) + 4 * g2.dot(m3),
                                        4 * m1.dot(m1) - 8 * m1.dot(g2) + 4 * g2.dot(g2));

    Eigen::Vector3d l = Eigen::Vector3d(S.determinant(), 2 * (S(0, 0) + S(1, 1)), 4);

    double a4 = p[0]-(l[0]*l[0]);
    double a3 = p[1]-(2*l[1]*l[0]);
    double a2 = p[2]-(l[1]*l[1]+2*l[0]*l[2]);
    double a1 = -(2*l[2]*l[1]);
    double a0 = -(l[2]*l[2]);

    /* q = p - l^2 */
    double q[5] = {p[0]-(l[0]*l[0]), p[1]-(2*l[1]*l[0]),
                   p[2]-(l[1]*l[1]+2*l[0]*l[2]), -(2*l[2]*l[1]), -(l[2]*l[2])};

    double lambda = poly_greatest_real_root(5,q);

    if(1) {
        printf("p = %f %f %f \n", p[2], p[1], p[0]);
        printf("l = %f %f %f \n", l[2], l[1], l[0]);
        printf("q = %f %f %f %f %f \n", q[4],  q[3],  q[2], q[1], q[0]);
        printf("lambda = %f \n", lambda);
    }

    Eigen::Matrix4d W = Eigen::Matrix4d::Zero();
    W.bottomRightCorner(2, 2) = Eigen::Matrix2d::Identity();
    Eigen::Vector4d x = -(bigM + 2 * lambda * W).inverse() * g;


    x_out[0] = x[0];
    x_out[1] = x[1];
    x_out[2] = atan2(x[3], x[2]);

    if(1) {
        printf("x =  %f  %f %f deg\n", x_out[0], x_out[1],x_out[2]*180/M_PI);
    }

    return 0;
}


double gpc_error(const struct gpc_corr*co, const double*x) {
    double c = cos(x[2]);
    double s = sin(x[2]);
    double e[2];
    e[0] = c*(co->p[0]) -s*(co->p[1]) + x[0] - co->q[0];
    e[1] = s*(co->p[0]) +c*(co->p[1]) + x[1] - co->q[1];
    return e[0]*e[0]*co->C[0][0]+2*e[0]*e[1]*co->C[0][1]+e[1]*e[1]*co->C[1][1];
}

double poly_greatest_real_root(unsigned int n, double*a) {
    double z[(n-1)*2];
    gsl_poly_complex_workspace * w  = gsl_poly_complex_workspace_alloc(n);
    gsl_poly_complex_solve (a, n, w, z);
    gsl_poly_complex_workspace_free (w);
    double lambda = 0; int set = 0;
    unsigned int i;
    for (i = 0; i < n-1; i++) {
//		printf ("z%d = %+.18f %+.18f\n", i, z[2*i], z[2*i+1]);
        // XXX ==0 is bad
        if( (z[2*i+1]==0) && (z[2*i]>lambda || !set)) {
            lambda = z[2*i];
            set = 1;
        }
    }
//	printf ("lambda = %+.18f \n", lambda);
    return lambda;
}