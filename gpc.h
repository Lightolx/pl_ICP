//
// Created by SENSETIME\luxiao on 19-4-2.
//

#ifndef PL_ICP_GPC_H
#define PL_ICP_GPC_H

#include <eigen3/Eigen/Eigen>


struct gpc_corr {  // 线段pq在直线法线方向上的投影
    Eigen::Vector2d p;
    Eigen::Vector2d q;
    Eigen::Vector2d normal;  // 匹配直线的法向量
};

/* This program solves the general point correspondences problem:
// to find a translation $t$ and rotation $\theta$ that minimize
//
//  \sum_k (rot(theta)*c[k].p+t-c[k].q)' * c[k].C * (rot(theta)*c[k].p+t-c[k].q)
//
// (see the attached documentation for details).
*/

int gpc_solve(int numPts, const struct gpc_corr*, double *x);

/* if valid[k] is 0, the correspondence is not used */
int gpc_solve_valid(int numPts, const struct gpc_corr*, double *x);

/* Some utilities functions */

/* Computes error for the correspondence */
double gpc_error(const struct gpc_corr*co, const double*x);

double poly_greatest_real_root(unsigned int n, double*);

#endif //PL_ICP_GPC_H
