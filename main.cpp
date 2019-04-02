#include <iostream>
#include <cmath>
#include <vector>
#include <sophus/so2.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include "gpc.h"

#define deg2rad(x) (x*M_PI/180)
using std::cout;
using std::endl;

struct RegisterError {
    RegisterError(Eigen::Vector2d _p, Eigen::Vector2d _q, Eigen::Vector2d _normal):
    p(_p), q(_q), normal(_normal)
    {}

    template <typename T>
    bool operator()(const T* pT, T* residual) const {
        T theta = pT[2];  // 第3个数表示旋转
        Eigen::Matrix<T, 2, 2> R = Eigen::Matrix<T, 2, 2>::Zero();
        R << ceres::cos(theta), -ceres::sin(theta),
             ceres::sin(theta),  ceres::cos(theta);
        Eigen::Map<const Eigen::Matrix<T, 2, 1>> t(pT);  // 前两个数表示平移
        Eigen::Matrix<T, 2, 1> p1 = R * p.template cast<T>() + t;
        Eigen::Matrix<T, 2, 1> pq = q.template cast<T>() - p1;

        residual[0] = pq.dot(normal.template cast<T>());

        return true;
    }


    Eigen::Vector2d p;
    Eigen::Vector2d q;
    Eigen::Vector2d normal;
};

int main() {
    // This is the true roto-translation
    double theta = deg2rad(20);
    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R << cos(theta), -sin(theta),
         sin(theta),  cos(theta);
    Eigen::Vector2d t(0.2, -0.3);
    cout << "ground truth is\n" << "theta = " << theta << ", t = " << t.transpose() << endl;

    int numPts = 200;
    std::vector<Eigen::Vector2d> vPts;
    vPts.resize(numPts);
    std::vector<Eigen::Vector2d> normals;
    normals.resize(numPts);
    struct gpc_corr c[numPts];
    Eigen::Vector2d p0(0, 1);
    Eigen::Vector2d p01(1, 1);
    Eigen::Vector2d v0(2, 0);
    Eigen::Vector2d p1(0, 0);
    Eigen::Vector2d p2(0, 0);
    for (int i = 0; i < numPts; ++i) {
        // 保证p和q不是同一个点，但相差的不多
        p1 = p0 + i * v0 + 0.05 * Eigen::Vector2d::Random();
        p2 = p01+ i * v0 + 0.05 * Eigen::Vector2d::Random();
//        p1 = p0 + i * v0;
//        p2 = p01+ i * v0;
        c[i].p = p1;
//        c[i].q = p2;
        c[i].q = R * p2 + t;
        c[i].normal = Eigen::Vector2d::Random().normalized();
        c[i].normal = Eigen::Vector2d(0, 1);
    }

    Eigen::Vector2d l = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    for (int i = 0; i < numPts - 1; ++i) {
        l = c[i+1].q - c[i].q;
        l.normalize();
        normal[0] = -l[1];
        normal[1] = l[0];
        c[i].normal = normal;
//        cout << "normal is " << normal.transpose() << endl;
    }
    c[numPts - 1].normal = normal;


//    double x[3];
//    gpc_solve(numPts,c,x);
//
//    printf("estimated x =  %f  %f  %f deg\n", x[0], x[1],x[2]*180/M_PI);


    //*******************************************************************//
    Sophus::SO2 SO2R(R);
    double alpha = SO2R.log();
    double pT[3];
//    pT[0] = t(0);
//    pT[1] = t(1);
//    pT[2] = alpha;
    pT[0] = 0;
    pT[1] = 0;
    pT[2] = 0;

    // 算之前检查一下error
    double sumError = 0;
    for (int i = 0; i < numPts; ++i) {
        // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
        auto p = c[i].p;
        auto q = c[i].q;
        auto normal = c[i].normal;
        double alpha = pT[2];
        Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
        R << cos(alpha), -sin(alpha),
                sin(alpha),  cos(alpha);
        Eigen::Vector2d t = Eigen::Vector2d(pT[0], pT[1]);
        auto p1 = R*p + t;
        auto pq = q - p1;
//        cout << "pq = " << pq.transpose() << endl;
        double error = pq.dot(normal);
        sumError += 0.5 * error * error;
    }
    cout << "sumError = " << sumError << endl;

    ceres::Problem problem;
    for (int i = 0; i < numPts; ++i) {
        // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
        ceres::CostFunction* pCostFunction = new ceres::AutoDiffCostFunction<RegisterError, 1, 3>(
                new RegisterError(c[i].p, c[i].q, c[i].normal));

        problem.AddResidualBlock(pCostFunction, nullptr, pT);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << endl;
    cout << "theta = " << pT[2] << ", t = " << pT[0] << " " << pT[1] << endl;

    return 0;
}