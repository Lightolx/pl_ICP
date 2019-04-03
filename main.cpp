#include <iostream>
#include <cmath>
#include <vector>
#include <sophus/so2.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <fstream>
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
        Eigen::Matrix<T, 2, 1> t = Eigen::Matrix<T, 2, 1>::Zero();
        t[0] = pT[0];
        t[1] = pT[1];
//        Eigen::Map<const Eigen::Matrix<T, 2, 1>> t(pT);  // 前两个数表示平移
        Eigen::Matrix<T, 2, 1> p1 = R * p.template cast<T>() + t;
        Eigen::Matrix<T, 2, 1> pq = q.template cast<T>() - p1;

        residual[0] = pq.dot(normal.template cast<T>());

        return true;
    }


    Eigen::Vector2d p;
    Eigen::Vector2d q;
    Eigen::Vector2d normal;
};

struct Correspondence {
    Eigen::Vector2d p;
    Eigen::Vector2d q;
    Eigen::Vector2d normal;
};

int main() {
    // Step1: 构造数据P, Q点集
    double theta = deg2rad(10);
    Eigen::Matrix2d R0 = Eigen::Matrix2d::Zero();
    R0 << cos(theta), -sin(theta),
         sin(theta),  cos(theta);
    Eigen::Vector2d t0(-0.2, 0.4);
    Eigen::Matrix3d T0 = Eigen::Matrix3d::Identity();
    T0.topLeftCorner(2, 2) = R0;
    T0.topRightCorner(2, 1) = t0;
    Eigen::Matrix3d T1 = T0.inverse();
    Eigen::Vector2d t1 = T1.topRightCorner(2, 1);
    cout << "ground truth is\n" << "theta = " << -theta << ", t = " << t1.transpose() << endl;

    // step1.1: p和q都是直线 y=1 上的点，范围[0, 400]
    int numPts = 100;
    std::vector<Eigen::Vector2d> vPts_P;
    std::vector<Eigen::Vector2d> vPts_Q;
    vPts_P.resize(numPts);
    vPts_Q.resize(numPts);
    Eigen::Vector2d p0(0, 1);
    Eigen::Vector2d v0(2, 0);
    Eigen::Vector2d p1(0, 0);
    Eigen::Vector2d p2(0, 0);
    std::ofstream fout1("p01.txt");
    std::ofstream fout2("q.txt");
    for (int i = 0; i < numPts; ++i) {
        // 保证p和q不是同一个点，但相差的不多
        p1 = p0 + i * v0 + 0.1 * Eigen::Vector2d::Random();
        p1 = p0 + 15 * Eigen::Vector2d(cos(deg2rad(i)), sin(deg2rad(i))) + 0.05 * Eigen::Vector2d::Random();
        vPts_P[i] = p1;
        fout1 << p1[0] << " " << p1[1] << endl;
        p2 = p0+ i * v0 + 0.1 * Eigen::Vector2d::Random();
        p2 = p0 + 15 * Eigen::Vector2d(cos(deg2rad(i)), sin(deg2rad(i))) + 0.05 * Eigen::Vector2d::Random();
//        p2 = p0+ i * v0;
        vPts_Q[i] = p2;
        fout2 << p2[0] << " " << p2[1] << endl;
    }
    fout1.close();
    fout2.close();

    // step1.2: 对点集p稍作平移及旋转
    std::ofstream fout4("p02.txt");
    for (int i = 0; i < numPts; ++i) {
        p1 = vPts_P[i];
        p2 = R0 * p1 + t0;
        fout4 << p2[0] << " " << p2[1] << endl;
        vPts_P[i] = p2;
    }
    fout4.close();

    // Step2: icp过程
    // step2.0: 初值，应该由别处给出
//    Sophus::SO2 SO2R(R);
//    double alpha = SO2R.log();
    double pT[3];
    pT[0] = t1(0) + (rand() % 10 - 5)/50.0;
    pT[1] = t1(1) + (rand() % 10 - 5)/50.0;
    pT[2] = -theta + (rand() % 10 - 5)/200.0;

//    pT[0] = 0;
//    pT[1] = 0;
//    pT[2] = 0;

    // 将点云Q做成kd_tree;
    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
    // Generate pointcloud data
    cloud->width = numPts;
    cloud->height = 1;
    cloud->points.resize (cloud->width * cloud->height);

    for (size_t i = 0; i < cloud->points.size (); ++i) {
        cloud->points[i].x = vPts_Q[i].x();
        cloud->points[i].y = vPts_Q[i].y();
    }

    pcl::KdTreeFLANN<pcl::PointXY> kdtree;
    kdtree.setInputCloud (cloud);
    pcl::PointXY searchPoint;

    // 这里迭代终止条件是固定迭代次数
    for (int iter = 0; iter < 20; ++iter) {
        std::vector<Correspondence> vCors;
        vCors.reserve(numPts);
        // step2.1: 首先寻找对应线，也就是两个最近的对应点
        Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
        double alpha = pT[2];
        R << cos(alpha), -sin(alpha),
             sin(alpha),  cos(alpha);
        Eigen::Vector2d t = Eigen::Vector2d(pT[0], pT[1]);

        int K = 2;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        Eigen::Vector2d q1 = Eigen::Vector2d::Zero();
        Eigen::Vector2d q2 = Eigen::Vector2d::Zero();
        std::ofstream fout3("p1.txt");
        for (auto p : vPts_P) {
            Eigen::Vector2d p1 = R * p + t;
            fout3 << p1[0] << " " << p1[1] << endl;
            // 寻找点云Q中离p1最近的两个点
            searchPoint.x = p1.x();
            searchPoint.y = p1.y();
            kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
            q1 = vPts_Q[pointIdxNKNSearch[0]];
            q2 = vPts_Q[pointIdxNKNSearch[1]];
//            cout << "q1 is " << q1.transpose() << ", q2 is " << q2.transpose() << endl;
            Eigen::Vector2d l = q1 - q2;
            l.normalize();
            Eigen::Vector2d normal(l[1], -l[0]);
            Correspondence cor;
            cor.p = p;
            cor.q = q1;
            cor.normal = normal;
            vCors.push_back(cor);
//            cout << "normal = " << normal.transpose() << endl;
        }
        fout3.close();

        // step2.2: 根据point_to_line的correspondence来计算相对变换R,t
        // 算之前检查一下error
        double sumError = 0;
        for (int i = 0; i < numPts; ++i) {
            // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
            auto p = vCors[i].p;
            auto q = vCors[i].q;
            auto normal = vCors[i].normal;
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
                    new RegisterError(vCors[i].p, vCors[i].q, vCors[i].normal));

            problem.AddResidualBlock(pCostFunction, nullptr, pT);
        }

//        problem.SetParameterBlockConstant(pT+1);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.BriefReport() << endl;
        cout << "theta = " << pT[2] << ", t = " << pT[0] << " " << pT[1] << endl << endl;
    }

    //*******************************************************************//




    return 0;
}