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
#include <cfloat>
#include "gpc.h"

#define deg2rad(x) (x*M_PI/180)
#define rad2deg(x) (x*180/M_PI)
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
//        Eigen::Matrix<T, 2, 1> t = Eigen::Matrix<T, 2, 1>::Zero();
//        t[0] = pT[0];
//        t[1] = pT[1];
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

struct Correspondence_pt2line {
    Eigen::Vector2d p;
    Eigen::Vector2d q;
    Eigen::Vector2d normal;
};

struct RegisterError_pt2pt {
    RegisterError_pt2pt(Eigen::Vector2d _p, Eigen::Vector2d _q):
            p(_p), q(_q)
    {}

    template <typename T>
    bool operator()(const T* pT, T* residual) const {
        T theta = pT[2];  // 第3个数表示旋转
        Eigen::Matrix<T, 2, 2> R = Eigen::Matrix<T, 2, 2>::Zero();
        R << ceres::cos(theta), -ceres::sin(theta),
                ceres::sin(theta),  ceres::cos(theta);
//        Eigen::Matrix<T, 2, 1> t = Eigen::Matrix<T, 2, 1>::Zero();
//        t[0] = pT[0];
//        t[1] = pT[1];
        Eigen::Map<const Eigen::Matrix<T, 2, 1>> t(pT);  // 前两个数表示平移
        Eigen::Matrix<T, 2, 1> p1 = R * p.template cast<T>() + t;
        Eigen::Matrix<T, 2, 1> pq = q.template cast<T>() - p1;

        residual[0] = pq.norm();

        return true;
    }

    Eigen::Vector2d p;
    Eigen::Vector2d q;
};

struct Correspondence_pt2pt {
    Eigen::Vector2d p;
    Eigen::Vector2d q;
};

int main(int argc, char** argv) {
    // Step1: 构造数据P, Q点集
    double theta0 = -30;
    double theta = deg2rad(theta0);
    Eigen::Matrix2d R0 = Eigen::Matrix2d::Zero();
    R0 << cos(theta), -sin(theta),
         sin(theta),  cos(theta);
    Eigen::Vector2d t0(4.1, -3.8);
    Eigen::Matrix3d T0 = Eigen::Matrix3d::Identity();
    T0.topLeftCorner(2, 2) = R0;
    T0.topRightCorner(2, 1) = t0;
    Eigen::Matrix3d T1 = T0.inverse();
    Eigen::Vector2d t1 = T1.topRightCorner(2, 1);
    cout << "ground truth is\n" << "theta = " << -theta0 << ", t = " << t1.transpose() << endl;

    // step1.1: 构造数据集Q，对应着高精地图
    std::vector<Eigen::Vector2d> vPts_Q;
    double sigma = 0.1;
    int numLanes = 2;
    // 直线段，从[15, -100]到[15, 0]，1m一个;从[18.5, -100]到[18.5, 0]，1m一个;
    for (int i = -100; i < 0; ++i) {
        for (int j = 0; j < numLanes; ++j) {
            vPts_Q.push_back(Eigen::Vector2d(15 + j * 3.5, i));
        }
    }

    // 弧线段，从[15, 0]到[0, 15]，1°一个;从[18.5, 0]到[0, 18.5]，1°一个
    Eigen::Vector2d c0(0, 0);
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    for (int i = 0; i < 90; ++i) {
        for (int j = 0; j < numLanes; ++j) {
            point = c0 + (15 + j * 3.5) * Eigen::Vector2d(cos(deg2rad(i)), sin(deg2rad(i)));
            vPts_Q.push_back(point);
        }
        i++;
    }


    // 直线段，从[0, 15]到[-100, 15]，1m一个;从[0, 18.5]到[-100, 18.5]，1m一个
    for (int i = 0; i >= -100; --i) {
        for (int j = 0; j < numLanes; ++j) {
            vPts_Q.push_back(Eigen::Vector2d(i, 15 + j * 3.5));
        }
    }

    // step1.2: 从点集Q中截取出来一段，构造点集P
    std::vector<Eigen::Vector2d> vPts_P;
    std::vector<bool> vbStopline;
//    for (int i = 537; i < 316; ++i) {        // [0, 29]
    for (int i = vPts_Q.size() - 57; i < vPts_Q.size(); ++i) {        // [0, 29]
        vPts_P.push_back(vPts_Q[i]);
        bool bStopline = false;
        for (int j = 0; j < numLanes; ++j) {
            if (i == vPts_Q.size() - j - 1) {
                bStopline = true;
            }
        }

        vbStopline.push_back(bStopline);

    }

    // 分别给P,Q添加噪声
    for (auto &p : vPts_Q) {
        p += 0.5*sigma * Eigen::Vector2d::Random();
    }

    for (auto &p : vPts_P) {
        p += 3 * sigma * Eigen::Vector2d::Random();
    }

    // 存储点集Q
    std::ofstream fout2("q.txt");
    for (auto p : vPts_Q) {
        fout2 << p[0] << " " << p[1] << endl;
    }
    fout2.close();

    // 存储点集P
    std::ofstream fout1("p01.txt");
    for (auto p : vPts_P) {
        fout1 << p[0] << " " << p[1] << endl;
    }
    fout1.close();

    // step1.3: 对点集p稍作平移及旋转
    std::ofstream fout4("p02.txt");
    int numPts = vPts_P.size();
    for (int i = 0; i < numPts; ++i) {
        point = vPts_P[i];
        point = R0 * point + t0;
        fout4 << point[0] << " " << point[1] << endl;
        vPts_P[i] = point;
    }
    fout4.close();

    // Step2: icp过程
    // step2.0: 初值，应该由别处给出
    double pT[3];

    pT[0] = 0;
    pT[1] = 0;
    pT[2] = 0;
//    pT[0] = t1(0) + (rand() % 10 - 5)/500.0;
//    pT[1] = t1(1) + (rand() % 10 - 5)/500.0;
//    pT[2] = -theta + (rand() % 10 - 5)/500.0;

    // 将点云Q做成kd_tree;
    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
    // Generate pointcloud data
    cloud->width = vPts_Q.size();
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
    bool bPL = std::atoi(argv[1]);
    for (int iter = 0; iter < 200; ++iter) {
        if (bPL) {
            std::vector<Correspondence_pt2line> vCors;
            vCors.reserve(vPts_P.size());
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
            std::stringstream ss;
//        ss << "p1" << iter << ".txt";
            ss << "p1" << 1 << ".txt";
            std::ofstream fout3(ss.str());
            for (int i = 0; i < vPts_P.size(); i++) {
                auto p = vPts_P[i];
                Eigen::Vector2d p1 = R * p + t;
                fout3 << p1[0] << " " << p1[1] << endl;
                // 寻找点云Q中离p1最近的两个点
                searchPoint.x = p1.x();
                searchPoint.y = p1.y();
                kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
                q1 = vPts_Q[pointIdxNKNSearch[0]];
                q2 = vPts_Q[pointIdxNKNSearch[1]];
                if (vbStopline[i]) {
                    q1 = vPts_Q[vPts_Q.size() - 1];
                    q2 = vPts_Q[vPts_Q.size() - 2];
                }

                Eigen::Vector2d l = q1 - q2;
                l.normalize();
                Eigen::Vector2d normal(l[1], -l[0]);
                Correspondence_pt2line cor;
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
            for (int i = 0; i < vPts_P.size(); ++i) {
                // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
                auto p = vCors[i].p;
                auto q = vCors[i].q;
                auto normal = vCors[i].normal;
                auto p1 = R*p + t;
                auto pq = q - p1;
                double error = pq.dot(normal);
                sumError += 0.5 * error * error;
            }
            cout << "sumError = " << sumError << endl;

            ceres::Problem problem;
            for (int i = 0; i < vPts_P.size(); ++i) {
                // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
                ceres::CostFunction* pCostFunction = new ceres::AutoDiffCostFunction<RegisterError, 1, 3>(
                        new RegisterError(vCors[i].p, vCors[i].q, vCors[i].normal));

                problem.AddResidualBlock(pCostFunction, nullptr, pT);
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
//            cout << "theta = " << rad2deg(pT[2]) << ", t = " << pT[0] << " " << pT[1] << endl << endl;

//            std::cout << summary.BriefReport() << endl;
//            if (summary.num_successful_steps == 1 && summary.num_unsuccessful_steps == 0) {
//                cout << "iterations = " << iter << endl;
//                break;
//            }
        } else {
            std::vector<Correspondence_pt2pt> vCors;
            vCors.reserve(vPts_P.size());
            // step2.1: 寻找最近的对应点
            Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
            double alpha = pT[2];
            R << cos(alpha), -sin(alpha),
                    sin(alpha),  cos(alpha);
            Eigen::Vector2d t = Eigen::Vector2d(pT[0], pT[1]);

            int K = 1;
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            Eigen::Vector2d q1 = Eigen::Vector2d::Zero();
            std::stringstream ss;
//        ss << "p1" << iter << ".txt";
            ss << "p1" << 1 << ".txt";
            std::ofstream fout3(ss.str());
            for (auto p : vPts_P) {
                Eigen::Vector2d p1 = R * p + t;
                fout3 << p1[0] << " " << p1[1] << endl;
                // 寻找点云Q中离p1最近的两个点
                searchPoint.x = p1.x();
                searchPoint.y = p1.y();
                kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
                q1 = vPts_Q[pointIdxNKNSearch[0]];
                Correspondence_pt2pt cor;
                cor.p = p;
                cor.q = q1;
                vCors.push_back(cor);
            }
            fout3.close();

            // step2.2: 根据point_to_line的correspondence来计算相对变换R,t
            // 算之前检查一下error
            double sumError = 0;
            for (int i = 0; i < vPts_P.size(); ++i) {
                // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
                auto p = vCors[i].p;
                auto q = vCors[i].q;
                auto p1 = R*p + t;
                auto pq = q - p1;
                double error = pq.norm();
                sumError += 0.5 * error * error;
            }
//            cout << "sumError = " << sumError << endl;

            ceres::Problem problem;
            for (int i = 0; i < vPts_P.size(); ++i) {
                // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
                ceres::CostFunction* pCostFunction = new ceres::AutoDiffCostFunction<RegisterError_pt2pt, 1, 3>(
                        new RegisterError_pt2pt(vCors[i].p, vCors[i].q));

                problem.AddResidualBlock(pCostFunction, nullptr, pT);
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
//        cout << "theta = " << pT[2] << ", t = " << pT[0] << " " << pT[1] << endl << endl;

//        std::cout << summary.BriefReport() << endl;
//        if (summary.num_successful_steps == 1 && summary.num_unsuccessful_steps == 0) {
//            cout << "iterations = " << iter << endl;
//            break;
//        }
        }


    }

    // step2.3: 最终输出pose
    cout << "theta = " << rad2deg(pT[2]) << ", t = [" << pT[0] << " " << pT[1] << "]" << endl;
    cout << "delta is: " << "theta = " << -theta0 - rad2deg(pT[2]) << ", t = [" << t1[0] - pT[0] << " " << t1[1] - pT[1] << "]" << endl << endl;

    //*******************************************************************//


    return 0;
}