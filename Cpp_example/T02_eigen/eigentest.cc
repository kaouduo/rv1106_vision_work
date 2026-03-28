#include <iostream>
#include <Eigen/Dense>  // Eigen 的核心模块

int main() {
    // 声明两个 3x3 矩阵
    Eigen::Matrix3f matA;
    Eigen::Matrix3f matB;

    // 初始化矩阵 A
    matA << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    // 初始化矩阵 B
    matB << 2, 0, 0,
            0, 2, 0,
            0, 0, 2;

    std::cout << "Matrix A:\n" << matA << std::endl;
    std::cout << "Matrix B:\n" << matB << std::endl;

    // 矩阵相乘
    Eigen::Matrix3f result = matA * matB;

    std::cout << "Result of A * B:\n" << result << std::endl;

    return 0;
}