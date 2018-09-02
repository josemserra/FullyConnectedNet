#include <Eigen/Dense>

#pragma once
class ActivationFunction
{
public:
	virtual void forwardPass(Eigen::MatrixXd &z) = 0;

	virtual void backwardPass(const Eigen::MatrixXd &dA, const Eigen::MatrixXd &Z, Eigen::MatrixXd & dZ) = 0;
};

