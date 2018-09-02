#pragma once
#include "ActivationFunction.h"
class ReLU_FCN :
	public ActivationFunction
{
public:
	ReLU_FCN();
	~ReLU_FCN();

	void forwardPass(Eigen::MatrixXd &z);

	void backwardPass(const Eigen::MatrixXd &dA, const Eigen::MatrixXd &Z, Eigen::MatrixXd & dZ);
};

