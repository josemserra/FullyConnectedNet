#pragma once
#include "ActivationFunction.h"
class Tanh_FCN :
	public ActivationFunction
{
public:
	Tanh_FCN();
	~Tanh_FCN();

	void forwardPass(Eigen::MatrixXd &z);

	void backwardPass(const Eigen::MatrixXd &dA, const Eigen::MatrixXd &Z, Eigen::MatrixXd & dZ);

};

