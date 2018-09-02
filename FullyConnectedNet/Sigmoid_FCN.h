#pragma once
#include "ActivationFunction.h"
class Sigmoid_FCN :
	public ActivationFunction
{
public:
	Sigmoid_FCN();
	~Sigmoid_FCN();

	void forwardPass(Eigen::MatrixXd &z);

	void backwardPass(const Eigen::MatrixXd &dA, const Eigen::MatrixXd &Z, Eigen::MatrixXd & dZ);
};

