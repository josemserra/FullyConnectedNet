#include "ReLU_FCN.h"

ReLU_FCN::ReLU_FCN()
{
}

ReLU_FCN::~ReLU_FCN()
{
}

void ReLU_FCN::forwardPass(Eigen::MatrixXd &z) {
	z = (z.array() > 0).select(z.array(), 0);
}

void ReLU_FCN::backwardPass(const Eigen::MatrixXd &dA, const Eigen::MatrixXd &Z, Eigen::MatrixXd &dZ) {
	Eigen::MatrixXd dZTemp;
	dZTemp = (Z.array() < 0).select(Eigen::MatrixXd::Zero(Z.array().rows(), Z.array().cols()).array(), Eigen::MatrixXd::Ones(Z.array().rows(), Z.array().cols()).array());

	dZ = dA.array()*dZTemp.array();
}