#include "Sigmoid_FCN.h"

Sigmoid_FCN::Sigmoid_FCN()
{
}

Sigmoid_FCN::~Sigmoid_FCN()
{
}

void Sigmoid_FCN::forwardPass(Eigen::MatrixXd &z) {
	z = (1.0 + (-1 * z.array()).exp()).inverse().matrix();
}

void Sigmoid_FCN::backwardPass(const Eigen::MatrixXd &dA, const Eigen::MatrixXd &Z, Eigen::MatrixXd & dZ) {
	Eigen::MatrixXd zTemp = Z;
	forwardPass(zTemp);

	dZ = dA.array()*(zTemp.array()*(1 - zTemp.array()));
}