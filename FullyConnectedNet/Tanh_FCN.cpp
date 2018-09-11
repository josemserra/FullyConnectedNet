#include "Tanh_FCN.h"

Tanh_FCN::Tanh_FCN()
{
}

Tanh_FCN::~Tanh_FCN()
{
}

void Tanh_FCN::forwardPass(Eigen::MatrixXd &z) {
	z = z.array().tanh().matrix();
}

void Tanh_FCN::backwardPass(const Eigen::MatrixXd &dA, const Eigen::MatrixXd &Z, Eigen::MatrixXd &dZ) {
	Eigen::MatrixXd zTemp = Z;
	forwardPass(zTemp);

	dZ = dA.array()*(1 - zTemp.array().square());
}
