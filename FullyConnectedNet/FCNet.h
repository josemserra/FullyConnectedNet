#pragma once
#ifndef _FC_NET_
#define _FC_NET_

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include "CImg.h"
#include <iostream>


class FCNet
{
public:
	FCNet();
	~FCNet();

	static enum ActivationFunctions {sigmoid_fcn, relu_fcn, tanh_fcn };
	static enum CostFunctions {cross_entropy_fcn, squared_error_fcn };
	static enum Optimiser {grad_desc_fcn };

	static void drawPlot(cimg_library::CImgDisplay& disp, std::vector<double> x, std::vector<double> y,
		double minX, double maxX, double minY, double maxY,
		std::string xLabel, std::string yLabel);


	void setInputLayerSize(int nX);
	void addLayer(int nh, ActivationFunctions actFunc);
	void setCostFunction(CostFunctions costFunc);
	void setOptimization(Optimiser opt, int batchSize);
	bool trainNetwork(Eigen::MatrixXd X, Eigen::MatrixXi Y, int numEpochs, double learningRate, bool plotCost);
	Eigen::MatrixXi predict(Eigen::MatrixXd X);

private:
	int inputLayerSize = 0;
	int numLayers = 0;
	std::vector<std::tuple<int, ActivationFunctions>> layersInfo;
	std::vector<Eigen::MatrixXd> layers_weights;
	std::vector<Eigen::VectorXd> layers_b;
	CostFunctions costFunc = CostFunctions::cross_entropy_fcn;
	Optimiser opt = Optimiser::grad_desc_fcn;

};

#endif
