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
		double minX = 0.0f, double maxX = 15.0f, double minY = 0.0f, double maxY = 1.0f,
		std::string xLabel = "xAxis", std::string yLabel = "yAxis");

	void setInputLayerSize(int nX);
	void addLayer(int nh, ActivationFunctions actFunc);
	void setCostFunction(CostFunctions costFunc);
	void setOptimization(Optimiser opt, int batchSize = 32);
	bool trainNetwork(Eigen::MatrixXd X, Eigen::MatrixXi Y, int numEpochs = 25, double learningRate = 0.001, bool plotCost = false);
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
