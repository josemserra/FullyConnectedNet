#pragma once
#ifndef _FC_NET_
#define _FC_NET_

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include "CImg.h"
#include <iostream>

#include "ActivationFunction.h"

class FCNet
{
public:
	FCNet();
	~FCNet();

	enum CostFunctions {cross_entropy_fcn};
	enum Optimiser {grad_desc_fcn };

	//Plot the cost function evolution
	static void drawPlot(cimg_library::CImgDisplay& disp, std::vector<double> x, std::vector<double> y,
		double minX = 0.0f, double maxX = 15.0f, double minY = 0.0f, double maxY = 1.0f,
		std::string xLabel = "xAxis", std::string yLabel = "yAxis");

	//Defines the input dimensions
	void setInputLayerSize(int nX);

	//Adds an hidden layer to the network with the desired activation function (sigmoid, relu...)
	void addLayer(int nh, ActivationFunction* actFunc);

	//Defines the costf function: cross entropy, squared error, etc...
	void setCostFunction(CostFunctions costFunc);

	//Defines optimization method used when training (gradient descent....). -1 means that the method should use all samples
	void setOptimization(Optimiser opt, int batchSize = 32);

	//Train method, which takes the samples (X) and expected output for these samples (Y), number of epochs, learning rate of the update and if the method should plot a cost graph
	bool trainNetwork(Eigen::MatrixXd X, Eigen::MatrixXi Y, int numEpochs = 25, double learningRate = 0.001, bool plotCost = false);
	
	//Given one (or more) samples it will generate a vector with predictions
	Eigen::MatrixXd predict(Eigen::MatrixXd X, double treshold = 0.5f);

private:
	//Internal methods

	//Initialises the weights and biases for each layer
	bool initialiseNetworkStructure();

	// Forward Pass on the current network for the given input samples X
	Eigen::MatrixXd ForwardPropagation(Eigen::MatrixXd &X, std::vector<Eigen::MatrixXd>& cache_Z, std::vector<Eigen::MatrixXd>& cache_A);

	//Network properties
	int inputLayerSize = 0;
	int numHiddenLayers = 0;
	std::vector<std::tuple<int, ActivationFunction*>> layersInfo;
	std::vector<Eigen::MatrixXd> layers_weights;
	std::vector<Eigen::VectorXd> layers_b;
	CostFunctions costFunc = CostFunctions::cross_entropy_fcn;
	Optimiser opt = Optimiser::grad_desc_fcn;
	int batchSize = -1; //-1 means use all samples

};

#endif
