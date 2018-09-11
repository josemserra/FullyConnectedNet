#include "FCNet.h"

FCNet::FCNet() {
}

FCNet::~FCNet() {
}

bool FCNet::initialiseNetworkStructure() {
	//Checks
	if (inputLayerSize == 0) {
		std::cout << "The input size was not specified. \n";
		return false;
	}

	if (numHiddenLayers == 0) {
		std::cout << "No hidden layers were added. Please add some hidden layers. \n";
		return false;
	}

	if (std::get<0>(layersInfo[layersInfo.size()-1]) != 1) {
		std::cout << "Currently, this project only supports binary classification or regression for one output value. The last layer can only have one node. \n";
		return false;
	}

	layers_weights.clear();
	layers_b.clear();

	srand(1); // Just to force random to always generate the same randoms (good for tests purposes)

	Eigen::MatrixXd weights = Eigen::MatrixXd::Random(std::get<0>(layersInfo[0]), inputLayerSize)*0.01; //keep values small
	Eigen::VectorXd b = Eigen::VectorXd::Zero(std::get<0>(layersInfo[0]), 1);

	layers_weights.push_back(weights);
	layers_b.push_back(b);

	for (int layerIdx = 1; layerIdx < numHiddenLayers; layerIdx++) {

		Eigen::MatrixXd weights_temp = Eigen::MatrixXd::Random(std::get<0>(layersInfo[layerIdx]), std::get<0>(layersInfo[layerIdx - 1]))*0.01; //keep values small
		Eigen::VectorXd b_temp = Eigen::VectorXd::Zero(std::get<0>(layersInfo[layerIdx]), 1);

		layers_weights.push_back(weights_temp);
		layers_b.push_back(b_temp);
	}

	return true;
}

void FCNet::setInputLayerSize(int nX) {
	inputLayerSize = nX;
}

void FCNet::addLayer(int nh, ActivationFunction* actFunc) {
	layersInfo.push_back(std::tuple<int, ActivationFunction*>(nh, actFunc));
	numHiddenLayers++;
}

void FCNet::setCostFunction(CostFunctions costFunc) {
	this->costFunc = costFunc;
}

void FCNet::setOptimization(Optimiser opt, int batchSize) {
	this->opt = opt;
	this->batchSize = batchSize;
}

bool FCNet::trainNetwork(Eigen::MatrixXd X, Eigen::MatrixXi Y, 
	int numEpochs, double learningRate, bool plotCost) {
	
	if (!initialiseNetworkStructure())
		return false;

	Eigen::MatrixXd Y_temp = Y.cast <double>();

	for (int epochIdx = 0; epochIdx < numEpochs; epochIdx++) {
		std::vector<Eigen::MatrixXd> cache_Z;
		std::vector<Eigen::MatrixXd> cache_A;

		Eigen::MatrixXd preds = ForwardPropagation(X, cache_Z, cache_A);

		//ToDo

	}	
	return true;
}

Eigen::MatrixXd FCNet::predict(Eigen::MatrixXd X, double treshold) {
	//Check
	if (layers_weights.size() == 0) {
		std::cout << "Something went wrong with training. Aborting predict. \n";
		return Eigen::MatrixXd();
	}

	//ToDo

	return Eigen::MatrixXd();
}

Eigen::MatrixXd FCNet::ForwardPropagation(Eigen::MatrixXd &X, std::vector<Eigen::MatrixXd>& cache_Z, std::vector<Eigen::MatrixXd>& cache_A) {
	
	cache_A.clear();
	cache_Z.clear();
	
	Eigen::MatrixXd A;
	A = X;

	for (int layerIdx = 0; layerIdx < numHiddenLayers; layerIdx++) {

		Eigen::MatrixXd Z = layers_weights[layerIdx] * A;
		Z.colwise() += layers_b[layerIdx];

		A = Z;
		std::get<1>(layersInfo[layerIdx])->forwardPass(A);

		cache_A.push_back(A);
		cache_Z.push_back(Z);
	}

	return A;
}



void FCNet::drawPlot(cimg_library::CImgDisplay& disp, std::vector<double> x, std::vector<double> y,
	double minX, double maxX, double minY, double maxY,
	std::string xLabel, std::string yLabel) {

	const unsigned char lineColour[] = { 0,0,0 };// i.e. black
	int bgFillColour = 255;// i.e. white

	if (x.size() != y.size()) {
		std::cout << "Both vectors need to have the same size. \n Will not draw anything \n";
	}

	int dispWidth = disp.width();
	int dispHeight = disp.height();

	cimg_library::CImg<unsigned char>  visu(dispWidth, dispHeight, 1, 3, 1);

	//Plot Drawing limits, i.e. what is drawn inside the axis lines
	int xMinAxis = 50;
	int yMinAxis = 10;
	int xMaxAxis = dispWidth - 10;
	int yMaxAxis = dispHeight - 50;

	//Validate the max. if any of the values in x or y are larger or smaller than the the max or min (respectively), increase the max and decrease the min (respectively).
	auto minX_it = std::min_element(std::begin(x), std::end(x));
	auto maxX_it = std::max_element(std::begin(x), std::end(x));
	auto minY_it = std::min_element(std::begin(y), std::end(y));
	auto maxY_it = std::max_element(std::begin(y), std::end(y));

	if (*minX_it < minX)
		minX = *minX_it - 1;
	if (*maxX_it > maxX) {
		maxX = *maxX_it + 1;
	}
	if (*minY_it < minY)
		minY = *minY_it - 1;
	if (*maxY_it > maxY)
		maxY = *maxY_it + 1;


	visu.fill(bgFillColour);

	//Draw Axis labels
	visu.rotate(90);
	visu.draw_text((int)dispHeight / 2, 10, yLabel.c_str(), lineColour, 0, 1, 30, 30);
	visu.rotate(-90);
	visu.draw_text((int)(dispWidth / 2 - (xLabel.size() / 2) * 10), dispHeight - 40, xLabel.c_str(), lineColour, 0, 1, 30, 30);

	//Draw Axis Lines
	visu.draw_line(xMinAxis, yMinAxis, xMinAxis, yMaxAxis, lineColour, 1);
	visu.draw_line(xMinAxis, yMaxAxis, xMaxAxis, yMaxAxis, lineColour, 1);

	//Convert Vector Values to the appropriate pix coordinates
	for (int idx = 0; idx < x.size(); idx++) {

		if (x[idx] > maxX)
			maxX = x[idx] + 5.0f;

		if (y[idx] > maxY)
			maxY = y[idx] + 1.0f;

		x[idx] = (x[idx] - minX) / maxX;
		x[idx] = x[idx] * (xMaxAxis - xMinAxis);

		y[idx] = (y[idx] - minY) / maxY;
		y[idx] = (1 - y[idx]) * (yMaxAxis - yMinAxis);

		if (idx > 1) { // Draw line
			visu.draw_line(x[idx - 1] + xMinAxis, y[idx - 1] + yMinAxis, x[idx] + xMinAxis, y[idx] + yMinAxis, lineColour, 1);
		}
	}

	visu.display(disp);

	//Alternatives, simpler with less control
	//Draw each point http://www.cplusplus.com/forum/general/82584/
	// locks after it draws https://stackoverflow.com/questions/39414084/plotting-a-vector-in-c-with-cimg
}