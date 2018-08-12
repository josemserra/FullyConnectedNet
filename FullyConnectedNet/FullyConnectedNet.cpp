#include <Windows.h> 
#include <iostream>
#include <fstream> 

#include <vector>
#include <string>
#include <algorithm> 

#include <time.h>

#include "CImg.h"
#include <Eigen/Dense>

#include "IO_Manager.h"
#include "FCNet.h"

using namespace cimg_library;

//Full preprocess of an image, that is load, resize and convert to eigen
Eigen::MatrixXd PreProcessImg(std::string imgFilePath, int imgRescaleValue) {

	//Load img cimg
	CImg<unsigned char> image(imgFilePath.c_str());
	//Resize img
	image.resize(imgRescaleValue, imgRescaleValue);
	//Convert to Eigen + flatten
	Eigen::MatrixXd flattenImg = IO_Manager::convertImg2Eigen(image);
	//Normalise. This could be the average image of the full DB, but it works just fine with 255
	flattenImg *= 1.0 / 255;
	return flattenImg;
}

//Loads a training/dev or test set for a simple binary classification task. You can specify the folder for each class and a rescale value to preprocess the images
//It returns the training samples as a <flattenedImgDim,NumSamples> matrix and a <numSamples,1> mat with the classes of the loaded images
void LoadSet(std::string classExamplesFolder, std::string nonClassExamplesFolder, int imgRescaleValue, Eigen::MatrixXd &outTrainingSamples, Eigen::MatrixXi &outTrainingSamplesClasses) {

	std::vector<std::string> trainImgFilesClass = IO_Manager::FindAllImgInFolder(classExamplesFolder);
	std::vector<std::string> trainImgFilesNotClass = IO_Manager::FindAllImgInFolder(nonClassExamplesFolder);

	outTrainingSamples = Eigen::MatrixXd(imgRescaleValue*imgRescaleValue * 3, trainImgFilesClass.size() + trainImgFilesNotClass.size());
	outTrainingSamplesClasses = Eigen::MatrixXi(1, trainImgFilesClass.size() + trainImgFilesNotClass.size());

	for (int imgIdx = 0; imgIdx < trainImgFilesClass.size(); imgIdx++) {
		//Preprocess Image
		Eigen::MatrixXd flattenImg = PreProcessImg(trainImgFilesClass[imgIdx], imgRescaleValue);
		//Add to trainingSamples
		outTrainingSamples.col(imgIdx) = flattenImg;
		//Create labels
		outTrainingSamplesClasses(0, imgIdx) = 1;
	}

	for (int imgIdx = 0; imgIdx < trainImgFilesNotClass.size(); imgIdx++) {
		//Load img cimg
		Eigen::MatrixXd flattenImg = PreProcessImg(trainImgFilesNotClass[imgIdx], imgRescaleValue);
		//Add to trainingSamples
		outTrainingSamples.col(trainImgFilesClass.size() + imgIdx) = flattenImg;
		//Create labels
		outTrainingSamplesClasses(0, trainImgFilesClass.size() + imgIdx) = 0;

	}

}

//Initialize all the weights with random values (small) and b with 0
void InitializeNeuron(int inputSize, Eigen::MatrixXd &weights, Eigen::VectorXd &b) {

	srand(1); // Just to force random to always generate the same randoms (good for tests purposes)

	weights = Eigen::MatrixXd::Random(inputSize*inputSize * 3, 1)*0.01; //keep values small
	b = Eigen::VectorXd::Zero(1);
}

//Activation function. Applies the sigmoid function element wise on a matrix. Changes the input
void Sigmoid(Eigen::MatrixXd &z) {
	z = (1.0 + (-1 * z.array()).exp()).inverse().matrix();
}

//Forward Propagation step for single neuron
Eigen::MatrixXd ForwardPropagation(Eigen::MatrixXd weights, Eigen::VectorXd b, Eigen::MatrixXd X) {

	Eigen::MatrixXd A = weights.transpose()*X;
	A.colwise() += b;

	Sigmoid(A);

	return A;
}

//Cross Entropy Loss Function. A are the predictions, Y are the training labels 
Eigen::MatrixXd CrossEntropy(Eigen::MatrixXd &A, Eigen::MatrixXd &Y) {
	Eigen::MatrixXd entropy = -Y.array()*((A.array()).log()) - (1 - Y.array())*((1 - A.array()).log());
	return entropy;
}

//Calculates the cost for all the samples in A
double CalculateCost(Eigen::MatrixXd A, Eigen::MatrixXd Y) {
	double E = 0.00000001;

	int m = A.cols();
	Eigen::MatrixXd entropy = CrossEntropy(A, Y);
	double cost = (1.0 / m)*(entropy.sum() + E);
	return cost;
}

//Calculates dJ/dW (dw) and dJ/db (db), which describe how much the weights should change to approximate the predictions of the true classes
void BackwardPropagation(Eigen::MatrixXd X, Eigen::MatrixXd A, Eigen::MatrixXd Y, Eigen::MatrixXd &dw, Eigen::MatrixXd &db) {

	Eigen::MatrixXd dz = A - Y;
	int m = dz.cols();

	Eigen::VectorXd dzV(Eigen::Map<Eigen::VectorXd>(dz.data(), m)); //otherwise I can't broadcast in the line below

	dw = X.array().rowwise() * dzV.transpose().array();
	Eigen::MatrixXd dTemp = (1.0 / m)*(dw.rowwise().sum()); // Eigen behaves strangly if I don't store the results in a temp variable
	dw = dTemp;

	dTemp = (1.0 / m)*(dz.rowwise().sum());
	db = dTemp;
}

void ShuffleMatrixCols(Eigen::MatrixXd X, Eigen::MatrixXi X_Classes, Eigen::MatrixXd &X_perm, Eigen::MatrixXi &X_Classes_Perm) {

	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X.cols());
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
	X_perm = X * perm; // permute columns
	X_Classes_Perm = X_Classes * perm; // permute columns
									   //Eigen::MatrixXi x2_perm2 = perm * x123; // permute rows

}

void GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXi X_Classes, Eigen::MatrixXd &weights, Eigen::VectorXd &b, int numEpochs = 25, double learningRate = 0.001, bool plotCost = false) {

	CImgDisplay main_disp;
	std::vector<double> x;
	std::vector<double> y;
	if (plotCost)
		main_disp = CImgDisplay(500, 400, "Cost Plot"); // display it


														//Gradient descent
	for (int itIdx = 0; itIdx < numEpochs; itIdx++) {

		Eigen::MatrixXd preds = ForwardPropagation(weights, b, X);

		double cost = CalculateCost(preds, X_Classes.cast <double>());

		if (plotCost) {
			x.push_back(itIdx);
			y.push_back(cost);
			FCNet::drawPlot(main_disp, x, y,
				0.0f, 15.0f, 0.0f, 1.0f,
				"Iterations", "Cost");
		}

		//Single Back Prop Step
		Eigen::MatrixXd dw;
		Eigen::MatrixXd db;
		BackwardPropagation(X, preds, X_Classes.cast <double>(), dw, db);

		//Update weights
		weights = weights - learningRate*dw;
		b = b - learningRate*db;
	}

	main_disp.wait(); // Wait for key any key input
}

void BatchGradientDescent(Eigen::MatrixXd X, Eigen::MatrixXi X_Classes, Eigen::MatrixXd &weights, Eigen::VectorXd &b, int batchSize = 32, int numEpochs = 25, double learningRate = 0.001, bool plotCost = false) {

	CImgDisplay main_disp;
	std::vector<double> x;
	std::vector<double> y;
	if (plotCost)
		main_disp = CImgDisplay(500, 400, "Cost Plot"); // display it


	int numTrainingSamples = X.cols();
	int counterPlot = 0;
	for (int itIdx = 0; itIdx < numEpochs; itIdx++) {

		//Random shuffle of samples
		Eigen::MatrixXd X_perm;
		Eigen::MatrixXi X_Classes_Perm;
		ShuffleMatrixCols(X, X_Classes, X_perm, X_Classes_Perm);

		//Process all batches aside from last one, which might have a different size than the others
		int processedBatches = 0;
		while ((processedBatches + batchSize) < numTrainingSamples) {

			Eigen::MatrixXd batch = X.block(0, processedBatches, X.rows(), numTrainingSamples - processedBatches);
			Eigen::MatrixXi batchClasses = X_Classes.block(0, processedBatches, 1, numTrainingSamples - processedBatches);

			Eigen::MatrixXd preds = ForwardPropagation(weights, b, batch);

			double cost = CalculateCost(preds, batchClasses.cast <double>());

			if (plotCost) {
				x.push_back(counterPlot);
				y.push_back(cost);
				FCNet::drawPlot(main_disp, x, y,
					0.0f, 15.0f, 0.0f, 1.0f,
					"Iterations", "Cost");
				counterPlot++;
			}


			//Single Back Prop Step
			Eigen::MatrixXd dw;
			Eigen::MatrixXd db;
			BackwardPropagation(batch, preds, batchClasses.cast <double>(), dw, db);

			weights = weights - learningRate*dw;
			b = b - learningRate*db;

			processedBatches += batchSize;
		}

		//Process the last batch
		Eigen::MatrixXd batch = X.block(0, processedBatches, X.rows(), numTrainingSamples - processedBatches);
		Eigen::MatrixXi batchClasses = X_Classes.block(0, processedBatches, 1, numTrainingSamples - processedBatches);

		Eigen::MatrixXd preds = ForwardPropagation(weights, b, batch);

		double cost = CalculateCost(preds, batchClasses.cast <double>());

		if (plotCost) {
			x.push_back(counterPlot);
			y.push_back(cost);
			FCNet::drawPlot(main_disp, x, y,
				0.0f, 15.0f, 0.0f, 1.0f,
				"Iterations", "Cost");
			counterPlot++;
		}


		Eigen::MatrixXd dw;
		Eigen::MatrixXd db;
		BackwardPropagation(batch, preds, batchClasses.cast <double>(), dw, db);

		//Update weights
		weights = weights - learningRate*dw;
		b = b - learningRate*db;

	}

	main_disp.wait(); // Wait for key any key input
}

Eigen::MatrixXd Predict(Eigen::MatrixXd weights, Eigen::VectorXd b, Eigen::MatrixXd X) {

	double threshold = 0.5f;

	Eigen::MatrixXd preds = ForwardPropagation(weights, b, X);

	preds = (preds.array() > threshold).select(1, preds);
	preds = (preds.array() <= threshold).select(0, preds);

	return preds;
}

double CalcError(Eigen::MatrixXd real_Y, Eigen::MatrixXd pred_Y) {

	if (real_Y.cols() != pred_Y.cols()) {
		std::cout << "Real Y and Pred Y need be same dimensions \n";
		return -1.0f;
	}

	double error = (real_Y - pred_Y).cwiseAbs().sum();
	error /= real_Y.cols();

	return error;
}



int main() {

	std::string trainFolderDogs = "../Img/Train/Dogs";
	std::string trainFolderNotDogs = "../Img/Train/Not Dogs";
	std::string devFolderDogs = "../Img/Dev/Dogs";
	std::string devFolderNotDogs = "../Img/Dev/Not Dogs";
	std::string testFolder = "../Img/Test";

	bool loadDBFromFiles = false;
	int imgRescaleValue = 64;

	//Initialisation Load Dataset -------------------
	Eigen::MatrixXd TrainingSamples;
	Eigen::MatrixXi TrainingSamplesClasses;
	//Class 1 - Dogs
	//Class 2 - Not Dogs
	if (loadDBFromFiles) {
		IO_Manager::Deserialise(TrainingSamples, "../Cereal Database/TrainingSamples.eigm");
		IO_Manager::Deserialise(TrainingSamplesClasses, "../Cereal Database/TrainingSamplesClasses.eigm");
	}
	else {
		LoadSet(trainFolderDogs, trainFolderNotDogs, imgRescaleValue, TrainingSamples, TrainingSamplesClasses);
		//Save Eigen Mat Files
		IO_Manager::Serialise(TrainingSamples, "../Cereal Database/TrainingSamples.eigm");
		IO_Manager::Serialise(TrainingSamplesClasses, "../Cereal Database/TrainingSamplesClasses.eigm");
	}

	Eigen::MatrixXd DevSamples;
	Eigen::MatrixXi DevSamplesClasses;
	//Class 1 - Dogs
	//Class 2 - Not Dogs
	if (loadDBFromFiles) {
		IO_Manager::Deserialise(DevSamples, "../Cereal Database/DevSamples.eigm");
		IO_Manager::Deserialise(DevSamplesClasses, "../Cereal Database/DevSamplesClasses.eigm");
	}
	else {
		LoadSet(devFolderDogs, devFolderNotDogs, imgRescaleValue, DevSamples, DevSamplesClasses);
		//Save Eigen Mat Files
		IO_Manager::Serialise(DevSamples, "../Cereal Database/DevSamples.eigm");
		IO_Manager::Serialise(DevSamplesClasses, "../Cereal Database/DevSamplesClasses.eigm");
	}

	//Initialisation --------------------------------
	Eigen::MatrixXd weights;
	Eigen::VectorXd b;
	InitializeNeuron(imgRescaleValue, weights, b);

	//Single Fwd Prop Step --------------------------
	Eigen::MatrixXd preds = ForwardPropagation(weights, b, TrainingSamples);

	//Calc cost after Fwd Prop ----------------------
	double cost = CalculateCost(preds, TrainingSamplesClasses.cast <double>());

	//Single Back Prop Step -------------------------
	Eigen::MatrixXd dw;
	Eigen::MatrixXd db;
	BackwardPropagation(TrainingSamples, preds, TrainingSamplesClasses.cast <double>(), dw, db);

	//Train with Gradient Descent -------------------
	InitializeNeuron(imgRescaleValue, weights, b);
	GradientDescent(TrainingSamples, TrainingSamplesClasses, weights, b, 150, 0.001, true);

	preds = ForwardPropagation(weights, b, TrainingSamples);
	cost = CalculateCost(preds, TrainingSamplesClasses.cast <double>());

	std::cout << "Final Training Cost: " << cost << "\n";
	preds = Predict(weights, b, TrainingSamples);
	std::cout << "Train set with Gradient Descent Accuracy: " << 100 - CalcError(TrainingSamplesClasses.cast <double>(), preds) * 100 << "\n";
	preds = Predict(weights, b, DevSamples);
	std::cout << "Dev set with Gradient Descent Accuracy: " << 100 - CalcError(DevSamplesClasses.cast <double>(), preds) * 100 << "\n";

	std::cout << "-------------------------------------------------------\n";

	//Train with Batch Gradient Descent -------------
	Eigen::MatrixXd weights_Batch;
	Eigen::VectorXd b_Batch;
	InitializeNeuron(imgRescaleValue, weights_Batch, b_Batch);
	BatchGradientDescent(TrainingSamples, TrainingSamplesClasses, weights_Batch, b_Batch, 32, 150, 0.001, true);

	preds = ForwardPropagation(weights_Batch, b_Batch, TrainingSamples);
	cost = CalculateCost(preds, TrainingSamplesClasses.cast <double>());

	std::cout << "Final Training Cost: " << cost << "\n";
	preds = Predict(weights_Batch, b_Batch, TrainingSamples);
	std::cout << "Train set with Batch Gradient Descent Accuracy: " << 100 - CalcError(TrainingSamplesClasses.cast <double>(), preds) * 100 << "\n";
	preds = Predict(weights_Batch, b_Batch, DevSamples);
	std::cout << "Dev set with Batch Gradient Descent Accuracy: " << 100 - CalcError(DevSamplesClasses.cast <double>(), preds) * 100 << "\n";

	std::cout << "-------------------------------------------------------\n";

	return 0;
}