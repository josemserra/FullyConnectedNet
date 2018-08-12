#pragma once
#ifndef _IO_MANAGER_
#define _IO_MANAGER_

//#include <Windows.h> 
//#include <iostream>
//#include <fstream> 
//
#include <vector>
#include <string>
//#include <algorithm> 


#include "CImg.h"
#include <Eigen/Dense>

namespace IO_Manager
{

	std::vector<std::string> FindAllImgInFolder(std::string folder);

	Eigen::MatrixXd convertImg2Eigen(cimg_library::CImg<unsigned char> img);

	template<typename T>
	void Serialise(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, std::string fileName = "matrix.eigm") {

		std::fstream writeFile;
		writeFile.open(fileName, std::ios::binary | std::ios::out);

		if (writeFile.is_open())
		{
			int rows, cols;
			rows = m.rows();
			cols = m.cols();

			writeFile.write((const char *)&(rows), sizeof(int));
			writeFile.write((const char *)&(cols), sizeof(int));

			writeFile.write((const char *)(m.data()), sizeof(T) * rows * cols);

			writeFile.close();
		}
	}

	template<typename T>
	void Deserialise(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, std::string fileName = "matrix.eigm") {
		std::fstream readFile;
		readFile.open(fileName, std::ios::binary | std::ios::in);
		if (readFile.is_open())
		{
			int rows, cols;
			readFile.read((char*)&rows, sizeof(int));
			readFile.read((char*)&cols, sizeof(int));

			m.resize(rows, cols);

			readFile.read((char*)(m.data()), sizeof(T) * rows * cols);

			readFile.close();

		}
	}
}

#endif