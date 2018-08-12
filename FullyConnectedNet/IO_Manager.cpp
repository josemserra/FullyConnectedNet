#include "IO_Manager.h"

Eigen::MatrixXd IO_Manager::convertImg2Eigen(cimg_library::CImg<unsigned char> img)
{

	int numPixels = img.height() * img.width();

	Eigen::MatrixXd channelR(img.height(), img.width());
	Eigen::MatrixXd channelG(img.height(), img.width());
	Eigen::MatrixXd channelB(img.height(), img.width());
	Eigen::MatrixXd returnValue(numPixels * 3, 1);

	//read into eigen mat
	cimg_forXY(img, colIdx, rowIdx) {
		channelR(rowIdx, colIdx) = img(colIdx, rowIdx, 0, 0); //Red
		channelG(rowIdx, colIdx) = img(colIdx, rowIdx, 0, 1); //Green	
		channelB(rowIdx, colIdx) = img(colIdx, rowIdx, 0, 2); //Blue
	}

	//flatten Channels
	channelR.resize(numPixels, 1);
	channelG.resize(numPixels, 1);
	channelB.resize(numPixels, 1);

	//Assign the blocks
	returnValue.block(0, 0, numPixels, 1) = channelR; //From row 0 to col (img.height() * img.width()=numPixels) is the R channel
	returnValue.block(numPixels, 0, numPixels, 1) = channelG; //From row numPixels to col 2*numPixels is the G channel
	returnValue.block(2 * numPixels, 0, numPixels, 1) = channelB; //From row 2*numPixels to col 3*numPixels is the B channel

	return returnValue;
}

// Returns a vector with a path for all the image files in the specified folder
std::vector<std::string> IO_Manager::FindAllImgInFolder(std::string folder) {

	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind;

	std::vector<std::string> returnVal;
	std::string newPath = folder;
	newPath.append("\\*.jpg");

	hFind = FindFirstFileA(newPath.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		printf("FindFirstFile failed (%d)\n", GetLastError());
		return returnVal;
	}
	else
	{
		do
		{
			//ignore current and parent directories
			if (strcmp(FindFileData.cFileName, ".") == 0 || strcmp(FindFileData.cFileName, "..") == 0)
				continue;

			if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				//ignore directories
			}
			else
			{
				//list the Files
				std::string temp = folder;
				temp.append("/");
				temp.append(FindFileData.cFileName);
				returnVal.push_back(temp);
			}
		} while (FindNextFile(hFind, &FindFileData));
		FindClose(hFind);
	}

	return returnVal;
}



