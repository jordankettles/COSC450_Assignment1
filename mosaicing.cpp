#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Timer.h"
#include "cmath"

cv::Mat imageGenerator(cv::Mat img1, cv::Mat H, double percent) {
	int height = (int) img1.rows * percent;
	int width = (int) img1.cols * percent;
	cv::Mat img2(img1.rows, img1.cols, CV_8UC3);
	cv::Mat largerCrop(img1.rows, img1.cols, CV_8UC3);
	cv::Mat croppedimage = img1(cv::Range(img1.rows - height, img1.rows), cv::Range(img1.cols - width, img1.cols));
	cv::resize(croppedimage, largerCrop, img1.size());
	// Crop image here.
	cv::warpPerspective(largerCrop, img2, H, img1.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
	return img2;
}

cv::Mat getTestImage1(int option) {
	cv::Mat returnImage;
	//Image given in example folder.
	if (option == 1) {
		returnImage = cv::imread("../testimages/testimage1.jpg");
	}
	if (option == 2) {
		returnImage = cv::imread("../testimages/testimage2.jpg");
	}
	if (option == 3) {
		returnImage = cv::imread("../testimages/testimage3.jpg");
	}
	return returnImage;
}

cv::Mat getHomography(int option) {
	cv::Mat H(3, 3, CV_64F);
	// Fridge
	if (option == 1) {
		H.at<double>(0, 0) = 1.005231550598348;
		H.at<double>(0, 1) = -0.001754330803839498;
		H.at<double>(0, 2) = 0.003508661608419048;
		H.at<double>(1, 0) = 0.004643058170193042;
		H.at<double>(1, 1) = 1.002900564761766;
		H.at<double>(1, 2) = -2.00580112952328;
		H.at<double>(2, 0) = 1.211007082951835e-05;
		H.at<double>(2, 1) = -4.060950934818833e-06;
		H.at<double>(2, 2) = 1.00000812190187;
	}
	// Dundas Street
	else if (option == 2) {
		H.at<double>(0, 0) = 1.007576066480183;
		H.at<double>(0, 1) = 0.00351684490917723;
		H.at<double>(0, 2) = -2.018668977868777;
		H.at<double>(1, 0) = -0.002359637216448551;
		H.at<double>(1, 1) = 1.009924728639805;
		H.at<double>(1, 2) = -1.005205454206651;
		H.at<double>(2, 0) = -4.096592389670027e-06;
		H.at<double>(2, 1) = 8.140844697163744e-06;
		H.at<double>(2, 2) = 1.000000052340082;
	}
	// Mural
	else if (option == 3) {
		H.at<double>(0, 0) = 0.9999999999999933;
		H.at<double>(0, 1) = -0.003484320557494391;
		H.at<double>(0, 2) = 0.006968641115726214;
		H.at<double>(1, 0) = -6.69588841507594e-16;
		H.at<double>(1, 1) = 0.996515679442499;
		H.at<double>(1, 2) = -1.99303135888475;
		H.at<double>(2, 0) = -3.815801005221341e-18;
		H.at<double>(2, 1) = -1.209833526908025e-05;
		H.at<double>(2, 2) = 1.000024196670538;
	}
	

	// Create translation matrix.
	cv::Mat trans(3, 3, CV_64F);
	trans.at<double>(0, 0) = 1;
	trans.at<double>(0, 1) = 0;
	trans.at<double>(0, 2) = -400;
	trans.at<double>(1, 0) = 0;
	trans.at<double>(1, 1) = 1;
	trans.at<double>(1, 2) = 0;
	trans.at<double>(2, 0) = 0;
	trans.at<double>(2, 1) = 0;
	trans.at<double>(2, 2) = 1;


	
	return H * trans;
}

/**
 * Don't make the step larger than the size of the first image.
 * These were the inputs used to generate the homographies.
 * double percent = 0.48;
 * cv::Mat trueH = generateHomography((int) image1.rows * percent, (int) image1.cols * percent, 3);
 */
cv::Mat generateHomography(int rows, int cols, int step) {
	srand(time(0));
	std::vector<cv::Point2f> points, warpedPoints;
	points.push_back(cv::Point2f(0, 0));
	points.push_back(cv::Point2f(0, cols));
	points.push_back(cv::Point2f(rows, 0));
	points.push_back(cv::Point2f(rows, cols));
	warpedPoints.push_back(cv::Point2f(0 + (rand() % step), 0 + (rand() % step)));
	warpedPoints.push_back(cv::Point2f(0 + (rand() % step), cols - (rand() % step)));
	warpedPoints.push_back(cv::Point2f(rows - (rand() % step) , 0 + (rand() % step)));
	warpedPoints.push_back(cv::Point2f(rows - (rand() % step), cols - (rand() % step)));

	cv::Mat H = cv::getPerspectiveTransform(points, warpedPoints, cv::INTER_LINEAR);

	return H.inv();
}

cv::Mat directLinearTransform(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints) {
	cv::Mat A(2*sourcePoints.size(), 9, CV_64F);
	cv::Mat h(9, 1, CV_64F);
	cv::Mat h3(3, 3, CV_64F);

	// fill in A.
	for (int n = 0; n < sourcePoints.size(); n++) {
		// first row.
		A.at<double>(n*2, 0) = 0;
		A.at<double>(n*2, 1) = 0;
		A.at<double>(n*2, 2) = 0;
		A.at<double>(n*2, 3) = sourcePoints[n].x; // u
		A.at<double>(n*2, 4) = sourcePoints[n].y; // v
		A.at<double>(n*2, 5) = 1;
		A.at<double>(n*2, 6) = -sourcePoints[n].x * destPoints[n].y; // -u * v prime
		A.at<double>(n*2, 7) = -sourcePoints[n].y * destPoints[n].y; // -v * v prime
		A.at<double>(n*2, 8) = -destPoints[n].y; // -v prime
		// second row.
		A.at<double>(n*2+1, 0) = -sourcePoints[n].x; //- u
		A.at<double>(n*2+1, 1) = -sourcePoints[n].y; // - v
		A.at<double>(n*2+1, 2) = -1;
		A.at<double>(n*2+1, 3) = 0;
		A.at<double>(n*2+1, 4) = 0;
		A.at<double>(n*2+1, 5) = 0;
		A.at<double>(n*2+1, 6) = sourcePoints[n].x * destPoints[n].x; // u * u prime
		A.at<double>(n*2+1, 7) = sourcePoints[n].y * destPoints[n].x; // v * u prime
		A.at<double>(n*2+1, 8) = destPoints[n].x; // u prime
	}

	cv::SVD::solveZ(A, h);
	// reshape
	for (int u = 0; u < h3.rows; u++) {
		for(int v = 0; v < h3.cols; v++) {
			h3.at<double>(u, v) = h.at<double>((u*3) + v);
		}
	}
	return h3;
}

cv::Mat getNormaliseMatrix(std::vector<cv::Point2f> inputPoints) {
	// Average Point.
	cv::Point2f average;

	// Calcluate average points.
	for (int i = 0; i < inputPoints.size(); i++) {
		average += inputPoints[i];
	}
	average.x /= inputPoints.size();
	average.y /= inputPoints.size();

	// Create translation matrix.
	cv::Mat trans(3, 3, CV_64F);
	trans.at<double>(0, 0) = 1;
	trans.at<double>(0, 1) = 0;
	trans.at<double>(0, 2) = -average.x;
	trans.at<double>(1, 0) = 0;
	trans.at<double>(1, 1) = 1;
	trans.at<double>(1, 2) = -average.y;
	trans.at<double>(2, 0) = 0;
	trans.at<double>(2, 1) = 0;
	trans.at<double>(2, 2) = 1;

	// Calcluate scale.
	double scale, length;
	for (int i = 0; i < inputPoints.size(); i++) {
		length += cv::norm(inputPoints[i] - average);
	}
	length /= inputPoints.size();
	scale = std::sqrt(2) / length;

	// Create scale matrix.
	cv::Mat scaleMatrix(3, 3, CV_64F);
	scaleMatrix.at<double>(0, 0) = scale;
	scaleMatrix.at<double>(0, 1) = 0;
	scaleMatrix.at<double>(0, 2) = 0;
	scaleMatrix.at<double>(1, 0) = 0;
	scaleMatrix.at<double>(1, 1) = scale;
	scaleMatrix.at<double>(1, 2) = 0;
	scaleMatrix.at<double>(2, 0) = 0;
	scaleMatrix.at<double>(2, 1) = 0;
	scaleMatrix.at<double>(2, 2) = 1;

	//Calculate the transform matrix.
	return scaleMatrix * trans;
}

double MSE(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints, cv::Mat estH, cv::Mat trueH) {
	std::vector<cv::Point2f> nrmlSourcePoints, nrmlDestPoints, estimatedDest, estimatedSource, trueDest, trueSource;
	cv::Mat scaledEstH, ScaledTrueH;

	// Scale the matrices to align the last element with 1. 
	scaledEstH = estH / estH.at<double>(2,2);
	ScaledTrueH = trueH / trueH.at<double>(2,2);

	cv::perspectiveTransform(sourcePoints, estimatedDest, scaledEstH);
	cv::perspectiveTransform(sourcePoints, trueDest, ScaledTrueH.inv());
	cv::perspectiveTransform(destPoints, estimatedSource, scaledEstH.inv());
	cv::perspectiveTransform(destPoints, trueSource, ScaledTrueH);

	double distance = 0.0;
	for (int point = 0; point < sourcePoints.size(); point++) {
			distance += pow(cv::norm(trueDest[point] - estimatedDest[point]), 2) + pow(cv::norm(trueSource[point] - estimatedSource[point]), 2);
	}
	return distance;
}

cv::Mat normalisedDLT(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints) {
		//Calculate the transform matrix.
		cv::Mat T = getNormaliseMatrix(sourcePoints);
		cv::Mat Tprime = getNormaliseMatrix(destPoints);
		
		// Normalise the points.
		std::vector<cv::Point2f> nrmlSourcePoints, nrmlDestPoints;
		cv::perspectiveTransform(sourcePoints, nrmlSourcePoints, T);
		cv::perspectiveTransform(destPoints, nrmlDestPoints, Tprime);

		// Normalised answer.
		cv::Mat HTilde = directLinearTransform(nrmlSourcePoints, nrmlDestPoints);
		return Tprime.inv() * HTilde * T;
}

double getIterations(int numInliers, int numPoints, double p) {
	double phi = (double)numInliers / (double)numPoints;
	return std::log(p) / std::log(1.0 - pow(phi, 4));
}

cv::Mat RANSAC(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints, double threshold) {
	srand(time(0));
	double p = 0.001; //Choose 0.001
	// Get the starting iteration value.
	double maxIter = getIterations(4, sourcePoints.size(), p);

	std::vector<cv::Point2f> bestSourceInliers, bestDestInliers;
	for (int i = 0; i < maxIter; i++) {
		std::vector<cv::Point2f> randomSourceVec, randomDestVec, inlierSourceVec, inlierDestVec;

		for (int j = 0; j < 4; j++) { //Pick 4 random points.
			int rNum = rand() % sourcePoints.size();
			randomSourceVec.push_back(sourcePoints[rNum]);
			randomDestVec.push_back(sourcePoints[rNum]);
		}

		//Create an estimate homography.
		cv::Mat estH = directLinearTransform(randomSourceVec, randomDestVec);
		std::vector<cv::Point2f> est1;
		cv::perspectiveTransform(sourcePoints, est1, estH);
		for (int point = 0; point < sourcePoints.size(); point++) { // See which points agree with the homography.
			double distance = cv::norm(destPoints[point] - est1[point]);
			if (distance < threshold) {
				inlierSourceVec.push_back(sourcePoints[point]);
				inlierDestVec.push_back(destPoints[point]);
			}
		}
		if (inlierSourceVec.size() > bestSourceInliers.size()) { // Update the consensus.
			bestSourceInliers = inlierSourceVec;
			bestDestInliers = inlierDestVec;
			maxIter = getIterations(bestSourceInliers.size(), sourcePoints.size(), p); // Update the number of trials.
		}
	}
	std::cout << "number of points: " << sourcePoints.size() << std::endl;
	std::cout << "Inliers: " << bestSourceInliers.size() << std::endl;
	std::cout << "Iterations: " << maxIter << std::endl;
	// Return a DLT on the consenus set.
	return directLinearTransform(bestSourceInliers, bestDestInliers);
}

cv::Mat normalisedRANSAC(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints, double threshold) {
	//Calculate the transform matrix.
	cv::Mat T = getNormaliseMatrix(sourcePoints);
	cv::Mat Tprime = getNormaliseMatrix(destPoints);

	// Normalise the points.
	std::vector<cv::Point2f> nrmlSourcePoints, nrmlDestPoints;
	cv::perspectiveTransform(sourcePoints, nrmlSourcePoints, T);
	cv::perspectiveTransform(destPoints, nrmlDestPoints, Tprime);

	cv::Mat HTilde = RANSAC(nrmlSourcePoints, nrmlDestPoints, threshold);
	// Return the homography.
	return Tprime.inv() * HTilde * T;
}

int main() {
	// Read in two images and display them on screen
	/** 
	 * Image choice 1: Image of a fridge.
	 * Image choice 2: Image of Dundas Street.
	 * Image choice 3: Image of a mural.
	 */
	int image_choice = 2;
	cv:: Mat trueH = getHomography(image_choice);
	cv::Mat image1 = getTestImage1(image_choice);
	cv::Mat image2(image1.rows, image1.cols, CV_8UC3);
	cv::warpPerspective(image1, image2, trueH, image1.size());

	cv::namedWindow("Display 1");
	cv::namedWindow("Display 2");

	cv::imshow("Display 1", image1);
	cv::imshow("Display 2", image2);
	cv::waitKey(10);

	// Detect SIFT features in the two images
	
	// cv::Ptr<cv::FeatureDetector> detector = cv::features2d::SIFT::create();
	cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	detector->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

	// Match the features using Brute-Force method

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();

	std::vector<std::vector<cv::DMatch>> matches;

	matcher->knnMatch(descriptors1, descriptors2, matches, 2);

	// Filter the matches to remove ambiguous/unreliable ones

	std::vector<cv::DMatch> goodMatches;
	std::vector<cv::Point2f> goodPoints1, goodPoints2;
	for (const auto& match : matches) {
		if (match[0].distance < 0.8 * match[1].distance) {
			goodMatches.push_back(match[0]);
			goodPoints1.push_back(keypoints1[match[0].queryIdx].pt);
			goodPoints2.push_back(keypoints2[match[0].trainIdx].pt);
		}
	}

	// Compute the homgraphy
	Timer timer;
	std::vector<unsigned char> inliers;
	timer.reset();
	// cv::Mat H = cv::findHomography(goodPoints2, goodPoints1, cv::RANSAC, 3.0);
	// cv::Mat H = directLinearTransform(goodPoints2, goodPoints1);
	// cv::Mat H = normalisedDLT(goodPoints2, goodPoints1);
	// cv::Mat H = RANSAC(goodPoints2, goodPoints1, 80);
	cv::Mat H = normalisedRANSAC(goodPoints2, goodPoints1, 0.2);

	double elapsedTime = timer.read();
	std::cout << "MSE: " << MSE(goodPoints2, goodPoints1, H, trueH) << std::endl;
	std::cout << "Homography estimation took " << elapsedTime << " seconds" << std::endl;
	// Figure out the extent of the final mosaic

	std::vector<cv::Point2d> corners;
	corners.push_back(cv::Point2d(0, 0));
	corners.push_back(cv::Point2d(image2.size().width, 0));
	corners.push_back(cv::Point2d(image2.size().width, image2.size().height));
	corners.push_back(cv::Point2d(0, image2.size().height));
	cv::perspectiveTransform(corners, corners, H);
	double minX = 0;
	double maxX = image1.size().width;
	double minY = 0;
	double maxY = image1.size().height;
	for (const auto& corner : corners) {
		if (corner.x < minX) minX = corner.x;
		if (corner.x > maxX) maxX = corner.x;
		if (corner.y < minY) minY = corner.y;
		if (corner.y > maxY) maxY = corner.y;
	}

	// Form a transform to put the images within the computed extent

	cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
	T.at<double>(0, 2) = -minX;
	T.at<double>(1, 2) = -minY;

	// Warp the images and display the final mosaic

	cv::Mat mosaic(cv::Size(int(maxX - minX), int(maxY - minY)), CV_8UC3);
	mosaic.setTo(cv::Scalar(0, 0, 0));
	cv::warpPerspective(image1, mosaic, T, mosaic.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
	cv::warpPerspective(image2, mosaic, T * H, mosaic.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

	cv::namedWindow("Mosaic");
	cv::imshow("Mosaic", mosaic);
	cv::waitKey();
	cv::destroyAllWindows();

	return 0;
}