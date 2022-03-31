#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Timer.h"
#include "cmath"

double RSS(std::vector<cv::Point2f> sourcePoints, cv::Mat estH, cv::Mat trueH) {
	std::vector<cv::Point2f> estPoints, truePoints;
	cv::perspectiveTransform(sourcePoints, estPoints, estH);
	cv::perspectiveTransform(sourcePoints, truePoints, trueH);
	double rss;
	for (int i = 0; i < sourcePoints.size(); i++) {
		rss += pow(cv::norm(estPoints[i] - truePoints[i]), 2);
	}
	return rss;
}

cv::Mat imageGenerator(cv::Mat img1, cv::Mat H) {
	cv::Mat img2(img1.rows, img1.cols, CV_8UC3);
	cv::warpPerspective(img1, img2, H, img2.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
	return img2;
}

cv::Mat getTestImage(int option) {
	cv::Mat returnImage;
	//Image given in example folder.
	if (option == 1) {
		returnImage = cv::imread("../image1.jpg");
	}
	return returnImage;
}

/**
 * Don't make the step larger than the size of the first image.
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

cv::Mat directLinearTransform(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints, std::vector<unsigned char> inliers) {
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

cv::Mat normalisedDLT(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints, std::vector<unsigned char> inliers) {
		//Calculate the transform matrix.
		cv::Mat T = getNormaliseMatrix(sourcePoints);
		cv::Mat Tprime = getNormaliseMatrix(destPoints);
		
		// Normalise the points.
		std::vector<cv::Point2f> nrmlSourcePoints, nrmlDestPoints;
		cv::perspectiveTransform(sourcePoints, nrmlSourcePoints, T);
		cv::perspectiveTransform(destPoints, nrmlDestPoints, Tprime);

		// Normalised answer.
		cv::Mat HTilde = directLinearTransform(nrmlSourcePoints, nrmlDestPoints, inliers);
		return Tprime.inv() * HTilde * T;
}

double getIterations(int numInliers, int numPoints, double p) {
	double phi = (double)numInliers / (double)numPoints;
	std::cout << "number of inliers / number of points: " << numInliers << " / " << numPoints << std::endl;
	std::cout << "number of iterations: " << std::log(p) / std::log(1.0 - pow(phi, 4)) << std::endl;
	return std::log(p) / std::log(1.0 - pow(phi, 4));
}

cv::Mat RANSACwithoutNrml(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints, std::vector<unsigned char> inliers, double threshold) {
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
		cv::Mat estH = directLinearTransform(randomSourceVec, randomDestVec, inliers);
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
	// Return a DLT on the consenus set.
	return directLinearTransform(bestSourceInliers, bestDestInliers, inliers);
}

cv::Mat normalisedRANSAC(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints, std::vector<unsigned char> inliers, double threshold) {
	//Calculate the transform matrix.
	cv::Mat T = getNormaliseMatrix(sourcePoints);
	cv::Mat Tprime = getNormaliseMatrix(destPoints);

	// Normalise the points.
	std::vector<cv::Point2f> nrmlSourcePoints, nrmlDestPoints;
	cv::perspectiveTransform(sourcePoints, nrmlSourcePoints, T);
	cv::perspectiveTransform(destPoints, nrmlDestPoints, Tprime);
	
	srand(time(0));
	double p = 0.001; //Choose 0.001
	// Get the starting iteration value.
	double maxIter = getIterations(4, nrmlSourcePoints.size(), p);

	std::vector<cv::Point2f> bestSourceInliers, bestDestInliers;
	for (int i = 0; i < maxIter; i++) {
		std::vector<cv::Point2f> randomSourceVec, randomDestVec, inlierSourceVec, inlierDestVec;

		for (int j = 0; j < 4; j++) { //Pick 4 random points.
			int rNum = rand() % nrmlSourcePoints.size();
			randomSourceVec.push_back(nrmlSourcePoints[rNum]);
			randomDestVec.push_back(nrmlDestPoints[rNum]);
		}

		//Create an estimate homography.
		cv::Mat estH = directLinearTransform(randomSourceVec, randomDestVec, inliers);
		std::vector<cv::Point2f> est1;
		cv::perspectiveTransform(nrmlSourcePoints, est1, estH);
		for (int point = 0; point < nrmlSourcePoints.size(); point++) { // See which points agree with the homography.
			double distance = cv::norm(nrmlDestPoints[point] - est1[point]);
			if (distance < threshold) {
				inlierSourceVec.push_back(nrmlSourcePoints[point]);
				inlierDestVec.push_back(nrmlDestPoints[point]);
			}
		}
		if (inlierSourceVec.size() > bestSourceInliers.size()) { // Update the consensus.
			bestSourceInliers = inlierSourceVec;
			bestDestInliers = inlierDestVec;
			maxIter = getIterations(bestSourceInliers.size(), nrmlSourcePoints.size(), p); // Update the number of trials.
		}
	}
	// Calculate normalised answer on the consensus set.
	cv::Mat HTilde = directLinearTransform(bestSourceInliers, bestDestInliers, inliers);
	// Return the homography.
	return Tprime.inv() * HTilde * T;
}

int main() {
	// Read in two images and display them on screen
	
	// Example Image.
	cv::Mat image1 = cv::imread("../image1.jpg");
	cv::Mat image2 = cv::imread("../image2.jpg");
	
	// cv::Mat image1 = getTestImage(1);
	cv::Mat trueH = generateHomography(image1.rows, image1.cols, 250);
	// cv::Mat image2 = imageGenerator(image1, trueH);


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

	/**********************************************************
	 * This is the bit you need to replace for the assignment *
	 **********************************************************/
	Timer timer;
	std::vector<unsigned char> inliers;
	timer.reset();
	// cv::Mat H = cv::findHomography(goodPoints2, goodPoints1, inliers, cv::RANSAC, 3.0);
	// cv::Mat H = directLinearTransform(goodPoints2, goodPoints1, inliers);
	// cv::Mat H = normalisedDLT(goodPoints2, goodPoints1, inliers);
	// cv::Mat H = RANSACwithoutNrml(goodPoints2, goodPoints1, inliers, 35); // 30 - 60ish
	cv::Mat H = normalisedRANSAC(goodPoints2, goodPoints1, inliers, 0.01); // 0.01 - 0.001

	double elapsedTime = timer.read();
	std::cout << "Residual Square Error " << RSS(goodPoints2, H, trueH) << std::endl;
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

	return 0;
}