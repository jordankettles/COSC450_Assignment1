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

cv::Mat directLinearTransform(
	std::vector<cv::Point2f> sourcePoints, 
	std::vector<cv::Point2f> destPoints,
	std::vector<unsigned char> inliers) {
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

cv::Mat normalisedDLT(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints, std::vector<unsigned char> inliers) {
		//Average Points.
		cv::Point2f avSource, avDest;
		
		// Calcluate average points.
		for (int i = 0; i < sourcePoints.size(); i++) {
			avSource += sourcePoints[i];
			avDest += destPoints[i];
		}
		avSource.x /= sourcePoints.size();
		avSource.y /= sourcePoints.size();
		avDest.x /= destPoints.size();
		avDest.y /= destPoints.size();
		
		// create translation matrices.
		cv::Mat trans(3, 3, CV_64F);
		cv::Mat transPrime(3, 3, CV_64F);
		trans.at<double>(0, 0) = 1;
		trans.at<double>(0, 1) = 0;
		trans.at<double>(0, 2) = -avSource.x;
		trans.at<double>(1, 0) = 0;
		trans.at<double>(1, 1) = 1;
		trans.at<double>(1, 2) = -avSource.y;
		trans.at<double>(2, 0) = 0;
		trans.at<double>(2, 1) = 0;
		trans.at<double>(2, 2) = 1;

		transPrime.at<double>(0, 0) = 1;
		transPrime.at<double>(0, 1) = 0;
		transPrime.at<double>(0, 2) = -avDest.x;
		transPrime.at<double>(1, 0) = 0;
		transPrime.at<double>(1, 1) = 1;
		transPrime.at<double>(1, 2) = -avDest.y;
		transPrime.at<double>(2, 0) = 0;
		transPrime.at<double>(2, 1) = 0;
		transPrime.at<double>(2, 2) = 1;

		// Calcluate scales.
		double scale, scalePrime, length, lengthPrime;
		for (int i = 0; i < sourcePoints.size(); i++) {
			length += cv::norm(sourcePoints[i] - avSource);
			lengthPrime += cv::norm(destPoints[i] - avDest);
		}
		length /= sourcePoints.size();
		lengthPrime /= destPoints.size();
		scale = std::sqrt(2) / length;
		scalePrime = std::sqrt(2) / lengthPrime;

		// Create scale matrices.
		cv::Mat scaleMatrix(3, 3, CV_64F);
		cv::Mat scalePrimeMatrix(3, 3, CV_64F);
		scaleMatrix.at<double>(0, 0) = scale;
		scaleMatrix.at<double>(0, 1) = 0;
		scaleMatrix.at<double>(0, 2) = 0;
		scaleMatrix.at<double>(1, 0) = 0;
		scaleMatrix.at<double>(1, 1) = scale;
		scaleMatrix.at<double>(1, 2) = 0;
		scaleMatrix.at<double>(2, 0) = 0;
		scaleMatrix.at<double>(2, 1) = 0;
		scaleMatrix.at<double>(2, 2) = 1;

		scalePrimeMatrix.at<double>(0, 0) = scalePrime;
		scalePrimeMatrix.at<double>(0, 1) = 0;
		scalePrimeMatrix.at<double>(0, 2) = 0;
		scalePrimeMatrix.at<double>(1, 0) = 0;
		scalePrimeMatrix.at<double>(1, 1) = scalePrime;
		scalePrimeMatrix.at<double>(1, 2) = 0;
		scalePrimeMatrix.at<double>(2, 0) = 0;
		scalePrimeMatrix.at<double>(2, 1) = 0;
		scalePrimeMatrix.at<double>(2, 2) = 1;

		//Calculate the transform matrices.
		cv::Mat T(3, 3, CV_64F);
		cv::Mat Tprime(3, 3, CV_64F);
		T = scaleMatrix * trans;
		Tprime = scalePrimeMatrix * transPrime;
		//Normalise the points
		std::vector<cv::Point2f> nrmlSourcePoints, nrmlDestPoints;
		cv::perspectiveTransform(sourcePoints, nrmlSourcePoints, T);
		cv::perspectiveTransform(destPoints, nrmlDestPoints, Tprime);

		// Normalised answer.
		cv::Mat HTilde = directLinearTransform(nrmlSourcePoints, nrmlDestPoints, inliers);
		return Tprime.inv() * HTilde * T;
}

void RANSAC(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> destPoints,	std::vector<unsigned char> inliers) {
		std::cout << "hello" << std::endl;
}

int main() {

	int testOption = 1;

	// Read in two images and display them on screen
	
	// Example Image.
	// cv::Mat image1 = cv::imread("../image1.jpg");
	// cv::Mat image2 = cv::imread("../image2.jpg");

	// Abstract Image
	// MIT License
	
	cv::Mat image1 = cv::imread("../image1.jpg");
	cv::Mat trueH = generateHomography(image1.rows, image1.cols, 250);
	cv::Mat image2 = imageGenerator(image1, trueH);


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
	// cv::perspectiveTransform(source, destination, trueH);

	std::cout << "trueH = " << std::endl;

	// Compute the homgraphy

	/**********************************************************
	 * This is the bit you need to replace for the assignment *
	 **********************************************************/
	Timer timer;
	std::vector<unsigned char> inliers;
	timer.reset();
	// cv::Mat H = cv::findHomography(goodPoints2, goodPoints1, inliers, cv::RANSAC, 3.0);


	// cv::Mat H = directLinearTransform(goodPoints2, goodPoints1, inliers); //image calculation
	// cv::Mat H = directLinearTransform(source, destination, inliers); // My testing calculation
	cv::Mat H = normalisedDLT(goodPoints2, goodPoints1, inliers);

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