#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Timer.h"

int main() {

	// Read in two images and display them on screen
	
	cv::Mat image1 = cv::imread("../image1.jpg");
	cv::Mat image2 = cv::imread("../image2.jpg");

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
	cv::Mat H = cv::findHomography(goodPoints2, goodPoints1, inliers, cv::RANSAC, 3.0);
	double elapsedTime = timer.read();

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