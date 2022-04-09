#pragma once


#include <opencv2/opencv.hpp>
#include <vector>


#include "Common.h"


class Sfm {

private:

	std::vector<cv::KeyPoint> imgpts1,
		imgpts2,
		fullpts1,
		fullpts2,
		imgpts1_good,
		imgpts2_good;


	cv::Mat descriptors_1;
	cv::Mat descriptors_2;


	cv::Mat left_im,
		left_im_orig,
		right_im,
		right_im_orig;


	cv::Matx34d P, P1;
	cv::Mat K;
	cv::Mat_<double> Kinv;


	cv::Mat cam_matrix, distortion_coeff;

	std::vector<CloudPoint> pointcloud;
	std::vector<cv::KeyPoint> correspImg1Pt;

	bool features_matched;

public:


	Sfm(const cv::Mat& left_im_, const cv::Mat& right_im_);

	void OnlyMatchFeatures(std::vector<cv::DMatch> &Matches);

	void reconstruct_two_views();

	std::vector<cv::Point3d> getPointCloud() { return CloudPointsToPoints(pointcloud); }


	const cv::Mat& getleft_im_orig() { return left_im_orig; }
	const cv::Mat& getright_im_orig() { return right_im_orig; }
	const std::vector<cv::KeyPoint>& getcorrespImg1Pt() { return correspImg1Pt; }
	const std::vector<cv::Vec3b>& getPointCloudRGB() { return std::vector<cv::Vec3b>(); }






};
