#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>


#include <vector>

#include "Common.h"


#define EPSILON 0.0001


cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,		
	cv::Matx34d P,		
	cv::Point3d u1,		
	cv::Matx34d P1		
);


cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	
	cv::Matx34d P,			
	cv::Point3d u1,			
	cv::Matx34d P1			
);


double TriangulatePoints(const std::vector<cv::KeyPoint>& pt_set1,
	const std::vector<cv::KeyPoint>& pt_set2,
	const cv::Mat& K,
	const cv::Mat& Kinv,
	const cv::Mat& distcoeff,
	const cv::Matx34d& P,
	const cv::Matx34d& P1,
	std::vector<CloudPoint>& pointcloud,
	std::vector<cv::KeyPoint>& correspImg1Pt);