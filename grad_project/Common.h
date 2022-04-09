#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>




struct CloudPoint {
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
};

void open_imgs_dir(char* dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names);

void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);
void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps);

std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches);


std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);


void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,const std::vector<cv::KeyPoint>& imgpts2,const std::vector<cv::DMatch>& matches,std::vector<cv::KeyPoint>& pt_set1,std::vector<cv::KeyPoint>& pt_set2);

bool hasEnding(std::string const &fullString, std::string const &ending);

bool hasEndingLower(std::string const &fullString_, std::string const &_ending);


void drawArrows(cv::Mat& frame, const std::vector<cv::Point2f>& prevPts, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status, const std::vector<float>& verror, const cv::Scalar& _line_color = cv::Scalar(0, 0, 255));

cv::Mat get_intrinsics(char* dir_name);


