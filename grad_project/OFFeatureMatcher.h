#pragma once


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "IFeatureMatcher.h"



class OFFeatureMatcher : public IFeatureMatcher {
	std::vector<cv::Mat>& imgs;
	std::vector<std::vector<cv::KeyPoint> >& imgpts;


public:
	OFFeatureMatcher(
		std::vector<cv::Mat>& imgs_,
		std::vector<std::vector<cv::KeyPoint> >& imgpts_);

	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches);

	std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }

};