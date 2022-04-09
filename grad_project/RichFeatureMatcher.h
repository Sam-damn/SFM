#pragma once

#include "IFeatureMatcher.h"

class RichFeatureMatcher : public IFeatureMatcher {
private:
	cv::Ptr<cv::FeatureDetector> detector;

	std::vector<cv::Mat> descriptors;
	std::vector<cv::Mat>& imgs;

	std::vector<std::vector<cv::KeyPoint> >& imgpts;
public:

	RichFeatureMatcher(std::vector<cv::Mat>& imgs, std::vector<std::vector<cv::KeyPoint> >& imgpts);

	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches = NULL);

	std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};
