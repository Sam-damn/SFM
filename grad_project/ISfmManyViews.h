#pragma once


#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>


class ISfmManyViews {

public :


	virtual void OnlyMatchFeatures() = 0;
	virtual void RecoverDepthFromImages() = 0;
	virtual std::vector<cv::Point3d> getPointCloud() = 0;
	virtual const std::vector<cv::Vec3b>& getPointCloudRGB() = 0;










};
