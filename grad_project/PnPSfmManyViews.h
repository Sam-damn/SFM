#pragma once


#include "SfmManyViews.h"

#include "Common.h"



class PnPSfmManyViews : public  SfmManyViews {

	std::vector<CloudPoint> pointcloud_beforeBA;
	std::vector<cv::Vec3b> pointCloudRGB_beforeBA;



public:

	PnPSfmManyViews(const std::vector<cv::Mat>& imgs_,
			const std::vector<std::string>& imgs_names_) :
		SfmManyViews(imgs_, imgs_names_) {

	}

	virtual void RecoverDepthFromImages();

	std::vector<cv::Point3d> getPointCloudBeforeBA() { return CloudPointsToPoints(pointcloud_beforeBA); }
	const std::vector<cv::Vec3b>& getPointCloudRGBBeforeBA() { return pointCloudRGB_beforeBA; }




private:

	void PruneMatchesBasedOnF();
	void GetBaseLineTriangulation();
	void AdjustCurrentBundle();

	bool TriangulatePointsBetweenViews(
		int working_view,
		int second_view,
		std::vector<struct CloudPoint>& new_triangulated,
		std::vector<int>& add_to_cloud
	);

	int FindHomographyInliers2Views(int vi, int vj);


	void Find2D3DCorrespondences(int working_view,
		std::vector<cv::Point3f>& ppcloud,
		std::vector<cv::Point2f>& imgPoints);
	bool FindPoseEstimation(
		int working_view,
		cv::Mat_<double>& rvec,
		cv::Mat_<double>& t,
		cv::Mat_<double>& R,
		std::vector<cv::Point3f> ppcloud,
		std::vector<cv::Point2f> imgPoints);

	int m_first_view;
	int m_second_view; //baseline's second view other to 0

	std::set<int> done_views;
	std::set<int> good_views;















};