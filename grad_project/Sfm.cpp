#include "Sfm.h"

#include "CameraFunctions.h"
#include "RichFeatureMatcher.h"
#include "OFFeatureMatcher.h"
#include "Triangulation.h"


Sfm::Sfm(const cv::Mat & left_im_, const cv::Mat & right_im_) : features_matched(false)
{


	left_im_.copyTo(left_im);
	right_im_.copyTo(right_im);
	left_im.copyTo(left_im_orig);
	cvtColor(left_im_orig, left_im, cv::COLOR_BGR2GRAY);
	right_im.copyTo(right_im_orig);
	cvtColor(right_im_orig, right_im, cv::COLOR_BGR2GRAY);

	P = cv::Matx34d(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);
	P1 = cv::Matx34d(1, 0, 0, 50,
		0, 1, 0, 0,
		0, 0, 1, 0);


	intrinsics(K , distortion_coeff , 0);

	cv::invert(K, Kinv);



}

void Sfm::OnlyMatchFeatures(std::vector<cv::DMatch> &Matches)
{
	imgpts1.clear(); imgpts2.clear(); fullpts1.clear(); fullpts2.clear();

	std::vector<cv::Mat> imgs; imgs.push_back(left_im); imgs.push_back(right_im);
	std::vector<std::vector<cv::KeyPoint> > imgpts; imgpts.push_back(imgpts1); imgpts.push_back(imgpts2);

/*
	RichFeatureMatcher rfm(imgs, imgpts);
	rfm.MatchFeatures(0, 1 , &Matches);

	imgpts1 = rfm.GetImagePoints(0);
	imgpts2 = rfm.GetImagePoints(1);*/





	OFFeatureMatcher offm(imgs, imgpts);
	offm.MatchFeatures(0, 1, &Matches);

	imgpts1 = offm.GetImagePoints(0);
	imgpts2 = offm.GetImagePoints(1);







	features_matched = true;


}

void Sfm::reconstruct_two_views()
{
	std::vector<cv::DMatch> matches;
	if (!features_matched)
		OnlyMatchFeatures(matches);


	std::vector<CloudPoint> emptypp;
	FindCameraMatrices(K, Kinv, distortion_coeff, imgpts1, imgpts2, imgpts1_good, imgpts2_good, P, P1, matches, emptypp);


	std::vector<cv::KeyPoint> pt_set1, pt_set2;
	GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, pt_set1, pt_set2);





	double error = TriangulatePoints(pt_set1, pt_set2, K, Kinv, distortion_coeff, P, P1, pointcloud, correspImg1Pt);


	std::cout << "reprojection error is :" << error<< std::endl;



}
