#include "RichFeatureMatcher.h"
#include "CameraFunctions.h"


#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <set>

using namespace std;
using namespace cv;




RichFeatureMatcher::RichFeatureMatcher(std::vector<cv::Mat>& imgs_,
	std::vector<std::vector<cv::KeyPoint> >& imgpts_) :
	imgpts(imgpts_), imgs(imgs_)
{
	detector = ORB::create();


	detector->detect(imgs, imgpts);
	detector->compute(imgs, imgpts, descriptors);

}

void RichFeatureMatcher::MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches)
{
	const vector<KeyPoint>& imgpts1 = imgpts[idx_i];
	const vector<KeyPoint>& imgpts2 = imgpts[idx_j];

	const Mat& descriptors_1 = descriptors[idx_i];
	const Mat& descriptors_2 = descriptors[idx_j];

	std::vector<DMatch> good_matches_, very_good_matches_;
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	keypoints_1 = imgpts1;
	keypoints_2 = imgpts2;

	//error handling
	if (descriptors_1.empty()) {
		CV_Error(0, "descriptors_1 is empty");
	}
	if (descriptors_2.empty()) {
		CV_Error(0, "descriptors_2 is empty");
	}
	

	BFMatcher matcher(NORM_HAMMING, true);
	std::vector<DMatch> matches_;

	if (matches == NULL) {
		matches = &matches_;
	}
	if (matches->size() == 0) {
		matcher.match(descriptors_1, descriptors_2, *matches);
	}


	vector<KeyPoint> imgpts1_good, imgpts2_good;


	std::set<int> existing_trainIdx;
	for (unsigned int i = 0; i < matches->size(); i++) {

		if ((*matches)[i].trainIdx <= 0) {
			(*matches)[i].trainIdx = (*matches)[i].imgIdx;
		}

		if (existing_trainIdx.find((*matches)[i].trainIdx) == existing_trainIdx.end() &&
			(*matches)[i].trainIdx >= 0 && (*matches)[i].trainIdx < (int)(keypoints_2.size()) )
		{
			good_matches_.push_back((*matches)[i]);
			imgpts1_good.push_back(keypoints_1[(*matches)[i].queryIdx]);
			imgpts2_good.push_back(keypoints_2[(*matches)[i].trainIdx]);
			existing_trainIdx.insert((*matches)[i].trainIdx);
		}

	}

	cout << "keypoints_1.size() " << keypoints_1.size() << " imgpts1_good.size() " << imgpts1_good.size() << endl;
	cout << "keypoints_2.size() " << keypoints_2.size() << " imgpts2_good.size() " << imgpts2_good.size() << endl;

	vector<KeyPoint> imgpts2_very_good, imgpts1_very_good;

	GetFundamentalMat(keypoints_1, keypoints_2, imgpts1_very_good, imgpts2_very_good, good_matches_);

	*matches = good_matches_;


	Mat img_matches;
	drawMatches(imgs[idx_i], keypoints_1, imgs[idx_j], keypoints_2,
		good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	stringstream ss; ss << "Feature Matches " << idx_i << "-" << idx_j;
	imshow(ss.str(), img_matches);
	

	waitKey(-1);
	destroyWindow(ss.str());


}
