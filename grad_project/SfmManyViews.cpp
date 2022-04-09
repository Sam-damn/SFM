#include "SfmManyViews.h"


#include "OFFeatureMatcher.h"
#include "RichFeatureMatcher.h"


SfmManyViews::SfmManyViews(const std::vector<cv::Mat>& imgs_, const std::vector<std::string>& imgs_names_):imgs_names(imgs_names_), features_matched(false)
{


	std::cout << "=========================== Load Images ===========================\n";
	//ensure images are CV_8UC3
	for (unsigned int i = 0; i < imgs_.size(); i++) {
		imgs_orig.push_back(cv::Mat_<cv::Vec3b>());
		if (!imgs_[i].empty()) {
			if (imgs_[i].type() == CV_8UC1) {
				cvtColor(imgs_[i], imgs_orig[i], cv::COLOR_BGR2GRAY);
			}
			else if (imgs_[i].type() == CV_32FC3 || imgs_[i].type() == CV_64FC3) {
				imgs_[i].convertTo(imgs_orig[i], CV_8UC3, 255.0);
			}
			else {
				imgs_[i].copyTo(imgs_orig[i]);
			}
		}

		imgs.push_back(cv::Mat());
		cvtColor(imgs_orig[i], imgs[i], cv::COLOR_BGR2GRAY);

		imgpts.push_back(std::vector<cv::KeyPoint>());
		imgpts_good.push_back(std::vector<cv::KeyPoint>());
		std::cout << ".";
	}
	std::cout << std::endl;



	intrinsics(K, distortion_coeff , 0);

	cv::invert(K, Kinv);


	distortion_coeff.convertTo(distcoeff_32f, CV_64F);
	K.convertTo(K_32f, CV_64F);



}


void SfmManyViews::OnlyMatchFeatures()
{

	if (features_matched) return;


	feature_matcher = new OFFeatureMatcher( imgs, imgpts);

	//feature_matcher = new  RichFeatureMatcher(imgs, imgpts);


	int loop1_top = imgs.size() - 1, loop2_top = imgs.size();
	int frame_num_i = 0;


#pragma omp parallel for
	for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
		for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
		{
			std::cout << "------------ Match " << imgs_names[frame_num_i] << "," << imgs_names[frame_num_j] << " ------------\n";
			std::vector<cv::DMatch> matches_tmp;
			feature_matcher->MatchFeatures(frame_num_i, frame_num_j, &matches_tmp);
			matches_matrix[std::make_pair(frame_num_i, frame_num_j)] = matches_tmp;

			std::vector<cv::DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
			matches_matrix[std::make_pair(frame_num_j, frame_num_i)] = matches_tmp_flip;
		}
	}

	features_matched = true;


}


void SfmManyViews::GetRGBForPointCloud(const std::vector<struct CloudPoint>& _pcloud, std::vector<cv::Vec3b>& RGBforCloud)
{

	RGBforCloud.resize(_pcloud.size());
	for (unsigned int i = 0; i < _pcloud.size(); i++) {
		unsigned int good_view = 0;
		std::vector<cv::Vec3b> point_colors;
		for (; good_view < imgs_orig.size(); good_view++) {
			if (_pcloud[i].imgpt_for_img[good_view] != -1) {
				int pt_idx = _pcloud[i].imgpt_for_img[good_view];
				if (pt_idx >= imgpts[good_view].size()) {
					std::cerr << "BUG: point id:" << pt_idx << " should not exist for img #" << good_view << " which has only " << imgpts[good_view].size() << std::endl;
					continue;
				}
				cv::Point _pt = imgpts[good_view][pt_idx].pt;
				assert(good_view < imgs_orig.size() && _pt.x < imgs_orig[good_view].cols && _pt.y < imgs_orig[good_view].rows);

				point_colors.push_back(imgs_orig[good_view].at<cv::Vec3b>(_pt));

				
			}
		}
		cv::Scalar res_color = cv::mean(point_colors);
		RGBforCloud[i] = (cv::Vec3b(res_color[0], res_color[1], res_color[2])); //bgr2rgb
		if (good_view == imgs.size()) //nothing found.. put red dot
			RGBforCloud.push_back(cv::Vec3b(255, 0, 0));
	}

}

