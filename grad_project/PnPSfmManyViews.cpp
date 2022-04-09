#include "PnPSfmManyViews.h"
using namespace std;

#include "Triangulation.h"
#include "BundleAdjuster.h"



bool sort_by_first(pair<int, pair<int, int> > a, pair<int, pair<int, int> > b) { return a.first < b.first; }


void PnPSfmManyViews::RecoverDepthFromImages()
{


	if (!features_matched)
		OnlyMatchFeatures();

	std::cout << "======================================================================\n";
	std::cout << "======================== Depth Recovery Start ========================\n";
	std::cout << "======================================================================\n";

	PruneMatchesBasedOnF();
	GetBaseLineTriangulation();
	AdjustCurrentBundle();


	cv::Matx34d P1 = Pmats[m_second_view];
	cv::Mat_<double> t = (cv::Mat_<double>(1, 3) << P1(0, 3), P1(1, 3), P1(2, 3));
	cv::Mat_<double> R = (cv::Mat_<double>(3, 3) << P1(0, 0), P1(0, 1), P1(0, 2),
		P1(1, 0), P1(1, 1), P1(1, 2),
		P1(2, 0), P1(2, 1), P1(2, 2));

	cv::Mat_<double> rvec(1, 3); 
	Rodrigues(R, rvec);

	done_views.insert(m_first_view);
	done_views.insert(m_second_view);
	good_views.insert(m_first_view);
	good_views.insert(m_second_view);



	for (int i = m_second_view + 1; i < imgs.size(); i++) {


		vector<cv::Point3f> points3D; vector<cv::Point2f>  points2D;

		Find2D3DCorrespondences(i, points3D, points2D);

		bool pose_estimated = FindPoseEstimation(i, rvec, t, R, points3D, points2D);

		if (!pose_estimated)
			continue;


		Pmats[i] = cv::Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
			R(1, 0), R(1, 1), R(1, 2), t(1),
			R(2, 0), R(2, 1), R(2, 2), t(2));


	for (set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view){
		int view = *done_view;
		if (view == i) continue; 

		cout << " -> " << imgs_names[view] << endl;

		vector<CloudPoint> new_triangulated;
		vector<int> add_to_cloud;
		bool good_triangulation = TriangulatePointsBetweenViews(i, view, new_triangulated, add_to_cloud);
		if (!good_triangulation) continue;

		std::cout << "before triangulation: " << pcloud.size();
		for (int j = 0; j < add_to_cloud.size(); j++) {
			if (add_to_cloud[j] == 1)
				pcloud.push_back(new_triangulated[j]);
		}
		std::cout << " after " << pcloud.size() << std::endl;

		}


		good_views.insert(i);
		done_views.insert(i);
		

		AdjustCurrentBundle();



	}


	cout << "======================================================================\n";
	cout << "========================= Depth Recovery DONE ========================\n";
	cout << "======================================================================\n";

	

}

void PnPSfmManyViews::PruneMatchesBasedOnF()
{
	for (int _i = 0; _i < imgs.size() - 1; _i++)
	{
		for (unsigned int _j = _i + 1; _j < imgs.size(); _j++) {
			int older_view = _i, working_view = _j;

			GetFundamentalMat(imgpts[older_view],
				imgpts[working_view],
				imgpts_good[older_view],
				imgpts_good[working_view],
				matches_matrix[std::make_pair(older_view, working_view)]);
			//update flip matches as well
#pragma omp critical
			matches_matrix[std::make_pair(working_view, older_view)] = FlipMatches(matches_matrix[std::make_pair(older_view, working_view)]);
		}
	}


}

void PnPSfmManyViews::GetBaseLineTriangulation()
{




	cv::Matx34d P(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0),
		P1(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0);

	std::vector<CloudPoint> tmp_pcloud;


	bool goodF = false;

	m_first_view = 0;
	m_second_view = 1;


	std::cout << " -------- " << imgs_names[m_first_view] << " and " << imgs_names[m_second_view] << " -------- " << std::endl;




		goodF = FindCameraMatrices(K, Kinv, distortion_coeff,
			imgpts[m_first_view],
			imgpts[m_second_view],
			imgpts_good[m_first_view],
			imgpts_good[m_second_view],
			P,
			P1,
			matches_matrix[std::make_pair(m_first_view, m_second_view)],
			tmp_pcloud);

		//See if the Fundamental Matrix between these two views is good


		if (goodF) {
			vector<CloudPoint> new_triangulated;
			vector<int> add_to_cloud;

			Pmats[m_first_view] = P;
			Pmats[m_second_view] = P1;



			bool good_triangulation = TriangulatePointsBetweenViews(m_second_view, m_first_view, new_triangulated, add_to_cloud);

			if (!good_triangulation || cv::countNonZero(add_to_cloud) < 10) {
				std::cout << "triangulation failed" << std::endl;
				goodF = false;
				Pmats[m_first_view] = cv::Matx34d::zeros();
				Pmats[m_second_view] = cv::Matx34d::zeros();
				//m_second_view++;
			}
			else {
				std::cout << "before triangulation: " << pcloud.size();
				for (unsigned int j = 0; j < add_to_cloud.size(); j++) {
					if (add_to_cloud[j] == 1)
						pcloud.push_back(new_triangulated[j]);
				}
				std::cout << " after " << pcloud.size() << std::endl;
			}



		}


		if (!goodF) {
		cerr << "Cannot find a good pair of images to obtain a baseline triangulation" << endl;
		exit(0);
		}

	cout << "Taking baseline from " << imgs_names[m_first_view] << " and " << imgs_names[m_second_view] << endl;


	std::cout << "=========================== Baseline triangulation ===========================\n";



}

void PnPSfmManyViews::AdjustCurrentBundle()
{

	cout << "======================== Bundle Adjustment ==========================\n";

	pointcloud_beforeBA = pcloud;
	GetRGBForPointCloud(pointcloud_beforeBA, pointCloudRGB_beforeBA);

	cv::Mat _cam_matrix = K;
	BundleAdjuster BA;
	BA.adjustBundle(pcloud, _cam_matrix, imgpts, Pmats);
	K = _cam_matrix;
	Kinv = K.inv();

	cout << "use new K " << endl << K << endl;

	GetRGBForPointCloud(pcloud, pointCloudRGB);


}

bool PnPSfmManyViews::TriangulatePointsBetweenViews(int working_view, int older_view, std::vector<struct CloudPoint>& new_triangulated, std::vector<int>& add_to_cloud)
{

	cout << " Triangulate " << imgs_names[working_view] << " and " << imgs_names[older_view] << endl;
	//get the left camera matrix
	//TODO: potential bug - the P mat for <view> may not exist? or does it...
	cv::Matx34d P = Pmats[older_view];
	cv::Matx34d P1 = Pmats[working_view];

	std::vector<cv::KeyPoint> pt_set1, pt_set2;
	std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(older_view, working_view)];
	GetAlignedPointsFromMatch(imgpts[older_view], imgpts[working_view], matches, pt_set1, pt_set2);


	//adding more triangulated points to general cloud
	double reproj_error = TriangulatePoints(pt_set1, pt_set2, K, Kinv, distortion_coeff, P, P1, new_triangulated, correspImg1Pt);
	std::cout << "triangulation reproj error " << reproj_error << std::endl;

	vector<uchar> trig_status;
	if (!TestTriangulation(new_triangulated, P, trig_status) || !TestTriangulation(new_triangulated, P1, trig_status)) {
		cerr << "Triangulation did not succeed" << endl;
		return false;
	}
	

		//filter out outlier points with high reprojection
	vector<double> reprj_errors;
	for (int i = 0; i < new_triangulated.size(); i++) { reprj_errors.push_back(new_triangulated[i].reprojection_error); }
	std::sort(reprj_errors.begin(), reprj_errors.end());
	//get the 80% precentile
	double reprj_err_cutoff = reprj_errors[4 * reprj_errors.size() / 5] * 2.4; //threshold from Snavely07 4.2

	vector<CloudPoint> new_triangulated_filtered;
	std::vector<cv::DMatch> new_matches;
	for (int i = 0; i < new_triangulated.size(); i++) {
		if (trig_status[i] == 0)
			continue; //point was not in front of camera
		if (new_triangulated[i].reprojection_error > 16.0) {
			continue; //reject point
		}
		if (new_triangulated[i].reprojection_error < 4.0 ||
			new_triangulated[i].reprojection_error < reprj_err_cutoff)
		{
			new_triangulated_filtered.push_back(new_triangulated[i]);
			new_matches.push_back(matches[i]);
		}
		else
		{
			continue;
		}
	}

	cout << "filtered out " << (new_triangulated.size() - new_triangulated_filtered.size()) << " high-error points" << endl;

	//all points filtered?
	if (new_triangulated_filtered.size() <= 0) return false;

	new_triangulated = new_triangulated_filtered;

	matches = new_matches;
	matches_matrix[std::make_pair(older_view, working_view)] = new_matches; //just to make sure, remove if unneccesary
	matches_matrix[std::make_pair(working_view, older_view)] = FlipMatches(new_matches);
	add_to_cloud.clear();
	add_to_cloud.resize(new_triangulated.size(), 1);
	int found_other_views_count = 0;
	int num_views = imgs.size();

	//scan new triangulated points, if they were already triangulated before - strengthen cloud
	//#pragma omp parallel for num_threads(1)
	for (int j = 0; j < new_triangulated.size(); j++) {
		new_triangulated[j].imgpt_for_img = std::vector<int>(imgs.size(), -1);

		//matches[j] corresponds to new_triangulated[j]
		//matches[j].queryIdx = point in <older_view>
		//matches[j].trainIdx = point in <working_view>
		new_triangulated[j].imgpt_for_img[older_view] = matches[j].queryIdx;	//2D reference to <older_view>
		new_triangulated[j].imgpt_for_img[working_view] = matches[j].trainIdx;		//2D reference to <working_view>
		bool found_in_other_view = false;
		for (unsigned int view_ = 0; view_ < num_views; view_++) {
			if (view_ != older_view) {
				//Look for points in <view_> that match to points in <working_view>
				std::vector<cv::DMatch> submatches = matches_matrix[std::make_pair(view_, working_view)];
				for (unsigned int ii = 0; ii < submatches.size(); ii++) {
					if (submatches[ii].trainIdx == matches[j].trainIdx &&
						!found_in_other_view)
					{
						//Point was already found in <view_> - strengthen it in the known cloud, if it exists there

						//cout << "2d pt " << submatches[ii].queryIdx << " in img " << view_ << " matched 2d pt " << submatches[ii].trainIdx << " in img " << i << endl;
						for (unsigned int pt3d = 0; pt3d < pcloud.size(); pt3d++) {
							if (pcloud[pt3d].imgpt_for_img[view_] == submatches[ii].queryIdx)
							{
								//pcloud[pt3d] - a point that has 2d reference in <view_>

								//cout << "3d point "<<pt3d<<" in cloud, referenced 2d pt " << submatches[ii].queryIdx << " in view " << view_ << endl;
#pragma omp critical 
								{
									pcloud[pt3d].imgpt_for_img[working_view] = matches[j].trainIdx;
									pcloud[pt3d].imgpt_for_img[older_view] = matches[j].queryIdx;
									found_in_other_view = true;
									add_to_cloud[j] = 0;
								}
							}
						}
					}
				}
			}
		}
#pragma omp critical
		{
			if (found_in_other_view) {
				found_other_views_count++;
			}
			else {
				add_to_cloud[j] = 1;
			}
		}
	}
	std::cout << found_other_views_count << "/" << new_triangulated.size() << " points were found in other views, adding " << cv::countNonZero(add_to_cloud) << " new\n";
	return true;
}

int PnPSfmManyViews::FindHomographyInliers2Views(int vi, int vj)
{
	vector<cv::KeyPoint> ikpts, jkpts; vector<cv::Point2f> ipts, jpts;
	GetAlignedPointsFromMatch(imgpts[vi], imgpts[vj], matches_matrix[make_pair(vi, vj)], ikpts, jkpts);
	KeyPointsToPoints(ikpts, ipts); KeyPointsToPoints(jkpts, jpts);

	double minVal, maxVal; cv::minMaxIdx(ipts, &minVal, &maxVal); //TODO flatten point2d?? or it takes max of width and height

	vector<uchar> status;
	cv::Mat H = cv::findHomography(ipts, jpts, status, cv::RANSAC, 0.004 * maxVal); //threshold from Snavely07
	return cv::countNonZero(status);
}

void PnPSfmManyViews::Find2D3DCorrespondences(int working_view, std::vector<cv::Point3f>& ppcloud, std::vector<cv::Point2f>& imgPoints)
{
	ppcloud.clear(); imgPoints.clear();

	vector<int> pcloud_status(pcloud.size(), 0);
	for (set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view)
	{
		int old_view = *done_view;
		//check for matches_from_old_to_working between i'th frame and <old_view>'th frame (and thus the current cloud)
		std::vector<cv::DMatch> matches_from_old_to_working = matches_matrix[std::make_pair(old_view, working_view)];

		for (unsigned int match_from_old_view = 0; match_from_old_view < matches_from_old_to_working.size(); match_from_old_view++) {
			// the index of the matching point in <old_view>
			int idx_in_old_view = matches_from_old_to_working[match_from_old_view].queryIdx;

			//scan the existing cloud (pcloud) to see if this point from <old_view> exists
			for (unsigned int pcldp = 0; pcldp < pcloud.size(); pcldp++) {
				// see if corresponding point was found in this point
				if (idx_in_old_view == pcloud[pcldp].imgpt_for_img[old_view] && pcloud_status[pcldp] == 0) //prevent duplicates
				{
					//3d point in cloud
					ppcloud.push_back(pcloud[pcldp].pt);
					//2d point in image i
					imgPoints.push_back(imgpts[working_view][matches_from_old_to_working[match_from_old_view].trainIdx].pt);

					pcloud_status[pcldp] = 1;
					break;
				}
			}
		}
	}
	cout << "found " << ppcloud.size() << " 3d-2d point correspondences" << endl;



}

bool PnPSfmManyViews::FindPoseEstimation(int working_view, cv::Mat_<double>& rvec, cv::Mat_<double>& t, cv::Mat_<double>& R, std::vector<cv::Point3f> ppcloud, std::vector<cv::Point2f> imgPoints)
{

	if (ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
		//something went wrong aligning 3D to 2D points..
		cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" << endl;
		return false;
	}

	vector<int> inliers;

		//use CPU
	double minVal, maxVal; cv::minMaxIdx(imgPoints, &minVal, &maxVal);

	
	//cv::solvePnPRansac(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, 1000, 0.006 * maxVal, 0.25 * (double)(imgPoints.size()), inliers, cv::SOLVEPNP_EPNP);



	cv::solvePnP(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, cv::SOLVEPNP_EPNP);
	
	

	vector<cv::Point2f> projected3D;
	cv::projectPoints(ppcloud, rvec, t, K, distortion_coeff, projected3D);

	if (inliers.size() == 0) { //get inliers
		for (int i = 0; i < projected3D.size(); i++) {
			if (norm(projected3D[i] - imgPoints[i]) < 10.0)
				inliers.push_back(i);
		}
	}

	//cv::Rodrigues(rvec, R);
	//visualizerShowCamera(R,t,0,255,0,0.1);

	if (inliers.size() < (double)(imgPoints.size()) / 5.0) {
		cerr << "not enough inliers to consider a good pose (" << inliers.size() << "/" << imgPoints.size() << ")" << endl;
		return false;
	}

	if (cv::norm(t) > 200.0) {
		// this is bad...
		cerr << "estimated camera movement is too big, skip this camera\r\n";
		return false;
	}

	cv::Rodrigues(rvec, R);
	if (!CheckCoherentRotation(R)) {
		cerr << "rotation is incoherent. we should try a different base view..." << endl;
		return false;
	}

	std::cout << "found t = " << t << "\nR = \n" << R << std::endl;
	return true;


	
}
