#include "Common.h"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



#include <iostream>
#include <windows.h>


#include <fstream>
#include <string>

using namespace std;
using namespace cv;


void open_imgs_dir(char * dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names)
{

	if (dir_name == NULL) {
		return;
	}

	string dir_name_ = string(dir_name);
	vector<string> files_;

	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA fdata;

	if (dir_name_[dir_name_.size() - 1] == '\\' || dir_name_[dir_name_.size() - 1] == '/') {
		dir_name_ = dir_name_.substr(0, dir_name_.size() - 1);
	}

	hFind = FindFirstFile(string(dir_name_).append("\\*").c_str(), &fdata);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (strcmp(fdata.cFileName, ".") != 0 &&
				strcmp(fdata.cFileName, "..") != 0)
			{
				if (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					continue; // a diretory
				}
				else
				{
					files_.push_back(fdata.cFileName);
				}
			}
		} while (FindNextFile(hFind, &fdata) != 0);
	}
	else {
		cerr << "can't open directory\n";
		return;
	}

	if (GetLastError() != ERROR_NO_MORE_FILES)
	{
		FindClose(hFind);
		cerr << "some other error with opening directory: " << GetLastError() << endl;
		return;
	}

	FindClose(hFind);
	hFind = INVALID_HANDLE_VALUE;


	for (unsigned int i = 0; i < files_.size(); i++) {
		if (files_[i][0] == '.' || !(hasEndingLower(files_[i], "jpg") || hasEndingLower(files_[i], "png") || hasEndingLower(files_[i], "ppm"))) {
			continue;
		}
		cv::Mat m_ = cv::imread(string(dir_name_).append("/").append(files_[i]));
		while (m_.size().width > 2 * 600) {
			cv::pyrDown(m_, m_);
		}
		//cv::resize(m_, m_, Size(), 0.5, 0.5);
		
		images_names.push_back(files_[i]);
		images.push_back(m_);
	}








}

void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps)
{
	ps.clear();
	for (unsigned int i = 0; i < kps.size(); i++) ps.push_back(kps[i].pt);

}

void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps)
{
	kps.clear();
	for (unsigned int i = 0; i < ps.size(); i++) kps.push_back(KeyPoint(ps[i], 1.0f));
}

std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches)
{

	std::vector<cv::DMatch> flip;
	for (int i = 0; i < matches.size(); i++) {
		flip.push_back(matches[i]);
		swap(flip.back().queryIdx, flip.back().trainIdx);
	}
	return flip;
}

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts)
{

	std::vector<cv::Point3d> out;
	for (unsigned int i = 0; i < cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1, const std::vector<cv::KeyPoint>& imgpts2, const std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& pt_set1, std::vector<cv::KeyPoint>& pt_set2)
{
	for (unsigned int i = 0; i < matches.size(); i++) {
		assert(matches[i].queryIdx < imgpts1.size());
		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		assert(matches[i].trainIdx < imgpts2.size());
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
	}

}

bool hasEnding(std::string const & fullString, std::string const & ending)
{

	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
	
}

bool hasEndingLower(string const & fullString_, string const & _ending)
{

	string fullstring = fullString_, ending = _ending;
	transform(fullString_.begin(), fullString_.end(), fullstring.begin(), ::tolower); // to lower
	return hasEnding(fullstring, ending);
	
}

#define intrpmnmx(val,min,max) (max==min ? 0.0 : ((val)-min)/(max-min))

void drawArrows(cv::Mat & frame, const std::vector<cv::Point2f>& prevPts, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status, const std::vector<float>& verror, const cv::Scalar & _line_color)
{
	double minVal, maxVal; minMaxIdx(verror, &minVal, &maxVal, 0, 0, status);
	int line_thickness = 1;

	for (size_t i = 0; i < prevPts.size(); ++i)
	{
		if (status[i])
		{
			double alpha = intrpmnmx(verror[i], minVal, maxVal); alpha = 1.0 - alpha;
			Scalar line_color(alpha*_line_color[0],
				alpha*_line_color[1],
				alpha*_line_color[2]);

			Point p = prevPts[i];
			Point q = nextPts[i];

			double angle = atan2((double)p.y - q.y, (double)p.x - q.x);

			double hypotenuse = sqrt((double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x));

			if (hypotenuse < 1.0)
				continue;

			// Here we lengthen the arrow by a factor of three.
			q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			// Now we draw the main line of the arrow.
			line(frame, p, q, line_color, line_thickness);

			// Now draw the tips of the arrow. I do some scaling so that the
			// tips look proportional to the main line of the arrow.

			p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
			p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
			line(frame, p, q, line_color, line_thickness);

			p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
			p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
			line(frame, p, q, line_color, line_thickness);
		}
	}




}

cv::Mat get_intrinsics(char * dir_name)
{
	std::vector<double> values = std::vector<double>(12);

	ifstream myfile(dir_name);

	std::string content((std::istreambuf_iterator<char>(myfile)),
		(std::istreambuf_iterator<char>()));


	std::istringstream in(content);

	std::string header;
	in >> header;

	if (header == "CONTOUR") {

		for (int i = 0; i < 12; i++) {
			in >> values[i];
		}

	}

	std::cout << header <<endl;
	for (int i = 0; i < 12; i++) {
		std::cout << values[i]<<endl;
	}

	 cv::Mat P(3, 4, CV_32FC1, values.data());



	 return P;

}
