#include <iostream>
#include <thread>








#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "common.h"
#include "ifeaturematcher.h"
#include "richfeaturematcher.h"
#include "sfm.h"
#include "PnPSfmManyViews.h"




#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/statistical_outlier_removal.h>






using namespace cv;
using namespace std;





pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}

pcl::visualization::PCLVisualizer::Ptr rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}


int main()
{

	bool simple(false), rgb(true) , write_to_file(true);

	char output_file[] = "cumfaggot.pcd";


	std::vector<cv::Mat> images;
	std::vector<std::string> images_names;
	
	char path[] = "d:\\projects\\grad_project\\pics\\dataset";
	char path2[] = "d:\\projects\\grad_project\\pics\\fountain_dataset";
	char path3[] = "d:\\projects\\grad_project\\pics\\scene1";
	char path5[] = "d:\\projects\\grad_project\\pics\\ichigo2";

	
	
	
	open_imgs_dir(path2, images, images_names);


	PnPSfmManyViews pipline_many_views(images, images_names);

	pipline_many_views.RecoverDepthFromImages();



	std::vector<cv::Point3d> pointcloud = pipline_many_views.getPointCloud();
	std::vector<cv::Vec3b> rgbcloud = pipline_many_views.getPointCloudRGB();

	std::cout << "size of rgb cloud" << rgbcloud.size() <<endl;



	/*
	Sfm pipeline(images[2], images[3]);
	
	pipeline.reconstruct_two_views();
	
	std::vector<cv::Point3d> pointcloud = pipeline.getPointCloud();*/


	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

	std::cout << "generating example point clouds.\n\n";
	
	
		for (auto i = 0; i < pointcloud.size(); i++) {

			cv::Vec3b rgbv(255, 255, 255);
			rgbv = rgbcloud[i];

	
			pcl::PointXYZ basic_point;

			basic_point.x = pointcloud[i].x;
			basic_point.y = pointcloud[i].y* -1;
			basic_point.z = pointcloud[i].z;
			basic_cloud_ptr->push_back(basic_point);



			pcl::PointXYZRGB rgb_point;
			rgb_point.x = pointcloud[i].x;
			rgb_point.y = pointcloud[i].y* -1;
			rgb_point.z = pointcloud[i].z;

			uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
			rgb_point.rgb = *reinterpret_cast<float*>(&rgb);

			
			point_cloud_ptr->points.push_back(rgb_point);

	
	
		}


		//outlier removal
		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
		sor.setInputCloud(point_cloud_ptr);
		sor.setMeanK(101);
		sor.setStddevMulThresh(1.0);
		sor.filter(*point_cloud_ptr);

		if (write_to_file && simple) {

			pcl::io::savePCDFileASCII(output_file, *basic_cloud_ptr);


		}

		else if  (write_to_file && rgb) {

			pcl::io::savePCDFileASCII("couch.pcd", *basic_cloud_ptr);
			pcl::io::savePCDFileASCII(output_file, *point_cloud_ptr);


		}
	


		std::cout << "size of point clouud :" << basic_cloud_ptr->size() <<endl;


		basic_cloud_ptr->width = basic_cloud_ptr->size();
		basic_cloud_ptr->height = 1;
		point_cloud_ptr->width = point_cloud_ptr->size();
		point_cloud_ptr->height = 1;
	


		pcl::visualization::PCLVisualizer::Ptr viewer;

		if (rgb) {
			
			viewer = rgbVis(point_cloud_ptr);

		}

		else if (simple) {

			viewer = simpleVis(basic_cloud_ptr);

		}
	

		while (!viewer->wasStopped())
		{

			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));



		}
	
	





}