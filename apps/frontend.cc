
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

Point2f pixel2cam( const Point2d& p, const Mat& K ){
      return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}
void pose_estimation_2d2d (
    const std::vector<cv::KeyPoint>& keypoints_1,
    const std::vector<cv::KeyPoint>& keypoints_2,
    const std::vector< cv::DMatch >& matches,
    cv::Mat& R, cv::Mat& t ){

    Point2d principal_point (367.215803962, 248.37534061);	
    int focal_length = 458;	
    cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 458.654880721, 0, principal_point.x , 0, 457.296696463, principal_point.y, 0, 0, 1 );
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );

    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );

    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );

    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point ); 

    }

void triangulation (
    const std::vector<cv::KeyPoint>& keypoint_1,
    const std::vector<cv::KeyPoint>& keypoint_2,
    const std::vector< cv::DMatch >& matches,
    const cv::Mat& R, const cv::Mat& t,
    std::vector<cv::Point3d>& points){
    Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    Point2d principal_point (367.215803962, 248.37534061);	
    int focal_length = 458;	
    cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 458.654880721, 0, principal_point.x , 0, 457.296696463, principal_point.y, 0, 0, 1 );
    vector<Point2f> pts_1, pts_2;
    for ( DMatch m:matches )
    {
        pts_1.push_back ( pixel2cam( keypoint_1[m.queryIdx].pt, K) );
        pts_2.push_back ( pixel2cam( keypoint_2[m.trainIdx].pt, K) );
    }
    
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }
}


class CameraData {
 public:
  CameraData(std::string timestamp_str, std::string data_file_path) {
    timestamp_ = timestamp_str;
    image_ = cv::imread(data_file_path, cv::IMREAD_GRAYSCALE);
  }

  std::string GetTimestamp() { 
    return timestamp_; 
  }
  
  cv::Mat GetImage() { 
    return image_; 
  }

 private:
  std::string timestamp_;   // we don't have to process time at this moment
  cv::Mat image_;
};

// This class wraps cv::KeyPoint in order to use std::map
class CVKeypoint {
 public:
 	CVKeypoint(cv::KeyPoint keypoint) {
 		keypoint_ = keypoint;
 		hash_value_ = keypoint.hash();
 	}

  float GetU() {
  	return keypoint_.pt.x;
  }

  float GetV() {
  	return keypoint_.pt.y;
  }

 	bool operator==(const CVKeypoint& kp) const{
    return hash_value_ == kp.hash_value_;
 	}

 	bool operator<(const CVKeypoint& kp) const{
    return hash_value_ < kp.hash_value_;
 	}

 private:
  cv::KeyPoint keypoint_;
  size_t hash_value_;
};

int main(int argc, char **argv) {

  /*** Step 0. Read configuration file ***/

  // for yaml file
  std::string config_file_path("../config/test.yaml");
  cv::FileStorage config_file(config_file_path, cv::FileStorage::READ);

  std::string time_window_begin(config_file["time_window"][0]);
  std::string time_window_end(config_file["time_window"][1]);

  size_t downsample_rate = (size_t)(int)(config_file["frontend"]["downsample_rate"]);

  std::cout << "Consider from " << time_window_begin << " to " << time_window_end << ": " << std::endl;

  /*** Step 1. Read image files ***/

  // the folder path
  std::string path((std::string)argv[1]+"/");
  // std::string path("../../../dataset/mav0/");
  std::string camera_data_folder("cam0/data/");

  std::vector<std::string> image_names;

  // boost allows us to work on image files directly
  for (auto iter = boost::filesystem::directory_iterator(path + camera_data_folder);
        iter != boost::filesystem::directory_iterator(); iter++) {

    if (!boost::filesystem::is_directory(iter->path())) {           // we eliminate directories
      image_names.push_back(iter->path().filename().string());
    } 
    else
      continue;
  }

  std::sort(image_names.begin(), image_names.end());

  std::vector<CameraData> camera_observation_data;                  // image and timestep

  size_t counter = 0;
  for (auto& image_names_iter: image_names) {	
  
    if (counter % downsample_rate == 0) {                           // downsample images for testing
      std::string time_stamp_str = image_names_iter.substr(0,19);   // remove ".png"

      if(time_window_begin <= time_stamp_str && time_stamp_str <= time_window_end) {
        std::string dataFilePath = path + camera_data_folder + image_names_iter;
        camera_observation_data.push_back(CameraData(time_stamp_str, dataFilePath));

        // cv::imshow(time_stamp_str, camera_observation_data.back().GetImage());
        // cv::waitKey(100);
      }
    }

    counter++;
  }

  size_t num_of_cam_observations = camera_observation_data.size();


  /*** Step 2. Extract features ***/

  cv::Ptr<cv::BRISK> brisk_detector =
    cv::BRISK::create(60, 0, 1.0f);


  // you can try to use ORB feature as well
  // std::shared_ptr<cv::FeatureDetector> orb_detector = cv::ORB::create();

  std::vector<std::vector<cv::KeyPoint>> image_keypoints(num_of_cam_observations);
  std::vector<cv::Mat> image_descriptions(num_of_cam_observations);

  for (size_t i=0; i<num_of_cam_observations; i++) {	

    brisk_detector->detect(camera_observation_data.at(i).GetImage(), image_keypoints.at(i));

    brisk_detector->compute(camera_observation_data.at(i).GetImage(), 
      image_keypoints.at(i), 
      image_descriptions.at(i));
  }
 
  /*** Step 3. Match features ***/

  cv::Ptr<cv::DescriptorMatcher>  matcher = 
    cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

  std::vector<std::vector<cv::DMatch>> image_matches(num_of_cam_observations-1);
  std::vector<std::vector<cv::DMatch>> image_good_matches(num_of_cam_observations-1);
  std::vector< vector<Point3d>> landmark_points_data;
  for (size_t i=0; i<num_of_cam_observations-1; i++) {

    matcher->match(image_descriptions.at(i), image_descriptions.at(i+1), image_matches.at(i));

    cv::Mat img_w_matches;
    for (size_t k=0; k<image_matches.at(i).size(); k++) {
      if (image_matches.at(i)[k].distance < 60) {
        image_good_matches.at(i).push_back(image_matches.at(i)[k]);
      }
    }
    Mat R,t;
    pose_estimation_2d2d (image_keypoints.at(i), 
                                                      image_keypoints.at(i+1), 
                                                      image_good_matches.at(i),
                                                      R, 
                                                      t);

    vector<Point3d> points;
    triangulation( image_keypoints.at(i),
                                  image_keypoints.at(i+1),
                                  image_good_matches.at(i),
                                  R, 
                                  t, 
                                  points );
    landmark_points_data.push_back(points);
    // cv::drawMatches(camera_observation_data.at(i).GetImage(), image_keypoints.at(i),
    //                 camera_observation_data.at(i+1).GetImage(), image_keypoints.at(i+1),
    //                 image_good_matches.at(i), img_w_matches);

    // cv::imshow("Matches between " + std::to_string(i) + " and " + std::to_string(i+1), img_w_matches);
    // cv::waitKey();
  }

  /*** Step 4. Obtain feature observation ***/

  std::map<CVKeypoint, size_t> pre_landmark_lookup_table;       // keypoint and landmark id
  std::map<CVKeypoint, size_t> next_landmark_lookup_table;

  std::vector<std::string> output_feature_observation;

  size_t landmakr_id_count = 0;
  size_t landmark_id = 0;

  for (size_t i=0; i<image_good_matches.size(); i++) {
    for (size_t m=0; m<image_good_matches.at(i).size(); m++) {
      
      size_t pre_keypoint_id = image_good_matches.at(i)[m].queryIdx;
      size_t next_keypoint_id = image_good_matches.at(i)[m].trainIdx;

      CVKeypoint pre_keypoint = CVKeypoint(image_keypoints.at(i)[pre_keypoint_id]);
      CVKeypoint next_keypoint = CVKeypoint(image_keypoints.at(i+1)[next_keypoint_id]);

      auto iterr = pre_landmark_lookup_table.find(pre_keypoint);
      if (iterr == pre_landmark_lookup_table.end()) {

        landmark_id = landmakr_id_count;

        pre_landmark_lookup_table.insert(std::pair<CVKeypoint, size_t>(pre_keypoint, landmark_id));
        ++landmakr_id_count;
      }
      else {
        landmark_id = iterr->second;
      }      	

    // output
    // timestamp [ns], landmark id, u [pixel], v [pixel]
    std::string output_str = camera_observation_data.at(i).GetTimestamp() + "," + std::to_string(landmark_id+1) + ","
                              + std::to_string(pre_keypoint.GetU()) + "," + std::to_string(pre_keypoint.GetV()) + "," + std::to_string(landmark_points_data.at(i)[m].x) + ","
                              + std::to_string(landmark_points_data.at(i)[m].y) + "," + std::to_string(landmark_points_data.at(i)[m].z) + "\n";
    output_feature_observation.push_back(output_str);
    // output_file << output_str;

      next_landmark_lookup_table.insert(std::pair<CVKeypoint, size_t>(next_keypoint, landmark_id));
    }

    std::swap(pre_landmark_lookup_table, next_landmark_lookup_table);
    next_landmark_lookup_table.clear();
  }


  /*** Step 5. Output observation ***/

  std::ofstream output_file;
  output_file.open ("feature_observation.csv");
  output_file << "timestamp [ns], landmark id, u [pixel], v [pixel], l_x, l_y, l_z\n";
  
  for (auto& output_str: output_feature_observation) { 
    // timestamp [ns], landmark id, u [pixel], v [pixel]
    output_file << output_str;
  }
  
  output_file.close();       

 
  return 0;
}
