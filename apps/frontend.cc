
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


class TimedImageData {
 public:
  TimedImageData(std::string timestamp_str, std::string data_file_path) {
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

class FeatureNode {
 public: 
  FeatureNode() {
    landmark_id_ = 0;
  }

  bool AddNeighbor(FeatureNode* feature_node_ptr) {
    neighbors_.push_back(feature_node_ptr);
    return true;
  }

  bool IsNeighborEmpty() {
    return neighbors_.empty();
  }

  size_t GetLandmarkId() {
    return landmark_id_;
  }

  void SetLandmarkId(size_t new_landmark_id) {
    landmark_id_ = new_landmark_id;
  }  

  bool AssignLandmarkId(size_t input_landmark_id) {
    if (input_landmark_id == 0) {
      std::cout << "invalid landmark id." << std::endl;
      return false;
    }
    else if (input_landmark_id == landmark_id_) {
      return true;
    }    
    else if (input_landmark_id != landmark_id_ && landmark_id_ == 0) {
      landmark_id_ = input_landmark_id;
      
      for (auto neighbor_ptr: neighbors_) {
        neighbor_ptr->AssignLandmarkId(input_landmark_id);
      }

      return true;
    }
    else {   // input_landmark_id != landmark_id_ && landmark_id_ != 0
      std::cout << "I don't know what happen!" << std::endl;
      return false;
    }

  }

 private:
  size_t landmark_id_;
  std::vector<FeatureNode *> neighbors_;   
};

class Frontend {

 public:
  Frontend(std::string config_folder_path) {
  
    std::string config_file_path = config_folder_path + "test.yaml";
    cv::FileStorage config_file(config_file_path, cv::FileStorage::READ);

    time_window_begin_ = std::string(config_file["time_window"][0]);
    time_window_end_ = std::string(config_file["time_window"][1]);
    downsample_rate_ = (size_t)(int)(config_file["frontend"]["downsample_rate"]);
    landmark_obs_count_threshold_ = (int)(config_file["frontend"]["landmark_obs_count_threshold"]);

    std::cout << "Consider from " << time_window_begin_ << " to " << time_window_end_ << ": " << std::endl;
  }

  bool ReadImages(std::string image_folder_path) {

    // boost allows us to work on image files directly
    for (auto iter = boost::filesystem::directory_iterator(image_folder_path);
          iter != boost::filesystem::directory_iterator(); iter++) {

      if (!boost::filesystem::is_directory(iter->path())) {           // we eliminate directories
        image_names_.push_back(iter->path().filename().string());
      } 
      else
        continue;
    }

    std::sort(image_names_.begin(), image_names_.end());

    size_t counter = 0;
    size_t selected_counter = 0;

    for (auto& image_names_iter: image_names_) { 
    
      if (counter % downsample_rate_ == 0) {                           // downsample images for testing
        std::string time_stamp_str = image_names_iter.substr(0,19);    // remove ".png"

        if(time_window_begin_ <= time_stamp_str && time_stamp_str <= time_window_end_) {
          std::string image_file_path = image_folder_path + image_names_iter;
          image_data_.push_back(TimedImageData(time_stamp_str, image_file_path));

          // cv::imshow(time_stamp_str, image_data_.back().GetImage());
          // cv::waitKey(100);

          selected_counter++;
        }
      }

      counter++;
    }

    std::cout << "number of processed images: " << selected_counter << std::endl;

    return true;
  }

  bool ExtractFeatures(std::shared_ptr<cv::FeatureDetector> detector) {

    size_t num_of_images = image_data_.size();

    image_keypoints_.resize(num_of_images);
    image_descriptions_.resize(num_of_images);

    for (size_t i=0; i<num_of_images; i++) {  

      detector->detect(image_data_.at(i).GetImage(), image_keypoints_.at(i));

      detector->compute(image_data_.at(i).GetImage(), 
        image_keypoints_.at(i), 
        image_descriptions_.at(i));

        /***
        cv::Mat img_w_keypoints;
        cv::drawKeypoints(image_data_.at(i).GetImage(), image_keypoints_.at(i), img_w_keypoints);

        cv::imshow("image with keypoints " + std::to_string(i) + "/" + std::to_string(num_of_images) , img_w_keypoints);
        cv::waitKey();
        ***/
    }

    return true;
  }


  bool MatchFeatures(std::shared_ptr<cv::DescriptorMatcher> matcher) {

    size_t num_of_images = image_data_.size();

    landmark_id_table_.resize(num_of_images);
    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        landmark_id_table_.at(i).push_back(new FeatureNode());
      }
    }


    // call opencv matcher
    for (size_t i=0; i<num_of_images; i++) {
      for (size_t j=i+1; j<num_of_images; j++) {

        std::vector<cv::DMatch> image_keypoint_temp_matches;
        std::vector<cv::DMatch> image_keypoint_matches;
  
        matcher->match(image_descriptions_.at(i), image_descriptions_.at(j), image_keypoint_temp_matches);

        // keep the matches that have smaller distance
        for (size_t k=0; k<image_keypoint_temp_matches.size(); k++) {
          if (image_keypoint_temp_matches[k].distance < 40) {   // 60

            image_keypoint_matches.push_back(image_keypoint_temp_matches[k]);

            // add edge to the graph
            size_t query_idx = image_keypoint_temp_matches[k].queryIdx;
            size_t train_idx = image_keypoint_temp_matches[k].trainIdx;

            landmark_id_table_.at(i).at(query_idx)->AddNeighbor(landmark_id_table_.at(j).at(train_idx));
            landmark_id_table_.at(j).at(train_idx)->AddNeighbor(landmark_id_table_.at(i).at(query_idx));
          }
        }  
      }
    }


    // assign landmark id to each matched features
    size_t landmark_count = 0;

    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        if (!landmark_id_table_.at(i).at(k)->IsNeighborEmpty() && landmark_id_table_.at(i).at(k)->GetLandmarkId()==0) {
          landmark_count++;
          landmark_id_table_.at(i).at(k)->AssignLandmarkId(landmark_count);
        }
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
  }

  /*** Step 4. Obtain feature observation ***/


    // count the number of observations for each landmark/feature
    std::vector<size_t> landmark_obs_count(landmark_count, 0);

    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        
        size_t temp_landmark_id = landmark_id_table_.at(i).at(k)->GetLandmarkId();
        if (temp_landmark_id > 0) {
          landmark_obs_count.at(temp_landmark_id-1)++;
        }
      }
    }

    // keep only those landmarks observed often
    size_t landmark_count_after_threshold = 0;
    std::vector<size_t> landmark_id_2_id_table(landmark_count, 0);

    for (size_t i=0; i<landmark_count; i++) {
      if (landmark_obs_count.at(i) > landmark_obs_count_threshold_) {
        landmark_count_after_threshold++;
        landmark_id_2_id_table.at(i) = landmark_count_after_threshold;
      }
    }

    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {

        size_t temp_landmark_id = landmark_id_table_.at(i).at(k)->GetLandmarkId();
        if (temp_landmark_id > 0) {
          landmark_id_table_.at(i).at(k)->SetLandmarkId(landmark_id_2_id_table.at(temp_landmark_id-1));
        }
      }
    }

    std::cout << "total landmark counts: " << landmark_count << std::endl;
    std::cout << "total landmark counts: " << landmark_count_after_threshold << std::endl;

    return true;
  }


  bool OutputFeatureObservation(std::string output_file_str) {

    std::ofstream output_file;
    output_file.open(output_file_str);
    // output_file << "timestamp [ns], landmark id, u [pixel], v [pixel]\n";
    output_file << "timestamp [ns], landmark id, u [pixel], v [pixel], l_x, l_y, l_z\n";
    
    size_t num_of_images = image_data_.size();

    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); ++k) {

        if (landmark_id_table_.at(i).at(k)->GetLandmarkId()!=0) {
          
          std::string output_str = image_data_.at(i).GetTimestamp() + "," 
                                   + std::to_string(landmark_id_table_.at(i).at(k)->GetLandmarkId()) + ","
                                   + std::to_string(image_keypoints_.at(i).at(k).pt.x) + ","
                                   + std::to_string(image_keypoints_.at(i).at(k).pt.y) + ","
                                   + std::to_string(landmark_points_data.at(i)[m].x) + ","
                                   + std::to_string(landmark_points_data.at(i)[m].y) + "," 
                                   + std::to_string(landmark_points_data.at(i)[m].z) + "\n";"\n";
          output_file << output_str;
        }
      }
      else {
        landmark_id = iterr->second;
      }      	
    }
  
    output_file.close();

    return true;
  }

 private: 
  std::string time_window_begin_;
  std::string time_window_end_;
  size_t downsample_rate_;
  int landmark_obs_count_threshold_;

  std::vector<std::string>                  image_names_;
  std::vector<TimedImageData>               image_data_;       

  std::vector<std::vector<cv::KeyPoint>>    image_keypoints_;
  std::vector<cv::Mat>                      image_descriptions_;           

  std::vector<std::vector<FeatureNode*>>    landmark_id_table_;
};

int main(int argc, char **argv) {

  /*** Step 0. Read configuration file ***/

  std::string config_folder_path("../config/");
  Frontend frontend(config_folder_path);                     // read configuration file


  /*** Step 1. Read image files ***/

  // std::string path(argv[1]);
  // std::string dataset_path("../../../dataset/mav0/");
  std::string dataset_path((std::string)argv[1]+"/")
  std::string camera_data_folder("cam0/data/");

  frontend.ReadImages(dataset_path + camera_data_folder);


  /*** Step 2. Extract features ***/
 //   cv::Ptr<cv::BRISK> brisk_detector =cv::BRISK::create(60, 0, 1.0f);
  std::shared_ptr<cv::FeatureDetector> brisk_detector = cv::BRISK::create(60, 0, 1.0f);
  std::shared_ptr<cv::FeatureDetector> orb_detector = cv::ORB::create();

  frontend.ExtractFeatures(brisk_detector);


  /*** Step 3. Match features ***/
  std::shared_ptr<cv::DescriptorMatcher> bf_hamming_matcher = 
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);  
  frontend.MatchFeatures(bf_hamming_matcher);


  /*** Step 4. Output observation ***/
  std::string output_file_str("feature_observation.csv");
  frontend.OutputFeatureObservation(output_file_str);


  return 0;
}
