  #include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include "landmark_parameter_block.h"
#include "timed_3d_parameter_block.h"
#include "timed_quat_parameter_block.h"
#include "imu_error.h"
#include "reprojection_error.h"

    int t_imu_lastframe = time_begin_;
    size_t i_imu_lastframe = 0;
for (size_t i_obs = 0; i_obs < obs_data_vec.size(); i_obs++){
        // parameters are from ground truth data currently, and can be estimatd in the optimization problem later
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      
    Eigen::Vector3d gyro_bias = Eigen::Vector3d(-0.003196, 0.021298, 0.078430);
    Eigen::Vector3d accel_bias = Eigen::Vector3d(-0.026176, 0.137568, 0.076295);

    while (t_imu_lastframe <= obs_data_vec.at(i_obs).GetTimestamp()){
        size_t i = i_imu_lastframe;
        t_imu_lastframe =  imu_data_vec.at(i).GetTimestamp();
        double time_diff = position_parameter_.at(i+1)->timestamp() - position_parameter_.at(i)->timestamp();

        Eigen::Vector3d accel_measurement = imu_data_vec.at(i).GetAccelMeasurement();
        Eigen::Vector3d gyro_measurement = imu_data_vec.at(i).GetGyroMeasurement();      
        Eigen::Vector3d accel_plus_gravity = rotation_parameter_.at(i)->estimate().normalized().toRotationMatrix()*(accel_measurement - accel_bias) + gravity;

        Eigen::Vector3d position_t_plus_1 = position_parameter_.at(i)->estimate() + time_diff*velocity_parameter_.at(i)->estimate() + (0.5*time_diff*time_diff)*accel_plus_gravity;
        Eigen::Vector3d velocity_t_plus_1 = velocity_parameter_.at(i)->estimate() + time_diff*accel_plus_gravity;
        Eigen::Quaterniond rotation_t_plus_1 = rotation_parameter_.at(i)->estimate().normalized() * Eigen::Quaterniond(1, 0.5*time_diff*(gyro_measurement(0)-gyro_bias(0)), 
                                                                                                                            0.5*time_diff*(gyro_measurement(1)-gyro_bias(1)), 
                                                                                                                        0.5*time_diff*(gyro_measurement(2)-gyro_bias(2)));

        position_parameter_.at(i+1)->setEstimate(position_t_plus_1);
        velocity_parameter_.at(i+1)->setEstimate(velocity_t_plus_1);
        rotation_parameter_.at(i+1)->setEstimate(rotation_t_plus_1);
        i_imu_lastframe++;
    } 
    t_imu_lastframe = imu_data_vec.at(i_imu_lastframe).GetTimestamp();
    size_t landmark_id  = obs_data_vec.at(i_obs).GetId() -1 ;
 TODO::Initial Landmark[0]
    //Add Observation constraint
    ceres::CostFunction* cost_function = new ReprojectionError(obs_data_vec.at(i_obs).GetFeaturePosition(),
                                                                 T_bc_,
                                                                 focal_length_,
                                                                 principal_point_);
    optimization_problem_.AddResidualBlock(cost_function,
                                            NULL,
                                            position_parameter_.at(i_imu_lastframe)->parameters(),
                                            rotation_parameter_.at(i_imu_lastframe)->parameters(),
                                            landmark_parameter_.at(landmark_id)->parameters()); 
    
    ceres::CostFunction* cost_function = new ImuError(imu_data_vec.at(i_imu_lastframe).GetGyroMeasurement(),
                                                        imu_data_vec.at(i_imu_lastframe).GetAccelMeasurement(),
                                                        t_imu_lastframe - obs_data_vec.at(i_obs).GetTimestamp());
    optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             position_parameter_.at(i_imu_lastframe+1)->parameters(),
                                             velocity_parameter_.at(i_imu_lastframe+1)->parameters(),
                                             rotation_parameter_.at(i_imu_lastframe+1)->parameters(),
                                             position_parameter_.at(i_imu_lastframe+1)->parameters(),
                                             velocity_parameter_.at(i_imu_lastframe+1)->parameters(),
                                             rotation_parameter_.at(i_imu_lastframe+1)->parameters());   


    optimization_options_.linear_solver_type = ceres::DENSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 6;
    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

}