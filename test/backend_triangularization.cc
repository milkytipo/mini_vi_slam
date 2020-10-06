
// This test file verifies reprojection_error.h

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "landmark_parameter_block.h"
#include "timed_3d_parameter_block.h"
#include "timed_quat_parameter_block.h"
#include "reprojection_error.h"


class TriangularizationProblem {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  TriangularizationProblem() {
    focal_length_ = 458.5;
    principal_point_[0] = 367.215803962;
    principal_point_[1] = 248.37534061;

    initial_landmark_position_ = 10.5 * Eigen::Vector3d(1, 1, 1);  // check between 8 and 9
    real_landmark_position_ = Eigen::Vector3d(13,13,13);
  }

  bool Initialize() {

    // state 0
    rotation_parameter_.push_back(new TimedQuatParameterBlock(Eigen::Quaterniond::UnitRandom(), 0.0));
    position_parameter_.push_back(new Timed3dParameterBlock(Eigen::Vector3d(1,1,1), 0.0));

    // state 1
    rotation_parameter_.push_back(new TimedQuatParameterBlock(Eigen::Quaterniond::UnitRandom(), 0.1));
    position_parameter_.push_back(new Timed3dParameterBlock(Eigen::Vector3d(-1,-1,-1), 0.1));

    // state 2
    rotation_parameter_.push_back(new TimedQuatParameterBlock(Eigen::Quaterniond::UnitRandom(), 0.2));
    position_parameter_.push_back(new Timed3dParameterBlock(Eigen::Vector3d(3,4,-6), 0.2));

    // one landmark
    landmark_parameter_.push_back(new LandmarkParameterBlock(initial_landmark_position_));

    return true;
  }

  bool SetupOptimizationProblem() {

    // add parameter blocks
    for (size_t i=0; i<3; ++i) {
      optimization_problem_.AddParameterBlock(position_parameter_.at(i)->parameters(), 3);
      // optimization_problem_.SetParameterBlockConstant(position_parameter_.at(i)->parameters());

      optimization_problem_.AddParameterBlock(rotation_parameter_.at(i)->parameters(), 4);
      // optimization_problem_.SetParameterBlockConstant(rotation_parameter_.at(i)->parameters());
    }

    // observation
    for (size_t i=0; i<3; ++i) {
      Eigen::Vector2d observation;

      Eigen::Transform<double, 3, Eigen::Affine> T_nb = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
      T_nb.rotate(rotation_parameter_.at(i)->estimate());
      T_nb.translate(position_parameter_.at(i)->estimate());

      Eigen::Vector3d landmark_c = T_nb.inverse() * real_landmark_position_;

      observation(0) = -focal_length_*landmark_c(0)/landmark_c(2) + principal_point_[0];
      observation(1) = -focal_length_*landmark_c(1)/landmark_c(2) + principal_point_[1];


      std::cout << "observation: " << observation << std::endl;

      Eigen::Vector2d observation_noise = Eigen::Vector2d::Random();
      observation = observation + observation_noise;

      std::cout << "observation w. noise: " << observation << std::endl;

      // establish constraints
      ceres::CostFunction* cost_function = new ReprojectionError(observation,
                                                                 Eigen::Transform<double, 3, Eigen::Affine>::Identity(),
                                                                 focal_length_,
                                                                 principal_point_);

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             rotation_parameter_.at(i)->parameters(),
                                             position_parameter_.at(i)->parameters(),
                                             landmark_parameter_.at(0)->parameters()); 
    }

    optimization_problem_.SetParameterLowerBound(landmark_parameter_.at(0)->parameters(), 0, 0.0);
    optimization_problem_.SetParameterLowerBound(landmark_parameter_.at(0)->parameters(), 1, 0.0);
    optimization_problem_.SetParameterLowerBound(landmark_parameter_.at(0)->parameters(), 2, 0.0);

    optimization_problem_.SetParameterUpperBound(landmark_parameter_.at(0)->parameters(), 0, 20.0);
    optimization_problem_.SetParameterUpperBound(landmark_parameter_.at(0)->parameters(), 1, 20.0);
    optimization_problem_.SetParameterUpperBound(landmark_parameter_.at(0)->parameters(), 2, 20.0);

    return true;
  }

  bool SolveOptimizationProblem() {

    std::cout << "Begin solving the optimization problem." << std::endl;

    optimization_options_.linear_solver_type = ceres::DENSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.function_tolerance = 1e-9;

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }


  bool OutputOptimizationResult() {

    std::cout << "The true landmark position: " << real_landmark_position_.transpose() <<  std::endl;
    std::cout << "The initial estimated landmark position: " << initial_landmark_position_.transpose() <<  std::endl;
    std::cout << "The final estimated landmark position: " << landmark_parameter_.at(0)->estimate().transpose() <<  std::endl;
    
    return true;
  }

 private:

  // camera intrinsic parameters
  double focal_length_;
  double principal_point_[2];

  // data storage (parameters to be optimized)
  std::vector<TimedQuatParameterBlock*> rotation_parameter_;
  std::vector<Timed3dParameterBlock*>   position_parameter_;
  std::vector<LandmarkParameterBlock*>  landmark_parameter_;

  Eigen::Vector3d real_landmark_position_;
  Eigen::Vector3d initial_landmark_position_;

  // ceres parameter
  ceres::Problem optimization_problem_;
  ceres::Solver::Options optimization_options_;
  ceres::Solver::Summary optimization_summary_;
};


int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  TriangularizationProblem triangularization_problem;
  triangularization_problem.Initialize();
  triangularization_problem.SetupOptimizationProblem();
  triangularization_problem.SolveOptimizationProblem();
  triangularization_problem.OutputOptimizationResult();

  return 0;
}