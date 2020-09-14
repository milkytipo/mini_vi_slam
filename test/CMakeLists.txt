
add_executable(backend_triangularization backend_triangularization.cc)

target_include_directories(backend_triangularization
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC backend_lib
)

target_link_libraries(backend_triangularization
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC backend_lib
)

add_executable(backend_test backend_v_02.cc)

target_include_directories(backend_test
  PUBLIC ${OpenCV_INCLUDE_DIRS}         
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC backend_lib
)

target_link_libraries(backend_test
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC backend_lib
)


