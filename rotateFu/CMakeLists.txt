cmake_minimum_required(VERSION 2.8.3)
project(ar_test)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  artoolkit
)

find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
include_directories(include)

link_directories("/home/pan/git_ws/devel/lib/")
catkin_package(INCLUDE_DIRS include
               CATKIN_DEPENDS roscpp artoolkit
)

add_executable(rotateFu src/rotateFu.cpp)
#add_dependencies(rotateFu ${artoolkit_EXPORTED_TARGETS})
target_link_libraries(rotateFu ${Catkin_LIBS} ${OpenCV_LIBS} AR ARvideo)


