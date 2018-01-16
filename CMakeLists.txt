cmake_minimum_required(VERSION 3.5)
project(node)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system)
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories( ${Boost_INCLUDE_DIR})


file(GLOB SOURCE_FILES
	"maxi_new.cpp"
	"drivers/*.c"
	"drivers/*.h"
	"include/xparameters.h"
        "detection/*.h"
        "detection/*.cpp"
        "detection/*.hpp"
        )

#set(SOURCE_FILES maxi_new.cpp drivers/xbacksub.c drivers/xbacksub.h drivers/xfeature.c drivers/xfeature.h include/xparameters.h detection/*.h detection/*.cpp)
add_executable(node ${SOURCE_FILES})

target_link_libraries( node ${OpenCV_LIBS} ${Boost_LIBRARIES} -lpthread -lboost_serialization -lboost_date_time)
