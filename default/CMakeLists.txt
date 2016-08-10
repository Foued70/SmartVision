set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})


include_directories(${SmartVision_SOURCE_DIR}/common)
include_directories(${SmartVision_SOURCE_DIR}/acquisition)
include_directories(${SmartVision_SOURCE_DIR}/faceDetection)

link_directories(${SmartVision_BINARY_DIR}/common)
link_directories(${SmartVision_BINARY_DIR}/acquisition)
link_directories(${SmartVision_BINARY_DIR}/faceDetection)





set(SOURCE_FILES main.cpp)
find_package (Threads)
# create executable
add_executable(default ${SOURCE_FILES})
add_executable(configure configure.cpp)
target_link_libraries (default ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(default common acquisition faceDetection ${OpenCV_LIBS})
target_link_libraries(configure common acquisition faceDetection ${OpenCV_LIBS})
set_target_properties(common PROPERTIES LINKER_LANGUAGE CXX) 
set_target_properties(configure PROPERTIES LINKER_LANGUAGE CXX) 	
