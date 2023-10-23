cmake_minimum_required(VERSION 3.10)

project( render_tgt_volume LANGUAGES CXX )

find_package( OpenCV REQUIRED )

find_package(OpenMP)

find_package( pybind11 )
pybind11_add_module(render_tgt_volume render_tgt_volume.cpp)

target_link_libraries(render_tgt_volume PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} )
