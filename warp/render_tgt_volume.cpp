// PyBind11 Includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// OpenCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// OpenMP Includes
#include <omp.h>

// C++ Includes
#include <stdlib.h>
#include <vector>

namespace py = pybind11;

using namespace std;
using namespace cv;

void display_depth(const Mat map, string filename) {
    Size size = map.size();
    // crop 20 pixels
    Mat cropped = map(Rect(14,14,size.width-30,size.height-30));

    int min = 425;
    int max = 937;

    cropped = (cropped-min) * 255 / (max-min);
    Mat output;
    threshold(cropped, output,0, 255, THRESH_TOZERO);
    imwrite(filename, output);
}

float sigmoid(float x, float scale) {
	float input = (scale*x);
	return 1 / (1 + exp(-input));
}

//vector<float> render(const vector<int> &vol_shape, const vector<float> depth_values, const vector<float> original_depth, const vector<float> original_conf, const vector<float> &reference_cam, const vector<float> &target_cam, const float &scale) {
//	// grab shape of volume
//	int depth_planes = vol_shape[0];
//	int rows = vol_shape[1];
//	int cols = vol_shape[2];
//
//	// return container initialization
//	vector<float> rendered_volume(depth_planes*rows*cols, 0);
//
//	int cam_shape[3] = {2,4,4};
//
//	//Mat depth_slice = Mat::zeros(rows,cols,CV_32F);
//
//	Mat P_ref = Mat::zeros(4,4,CV_32F);
//	Mat K_ref = Mat::zeros(4,4,CV_32F);
//	Mat P_tgt = Mat::zeros(4,4,CV_32F);
//	Mat K_tgt = Mat::zeros(4,4,CV_32F);
//
////#pragma omp parallel num_threads(8)
////{
////	#pragma omp for collapse(3)
//	for (int i=0; i<cam_shape[0]; ++i) {
//		for (int j=0; j<cam_shape[1]; ++j) {
//			for (int k=0; k<cam_shape[2]; ++k) {
//				int ind = (i*cam_shape[1]*cam_shape[2]) + (j*cam_shape[2]) + (k);
//				if(i==0){
//					P_ref.at<float>(j,k) = reference_cam[ind];
//					P_tgt.at<float>(j,k) = target_cam[ind];
//				} else if(i==1) {
//					K_ref.at<float>(j,k) = reference_cam[ind];
//					K_tgt.at<float>(j,k) = target_cam[ind];
//				}
//			}
//		}
//	}
//
//	// correct the last row of the intrinsics
//	K_ref.at<float>(3,0) = 0;
//	K_ref.at<float>(3,1) = 0;
//	K_ref.at<float>(3,2) = 0;
//	K_ref.at<float>(3,3) = 1;
//
//	K_tgt.at<float>(3,0) = 0;
//	K_tgt.at<float>(3,1) = 0;
//	K_tgt.at<float>(3,2) = 0;
//	K_tgt.at<float>(3,3) = 1;
//
////} //omp parallel
//
//	// compute the rotation, translation, and camera centers for the target view
//	Mat R_tgt = P_tgt(Rect(0,0,3,3));
//	Mat z_tgt = R_tgt(Rect(0,2,3,1));
//	Mat t_tgt = P_tgt(Rect(3,0,1,3));
//	Mat C_tgt = -R_tgt.t()*t_tgt;
//
//	// compute the backwards and forwards projections
//	Mat b_proj =  P_ref.inv() * K_ref.inv();
//	Mat f_proj =  K_tgt * P_tgt;
//
////#pragma omp parallel num_threads(8)
////{
////    #pragma omp for collapse(3)
//	for (int d=0; d<depth_planes; ++d) {
//		for (int r=0; r<rows; ++r) {
//			for (int c=0; c<cols; ++c) {
//				float depth = depth_values[d];
//
//				// compute 3D world coord of back projection
//				Mat x_1(4,1,CV_32F);
//				x_1.at<float>(0,0) = depth * c;
//				x_1.at<float>(1,0) = depth * r;
//				x_1.at<float>(2,0) = depth;
//				x_1.at<float>(3,0) = 1;
//
//				Mat X_world = b_proj * x_1;
//				X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
//				X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
//				X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);
//
//				// calculate pixel location in target image
//				Mat x_2 = f_proj * X_world;
//
//				x_2.at<float>(0,0) = x_2.at<float>(0,0)/x_2.at<float>(2,0);
//				x_2.at<float>(1,0) = x_2.at<float>(1,0)/x_2.at<float>(2,0);
//
//				// take the floor to get the row and column pixel locations
//				int c_p = (int) floor(x_2.at<float>(0,0));
//				int r_p = (int) floor(x_2.at<float>(1,0));
//
//				// ignore if pixel projection falls outside the image
//				if (c_p < 0 || c_p >= cols || r_p < 0 || r_p >= rows) {
//					continue;
//				}
//
//				// calculate the projection depth from reference image plane
//				Mat diff = Mat::zeros(3,1,CV_32F);
//				diff.at<float>(0,0) = X_world.at<float>(0,0) - C_tgt.at<float>(0,0);
//				diff.at<float>(0,1) = X_world.at<float>(0,1) - C_tgt.at<float>(0,1);
//				diff.at<float>(0,2) = X_world.at<float>(0,2) - C_tgt.at<float>(0,2);
//
//				//project on z-axis of target cam
//				Mat projection = z_tgt * diff;
//				float proj_depth = projection.at<float>(0);
//
//				// calculate the current index
//				int ind = (d*rows*cols) + (r*cols) + c;
//				int proj_ind = (r_p*cols) + c_p;
//
//				float depth_diff = original_depth[proj_ind] - proj_depth;
//				float sig_output = sigmoid(depth_diff, scale);
//				if(d==54 && c==120 && r==120) {
//					cout << proj_depth << endl;
//					cout << original_depth[proj_ind] << endl;
//					cout << depth_diff << endl;
//					cout << sig_output << endl;
//					exit(0);
//				}
//
//				rendered_volume[ind] = original_conf[proj_ind] * sig_output;
//			}
//		}
//
//	}
////} //omp parallel
//
//	return rendered_volume;
//}
// all input should be list
vector<float> render_to_ref(const vector<int> &shape, const vector<float> depth_map,  const vector<float> &reference_cam, const vector<float> &target_cam) {
	// grab depth map shape (h,w)
	int rows = shape[0];
	int cols = shape[1];

	// return container initialization
	vector<float> rendered_map(2*rows*cols, 0);

	int cam_shape[3] = {2,4,4}; // 0 is P, 1 is K

	Mat P_ref = Mat::zeros(4,4,CV_32F);
	Mat K_ref = Mat::zeros(4,4,CV_32F);
	Mat P_tgt = Mat::zeros(4,4,CV_32F);
	Mat K_tgt = Mat::zeros(4,4,CV_32F);

#pragma omp parallel num_threads(8)
{
	#pragma omp for collapse(3)
	for (int i=0; i<cam_shape[0]; ++i) {
		for (int j=0; j<cam_shape[1]; ++j) {
			for (int k=0; k<cam_shape[2]; ++k) {
				int ind = (i*cam_shape[1]*cam_shape[2]) + (j*cam_shape[2]) + (k);
				if(i==0){
					P_ref.at<float>(j,k) = reference_cam[ind];
					P_tgt.at<float>(j,k) = target_cam[ind];
				} else if(i==1) {
					K_ref.at<float>(j,k) = reference_cam[ind];
					K_tgt.at<float>(j,k) = target_cam[ind];
				}
			}
		}
	}

	// correct the last row of the intrinsics
	K_ref.at<float>(3,0) = 0;
	K_ref.at<float>(3,1) = 0;
	K_ref.at<float>(3,2) = 0;
	K_ref.at<float>(3,3) = 1;

	K_tgt.at<float>(3,0) = 0;
	K_tgt.at<float>(3,1) = 0;
	K_tgt.at<float>(3,2) = 0;
	K_tgt.at<float>(3,3) = 1;

} //omp parallel

	// compute the rotation, translation, and camera centers for the target view
	Mat R_ref = P_ref(Rect(0,0,3,3));
	Mat z_ref = R_ref(Rect(0,2,3,1));
	Mat t_ref = P_ref(Rect(3,0,1,3));
	Mat C_ref = -R_ref.t()*t_ref;

	// compute the backwards and forwards projections
	Mat b_proj =  P_tgt.inv() * K_tgt.inv();
	Mat f_proj =  K_ref * P_ref;

#pragma omp parallel num_threads(8)
{
    #pragma omp for collapse(2)
	for (int r=0; r<rows; ++r) {
		for (int c=0; c<cols; ++c) {
			// calculate the current index
			int ind = (r*cols) + c;

			float depth = depth_map[ind];

			// compute 3D world coord of back projection
			Mat x_1(4,1,CV_32F);
			x_1.at<float>(0,0) = depth * c;
			x_1.at<float>(1,0) = depth * r;
			x_1.at<float>(2,0) = depth;
			x_1.at<float>(3,0) = 1;

			Mat X_world = b_proj * x_1;
			X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
			X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
			X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);
			X_world.at<float>(0,3) = X_world.at<float>(0,3) / X_world.at<float>(0,3);

			// calculate pixel location in target image
			Mat x_2 = f_proj * X_world;

			x_2.at<float>(0,0) = x_2.at<float>(0,0)/x_2.at<float>(2,0);
			x_2.at<float>(1,0) = x_2.at<float>(1,0)/x_2.at<float>(2,0);
			x_2.at<float>(2,0) = x_2.at<float>(2,0)/x_2.at<float>(2,0);

			// take the floor to get the row and column pixel locations
			int c_p = (int) floor(x_2.at<float>(0,0));
			int r_p = (int) floor(x_2.at<float>(1,0));
			
			// ignore if pixel projection falls outside the image
			if (c_p < 0 || c_p >= cols || r_p < 0 || r_p >= rows) {
				continue;
			}

			// calculate the projection depth from reference image plane
			Mat diff = Mat::zeros(3,1,CV_32F);
			diff.at<float>(0,0) = X_world.at<float>(0,0) - C_ref.at<float>(0,0);
			diff.at<float>(0,1) = X_world.at<float>(0,1) - C_ref.at<float>(0,1);
			diff.at<float>(0,2) = X_world.at<float>(0,2) - C_ref.at<float>(0,2);

			//project on z-axis of target cam
			Mat projection = z_ref * diff;
			float proj_depth = projection.at<float>(0);

			// compute projection index
			int proj_ind = (r_p*cols) + c_p;

			/* 
			 * Keep the closer (smaller) projection depth.
			 * A previous projection could have already populated the current pixel.
			 * If it is 0, no previous projection to this pixel was seen.
			 * Otherwise, we need to overwrite only if the current estimate is closer (smaller value).
			 */
			if (rendered_map[proj_ind] > 0) {
				if(rendered_map[proj_ind] > proj_depth) {
					rendered_map[proj_ind] = proj_depth;
					//rendered_map[proj_ind+(rows*cols)] = conf_map[ind];
				}
			} else {
				rendered_map[proj_ind] = proj_depth;
				//rendered_map[proj_ind+(rows*cols)] = conf_map[ind];
			}
		}
	}

} //omp parallel

	return rendered_map;
}
/*
vector<float> render_batch(const vector<int> &vol_shape, const vector<float> depth_values, const vector<float> original_depth, const vector<float> original_conf, const vector<float> &reference_cam, const vector<float> &target_cam, const float &scale, const float &shift) {
	// grab shape of volume
	int batches = vol_shape[0];
	int depth_planes = vol_shape[1];
	int rows = vol_shape[2];
	int cols = vol_shape[3];

	// return container initialization
	vector<float> rendered_volume(batches*depth_planes*rows*cols, 0);

	int cam_shape[3] = {2,4,4};

	Mat depth_slice = Mat::zeros(rows,cols,CV_32F);

    vector<Mat> P_refs(batches);
    vector<Mat> K_refs(batches);
    vector<Mat> P_tgts(batches);
    vector<Mat> K_tgts(batches);

	for (int b=0; b<batches; ++b) {
        Mat P_ref = Mat::zeros(4,4,CV_32F);
        Mat K_ref = Mat::zeros(4,4,CV_32F);
        Mat P_tgt = Mat::zeros(4,4,CV_32F);
        Mat K_tgt = Mat::zeros(4,4,CV_32F);

#pragma omp parallel num_threads(12)
{
	    #pragma omp for collapse(3)
        for (int i=0; i<cam_shape[0]; ++i) {
            for (int j=0; j<cam_shape[1]; ++j) {
                for (int k=0; k<cam_shape[2]; ++k) {
                    int ind =(b*cam_shape[0]*cam_shape[1]*cam_shape[2]) + (i*cam_shape[1]*cam_shape[2]) + (j*cam_shape[2]) + (k);
                    if(i==0){
                        P_ref.at<float>(j,k) = reference_cam[ind];
                        P_tgt.at<float>(j,k) = target_cam[ind];
                    } else if(i==1) {
                        K_ref.at<float>(j,k) = reference_cam[ind];
                        K_tgt.at<float>(j,k) = target_cam[ind];
                    }
                }
            }
        }
        // correct the last row of the intrinsics
        K_ref.at<float>(3,0) = 0;
        K_ref.at<float>(3,1) = 0;
        K_ref.at<float>(3,2) = 0;
        K_ref.at<float>(3,3) = 1;

        K_tgt.at<float>(3,0) = 0;
        K_tgt.at<float>(3,1) = 0;
        K_tgt.at<float>(3,2) = 0;
        K_tgt.at<float>(3,3) = 1;

        // store all the camera params
        P_refs[b] = P_ref;
        K_refs[b] = K_ref;
        P_tgts[b] = P_tgt;
        K_tgts[b] = K_tgt;
} //omp parallel

    }

	// compute the rotation, translation, and camera centers for the target view
    vector<Mat> R_tgts(batches);
    vector<Mat> z_tgts(batches);
    vector<Mat> t_tgts(batches);
    vector<Mat> C_tgts(batches);

	for (int b=0; b<batches; ++b) {
        R_tgts[b] = P_tgts[b](Rect(0,0,3,3));
		z_tgts[b] = R_tgts[b](Rect(0,2,3,1));
        t_tgts[b] = P_tgts[b](Rect(3,0,1,3));
        C_tgts[b] = -R_tgts[b].t()*t_tgts[b];
    }

	// compute the backwards and forwards projections
    vector<Mat> b_projs(batches);
    vector<Mat> f_projs(batches);

	for (int b=0; b<batches; ++b) {
		b_projs[b] =  P_refs[b].inv() * K_refs[b].inv();
		f_projs[b] =  K_tgts[b] * P_tgts[b];
	}

#pragma omp parallel num_threads(12)
{
    #pragma omp for collapse(4)
	for (int b=0; b<batches; ++b) {
        for (int d=0; d<depth_planes; ++d) {
            for (int r=0; r<rows; ++r) {
                for (int c=0; c<cols; ++c) {
                    float depth = depth_values[d];

                    // compute 3D world coord of back projection
                    Mat x_1(4,1,CV_32F);
                    x_1.at<float>(0,0) = depth * c;
                    x_1.at<float>(1,0) = depth * r;
                    x_1.at<float>(2,0) = depth;
                    x_1.at<float>(3,0) = 1;

                    Mat X_world = b_projs[b] * x_1;
                    X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
                    X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
                    X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);

                    // calculate pixel location in target image
                    Mat x_2 = f_projs[b] * X_world;

                    x_2.at<float>(0,0) = x_2.at<float>(0,0)/x_2.at<float>(2,0);
                    x_2.at<float>(1,0) = x_2.at<float>(1,0)/x_2.at<float>(2,0);

                    // take the floor to get the row and column pixel locations
                    int c_p = (int) floor(x_2.at<float>(0,0));
                    int r_p = (int) floor(x_2.at<float>(1,0));
                    
                    // ignore if pixel projection falls outside the image
                    if (c_p < 0 || c_p >= cols || r_p < 0 || r_p >= rows) {
                        continue;
                    }

                    // calculate the projection depth from reference image plane
                    Mat diff = Mat::zeros(3,1,CV_32F);
                    diff.at<float>(0,0) = X_world.at<float>(0,0) - C_tgts[b].at<float>(0,0);
                    diff.at<float>(0,1) = X_world.at<float>(0,1) - C_tgts[b].at<float>(0,1);
                    diff.at<float>(0,2) = X_world.at<float>(0,2) - C_tgts[b].at<float>(0,2);

					//project on z-axis of target cam
					Mat projection = z_tgts[b] * diff;
					float proj_depth = projection.at<float>(0);

                    // calculate the current index
                    int ind = (b*depth_planes*rows*cols) + (d*rows*cols) + (r*cols) + c;
                    int proj_ind = (b*rows*cols) + (r_p*cols) + c_p;

					float depth_diff = original_depth[proj_ind] - proj_depth;
					float sig_output = sigmoid(depth_diff, scale, shift);

                    rendered_volume[ind] = original_conf[proj_ind] * sig_output;
                }
            }

        }
    }
} //omp parallel

	return rendered_volume;
}
*/

PYBIND11_MODULE(render_tgt_volume, m) {
//	m.doc() = "MVS Utilities C++ Pluggin";
	// m.def("render", &render, "A function which renders a reference volume onto a target volume", py::arg("vol_shape"), py::arg("depth_values"), py::arg("original_depth"), py::arg("original_conf"), py::arg("reference_cam"), py::arg("target_cam"), py::arg("scale"));
	m.def("render_to_ref", &render_to_ref, "A function which renders a target depth map into a reference view", py::arg("shape"),py::arg("depth_map"), py::arg("reference_cam"), py::arg("target_cam"));
}
