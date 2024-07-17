/*
 * view_morphing.h - view morphing algorithm inplementation
 *
 * Author: Zong Wei <zongwave@hotmail.com>
 */

#ifndef _FRAME_VIEW_MORPHING_H_
#define _FRAME_VIEW_MORPHING_H_

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>

#include <calibration_parser.h>
#include <image_projector.h>
#include <interface/blender.h>

#include "freeview_synthesizer.h"

using namespace std;
using namespace XCam;

#define DEBUG_MORPH 0

typedef struct _XCamMorphInfo {
      uint32_t camera_id;
      XCam::CalibrationInfo cam_calib;
      cv::Mat intrinsic;
      cv::Mat rotation;
      cv::Mat translation;
      cv::Mat distortion;
      cv::Mat fundamental;
      cv::Mat essential;
      cv::Mat homo;

      cv::Mat orig_image;
      cv::Mat rectify_image;
      cv::Size rectified_image_size;
      cv::Mat prewarp_image;
      cv::Mat morph_image;

      uint32_t key_point_size;
      uint32_t feature_size;
      std::vector<cv::Point2f> matched_points;
      std::vector<cv::KeyPoint> key_points;
      std::vector<cv::Point2f> rec_points;

      std::vector<cv::Point3f> lines;
} XCamMorphInfo;

class ViewMorphing
    : public FreeviewSynthesizer
{
public:
    explicit ViewMorphing ();
    virtual ~ViewMorphing ();

    bool synthesis_buffers (const std::vector<cv::Mat>& in_bufs, cv::Mat& out_image, bool enable_blend = false, bool update_inputs = true);
    void set_blend_factors (float shape_ratio = 0.5, float color_ratio = -1) {
        _shape_ratio = shape_ratio;
        _color_ratio = color_ratio;
    }

    void set_pyramid_blender (float blend_ratio, uint32_t levels);

    XCam::Mat3d cal_essential_matrix (XCam::Mat3d fundamental, XCam::Mat3d intrinsic);
    bool cal_fundamental_matrix (std::vector<cv::Point2f> points0, std::vector<cv::Point2f> points1);

protected:
    void feature_match (cv::Mat &img_left, cv::Mat &img_right);
    void pre_warp ();
    void pre_warp_uncalibrated ();

    //!
    //! \brief    This function calculate the size of rectified image automatically
    //!
    //! \param    cv::Mat&   h0,         input, Homography matrix of image0
    //! \param    cv::Mat&   h1,         input, Homography matrix of image1
    //!
    //! \return   void
    //!
    void cal_rectified_size (cv::Mat& h0, cv::Mat& h1);

    void morph (float shape_ratio, float color_ratio);
    bool blend_images (cv::Mat& left_image, cv::Mat& right_image, cv::Mat& out_image);
    bool post_warp (cv::Mat& out_image);

private:
    void set_calibration (XCamMorphInfo &morph_info);

    void morph_points (const std::vector<cv::Point2f>& points0,
        const std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& output_points,
        float s = 0.5f);

    void get_triangle_vertices (const cv::Subdiv2D& sub_div,
        const std::vector<cv::Point2f>& points,
        std::vector<cv::Vec3i>& triangle_vertices);
    void trans_triangler_points (const std::vector<cv::Vec3i>& triangle_vertices,
        const std::vector<cv::Point2f>& points,
        std::vector<std::vector<cv::Point2f>>& triangler_pts);

    void solve_homography (const std::vector<cv::Point2f>& left_points,
        const std::vector<cv::Point2f>& right_points,
        cv::Mat& homography);
    void solve_homography (const std::vector<std::vector<cv::Point2f>>& left_points,
        const std::vector<std::vector<cv::Point2f>>& right_points,
        std::vector<cv::Mat>& homographys);
    cv::Mat cartesian_to_homogeneous (const std::vector<cv::Point2f>& pts);

    void morph_homography (const cv::Mat& h_mat,
        cv::Mat& h_mat0,
        cv::Mat& h_mat1,
        float blend_ratio);
    void morph_homography (const std::vector<cv::Mat>& h_mats,
        std::vector<cv::Mat>& h_mats0,
        std::vector<cv::Mat>& h_mats1,
        float blend_ratio);

    void create_map (const cv::Mat& triangle_map,
        const std::vector<cv::Mat>& h_mats,
        cv::Mat& map_x,
        cv::Mat& map_y);

    void paint_triangles (cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& triangles);
    void draw_triangles (cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& triangles);

    void norm_det (cv::Mat src, cv::Mat &dst);

private:
    float _shape_ratio;
    float _color_ratio;
    bool _enable_blend;
    XCam::SmartPtr<ImageProjector> _projector;
    XCam::SmartPtr<Blender> _blender;

    XCamMorphInfo _morph_info[2];
};

#endif

