/*
 * MIT License

Copyright (c) 2020 Zzh2000

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/*
 * view_morphing.cpp - view morphing algorithm inplementation
 *
 * Author: Zong Wei <zongwave@hotmail.com>
 */

#if !defined CREATE_MAP_USE_SIMD && !defined CREATE_MAP_USE_PLAIN_CODE
#warning "None of CREATE_MAP_USE_SIMD or CREATE_MAP_USE_PLAIN_CODE defined, SIMD will be used!"
#define CREATE_MAP_USE_SIMD
#endif

#if defined CREATE_MAP_USE_SIMD && defined CREATE_MAP_USE_PLAIN_CODE
#warning "Both CREATE_MAP_USE_SIMD and CREATE_MAP_USE_PLAIN_CODE defined, SIMD will be used!"
#undef CREATE_MAP_USE_PLAIN_CODE
#endif

#if defined CREATE_MAP_USE_SIMD
#include <immintrin.h>
#endif

#include "view_morphing.h"
#include "frame_interpolator.h"


#if defined CREATE_MAP_USE_SIMD
#define DEBUG_AVX_PS(num,m,msg)                                         \
                        {                                               \
                                float tmp[num/32];                      \
                                memcpy(tmp,&m,sizeof(tmp));             \
                                printf(msg"\n");                        \
                                for(int a=0;a<num/32;a++)               \
                                        printf("%f\t",tmp[a]);          \
                                printf("\n");                           \
                        }

#define DEBUG_AVX_EPI(num,m,msg)                                        \
                        {                                               \
                                int tmp[num/32];                        \
                                memcpy(tmp,&m,sizeof(tmp));             \
                                printf(msg"\n");                        \
                                for(int a=0;a<num/32;a++)               \
                                        printf("%d\t",tmp[a]);          \
                                printf("\n");                           \
                        }

#define PRINT_M512_I(m,msg)     DEBUG_AVX_EPI(512,m,msg)
#define PRINT_M512_S(m,msg)     DEBUG_AVX_PS(512,m,msg)
#define PRINT_M128_S(m,msg)     DEBUG_AVX_PS(128,m,msg)
#endif

using namespace cv;

ViewMorphing::ViewMorphing ()
    : _shape_ratio (0.5)
    , _color_ratio (-1)
{
}

ViewMorphing::~ViewMorphing ()
{
}

void
ViewMorphing::set_pyramid_blender (float blend_ratio, uint32_t levels)
{
    if (_blender.ptr ()) {
        _blender.release ();
    }
    _blender = Blender::create_soft_blender ();
}

void
ViewMorphing::feature_match (cv::Mat &img_left, cv::Mat &img_right)
{
    cv::resize(img_left, img_left, cv::Size(0,0), 0.5, 0.5, cv::INTER_LINEAR);
    cv::resize(img_right, img_right, cv::Size(0,0), 0.5, 0.5, cv::INTER_LINEAR);

    cv::medianBlur( img_left, img_left, 3);
    cv::medianBlur( img_right, img_right, 3);

    std::vector<cv::Point2f> left_corners, right_corners;
    std::vector<float> err;
    std::vector<uchar> status;

    left_corners.clear ();
    right_corners.clear ();
    
    // shi-tomashi
    cv::goodFeaturesToTrack (img_left, left_corners, 200, 0.03, 7.5);
    cv::cornerSubPix(img_left,left_corners,cv::Size (10, 10),cv::Size (-1, -1),
                    cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03));

    // OpticalFlow
    cv::calcOpticalFlowPyrLK (
        img_left, img_right, left_corners, right_corners, status, err, cv::Size (21, 21),
        3, cv::TermCriteria (cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20, 0.3f));

    _morph_info[0].matched_points.clear();
    _morph_info[1].matched_points.clear();

    for (uint32_t i = 0; i < status.size (); i++) {
        if (!status[i] || std::fabs(left_corners[i].y - right_corners[i].y) >= 10) continue;
        _morph_info[0].matched_points.push_back (cv::Point2f (left_corners[i].x * 2, left_corners[i].y * 2 ));
        _morph_info[1].matched_points.push_back (cv::Point2f (right_corners[i].x * 2, right_corners[i].y * 2));
    }
}

void
ViewMorphing::set_calibration (XCamMorphInfo &morph_info)
{
    XCam::IntrinsicParameter intrinsic = morph_info.cam_calib.intrinsic;
    morph_info.intrinsic = cv::Mat_<double>(3, 3) << (intrinsic.fx, intrinsic.skew, intrinsic.cx,
                                      0, intrinsic.fy, intrinsic.cy,
                                      0, 0, 1);

    const XCam::Vec3d euler (morph_info.cam_calib.extrinsic.roll,
                             morph_info.cam_calib.extrinsic.pitch,
                             morph_info.cam_calib.extrinsic.yaw);
    const XCam::Vec3d trans (morph_info.cam_calib.extrinsic.trans_x,
                             morph_info.cam_calib.extrinsic.trans_y,
                             morph_info.cam_calib.extrinsic.trans_z);
    XCam::Mat3d r = _projector->calc_camera_extrinsics (euler, trans);

    morph_info.rotation = cv::Mat_<double>(3, 3) << (r(0, 0), r(0, 1), r(0, 2),
                           r(1, 0), r(1, 1), r(1, 2),
                           r(2, 0), r(2, 1), r(2, 2));

    morph_info.translation = cv::Mat_<double>(1, 3) << (trans[0], trans[1], trans[2]);
}

XCam::Mat3d
ViewMorphing::cal_essential_matrix (XCam::Mat3d fundamental, XCam::Mat3d intrinsic)
{
    XCam::Mat3d essential;

    essential = intrinsic.transpose () * fundamental * intrinsic;

    XCAM_LOG_DEBUG ("Essential Matrix: ");
    for ( int i = 0; i < 3; i++ ) {
        XCAM_LOG_DEBUG ("[%d 0] = %f [%d 1] = %f [%d 2] = %f", i, essential (i, 0), i, essential (i, 1), i, essential (i, 2));
    }

    return essential;
}

bool
ViewMorphing::cal_fundamental_matrix (std::vector<cv::Point2f> points0, std::vector<cv::Point2f> points1)
{

    vector<cv::Point2f> &match_points0 = _morph_info[0].matched_points;
    vector<cv::Point2f> &match_points1 = _morph_info[1].matched_points;
    vector<uchar> status_LMEDS;

    
    cv::Mat f_mat;

    if (match_points0.size () > 0 && match_points1.size () > 0) {
#ifdef OCV_VERSION_3
        f_mat = cv::findFundamentalMat (match_points0, match_points1, status_LMEDS, CV_FM_LMEDS);
#endif

#ifdef OCV_VERSION_4
        f_mat = cv::findFundamentalMat (match_points0, match_points1, status_LMEDS, FM_LMEDS);
#endif
        if( _morph_info[0].fundamental.empty() ||
            cv::determinant(_morph_info[0].fundamental) == 0 || 
            std::abs(cv::determinant( f_mat - _morph_info[0].fundamental)) < 1e-15){

            _morph_info[0].fundamental = f_mat;
            _morph_info[1].fundamental = f_mat;
        }

    } else {
        XCAM_LOG_ERROR ("match point0:(%d) match point1(%d) ", match_points0.size (), match_points1.size ());
        return false;
    }
#ifdef DEBUG_MORPH
    vector<cv::Point2f> left_match;
    vector<cv::Point2f> right_match;
    cv::Mat mask;
    XCam::Mat3d fundamental;
    XCAM_LOG_DEBUG ("cv::findFundamentalMat fundamental matrix match points0 size:%ld, points1 size:%ld ", match_points0.size(), match_points1.size());
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                fundamental (i, j) = f_mat.at<double>(i, j);
            }
            XCAM_LOG_DEBUG ("[%d 0] = %f [%d 1] = %f [%d 2] = %f", i, fundamental (i, 0), i, fundamental (i, 1), i, fundamental (i, 2));
        }

    int pt_count = (int)points0.size ();
    int outliner_count = 0;
    for (int i = 0; i < pt_count; i++) {
        if (status_LMEDS[i] == 0) {
            outliner_count++;
        }
    }
    XCAM_LOG_DEBUG ("outliner count:%d", outliner_count);

    vector<cv::Point2f> left_inlier;
    vector<cv::Point2f> right_inlier;
    vector<cv::DMatch> inlier_matches;

    int inliner_count = pt_count - outliner_count;
    inlier_matches.resize (inliner_count);
    left_inlier.resize (inliner_count);
    right_inlier.resize (inliner_count);
    inliner_count = 0;
    for (int i = 0; i < pt_count; i++) {
        if (status_LMEDS[i] != 0) {
            left_inlier[inliner_count].x = match_points0[i].x;
            left_inlier[inliner_count].y = match_points0[i].y;
            right_inlier[inliner_count].x = match_points1[i].x;
            right_inlier[inliner_count].y = match_points1[i].y;
            inlier_matches[inliner_count].queryIdx = inliner_count;
            inlier_matches[inliner_count].trainIdx = inliner_count;
            inliner_count++;
        }
    }

    vector<cv::KeyPoint> key0 (inliner_count);
    vector<cv::KeyPoint> key1 (inliner_count);
    cv::KeyPoint::convert (left_inlier, key0);
    cv::KeyPoint::convert (right_inlier, key1);

    cv::Mat out_image;
    cv::drawMatches (_morph_info[0].orig_image, key0, _morph_info[1].orig_image, key1, inlier_matches, out_image);
    cv::imwrite ("images/match_features.jpg", out_image);


    vector<cv::Vec3f> lines0;
    cv::computeCorrespondEpilines (match_points0, 1, f_mat, lines0);
    for (int i = 0; i < lines0.size (); i++) {
        cv::line (_morph_info[1].orig_image,
                  cv::Point (0, -lines0[i][2] / lines0[i][1]),
                  cv::Point (_morph_info[1].orig_image.cols, -(lines0[i][2] + lines0[i][0] * _morph_info[1].orig_image.cols) / lines0[i][1]),
                  cv::Scalar (255, 255, 255));
    }

    vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines (match_points1, 2, f_mat, lines1);
    for (int i = 0; i < lines1.size (); i++) {
        cv::line (_morph_info[0].orig_image,
                 cv::Point (0, -lines1[i][2] / lines1[i][1]),
                 cv::Point (_morph_info[0].orig_image.cols, -(lines1[i][2] + lines1[i][0] * _morph_info[0].orig_image.cols) / lines1[i][1]),
                 cv::Scalar (255, 255, 255));
    }

    cv::imwrite ("images/left_img_epipolar_line.jpg", _morph_info[0].orig_image);
    cv::imwrite ("images/right_img_epipolar_line.jpg", _morph_info[1].orig_image);
#endif

    return true;
}

void
ViewMorphing::pre_warp ()
{
    cv::Mat R0, R1, P0, P1, Q, mapx0, mapx1, mapy0, mapy1;

    cv::stereoRectify (_morph_info[0].intrinsic, _morph_info[0].distortion,
                       _morph_info[1].intrinsic, _morph_info[1].distortion, _morph_info[0].orig_image.size (),
                       _morph_info[0].rotation, _morph_info[0].translation, R0, R1, P0, P1, Q,
                       cv::CALIB_ZERO_DISPARITY);

    cv::initUndistortRectifyMap (_morph_info[0].intrinsic, _morph_info[0].distortion, R0, P0, _morph_info[0].orig_image.size (), CV_32FC1, mapx0, mapy0);
    cv::initUndistortRectifyMap (_morph_info[1].intrinsic, _morph_info[1].distortion, R1, P1, _morph_info[1].orig_image.size (), CV_32FC1, mapx1, mapy1);

#ifdef OCV_VERSION_4
    cv::remap (_morph_info[0].orig_image, _morph_info[0].prewarp_image, mapx0, mapy0, INTER_LINEAR);
    cv::remap (_morph_info[1].orig_image, _morph_info[1].prewarp_image, mapx1, mapy1, INTER_LINEAR);
#else
    cv::remap (_morph_info[0].orig_image, _morph_info[0].prewarp_image, mapx0, mapy0, CV_INTER_LINEAR);
    cv::remap (_morph_info[1].orig_image, _morph_info[1].prewarp_image, mapx1, mapy1, CV_INTER_LINEAR);
#endif
}

void
ViewMorphing::cal_rectified_size (cv::Mat& h0, cv::Mat& h1)
{
    h0 /= cv::cubeRoot(determinant (h0));
    h1 /= cv::cubeRoot(determinant (h1));

    std::vector<cv::Point2f> obj_corners(4), scene0_corners(4), scene1_corners(4);
    obj_corners[0] = Point(0, 0);
    obj_corners[1] = Point(_morph_info[0].orig_image.cols, 0);
    obj_corners[2] = Point(_morph_info[0].orig_image.cols, _morph_info[0].orig_image.rows);
    obj_corners[3] = Point(0, _morph_info[0].orig_image.rows);

    const int count = 10;
    for(int i = 0; i < 2; i++){
        _morph_info[i].matched_points.insert(_morph_info[i].matched_points.end(), obj_corners.begin(), obj_corners.end());
        for(int j = 1; j < count; j++){
            _morph_info[i].matched_points.push_back(Point(0, _morph_info[i].orig_image.rows * j / count));
            _morph_info[i].matched_points.push_back(Point(_morph_info[i].orig_image.cols * j / count, _morph_info[i].orig_image.rows));
            _morph_info[i].matched_points.push_back(Point(_morph_info[i].orig_image.cols, _morph_info[i].orig_image.rows * j / count));
            _morph_info[i].matched_points.push_back(Point(_morph_info[i].orig_image.cols * j / count, 0));
        }
    }

    cv::perspectiveTransform(obj_corners, scene0_corners, h0);
    cv::Rect box0=cv::boundingRect(scene0_corners);
    
    cv::perspectiveTransform(obj_corners, scene1_corners, h1);
    cv::Rect box1=cv::boundingRect(scene1_corners);

    _morph_info[0].rectified_image_size.width = box0.width < box1.width ? box1.width : box0.width;
    _morph_info[0].rectified_image_size.height = box0.height < box1.height ? box1.height : box0.height;

    if( _morph_info[0].rectified_image_size.width > _morph_info[0].orig_image.cols * 1.5 ||
        _morph_info[0].rectified_image_size.height > _morph_info[0].orig_image.rows * 1.5){

        _morph_info[0].rectified_image_size.width = _morph_info[0].orig_image.cols * 1.5;
        _morph_info[0].rectified_image_size.height = _morph_info[0].orig_image.rows * 1.5;
        _morph_info[0].fundamental.setTo(0);
        _morph_info[1].fundamental.setTo(0);

    }

    _morph_info[1].rectified_image_size = _morph_info[0].rectified_image_size;
    
    vector<int> x0, x1, y0, y1;
    for(int i=0; i<4; i++){
        x0.push_back(scene0_corners[i].x);
        y0.push_back(scene0_corners[i].y);
        x1.push_back(scene1_corners[i].x);
        y1.push_back(scene1_corners[i].y);
    }

    cv::Mat rec = (cv::Mat_<double>( 3 ,3 ) << 1, 0, -*(std::min_element(x0.begin(), x0.end())),
                                               0, 1, -*(std::min_element(y0.begin(), y0.end())),
                                               0, 0, 1 );
    h0 = rec * h0;
    h0 /= cv::cubeRoot(determinant (h0));

    rec.at<double>(0, 2)= -*(std::min_element(x1.begin(), x1.end()));
    rec.at<double>(1, 2)= -*(std::min_element(y1.begin(), y1.end()));
    h1 = rec * h1;
    h1 /= cv::cubeRoot(determinant (h1));
}

void
ViewMorphing::pre_warp_uncalibrated ()
{
    Mat h0, h1;
    cv::stereoRectifyUncalibrated (_morph_info[0].matched_points,
                                   _morph_info[1].matched_points,
                                   _morph_info[0].fundamental,
                                   _morph_info[0].orig_image.size (),
                                   h0,
                                   h1);

    cal_rectified_size(h0, h1);

    _morph_info[0].homo = h0;
    _morph_info[1].homo = h1;

    _morph_info[0].rec_points.clear();
    _morph_info[1].rec_points.clear();

    cv::perspectiveTransform(_morph_info[0].matched_points, _morph_info[0].rec_points, h0);
    cv::perspectiveTransform(_morph_info[1].matched_points, _morph_info[1].rec_points, h1);

    cv::warpPerspective (_morph_info[0].orig_image, _morph_info[0].rectify_image, h0, _morph_info[0].rectified_image_size, INTER_LINEAR, BORDER_REPLICATE);
    cv::warpPerspective (_morph_info[1].orig_image, _morph_info[1].rectify_image, h1, _morph_info[1].rectified_image_size, INTER_LINEAR, BORDER_REPLICATE);

#ifdef DEBUG_MORPH
    vector<cv::KeyPoint> left_kp;
    vector<cv::KeyPoint> right_kp;

    for (int i = 0; i < _morph_info[0].matched_points.size (); i++) {
        left_kp.push_back (KeyPoint(_morph_info[0].matched_points[i], 4));
    }
    for (int i = 0; i < _morph_info[1].matched_points.size (); i++) {
        right_kp.push_back (KeyPoint(_morph_info[1].matched_points[i], 4));
    }
    cv::drawKeypoints (_morph_info[0].orig_image, left_kp, _morph_info[0].prewarp_image);
    cv::drawKeypoints (_morph_info[1].orig_image, right_kp, _morph_info[1].prewarp_image);

    cv::Mat prewarp_image0;
    cv::cvtColor (_morph_info[0].prewarp_image, prewarp_image0, COLOR_BGR2GRAY);
    cv::Mat prewarp_image1;
    cv::cvtColor (_morph_info[1].prewarp_image, prewarp_image1, COLOR_BGR2GRAY);

    //imwrite ("images/left_morph_kp.jpg", _morph_info[0].prewarp_image);
    //imwrite ("images/right_morph_kp.jpg", _morph_info[1].prewarp_image);
    imwrite ("images/left_morph_kp.jpg", prewarp_image0);
    imwrite ("images/right_morph_kp.jpg", prewarp_image1);

    imwrite ("images/left_rectified_image.jpg", _morph_info[0].rectify_image);
    imwrite ("images/right_rectified_image.jpg", _morph_info[1].rectify_image);
#endif
}

void
ViewMorphing::morph (float shape_ratio, float color_ratio)
{
    const std::vector<cv::Point2f>& points0 = _morph_info[0].rec_points;
    const std::vector<cv::Point2f>& points1 = _morph_info[1].rec_points;

    std::vector<cv::Point2f> src_points[2];
    src_points[0].insert (src_points[0].end (), points0.begin (), points0.end ());
    src_points[1].insert (src_points[1].end (), points1.begin (), points1.end ());

    // Morph points
    std::vector<cv::Point2f> morphed_points;
    morph_points (src_points[0], src_points[1], morphed_points, shape_ratio);

    // Generate Delaunay Triangles from the morphed points
    int num_points = morphed_points.size ();
    cv::Subdiv2D sub_div(cv::Rect (0, 0, _morph_info[0].rectified_image_size.width + 1, _morph_info[0].rectified_image_size.height + 1));
    for(int i=0; i< num_points; i++){
        if(morphed_points[i].x >= 0 && morphed_points[i].y >= 0 && 
        morphed_points[i].x < _morph_info[0].rectified_image_size.width + 1 && 
        morphed_points[i].y < _morph_info[0].rectified_image_size.height + 1)

        sub_div.insert (morphed_points[i]);
    }

    // Get the ID list of corners of Delaunay traiangles.
    std::vector<cv::Vec3i> triangle_indices;
    get_triangle_vertices (sub_div, morphed_points, triangle_indices);

    // Get coordinates of Delaunay corners from ID list
    std::vector<std::vector<cv::Point2f>> triangle_src[2], triangle_morph;
    trans_triangler_points (triangle_indices, src_points[0], triangle_src[0]);
    trans_triangler_points (triangle_indices, src_points[1], triangle_src[1]);
    trans_triangler_points (triangle_indices, morphed_points, triangle_morph);

    // Create a map of triangle ID in the morphed image.
    cv::Mat triangle_map = cv::Mat::zeros (_morph_info[0].rectified_image_size, CV_32SC1);
    paint_triangles (triangle_map, triangle_morph);

#ifdef DEBUG_MORPH
    cv::imwrite ("images/paint_triangles.jpg", triangle_map);

    //draw_triangles (triangle_map, triangle_morph);
    //cv::imwrite ("images/draw_triangles.jpg", triangle_map);
#endif

    // Compute Homography matrix of each triangle.
    std::vector<cv::Mat> h_mat, morph_hom[2];
    cv::Mat trans_image[2], trans_map_x, trans_map_y;

    if (_enable_blend){    
        solve_homography (triangle_src[0], triangle_src[1], h_mat);
        morph_homography (h_mat, morph_hom[0], morph_hom[1], shape_ratio);

        for (int i = 0; i < 2; i++) {
            create_map (triangle_map, morph_hom[i], trans_map_x, trans_map_y);
            cv::remap (_morph_info[i].rectify_image, trans_image[i], trans_map_x, trans_map_y, cv::INTER_LINEAR);
        }
#ifdef DEBUG_MORPH
    imwrite ("images/left_trans_image.jpg", trans_image[0]);
    imwrite ("images/right_trans_image.jpg", trans_image[1]);
#endif
        // Blend 2 input images
        float blend = (color_ratio < 0) ? shape_ratio : color_ratio;

        if (_blender.ptr ()) {
            blend_images (trans_image[0], trans_image[1], _morph_info[0].morph_image);
        } else {
            _morph_info[0].morph_image = trans_image[0] * (1.0 - blend) + trans_image[1] * blend;
        }
    }else{
        int i = shape_ratio >= 0.5 ? 1: 0;
        solve_homography (triangle_src[i], triangle_morph, morph_hom[i]);
        create_map (triangle_map, morph_hom[i], trans_map_x, trans_map_y);
        cv::remap (_morph_info[i].rectify_image, _morph_info[0].morph_image, trans_map_x, trans_map_y, cv::INTER_LINEAR);
    } 
    std::vector<cv::Point2f> dest_points; 
    dest_points.clear ();
    dest_points.insert (dest_points.end (), morphed_points.begin (), morphed_points.end ());
}

bool
ViewMorphing::blend_images (cv::Mat& left_image, cv::Mat& right_image, cv::Mat& out_image)
{
    if (NULL == _blender.ptr ()) {
        _blender = Blender::create_soft_blender ();
    }

    XCam::Rect area;
    area.pos_x = 0;
    area.pos_y = 0;
    area.width = left_image.cols;
    area.height = left_image.rows;

    _blender->set_output_size (area.width, area.height);
    _blender->set_merge_window (area);
    _blender->set_input_merge_area (area, 0);
    _blender->set_input_merge_area (area, 1);

    SmartPtr<VideoBuffer> left_buf;
    SmartPtr<VideoBuffer> right_buf;
    SmartPtr<VideoBuffer> out_buf;

    convert_mat_to_video_buffer (left_image, left_buf);
    convert_mat_to_video_buffer (right_image, right_buf);

    if (XCAM_RETURN_NO_ERROR != _blender->blend (left_buf, right_buf, out_buf)) {
        XCAM_LOG_ERROR ("blend buffer failed");
        return false;
    }
    convert_to_mat (out_buf, out_image);
    return true;
}

void
ViewMorphing::morph_points (const std::vector<cv::Point2f>& points0,
    const std::vector<cv::Point2f>& points1,
    std::vector<cv::Point2f>& output_points,
    float s)
{
    assert(points0.size () == points1.size ());

    int num_pts = points0.size ();

    output_points.resize (num_pts);
    for (uint32_t i = 0; i < num_pts; i++) {
        output_points[i].x = (1.0 - s) * points0[i].x + s * points1[i].x;
        output_points[i].y = (1.0 - s) * points0[i].y + s * points1[i].y;
    }
}

void
ViewMorphing::morph_homography (const cv::Mat& h_mat,
    cv::Mat& h_mat0,
    cv::Mat& h_mat1,
    float blend_ratio)
{
    cv::Mat inv_h = h_mat.inv ();
    h_mat0 = cv::Mat::eye (3,3,CV_32FC1) * (1.0 - blend_ratio) + h_mat * blend_ratio;
    h_mat1 = cv::Mat::eye (3,3,CV_32FC1) * blend_ratio + inv_h * (1.0 - blend_ratio);
}

void
ViewMorphing::morph_homography (const std::vector<cv::Mat>& h_mats,
    std::vector<cv::Mat>& h_mats0,
    std::vector<cv::Mat>& h_mats1,
    float blend_ratio)
{
    int hom_num = h_mats.size ();
    h_mats0.resize (hom_num);
    h_mats1.resize (hom_num);
    for (int i = 0; i < hom_num; i++) {
        morph_homography (h_mats[i], h_mats0[i], h_mats1[i], blend_ratio);
    }
}

void
ViewMorphing::get_triangle_vertices (const cv::Subdiv2D& sub_div,
                                          const std::vector<cv::Point2f>& points,
                                          std::vector<cv::Vec3i>& triangle_vertices)
{
    std::vector<cv::Vec6f> triangles;
    sub_div.getTriangleList (triangles);

    int num_triangles = triangles.size ();
    triangle_vertices.clear ();
    triangle_vertices.reserve (num_triangles);
    for (int i = 0; i < num_triangles; i++) {
        std::vector<cv::Point2f>::const_iterator vert1, vert2, vert3;
        vert1 = std::find (points.begin (), points.end (), cv::Point2f (triangles[i][0], triangles[i][1]));
        vert2 = std::find (points.begin (), points.end (), cv::Point2f (triangles[i][2], triangles[i][3]));
        vert3 = std::find (points.begin (), points.end (), cv::Point2f (triangles[i][4], triangles[i][5]));

        cv::Vec3i vertex;
        if (vert1 != points.end () && vert2 != points.end () && vert3 != points.end ()) {
            vertex[0] = vert1 - points.begin ();
            vertex[1] = vert2 - points.begin ();
            vertex[2] = vert3 - points.begin ();
            triangle_vertices.push_back (vertex);
        }
    }
}

void
ViewMorphing::trans_triangler_points (const std::vector<cv::Vec3i>& triangle_vertices,
                                      const std::vector<cv::Point2f>& points,
                                      std::vector<std::vector<cv::Point2f>>& triangler_pts)
{
    int num_triangle = triangle_vertices.size ();
    triangler_pts.resize (num_triangle);
    for (int i = 0; i < num_triangle; i++) {
        std::vector<cv::Point2f> triangle;
        for (int j = 0; j < 3; j++) {
            triangle.push_back (points[triangle_vertices[i][j]]);
        }
        triangler_pts[i] = triangle;
    }
}

void
ViewMorphing::solve_homography (const std::vector<cv::Point2f>& points0,
                                const std::vector<cv::Point2f>& points1,
                                cv::Mat& homography)
{
    assert (points0.size () == points1.size ());

    homography = cartesian_to_homogeneous (points1) * cartesian_to_homogeneous (points0).inv ();
}

void
ViewMorphing::solve_homography (const std::vector<std::vector<cv::Point2f>>& points0,
                                const std::vector<std::vector<cv::Point2f>>& points1,
                                std::vector<cv::Mat>& homographys)
{
    assert (points0.size () == points1.size ());

    int pts_num = points0.size ();
    homographys.clear ();
    homographys.reserve (pts_num);
    for (int i = 0; i < pts_num; i++) {
        cv::Mat h_mat;
        solve_homography (points0[i], points1[i], h_mat);
        homographys.push_back (h_mat);
    }
}

void
ViewMorphing::create_map (const cv::Mat& triangle_map,
    const std::vector<cv::Mat>& h_mats,
    cv::Mat& map_x,
    cv::Mat& map_y)
{
    assert (triangle_map.type () == CV_32SC1);

    // Allocate cv::Mat for the map
    map_x.create (triangle_map.size (), CV_32FC1);
    map_y.create (triangle_map.size (), CV_32FC1);

    int h_mat_array_size = h_mats.size();
    // Compute inverse matrices
    std::vector<cv::Mat> inv_h_mats (h_mat_array_size);
#if defined CREATE_MAP_USE_SIMD
    std::vector<int> h_ele_dist(h_mat_array_size);
#endif
    for (int i = 0; i < h_mats.size (); i++) {
#if defined CREATE_MAP_USE_SIMD
        inv_h_mats[i] = h_mats[i].inv ();
        h_ele_dist[i] = (char *)inv_h_mats[i].ptr<float>(0,0) - (char *)inv_h_mats[0].ptr<float>(0,0);
#else
        inv_h_mats[i] = h_mats[i].inv ();
#endif
    }
#if defined CREATE_MAP_USE_PLAIN_CODE
    for (int x = 0; x < triangle_map.cols; x++) {
        for (int y = 0; y < triangle_map.rows; y++) {
            int idx = triangle_map.at<int> (y, x) - 1;
            if (idx >= 0) {
                const cv::Mat& H = inv_h_mats[triangle_map.at<int> (y, x) - 1];
                float z = H.at<float> (2, 0) * x + H.at<float> (2, 1) * y + H.at<float> (2, 2);
                if (z == 0)
                    z = 0.00001;
                map_x.at<float> (y, x) = (H.at<float> (0, 0) * x + H.at<float> (0, 1) * y + H.at<float> (0, 2)) / z;
                map_y.at<float> (y, x) = (H.at<float> (1, 0) * x + H.at<float> (1, 1) * y + H.at<float> (1, 2)) / z;
            } else {
                map_x.at<float> (y, x) = x;
                map_y.at<float> (y, x) = y;
            }
        }
    }
#elif defined CREATE_MAP_USE_SIMD
    const __m512 vzero = _mm512_set_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    const __m512 vsub = _mm512_set_ps(0.000001,0.000001,0.000001,0.000001,
                                        0.000001,0.000001,0.000001,0.000001,
                                        0.000001,0.000001,0.000001,0.000001,
                                        0.000001,0.000001,0.000001,0.000001);
    const __m512i vonei = _mm512_set_epi32(1,1,1,1,1,1,1,1,
                                                1,1,1,1,1,1,1,1);
    const __m512 v16 = _mm512_set_ps(16,16,16,16,16,16,16,16,
                                                16,16,16,16,16,16,16,16);
    const __m512i vzeroi = _mm512_set_epi32(0,0,0,0,0,0,0,0,
                                                0,0,0,0,0,0,0,0);
    __m512i vH_offset, vH_offset_save;
    int x,y,i=0,last_x=-1,last_y=-1;
     __m512 vx,vy,vz;
    ((float *)&vy)[0]=-1;
    for (y = 0; y < triangle_map.rows; y++) {
        for (x = 0; x < triangle_map.cols;){
        #define CYCLE_REDUCE_FURTHER
        #ifndef CYCLE_REDUCE_FURTHER
            ((float *)&vx)[i]=x++;
            ((float *)&vy)[i++]=y;

            if(i<16)
                continue;
        #else
            /*
             * Optimization based on row-crossing iteration or not
             * ppppp...pppuuuuuuuuuuuuuuuu......[eol] -> fast path
             * pppppppp...ppp...ppppppppuuuuuuuu[eol] -> slow path
             * p:processed
             * u:unprocessed
             * */
            if(last_x>triangle_map.cols-16) {
                /* slow path */
                ((float *)&vx)[i]=x++;
                ((float *)&vy)[i++]=y;
                if(i<16)
                    continue;
            }
            else {
                /* fast path */
                if(((float *)&vy)[0]!=y) {
                    /* fist time solely hit a new row */
                    vx=_mm512_set_ps(x+15,x+14,x+13,x+12,x+11,x+10,x+9,x+8,
                                      x+7,x+6,x+5,x+4,x+3,x+2,x+1,x);
                    vy=_mm512_set_ps(y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y);
                } else {
                    vx=_mm512_add_ps(vx,v16);
                    // vy doesn't change
                }
                x+=16;
            }
        #endif

            /* reset */
            i=0;last_x=x,last_y=y;

            /* We got 16 elements */
            __m512 vmap_x, vmap_y;

            __m512 vH00, vH01, vH02;
            __m512 vH10, vH11, vH12;
            __m512 vH20, vH21, vH22;

            __mmask16 mask,mask_idx;

            int row = ((float *)&vy)[0];
            int col = ((float *)&vx)[0];
            __m512i vidxi = _mm512_load_epi32(triangle_map.ptr<int>(row, col));
            vidxi = _mm512_sub_epi32(vidxi, vonei); //idx = triangle_map.at<int> (y,x) -1
            mask_idx = _mm512_cmp_epi32_mask(vidxi, vzeroi, _MM_CMPINT_LT); // idx < 0
            vidxi = _mm512_mask_blend_epi32(mask_idx, vidxi, vzeroi);
            vH_offset = _mm512_i32gather_epi32(vidxi, h_ele_dist.data(), 4);
            mask = _mm512_cmp_epi32_mask(vH_offset,vH_offset_save,_MM_CMPINT_NE);
            if( mask || (row==0 && col==0)) {
                vH_offset_save = vH_offset;
                //PRINT_M512_I(vidxi, "index:");
                //PRINT_M512_I(vH_offset, "offset:");
                vH00 = _mm512_mask_i32gather_ps(vH00, mask, vH_offset, inv_h_mats[0].ptr<float>(0,0), 1);
                vH01 = _mm512_mask_i32gather_ps(vH01, mask, vH_offset, inv_h_mats[0].ptr<float>(0,1), 1);
                vH02 = _mm512_mask_i32gather_ps(vH02, mask, vH_offset, inv_h_mats[0].ptr<float>(0,2), 1);
                vH10 = _mm512_mask_i32gather_ps(vH10, mask, vH_offset, inv_h_mats[0].ptr<float>(1,0), 1);
                vH11 = _mm512_mask_i32gather_ps(vH11, mask, vH_offset, inv_h_mats[0].ptr<float>(1,1), 1);
                vH12 = _mm512_mask_i32gather_ps(vH12, mask, vH_offset, inv_h_mats[0].ptr<float>(1,2), 1);
                vH20 = _mm512_mask_i32gather_ps(vH20, mask, vH_offset, inv_h_mats[0].ptr<float>(2,0), 1);
                vH21 = _mm512_mask_i32gather_ps(vH21, mask, vH_offset, inv_h_mats[0].ptr<float>(2,1), 1);
                vH22 = _mm512_mask_i32gather_ps(vH22, mask, vH_offset, inv_h_mats[0].ptr<float>(2,2), 1);
            }
            //PRINT_M512_S(vH00,"vH00: "); PRINT_M512_S(vH01,"vH01: "); PRINT_M512_S(vH02,"vH02: ");
            //PRINT_M512_S(vH10,"vH10: "); PRINT_M512_S(vH11,"vH11: "); PRINT_M512_S(vH12,"vH12: ");
            //PRINT_M512_S(vH20,"vH20: "); PRINT_M512_S(vH21,"vH21: "); PRINT_M512_S(vH22,"vH22: ");

            // float z = H.at<float> (2, 0) * x + H.at<float> (2, 1) * y + H.at<float> (2, 2);
            vz = _mm512_fmadd_ps(vH20, vx, vH22);
            vz = _mm512_fmadd_ps(vH21, vy, vz);

            // Handle the case where z = 0
            mask = _mm512_cmp_ps_mask(vz, vzero, _MM_CMPINT_EQ);
            vz = _mm512_mask_blend_ps(mask, vz, vsub);

            // map_x.at<float> (y, x) = (H.at<float> (0, 0) * x + H.at<float> (0, 1) * y + H.at<float> (0, 2)) / z;
            vmap_x = _mm512_fmadd_ps(vH00, vx, vH02);
            vmap_x = _mm512_fmadd_ps(vH01, vy, vmap_x);
            vmap_x = _mm512_div_ps(vmap_x, vz);
            // map_y.at<float> (y, x) = (H.at<float> (1, 0) * x + H.at<float> (1, 1) * y + H.at<float> (1, 2)) / z;
            vmap_y = _mm512_fmadd_ps(vH10, vx, vH12);
            vmap_y = _mm512_fmadd_ps(vH11, vy, vmap_y);
            vmap_y = _mm512_div_ps(vmap_y, vz);
            //handle the case where idx < 0
            vmap_x = _mm512_mask_blend_ps(mask_idx,vmap_x,vx);
            vmap_y = _mm512_mask_blend_ps(mask_idx,vmap_y,vy);
            // Unpack vmap_x, vmap_y, and store to map_x, map_y
            _mm512_store_ps((void *)map_x.ptr<float>(row,col), vmap_x);
            _mm512_store_ps((void *)map_y.ptr<float>(row,col), vmap_y);
        }
    }
    /*
     * Finish what was left out, reversely
     * We know that cols must be greater than 16
     * So all the remnant is on the last row
     * */
    for(y--,x--;x>triangle_map.cols-1-i;x--) {
        int idx = triangle_map.at<int> (y,x) - 1;
        if (idx >= 0) {
            const cv::Mat &H = inv_h_mats[idx];
            //cv::Mat &H = inv_h_mats.at(triangle_map.at<int> (y, x) - 1);
            float z = H.at<float> (2, 0) * x + H.at<float> (2, 1) * y + H.at<float> (2, 2);
            if (z == 0)
                z = 0.00001;
            map_x.at<float> (y, x) = (H.at<float> (0, 0) * x + H.at<float> (0, 1) * y + H.at<float> (0, 2)) / z;
            map_y.at<float> (y, x) = (H.at<float> (1, 0) * x + H.at<float> (1, 1) * y + H.at<float> (1, 2)) / z;
        } else {
            map_x.at<float> (y, x) = x;
            map_y.at<float> (y, x) = y;
        }

    }
#endif
}

bool
ViewMorphing::post_warp (cv::Mat& out_image)
{
    cv::Mat &morph_image = _morph_info[0].morph_image;
    cv::Mat th_inv = (_shape_ratio * _morph_info[1].homo + (1 - _shape_ratio) * _morph_info[0].homo).inv ();
    cv::warpPerspective (morph_image, morph_image, th_inv, morph_image.size (), INTER_LINEAR, BORDER_REPLICATE);
    cv::Rect rect (0, 0, _morph_info[0].orig_image.cols, _morph_info[0].orig_image.rows);
    out_image = cv::Mat (morph_image, rect);
#ifdef DEBUG_MORPH
    static uint32_t id = 0;
    char name_str[64] = {'\0'};
    std::snprintf (name_str, 64, "images/morph_image_%03d_%03f.jpg", id++, _shape_ratio);
    imwrite (name_str, out_image);
#endif
    return true;
}

bool
ViewMorphing::synthesis_buffers (const std::vector<cv::Mat> &in_images, cv::Mat& out_image,  bool enable_blend, bool update_inputs)
{
    _enable_blend = enable_blend;

    _morph_info[0].orig_image = in_images[0];
    _morph_info[1].orig_image = in_images[1];

    cv::Mat img_left, img_right;
    cv::cvtColor(_morph_info[0].orig_image, img_left, COLOR_BGR2GRAY);
    cv::cvtColor(_morph_info[1].orig_image, img_right, COLOR_BGR2GRAY);

    if (update_inputs) {
        feature_match (img_left, img_right);

        cal_fundamental_matrix (_morph_info[0].matched_points, _morph_info[0].matched_points);
    }

    if (_morph_info[0].matched_points.size () == 0 ||
        _morph_info[1].matched_points.size () == 0 ||
        _morph_info[0].matched_points.size () != _morph_info[1].matched_points.size ()) {
        XCAM_LOG_ERROR ("feature matched points size 0 !! \n");
        return false;
    }

    pre_warp_uncalibrated ();
    
    morph (_shape_ratio, _color_ratio);

    if (false == post_warp (out_image)) {
        return false;
    }

    return true;
}

void
ViewMorphing::paint_triangles (cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& triangles)
{
    int num_triangle = triangles.size ();

    for (int i = 0; i < num_triangle; i++) {
        std::vector<cv::Point> poly (3);

        for (int j = 0; j < 3; j++) {
            poly[j] = cv::Point (cvRound (triangles[i][j].x), cvRound (triangles[i][j].y));
        }
        cv::fillConvexPoly (image, poly, cv::Scalar (i+1));
    }
}

void
ViewMorphing::draw_triangles (cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& triangles)
{
    int num_triangle = triangles.size ();

    std::vector<std::vector<cv::Point>> polies;
    for (int i = 0; i < num_triangle; i++) {
        std::vector<cv::Point> poly (3);

        for (int j = 0; j < 3; j++) {
            poly[j] = cv::Point (cvRound (triangles[i][j].x), cvRound (triangles[i][j].y));
        }
        polies.push_back (poly);
    }
    cv::polylines (image, polies, true, cv::Scalar (255, 0, 255));
}

void ViewMorphing::norm_det (cv::Mat src, cv::Mat &dst) {

    if (determinant (src) < 0) {
        dst = src / -pow (-determinant (src), 1.0 / src.rows);
    } else {
        dst = src / pow (determinant (src), 1.0 / src.rows);
    }
}

cv::Mat
ViewMorphing::cartesian_to_homogeneous (const std::vector<cv::Point2f>& pts)
{
    int num_pts = pts.size ();
    cv::Mat h_mat (3, num_pts, CV_32FC1);
    for (int i = 0; i < num_pts; i++) {
        h_mat.at<float> (0,i) = pts[i].x;
        h_mat.at<float> (1,i) = pts[i].y;
        h_mat.at<float> (2,i) = 1.0;
    }
    return h_mat;
}
