/*
 * frame_interpolator.cpp - view morphing frame interpolator
 *
 * Author: Zong Wei <zongwave@hotmail.com>
 */

#include "frame_interpolator.h"
#include "view_morphing.h"
#include "freeview_synthesizer.h"

using namespace cv;

#define CAMERA_CALIBRATON_FILE_PATH "data/paras.txt"

Interpolator::Interpolator (SyntheizerModule module)
{
    create_synthesizer (module);

    _projector = new ImageProjector ();

    _calib_parser = new CalibrationParser ();
    _calib_info.resize (INPUT_COUNT);

    _interp_count = 2;
    _interp_index = 0;

    if (_calib_parser.ptr ()) {
        _calib_parser->parse_calib_file (CAMERA_CALIBRATON_FILE_PATH, _calib_info, INPUT_COUNT);
    }

    for (uint32_t cam_idx = 0; cam_idx < _calib_info.size (); cam_idx++) {
        XCAM_LOG_DEBUG ("camera id:%d", _calib_info[cam_idx].camera_id);
        XCAM_LOG_DEBUG ("intrinsic:");
        XCAM_LOG_DEBUG ("cx:%f, cy:%f", _calib_info[cam_idx].intrinsic.cx, _calib_info[cam_idx].intrinsic.cy);
        XCAM_LOG_DEBUG ("fx:%f, fy:%f", _calib_info[cam_idx].intrinsic.fx, _calib_info[cam_idx].intrinsic.fy);
        XCAM_LOG_DEBUG ("skew:%f", _calib_info[cam_idx].intrinsic.skew);
        XCAM_LOG_DEBUG ("extrinsic:");
        XCAM_LOG_DEBUG ("tx:%f, ty:%f, tz:%f", _calib_info[cam_idx].extrinsic.trans_x, _calib_info[cam_idx].extrinsic.trans_y, _calib_info[cam_idx].extrinsic.trans_z);
        XCAM_LOG_DEBUG ("pitch:%f, yaw:%f, roll:%f", _calib_info[cam_idx].extrinsic.pitch, _calib_info[cam_idx].extrinsic.yaw, _calib_info[cam_idx].extrinsic.roll);
    }

    for (uint32_t cam_idx = 0; cam_idx < _calib_info.size (); cam_idx++) {
        //calc_projective (cam_idx);
    }
}

Interpolator::~Interpolator ()
{
    if (_projector.ptr ()) {
        _projector.release ();
    }
}

bool
Interpolator::create_synthesizer (SyntheizerModule module)
{
    switch (module) {
    case SyntheizerViewMorphing :
    default :
        _freeview_synthesizer = new ViewMorphing ();
        XCAM_ASSERT (_freeview_synthesizer.ptr ());
    break;

    }
    return true;
}

bool
Interpolator::process_frames (VideoBufferList& in_buffers, std::vector<XCam::SmartPtr<VideoBuffer>>& out_buffers)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_interp_index >= 0 && _interp_index < in_buffers.size () - 1);
    XCAM_ASSERT (_interp_count > 0);

    _in_bufs.clear ();
    uint32_t count = 0;
    for (VideoBufferList::const_iterator iter = in_buffers.begin (); iter != in_buffers.end (); ++iter) {
        if (count == _interp_index || count == _interp_index + 1) {
            SmartPtr<VideoBuffer> buf = *iter;
            XCAM_ASSERT (buf.ptr ());
            _in_bufs.push_back (buf);
        }
    }

    //XCam::Mat3d extrinsic_mat0 = calc_camera_extrinsics (0);
    //XCam::Mat3d intrinsic_mat0 = calc_camera_intrinsics (0);
    //XCam::Mat3d extrinsic_mat1 = calc_camera_extrinsics (1);
    //XCam::Mat3d intrinsic_mat1 = calc_camera_intrinsics (1);

    bool update_inputs = true;
    bool enable_blend = false;
    for (int idx = 0; idx < _interp_count; idx++) {
        float shape_ratio = (1.0f / (_interp_count + 1)) * (idx + 1);

        _freeview_synthesizer->set_blend_factors (shape_ratio);

        std::vector<cv::Mat> in_images;
        in_images.push_back (cv::Mat());
        in_images.push_back (cv::Mat());

        cv::Mat bgr_mat;

        convert_to_mat (_in_bufs[0], in_images[0]);
        convert_to_mat (_in_bufs[1], in_images[1]);

        if (false == _freeview_synthesizer->synthesis_buffers (in_images, bgr_mat, enable_blend, update_inputs)) {
            XCAM_LOG_WARNING ("Morph buffer failed! \n");
            return false;
        }

        if (bgr_mat.cols > 0 && bgr_mat.rows > 0) {
            write_frame (out_buffers[idx], bgr_mat, idx);
        } else {
            XCAM_LOG_WARNING ("Morph buffer return NULL frame! \n");
            return false;
        }
        update_inputs = false;
    }

    return true;
}

XCamReturn
Interpolator::write_frame (SmartPtr<VideoBuffer> &video_buf, cv::Mat &frame, uint32_t interp_index)
{
    if (!video_buf.ptr ()) {
        XCAM_LOG_ERROR ("interp buffer NULL!");

        return XCAM_RETURN_ERROR_PARAM;
    }

    convert_mat_to_video_buffer (frame, video_buf);

    return XCAM_RETURN_NO_ERROR;
}

XCam::Mat3d
Interpolator::calc_camera_intrinsics (uint32_t cam_idx)
{
    XCam::Mat3d intrinsic;
    if (_projector.ptr () && cam_idx < INPUT_COUNT) {
        double focal_x = _calib_info[cam_idx].intrinsic.fx;
        double focal_y = _calib_info[cam_idx].intrinsic.fy;
        double offset_x = _calib_info[cam_idx].intrinsic.cx;
        double offset_y = _calib_info[cam_idx].intrinsic.cy;
        double skew = _calib_info[cam_idx].intrinsic.skew;

        intrinsic = _projector->calc_camera_intrinsics (focal_x, focal_y, offset_x, offset_y, skew);
    }

    return intrinsic;
}

XCam::Mat3d
Interpolator::calc_camera_extrinsics (uint32_t cam_idx)
{
    XCam::Mat3d extrinsic;
    if (_projector.ptr () && cam_idx < INPUT_COUNT) {
        XCam::Vec3d euler_angles (_calib_info[cam_idx].extrinsic.pitch,
                                  _calib_info[cam_idx].extrinsic.yaw,
                                  _calib_info[cam_idx].extrinsic.roll);

        XCAM_LOG_DEBUG ("euler angles pitch:%f, yaw:%f, roll:%f", euler_angles[0], euler_angles[1], euler_angles[2]);

        XCam::Vec3d translation (_calib_info[cam_idx].extrinsic.trans_x,
                                 _calib_info[cam_idx].extrinsic.trans_y,
                                 _calib_info[cam_idx].extrinsic.trans_z);

        extrinsic = _projector->calc_camera_extrinsics (euler_angles, translation);
    }

    return extrinsic;
}

XCam::Mat3d
Interpolator::calc_projective (uint32_t cam_idx)
{
    XCam::Mat3d extrinsic = calc_camera_extrinsics (cam_idx);
    XCam::Mat3d intrinsic = calc_camera_intrinsics (cam_idx);

    return extrinsic * intrinsic;
}

cv::Mat
Interpolator::get_projection_matrix (int idx)
{
    return cv::Mat();
}
