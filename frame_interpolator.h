/*
 * frame_interpolator.h - view morphing frame interpolator interface
 *
 * Author: Zong Wei <zongwave@hotmail.com>
 */

#ifndef _FRAME_INTEPOLATOR_H_
#define _FRAME_INTEPOLATOR_H_

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <xcam/buffer_pool.h>
#include <xcam/image_file.h>
#include <xcam/calibration_parser.h>
#include <xcam/image_projector.h>

#include "stream_handler.h"

using namespace std;
using namespace XCam;

class FreeviewSynthesizer;

enum SyntheizerModule {
    SyntheizerNone    = 0,
    SyntheizerViewMorphing,
};

class Interpolator
{
public:
    explicit Interpolator (SyntheizerModule module);
    virtual ~Interpolator ();

    bool process_frames (VideoBufferList& in_buffers, std::vector<XCam::SmartPtr<VideoBuffer>>& out_buffers);
    XCamReturn write_frame (SmartPtr<VideoBuffer> &video_buf, cv::Mat &frame, uint32_t interp_index);

    XCam::Mat3d calc_camera_intrinsics (uint32_t cam_idx);
    XCam::Mat3d calc_camera_extrinsics (uint32_t cam_idx);
    XCam::Mat3d calc_projective (uint32_t cam_idx);

    void set_interp_frame_count (float interp_count = 2) {
        _interp_count = interp_count;
    }
    void set_interp_frame_index (float interp_index = 0) {
        _interp_index = interp_index;
    }
protected:
    bool create_synthesizer (SyntheizerModule module = SyntheizerViewMorphing);
    cv::Mat get_projection_matrix (int idx);

private:
    uint32_t _interp_count;
    uint32_t _interp_index;

    std::vector<SmartPtr<VideoBuffer>> _in_bufs;
    std::vector<SmartPtr<VideoBuffer>> _out_bufs;

    SmartPtr<FreeviewSynthesizer> _freeview_synthesizer;
    SmartPtr<ImageProjector> _projector;

    SmartPtr<CalibrationParser> _calib_parser;
    std::vector<CalibrationInfo> _calib_info;
};

#endif

