/*
 * freeview_synthesizer.h - freeview synthsis abstract interface
 *
 * Author: Zong Wei <zongwave@hotmail.com>
 */

#ifndef _FREEVIEW_SYNTHESIZER_H_
#define _FREEVIEW_SYNTHESIZER_H_

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

class FreeviewSynthesizer
{
public:
    explicit FreeviewSynthesizer () {};
    virtual ~FreeviewSynthesizer () {};

    virtual bool synthesis_buffers (const std::vector<cv::Mat>& in_images, cv::Mat& out_image, bool enable_blend = false, bool updat_inputs = true) = 0;
    virtual void set_blend_factors (float shape_ratio, float color_ratio = -1) = 0;

protected:

private:

};

#endif
