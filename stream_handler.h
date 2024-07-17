/*
 * stream_handler.h - video stream handler
 *
 * Author: Zong Wei <zongwave@hotmail.com>
 */

#ifndef _STREAM_HANDLER_H_
#define _STREAM_HANDLER_H_

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <xcam/buffer_pool.h>
#include <xcam/image_file.h>

using namespace std;
using namespace XCam;

#define INPUT_COUNT  4

enum StreamFormat {
    FormatNone,
    FormatNV12,
    FormatYUV420,
    FormatVideo,
    FormatImage,
};

static bool convert_to_mat (const SmartPtr<VideoBuffer> &buffer, cv::Mat &img)
{
    VideoBufferInfo info = buffer->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR,
        (info.format == V4L2_PIX_FMT_NV12) ||
        (info.format == V4L2_PIX_FMT_BGR24) ||
        (info.format == V4L2_PIX_FMT_RGB24) ||
        (info.format == V4L2_PIX_FMT_YUV420),
        false,
        "convert_to_mat only support NV12 & BGR24 & RGB24 format");

    uint8_t *mem = buffer->map ();
    XCAM_FAIL_RETURN (ERROR, mem, false, "convert_to_mat buffer map failed");

    if (info.format == V4L2_PIX_FMT_NV12) {
        cv::Mat mat = cv::Mat (info.aligned_height * 3 / 2, info.width, CV_8UC1, mem, info.strides[0]);
        cv::cvtColor (mat, img, cv::COLOR_YUV2BGR_NV12);
    } else if (info.format == V4L2_PIX_FMT_YUV420) {
        cv::Mat mat = cv::Mat (info.aligned_height * 3 / 2, info.width, CV_8UC1, mem, info.strides[0]);
        cv::cvtColor (mat, img, cv::COLOR_YUV2BGR_I420);
    } else if ((info.format == V4L2_PIX_FMT_BGR24) || (info.format == V4L2_PIX_FMT_RGB24)) {
        img = cv::Mat (info.aligned_height, info.width, CV_8UC3, mem);
    }

    buffer->unmap ();

    return true;
}

static void convert_I420_to_NV12 (const cv::Mat& input, cv::Mat& output)
{
    int width = input.cols;
    int height = input.rows * 2 / 3;

    //Rows bytes stride - in most cases equal to width
    int stride = (int)input.step[0];

    input.copyTo (output);

    //Y Channel
    // YYYYYYYYYYYYYYYY
    // YYYYYYYYYYYYYYYY
    // YYYYYYYYYYYYYYYY
    // YYYYYYYYYYYYYYYY
    // YYYYYYYYYYYYYYYY
    // YYYYYYYYYYYYYYYY

    // Input U color channel (in I420 U is above V).
    // UUUUUUUU
    // UUUUUUUU
    // UUUUUUUU
    cv::Mat in_u = cv::Mat (cv::Size (width / 2, height / 2), CV_8UC1, (unsigned char*)input.data + stride * height, stride / 2);

    //Input V color channel (in I420 V is below U).
    // VVVVVVVV
    // VVVVVVVV
    // VVVVVVVV
    cv::Mat in_v = cv::Mat(cv::Size(width / 2, height / 2), CV_8UC1, (unsigned char*)input.data + stride * height + (stride / 2) * (height / 2), stride / 2);

    for (int row = 0; row < height / 2; row ++) {
        for (int col = 0; col < width / 2; col ++) {
            output.at<uchar>(height + row, 2 * col) = in_u.at<uchar>(row, col);
            output.at<uchar>(height + row, 2 * col + 1) = in_v.at<uchar>(row, col);
        }
    }
}

static XCamReturn convert_mat_to_video_buffer (const cv::Mat &rgb_mat, SmartPtr<VideoBuffer> &video_buf)
{
    if (!video_buf.ptr ()) {
        XCAM_LOG_ERROR ("output buffer NULL!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    const VideoBufferInfo &info = video_buf->get_video_info ();
    VideoBufferPlanarInfo planar;

    uint8_t* memory = video_buf->map ();
    if (NULL == memory) {
        XCAM_LOG_ERROR ("map buffer failed");
        video_buf->unmap ();
        return XCAM_RETURN_ERROR_MEM;
    }

    cv::Mat yuv_mat;
    cv::cvtColor (rgb_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t* ptr = yuv_mat.data;

    cv::Mat nv12_mat;
    if (info.format == V4L2_PIX_FMT_NV12) {
        convert_I420_to_NV12 (yuv_mat, nv12_mat);
        ptr = nv12_mat.data;
    }

    //char yuv_filename[20];
    //static uint32_t index = 0;
    //sprintf (yuv_filename, "yuv_mat_%d.jpg", index++);
    //cv::imwrite (yuv_filename, yuv_mat);

    for (uint32_t comp = 0; comp < info.components; comp++) {
        info.get_planar_info (planar, comp);
        uint32_t row_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t row = 0; row < planar.height; row++) {
            uint32_t offset = info.offsets [comp] + row * info.strides [comp];
            memcpy (memory + offset, ptr + info.offsets [comp] + row_bytes * row, row_bytes);
        }
    }

    video_buf->unmap ();

    return XCAM_RETURN_NO_ERROR;
}

template<class T>
T convert_cv_mat_to_xcam_mat (const cv::Mat cv_mat)
{
    T xcam_mat;

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            xcam_mat (i, j) = cv_mat.at<float>(i, j);
        }
    }

    return xcam_mat;
}

template<class T>
void convert_xcam_mat_to_cv_mat (T xcam_mat, cv::Mat& cv_mat)
{
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            cv_mat.at<float>(i, j) = xcam_mat (i, j);
        }
    }

    return;
}

class Stream {
public:
    explicit Stream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~Stream ();

    void set_buf_size (uint32_t width, uint32_t height);
    uint32_t get_frame_width () const {
        return _frame_width;
    }
    uint32_t get_frame_height () const {
        return _frame_height;
    }
    float get_frame_count () const {
        return _frame_count;
    }
    float get_fps () const {
        return _fps;
    }
    const char *get_file_name () const {
        return _file_name;
    }
    void set_file (const SmartPtr<ImageFile> &file) {
        _file = file;
    }
    void set_format (StreamFormat format) {
        _format = format;
    }
    const StreamFormat get_format () const {
        return _format;
    }

    const char *get_name () const {
        return _file_name;
    }

    const SmartPtr<BufferPool> &get_buffer_pool () const {
        return _pool;
    }

    uint32_t get_free_buffer_size () {
        return _pool->get_free_buffer_size ();
    }

    SmartPtr<VideoBuffer> &get_buf ();
    SmartPtr<VideoBuffer> get_free_buf ();

    XCamReturn estimate_file_format ();

    XCamReturn open_reader (const char *option);
    XCamReturn open_writer (const char *option);
    XCamReturn close_file ();
    XCamReturn rewind ();

    XCamReturn read_frame ();
    XCamReturn read_frame (const char* image);
    XCamReturn write_frame ();
    XCamReturn write_frame (SmartPtr<VideoBuffer> &buf);

    XCamReturn create_buf_pool (uint32_t reserve_count, uint32_t format = V4L2_PIX_FMT_NV12);

    void debug_write_image (char *img_name, char *frame_str = NULL, char *idx_str = NULL);

protected:
    void set_buf_pool (const SmartPtr<BufferPool> &pool) {
        _pool = pool;
    }

private:
    XCamReturn cv_open_writer (const char *file_name);
    void cv_write_frame (const cv::Mat &mat);

    XCamReturn cv_open_reader (const char *file_name);
    cv::Mat cv_read_frame ();

private:
    XCAM_DEAD_COPY (Stream);

private:
    char                    *_file_name;
    SmartPtr<ImageFile>      _file;

    uint32_t                 _frame_width;
    uint32_t                 _frame_height;
    float                    _frame_count;
    float                    _fps;

    SmartPtr<VideoBuffer>    _buf;
    SmartPtr<BufferPool>     _pool;

    cv::VideoWriter          _cv_writer;
    cv::VideoCapture         _cv_reader;
    StreamFormat             _format;
};

typedef std::vector<SmartPtr<Stream>> Streams;

#endif

