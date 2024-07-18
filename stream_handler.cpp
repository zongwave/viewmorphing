/*
 * stream_handler.cpp - video stream handler
 *
 * Author: Zong Wei <zongwave@hotmail.com>
 */

#include "stream_handler.h"
#include <soft/soft_video_buf_allocator.h>


#define FREEVIEW_TEST_MAX_STR_SIZE 256

using namespace cv;

Stream::Stream (const char *file_name, uint32_t width, uint32_t height)
    : _file_name (NULL)
    , _frame_width (width)
    , _frame_height (height)
    , _frame_count (-1)
    , _fps (-1)
    , _format (FormatYUV420)
{
    if (file_name)
        _file_name = strndup (file_name, FREEVIEW_TEST_MAX_STR_SIZE);
}

Stream::~Stream ()
{
    if (_file.ptr ()) {
        _file->close ();
        _file.release ();
    }

    if (_file_name) {
        xcam_free (_file_name);
        _file_name = NULL;
    }
}

void
Stream::set_buf_size (uint32_t width, uint32_t height)
{
    _frame_width = width;
    _frame_height = height;
}

XCamReturn
Stream::open_reader (const char *option)
{
    XCAM_LOG_DEBUG ("Stream::open_reader filename:%s, option:%s ", _file_name, option);
    XCAM_FAIL_RETURN (
        ERROR, (_format == FormatNV12) || (_format == FormatYUV420) || (_format == FormatVideo) || (_format == FormatImage), XCAM_RETURN_ERROR_PARAM,
        "stream(%s) only support NV12 or YUV420 input format", _file_name);

    if (_format == FormatNV12 || _format == FormatYUV420) {
        if (!_file.ptr ()) {
            SmartPtr<ImageFile> file = new ImageFile ();
            XCAM_ASSERT (file.ptr ());
            _file = file;
        }

        if (_file->open (_file_name, option) != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_ERROR ("stream(%s) open failed", _file_name);
            return XCAM_RETURN_ERROR_FILE;
        }
    } else if (_format == FormatVideo || _format == FormatImage) {
        XCamReturn ret = cv_open_reader (_file_name);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "stream(%s) cv open reader failed", _file_name);
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %d", _file_name, (int)_format);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::open_writer (const char *option)
{
    XCAM_ASSERT (_format != FormatNone);

    if (_format == FormatNV12 || _format == FormatYUV420) {
        if (!_file.ptr ()) {
            SmartPtr<ImageFile> file = new ImageFile ();
            XCAM_ASSERT (file.ptr ());
            _file = file;
        }

        if (_file->open (_file_name, option) != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_ERROR ("stream(%s) open failed", _file_name);
            return XCAM_RETURN_ERROR_FILE;
        }
    } else if (_format == FormatVideo || _format == FormatImage) {
        XCamReturn ret = cv_open_writer (_file_name);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "stream(%s) cv open writer failed", _file_name);
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %d", _file_name, (int)_format);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::close_file ()
{
    return _file->close ();
}

XCamReturn
Stream::rewind ()
{
    if (_format == FormatNV12 || _format == FormatYUV420) {
        return _file->rewind ();
    } else {
        return XCAM_RETURN_NO_ERROR;
    }
}

XCamReturn
Stream::create_buf_pool (uint32_t reserve_count, uint32_t format)
{
    XCAM_ASSERT (get_frame_width () && get_frame_height ());

    VideoBufferInfo info;
    info.init (format, get_frame_width (), get_frame_height ());

    SmartPtr<BufferPool> pool;
    pool = new SoftVideoBufAllocator (info);
    XCAM_ASSERT (pool.ptr ());

    if (!pool->reserve (reserve_count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::read_frame ()
{
    XCAM_ASSERT (_pool.ptr ());

    _buf = _pool->get_buffer (_pool);
    XCAM_ASSERT (_buf.ptr ());

    if (_format == FormatNV12 || _format == FormatYUV420) {
        return _file->read_buf (_buf);
    } else if (_format == FormatVideo || _format == FormatImage) {
        cv::Mat rgb_mat = cv_read_frame ();
        if (rgb_mat.rows == _frame_height && rgb_mat.cols == _frame_width) {
            convert_mat_to_video_buffer (rgb_mat, _buf);
        } else {
            XCAM_LOG_WARNING ("cv read NULL frame size: %dx%d", rgb_mat.rows, rgb_mat.cols);
            return XCAM_RETURN_BYPASS;
        }
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %d", XCAM_STR(get_file_name ()), (int)_format);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::read_frame (const char* image)
{
     XCAM_ASSERT (_pool.ptr ());

    _buf = _pool->get_buffer (_pool);
    XCAM_ASSERT (_buf.ptr ());

    const VideoBufferInfo &info = _buf->get_video_info ();
    VideoBufferPlanarInfo planar;

    uint8_t *memory = _buf->map ();
    if (NULL == memory) {
        XCAM_LOG_ERROR ("ImageFile map buffer failed");
        _buf->unmap ();
        return XCAM_RETURN_ERROR_MEM;
    }

    char* ptr_image = (char*)image;
    for (uint32_t comp = 0; comp < info.components; comp++) {
        info.get_planar_info (planar, comp);
        uint32_t row_bytes = planar.width * planar.pixel_bytes;
        for (uint32_t i = 0; i < planar.height; i++) {
            memcpy (memory + info.offsets [comp] + i * info.strides [comp], ptr_image, row_bytes);
            ptr_image += row_bytes;
        }
    }
    _buf->unmap ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::write_frame (SmartPtr<VideoBuffer> &buf)
{
    if (!buf.ptr ()) {
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (_format == FormatNV12 || _format == FormatYUV420) {
        _file->write_buf (buf);
    } else if (_format == FormatVideo) {
        cv::Mat mat;
        convert_to_mat (buf, mat);
        cv_write_frame (mat);
    } else if (_format == FormatImage) {
        cv::Mat mat;
        convert_to_mat (buf, mat);
        static uint32_t idx = 0;
        char file_name[256];

        std::string base_name = get_base_name ();
        std::string extension = get_file_extension ();
        // Format the new file name with the index before the extension
        sprintf(file_name, "%s_%d%s", base_name.c_str(), idx++, extension.c_str());

        cv::imwrite (file_name, mat);
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %d", XCAM_STR(get_file_name ()), (int)_format);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::write_frame ()
{
    if (_format == FormatNV12 || _format == FormatYUV420) {
        _file->write_buf (_buf);
    } else if (_format == FormatVideo) {
        cv::Mat mat;
        convert_to_mat (_buf, mat);
        cv_write_frame (mat);
    } else if (_format == FormatImage) {
        cv::Mat mat;
        convert_to_mat (_buf, mat);
        static uint32_t idx = 0;
        char file_name[20];
        sprintf (file_name, "%d_%s", idx++, get_file_name ());
        cv::imwrite (file_name, mat);
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %d", XCAM_STR(get_file_name ()), (int)_format);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<VideoBuffer> &
Stream::get_buf ()
{
    if (!_buf.ptr () && _pool.ptr ()) {
        _buf = _pool->get_buffer (_pool);
        XCAM_ASSERT (_buf.ptr ());
    }

    return _buf;
}

SmartPtr<VideoBuffer>
Stream::get_free_buf ()
{
    XCAM_FAIL_RETURN (
        ERROR, _pool.ptr (), NULL,
        "Stream(%s) get free buffer failed since allocator was not initilized", XCAM_STR(get_file_name ()));

    return _pool->get_buffer (_pool);
}

XCamReturn
Stream::estimate_file_format ()
{
    XCAM_ASSERT (get_file_name ());

    char suffix[FREEVIEW_TEST_MAX_STR_SIZE] = {'\0'};
    const char *ptr = strrchr (get_file_name (), '.');
    snprintf (suffix, FREEVIEW_TEST_MAX_STR_SIZE, "%s", ptr + 1);

    if (!strcasecmp (suffix, "nv12")) {
        _format = FormatNV12;
    } else if (!strcasecmp (suffix, "yuv")) {
        _format = FormatYUV420;
    } else if (!strcasecmp (suffix, "mp4")) {
        _format = FormatVideo;
    } else if (!strcasecmp (suffix, "jpg")) {
        _format = FormatImage;
    } else if (!strcasecmp (suffix, "bmp")) {
        _format = FormatImage;
    } else if (!strcasecmp (suffix, "png")) {
        _format = FormatImage;
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %s", _file_name, suffix);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::cv_open_writer (const char *file_name)
{
    XCAM_FAIL_RETURN (
        ERROR, _frame_width && _frame_height, XCAM_RETURN_ERROR_PARAM,
        "stream(%s) invalid size width:%d height:%d", file_name, _frame_width, _frame_height);

    cv::Size frame_size = cv::Size (_frame_width, _frame_height);
    if (!_cv_writer.open (file_name, cv::VideoWriter::fourcc ('X', '2', '6', '4'), 30, frame_size)) {
        XCAM_LOG_ERROR ("stream(%s) open file failed", file_name);
        return XCAM_RETURN_ERROR_FILE;
    }

    return XCAM_RETURN_NO_ERROR;
}

void
Stream::cv_write_frame (const cv::Mat &mat)
{
    if (_cv_writer.isOpened () && !mat.empty()) {
        _cv_writer.write (mat);
    }
}

XCamReturn
Stream::cv_open_reader (const char *file_name)
{
    _cv_reader.open (file_name);
    XCAM_LOG_DEBUG ("video is opened: %d ", _cv_reader.isOpened ());

#ifdef OCV_VERSION_3
    _frame_width = _cv_reader.get (CV_CAP_PROP_FRAME_WIDTH);
    _frame_height = _cv_reader.get (CV_CAP_PROP_FRAME_HEIGHT);
    _frame_count = _cv_reader.get (CV_CAP_PROP_FRAME_COUNT);
    _fps = _cv_reader.get (CV_CAP_PROP_FPS);
#endif

#ifdef OCV_VERSION_4
    _frame_width = _cv_reader.get (CAP_PROP_FRAME_WIDTH);
    _frame_height = _cv_reader.get (CAP_PROP_FRAME_HEIGHT);
    _frame_count = _cv_reader.get (CAP_PROP_FRAME_COUNT);
    _fps = _cv_reader.get (CAP_PROP_FPS);
#endif
    printf ("video width:%d, height:%d, frame count:%f, fps:%f \n", _frame_width, _frame_height, _frame_count, _fps);

    return XCAM_RETURN_NO_ERROR;
}

Mat
Stream::cv_read_frame ()
{
    Mat frame;

    if (false == _cv_reader.read (frame)) {
        XCAM_LOG_WARNING ("read NULL frame !");
    }
    return frame;
}

void
Stream::debug_write_image (char *img_name, char *frame_str, char *idx_str)
{
    XCAM_ASSERT (img_name);
    const cv::Scalar color = cv::Scalar (0, 0, 255);
    const int fontFace = cv::FONT_HERSHEY_COMPLEX;

    cv::Mat mat;
    convert_to_mat (_buf, mat);

    if(frame_str)
        cv::putText (mat, frame_str, cv::Point(20, 50), fontFace, 2.0, color, 2, 8, false);
    if(idx_str)
        cv::putText (mat, idx_str, cv::Point(20, 110), fontFace, 2.0, color, 2, 8, false);

    cv::imwrite (img_name, mat);
}

