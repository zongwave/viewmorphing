/*
 * test-freeview.cpp - freeview test application
 *
 * Author: Zong Wei <zongwave@hotmail.com>
 */

#include <string>
#include <stdexcept>
#include <unistd.h>
#include <getopt.h>

#include "stream_handler.h"
#include "frame_interpolator.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ATTACH_STREAM(Type, streams, file_name) \
    {                                                  \
        SmartPtr<Type> stream = new Type (file_name);  \
        XCAM_ASSERT (stream.ptr ());                   \
        streams.push_back (stream);                    \
    }

#define CAMERA_COUNT   12

static int interpolate_frames (
    const SmartPtr<Interpolator> &interpolator,
    const Streams &ins,
    const Streams &outs,
    const uint32_t interp_index,
    const uint32_t interp_count,
    const bool save_out,
    const bool continuous)
{
    printf ("interpolation index:%d, interpolation count:%d \n", interp_index, interp_count);

    XCAM_ASSERT (interp_index >= 0);
    XCAM_ASSERT (interp_count > 0);

    XCAM_ASSERT (interp_index < ins.size () - 1);
    XCAM_ASSERT (interp_index <= outs.size () - 1);

    interpolator->set_interp_frame_index (interp_index);
    interpolator->set_interp_frame_count (interp_count);

    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    VideoBufferList in_buffers;

    int processed_count = 0;

    for (uint32_t i = 0; i < ins.size (); ++i) {
        if (XCAM_RETURN_NO_ERROR != ins[i]->rewind ()) {
            printf ("rewind buffer from file(%s) failed \n", ins[i]->get_file_name ());
        }
    }
    
    do {
        printf ("interpolate_frames frame count: %d \n", processed_count);

        in_buffers.clear ();

        for (uint32_t i = 0; i < ins.size (); ++i) {
            ret = ins[i]->read_frame ();
            if (XCAM_RETURN_BYPASS == ret) {
                printf ("Read EOF file(%s) \n", ins[i]->get_file_name ());
                return processed_count;
            }
            XCAM_ASSERT (ins[i]->get_buf ().ptr ());
            in_buffers.push_back (ins[i]->get_buf ());
        }

        {
            std::vector<XCam::SmartPtr<VideoBuffer>> out_buffers;

            for (uint32_t i = 0; i < outs.size (); ++i) {
                for (uint32_t j = 0; j < interp_count; j++) {
                    SmartPtr<VideoBuffer> buf = outs[i]->get_free_buf ();
                    out_buffers.push_back (buf);
                }
            }

            // start to run
            if (interpolator->process_frames (in_buffers, out_buffers)) {
                if (save_out) {
                    printf ("save output buffer continuous:%d, size:%ld \n", continuous, out_buffers.size ());
                    if (continuous) {
                        for (uint32_t i = 0; i < outs.size (); i++) {
                            SmartPtr<VideoBuffer> buf = out_buffers[i];
                            XCAM_ASSERT (buf.ptr ());
                            outs[i]->write_frame (buf);
                        }
                    } else {
                        std::vector<SmartPtr<VideoBuffer>> key_frames;
                        uint32_t count = 0;
                        for (VideoBufferList::const_iterator iter = in_buffers.begin (); iter != in_buffers.end (); ++iter) {
                            if (count == interp_index || count == interp_index + 1) {
                                SmartPtr<VideoBuffer> buf = *iter;
                                XCAM_ASSERT (buf.ptr ());
                                key_frames.push_back (buf);
                            }
                        }
                        outs[interp_index]->write_frame (key_frames[0]);
                        for (uint32_t i = 0; i < out_buffers.size (); i++) {
                            SmartPtr<VideoBuffer> buf = out_buffers[i];
                            XCAM_ASSERT (buf.ptr ());
                            outs[interp_index]->write_frame (buf);
                        }
                        outs[interp_index]->write_frame (key_frames[1]);
                    }
                }                    
                out_buffers.clear ();
            }else break;
        }

        processed_count++;

    } while (true);
    return processed_count;
}

static void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --input input0.jpg --input input1.jpg --input input2.jpg ..."
            "\t    --output output.jpg ...\n"
            "\t--input             input image(JPEG/PNG/BMP/YUV/NV12)\n"
            "\t--output            output image(JPEG/PNG/BMP/YUV/NV12/MP4)\n"
            "\t--save              save output images\n"
            "\t--interp-count      optional, interpolation frame count: 2\n"
            "\t--in-w              optional, input width, default: 1280\n"
            "\t--in-h              optional, input height, default: 800\n"
            "\t--out-w             optional, output width, default: 1920\n"
            "\t--out-h             optional, output height, default: 640\n"
            "\t--continuous        optional, continuous output image(YUV/NV12/MP4)\n",
            arg0
           );
}

int main (int argc, char *argv[])
{
    Streams input_streams;
    uint32_t input_width = 1920;
    uint32_t input_height = 1080;
    uint32_t input_format = V4L2_PIX_FMT_YUV420;
    uint32_t output_format = V4L2_PIX_FMT_YUV420;

    Streams output_streams;
    uint32_t output_width = 1920;
    uint32_t output_height = 1080;

    uint32_t interp_index = 0;
    uint32_t interp_count = 1;
    bool use_gpu = false;
    bool save_out = true;
    bool continuous = false;

    SyntheizerModule syntheizer_module = SyntheizerViewMorphing;

    const struct option long_opts[] = {
        {"gpu", required_argument, NULL, 'g'},
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"out-w", required_argument, NULL, 'W'},
        {"out-h", required_argument, NULL, 'H'},
        {"in-format", required_argument, NULL, 'p'},
        {"save", required_argument, NULL, 's'},
        {"interp-idx", required_argument, NULL, 'I'},
        {"interp-count", required_argument, NULL, 'c'},
        {"repeat", required_argument, NULL, 'R'},
        {"continuous", required_argument, NULL, 'C'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    char outstr[64] = {'\0'};
    printf ("Parse command line \n");
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'g':
            use_gpu = (strcasecmp (optarg, "false") == 0 ? false : true);
            cout << "Use GPU: " << use_gpu << endl;
            break;
        case 'i':
            cout << "input file: " << optarg << endl;
            ATTACH_STREAM (Stream, input_streams, optarg);
            break;
        case 'o':
            output_streams.clear();
            cout << "output file: " << optarg << endl;
            ATTACH_STREAM (Stream, output_streams, optarg);
            break;
        case 's':
            save_out = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'w':
            input_width = atoi(optarg);
            break;
        case 'h':
            input_height = atoi(optarg);
            break;
        case 'W':
            output_width = atoi(optarg);
            break;
        case 'H':
            output_height = atoi(optarg);
            break;
        case 'I':
            interp_index = atoi(optarg);
            break;
        case 'c':
            interp_count = atoi(optarg);
            break;
        case 'C':
            output_streams.clear();
            for(int i = 1; i <= interp_count; i++){
                snprintf(outstr, 64, "%s_%d_%d.yuv", optarg, i, interp_count);
                ATTACH_STREAM (Stream, output_streams, outstr);
            }
            continuous = true;
            break;
        default:
            printf ("getopt_long return unknown value: %c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (optind < argc || argc < 2) {
        printf ("unknown option %s \n", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);
    printf ("interpolation count:\t\t%d\n", interp_count);
    printf ("interpolation index:\t\t%d\n", interp_index);

    for (uint32_t i = 0; i < input_streams.size (); ++i) {
        input_streams[i]->set_buf_size (input_width, input_height);
        if (XCAM_RETURN_NO_ERROR == input_streams[i]->estimate_file_format ()) {
            switch (input_streams[i]->get_format ()) {
            case FormatNV12 :
                input_format = V4L2_PIX_FMT_NV12;
            break;
            case FormatYUV420 :
            default :
                input_format = V4L2_PIX_FMT_YUV420;
            break;
            }
        } else {
            printf ("%s: estimate input file format failed \n", input_streams[i]->get_file_name ());
            return -1;
        }
        if (XCAM_RETURN_NO_ERROR != input_streams[i]->create_buf_pool (CAMERA_COUNT, input_format)) {
            printf ("create input buffer pool failed \n");
            return -1;
        }

        if (XCAM_RETURN_NO_ERROR != input_streams[i]->open_reader ("rb")) {
            printf ("open input file(%s) failed \n", input_streams[i]->get_file_name ());
            return -1;
        }
    }

    for (uint32_t i = 0; i < output_streams.size (); ++i) {
        output_streams[i]->set_buf_size (output_width, output_height);
            if (XCAM_RETURN_NO_ERROR == output_streams[i]->estimate_file_format ()) {
                switch (output_streams[i]->get_format ()) {
                case FormatNV12 :
                    output_format = V4L2_PIX_FMT_NV12;
                break;
                case FormatYUV420 :
                default :
                    output_format = V4L2_PIX_FMT_YUV420;
                break;
                }
            } else {
                printf ("%s: estimate output file format failed \n", output_streams[i]->get_file_name ());
                return -1;
            }
            if (XCAM_RETURN_NO_ERROR != output_streams[i]->create_buf_pool (interp_count, output_format)) {
                printf ("create output buffer pool failed \n");
                return -1;
            }

        if (save_out) {

            if (XCAM_RETURN_NO_ERROR != output_streams[i]->open_writer ("wb")) {
                printf ("open output file(%s) failed \n", output_streams[i]->get_file_name ());
                return -1;
            }
        }
    }

    SmartPtr<Interpolator> interpolator = new Interpolator (syntheizer_module);
    XCAM_ASSERT (interpolator.ptr ());

    int ret = interpolate_frames (interpolator, input_streams, output_streams, interp_index, interp_count, save_out, continuous);

    return ret;
}

