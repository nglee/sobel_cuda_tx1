/*
 * Author: Lee Namgoo
 * E-Mail: lee.namgoo@sualab.com
 */

#include <Argus/Argus.h> // @: tegra_multimedia_api/include
#include <EGLStream/EGLStream.h> // FrameConsumer
#include <EGLStream/ArgusCaptureMetadata.h> // EGLStream::IArgusCaptureMetadata
#include <EGLStream/NV/ImageNativeBuffer.h> // EGLStream::NV::IImageNativeBuffer

#include <nvbuf_utils.h>
#include <NvUtils.h>
#include <NvJpegEncoder.h>

#include <opencv2/opencv.hpp>

#include <time.h>
#include <math.h>
#include <errno.h> // strerror, errno
#include <stdio.h>
#include <stdlib.h> // EXIT_FAILURE, EXIT_SUCCESS
#include <unistd.h> // sleep
#include <sys/mman.h> // mmap, munmap
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace Argus;

#define IMAGEWIDTH              1920
#define IMAGEHEIGHT             1080

#define TEST_ERROR_RETURN(cond, _str, ...) \
    do { \
        if (cond) { \
            fprintf(stderr, "[ERROR] %s (%s:%d) : ", __FUNCTION__, __FILE__, __LINE__); \
            fprintf(stderr, _str "\n", ##__VA_ARGS__); \
            return false; \
        }} while(0)
#define TEST_WARNING(cond, _str, ...) \
    do { \
        if (cond) { \
            fprintf(stderr, "[WARNING] %s (%s:%d) : ", __FUNCTION__, __FILE__, __LINE__); \
            fprintf(stderr, _str "\n", ##__VA_ARGS__); \
        }} while(0)
#define OPENCV_IMWRITE(bmp_path, mat) \
    try { \
        cv::imwrite(bmp_path, mat); \
    } catch (runtime_error& ex) { \
        fprintf(stderr, "Exception converting image to BMP format: %s\n", ex.what()); \
        return false; \
    }

extern bool sobel_cuda(const cv::Mat&, cv::Mat&, int);

#if 0
void printCameraProperties(CameraDevice*& camera, int index)
{
    cout << setfill('=') << setw(22) << right << " "
        << "Camera properties for camera #" << index
        << setw(22) << left << " " << setfill(' ') << "\n";

    ICameraProperties *iCameraProperties = interface_cast<ICameraProperties>(camera);
    cout << setw(35) << left << "Max AE Regions" << ": " << iCameraProperties->getMaxAeRegions()
        << " (0=only the entire image supported)\n";
    cout << setw(35) << left << "Max AWB Regions" << ": " << iCameraProperties->getMaxAwbRegions()
        << " (0=only the entire image supported)\n";

    vector<SensorMode*> sensorModes;
    iCameraProperties->getSensorModes(&sensorModes);
    cout << setw(35) << left << "Number of sensor modes" << ": " << sensorModes.size() << endl;
    for (int i = 0; i < sensorModes.size(); i++) {
        ISensorMode *iSensorMode = interface_cast<ISensorMode>(sensorModes[i]);
        Size resolution = iSensorMode->getResolution();
        cout << "Resolution for sensor [" << i << setw(11) << left << "]" << ": ("
            << resolution.width << ", " << resolution.height << ")\n";
    }

    Range<int32_t> f = iCameraProperties->getFocusPositionRange();
    cout << setw(35) << left << "Valid range of focuser positions" << ": ["
        << f.min << ", " << f.max << "]\n";

    Range<float> a = iCameraProperties->getLensApertureRange();
    cout << setw(35) << left << "Supported aperture range" << ": ["
        << showpoint << setprecision(2) << a.min << ", " << a.max << "]\n";
}
#endif

#if 0
bool printImageProperties(EGLStream::Image *image)
{
    Argus::Status status;

    EGLStream::IImage *iImage = interface_cast<EGLStream::IImage>(image);
    TEST_ERROR_RETURN(!iImage, "Failed to get an IImage");
    EGLStream::IImage2D *iImage2D = interface_cast<EGLStream::IImage2D>(image);
    TEST_ERROR_RETURN(!iImage2D, "Failed to get an IImage2D");

    uint32_t                bufferCount = iImage->getBufferCount();
    uint64_t*               bufferSize  = (uint64_t*)malloc(bufferCount * sizeof(uint64_t));
    Size*                   imageSize   = (Size*)malloc(bufferCount * sizeof(Size));
    uint32_t*               imageStride = (uint32_t*)malloc(bufferCount * sizeof(uint32_t));
    const unsigned char**   buf         = (const unsigned char**)malloc(bufferCount
                                                * sizeof(const unsigned char*));

    cout << "\nNumber of buffers : " << bufferCount << "\n";

    for (int i = 0; i < bufferCount; i++) {
        bufferSize[i] = iImage->getBufferSize(i);
        imageSize[i] = iImage2D->getSize(i);
        imageStride[i] = iImage2D->getStride(i);
        buf[i] = (const unsigned char *)iImage->mapBuffer(i, &status);
        TEST_ERROR_RETURN(status != STATUS_OK, "(%d) Failed to map buffer %d", status, i);

        cout << setw(13) << left << "buffer size" << "[" << i << "] : "
                << bufferSize[i] << "\n";
        cout << setw(13) << left << "image size" << "[" << i << "] : "
                << imageSize[i].width << " x " << imageSize[i].height << "\n";
        cout << setw(13) << left << "image stride" << "[" << i << "] : "
                << imageStride[i] << "\n";

#if 0
        for (int j = 0; j < imageSize[i].height; j++) {
            for (int k = 0; k < imageSize[i].width; k++)
                cout << setw(3) << right <<
                    (unsigned int)*((unsigned char *)buf + j * imageStride[i] + k) << " ";
            cout << endl;
        }
#endif
    }

    cout << endl;

    free(bufferSize);
    free(imageSize);
    free(imageStride);
    free(buf);
}
#endif

#if 0
bool vflip(const int dmabuf_fd)
{
    int ret;
    NvBufferParams params;

    ret = NvBufferGetParams(dmabuf_fd, &params);
    TEST_ERROR_RETURN(ret < 0, "Failed to get a native buffer parameters");

    if (!printNvBufferParams(&params))
        return false;

    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    printf("\n[SYSTEM] page size = 0x%lx (%lu Bytes)\n", page_size, page_size);

    for (int i = 0; i < params.num_planes; i++) {
        uint32_t width = params.width[i];
        uint32_t height = params.height[i];
        uint32_t pitch = params.pitch[i];

        TEST_WARNING(pitch % sizeof(uint64_t), "pitch[%d] is not a multiple of %ld", i, sizeof(uint64_t));

        uint8_t* data_mem;
        size_t fsize = pitch * height;

        data_mem = (uint8_t*)mmap(0, fsize, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, params.offset[i]);
        TEST_ERROR_RETURN(data_mem == MAP_FAILED, "mmap failed - %s", strerror(errno));
        printf("[SYSTEM] mmap to %p with size 0x%lx (%lu Bytes)\n", data_mem, fsize, fsize);

        for (int j = 0; j < (height - 2) / 2; j++) {
            /* Hope we can benefit from the 64bit architecture */
            for (int k = 0; k < pitch/sizeof(uint64_t); k++) {
                uint64_t* a = (uint64_t*)(data_mem + j * pitch + k * sizeof(uint64_t));
                uint64_t* b = (uint64_t*)(data_mem + (height - 1 - j) * pitch + k * sizeof(uint64_t));
                uint64_t tmp = *a;
                *a = *b;
                *b = tmp;
            }
        }

        ret = msync(data_mem, fsize, MS_ASYNC);
        TEST_ERROR_RETURN(ret < 0, "msync failed - %s", strerror(errno));
        ret = munmap(data_mem, fsize);
        TEST_ERROR_RETURN(ret < 0, "munmap failed - %s", strerror(errno));
    }
    sleep(1);

    unsigned long out_buf_size = IMAGEWIDTH * IMAGEHEIGHT;
    unsigned char *out_buf = new unsigned char[out_buf_size];

    NvJPEGEncoder *jpegenc = NvJPEGEncoder::createJPEGEncoder("jpegenc");
    TEST_ERROR_RETURN(!jpegenc, "Failed to create a JPEG encoder");

    ret = jpegenc->encodeFromFd(dmabuf_fd, JCS_YCbCr, &out_buf, out_buf_size);
    TEST_ERROR_RETURN(ret < 0, "Failed to encode JPEG from fd");

    ofstream *out_file = new ofstream("vflip.jpg");
    TEST_ERROR_RETURN(!out_file->is_open(), "Could not open vflip.jpg");

    out_file->write((char *)out_buf, out_buf_size);

    delete[] out_buf;
    delete out_file;

    return true;
}
#endif

bool capture(vector<cv::Mat>& ycbcr_split)
{
    int ret;
    Argus::Status status;
    vector<CameraDevice*> cameraDevices;

    /*
     * Create a camera provider object and get its interface.
     * This establishes connection with the libargus driver.
     */
    UniqueObj<CameraProvider> cameraProvider {CameraProvider::create()};
    ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
    TEST_ERROR_RETURN(!iCameraProvider, "Failed to create an ICameraProvider");

    /* Get the camera device from the camera provider */
    status = iCameraProvider->getCameraDevices(&cameraDevices);
    TEST_ERROR_RETURN(status != STATUS_OK, "Function getCameraDevices fail");
    TEST_ERROR_RETURN(cameraDevices.size() == 0, "No cameras available");
#if 0
    cout << "\nNumber of cameras : " << cameraDevices.size() << endl;
    for (int i = 0; i < cameraDevices.size(); i++) {
        printCameraProperties(cameraDevices[i], i);
    }
    cout << setfill('=') << setw(76) << right << "\n";
    cout << setfill(' ') << setw(76)
        << "(There are other properties for sensor modes not displayed here)\n" << endl;
#endif

    /*
     * Create a capture session using the first camera device.
     * Capture session is the heart of an argus capture pipe-line.
     * A capture session has an exclusive connection to one or more camera
     * devices. A capture session must be destroyed before its camera devices
     * are ready for use by another capture session.
     * A capture session creates OutputStreamSettings, OutputStream, and Request.
     */
    UniqueObj<CaptureSession> captureSession {
            iCameraProvider->createCaptureSession(cameraDevices[0], &status)};
    TEST_ERROR_RETURN(status != STATUS_OK, "Function createCaptureSession fail");
    ICaptureSession *iCaptureSession = interface_cast<ICaptureSession>(captureSession);
    TEST_ERROR_RETURN(!iCaptureSession, "Failed to create an ICaptureSession");

    /*
     * Create an output stream
     * An output stream is the destination stream for capture request outputs.
     */
    UniqueObj<OutputStreamSettings> outputStreamSettings {
            iCaptureSession->createOutputStreamSettings()};
    IOutputStreamSettings *iOutputStreamSettings
            = interface_cast<IOutputStreamSettings>(outputStreamSettings);
    TEST_ERROR_RETURN(!iOutputStreamSettings, "Failed to create an IOutputStreamSettings");

    status = iOutputStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    TEST_ERROR_RETURN(status != STATUS_OK, "Function setPixelFormat fail");
    status = iOutputStreamSettings->setResolution(Size {IMAGEWIDTH, IMAGEHEIGHT});
    TEST_ERROR_RETURN(status != STATUS_OK, "Function setResolution fail");

    UniqueObj<OutputStream> outputStream {
            iCaptureSession->createOutputStream(outputStreamSettings.get(), &status)};
    TEST_ERROR_RETURN(status != STATUS_OK, "Function createOutputStream fail");
    /* There exists an interface IStream for OutputStream but never used here */

    /* Create a frame consumer */
    UniqueObj<EGLStream::FrameConsumer> frameConsumer {
            EGLStream::FrameConsumer::create(outputStream.get())};
    EGLStream::IFrameConsumer *iFrameConsumer
            = interface_cast<EGLStream::IFrameConsumer>(frameConsumer);
    TEST_ERROR_RETURN(!iFrameConsumer, "Failed to create an IFrameConsumer");

    /* Create a request */
    UniqueObj<Request> request {
        iCaptureSession->createRequest(CAPTURE_INTENT_STILL_CAPTURE, &status)};
    TEST_ERROR_RETURN(status != STATUS_OK, "Function createRequest fail");
    IRequest *iRequest = interface_cast<IRequest>(request);
    TEST_ERROR_RETURN(!iRequest, "Failed to create an IRequest");

    /*
     * Connect the request with the output stream.
     * Captures made with this request will produce output on that stream.
     */
    status = iRequest->enableOutputStream(outputStream.get());
    TEST_ERROR_RETURN(status != STATUS_OK, "Failed to enable the output stream for the capture request");

    /* Submit a capture request */
    uint32_t requestId = iCaptureSession->capture(request.get());
    TEST_ERROR_RETURN(!requestId, "Failed to submit a capture request");

    /* Acquire the frame generated by the capture request */
    UniqueObj<EGLStream::Frame> frame {
            iFrameConsumer->acquireFrame(1000000000, &status)}; /* Timeout 1 sec */
    TEST_ERROR_RETURN(status != STATUS_OK, "Failed acquiring a frame from the frame consumer");
    EGLStream::IFrame *iFrame = interface_cast<EGLStream::IFrame>(frame);
    TEST_ERROR_RETURN(!iFrame, "Failed to create an IFrame");

    /* Get the image from the frame */
    EGLStream::Image *image  = iFrame->getImage();
    TEST_ERROR_RETURN(!image, "Failed to get the image from the frame");
#if 0
    if (!printImageProperties(image))
        return false;
#endif

    /* Save the original image to "original.jpg" */
    EGLStream::IImageJPEG *iImageJPEG = interface_cast<EGLStream::IImageJPEG>(image);
    TEST_ERROR_RETURN(!iImageJPEG, "Failed to get an IImageJPEG");
    status = iImageJPEG->writeJPEG("0_original.jpg");
    TEST_ERROR_RETURN(status != STATUS_OK, "Failed to write to original.jpg");

    /* Create an NvBuffer */
    EGLStream::NV::IImageNativeBuffer *iImageNativeBuffer
            = interface_cast<EGLStream::NV::IImageNativeBuffer>(image);
    TEST_ERROR_RETURN(!iImageNativeBuffer, "Failed to create an IImageNativeBuffer");

    int dmabuf_fd = iImageNativeBuffer->createNvBuffer(Size {IMAGEWIDTH, IMAGEHEIGHT},
            NvBufferColorFormat_YUV420, NvBufferLayout_Pitch, &status);
    if (status != STATUS_OK)
    TEST_ERROR_RETURN(status != STATUS_OK, "Failed to create a native buffer");

    NvBufferParams params;
    ret = NvBufferGetParams(dmabuf_fd, &params);
    TEST_ERROR_RETURN(ret < 0, "Failed to get a native buffer parameters");

    /* Copy the image data to OpenCV Mat format */
    for (int i = 0; i < params.num_planes; i++) {
        uint32_t width = params.width[i];
        uint32_t height = params.height[i];
        uint32_t pitch = params.pitch[i];

        TEST_WARNING(pitch % sizeof(uint64_t), "pitch[%d] is not a multiple of %ld", i, sizeof(uint64_t));

        size_t fsize = pitch * height;
        uint8_t* data_mem = (uint8_t*)mmap(0, fsize, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, params.offset[i]);
        TEST_ERROR_RETURN(data_mem == MAP_FAILED, "mmap failed - %s", strerror(errno));
        printf("[SYSTEM] mmap to %p with size 0x%lx (%lu Bytes)\n", data_mem, fsize, fsize);
#if 0
        /* Initialize OpenCV Mat */
        ycbcr_split[i] = cv::Mat (height, width, CV_8UC1);

        /* Copy each row */
        for (int j = 0; j < height; j++) {
            const uint8_t* origptr = data_mem + j * pitch;
            uint8_t* destptr = static_cast<uint8_t*>(ycbcr_split[i].ptr(j));
            memcpy(destptr, origptr, sizeof(uint8_t) * width);
        }
#else
        ycbcr_split[i] = cv::Mat (height, width, CV_8UC1, data_mem, pitch);
#endif
        /*
         * If you unmap data_mem, segmentation fault arize because above copy
         * is a "shallow" copy. It's just making a "header" for the
         * "user-allocated" data.
         */
        //munmap(data_mem, fsize);
    }

    /* Destroy NvBuffer */
    NvBufferDestroy(dmabuf_fd);
    /*
     * NOTE: (from man munmap)
     * "Closing the file descriptor does not unmap the region."
     */

    OPENCV_IMWRITE("1_greyscale_Y.bmp", ycbcr_split[0]);

    return true;
}

inline bool vflip(cv::Mat& image)
{
    cv::flip(image, image, 0);

    OPENCV_IMWRITE("2_vflip.bmp", image);

    return true;
}

inline bool border(const cv::Mat& input, cv::Mat& output, int type)
{
    output = cv::Mat (input.rows, input.cols, CV_8UC1);

    if (!sobel_cuda(input, output, type))
        return false;

    switch (type) {
    case 0:
        OPENCV_IMWRITE("5_Sobel.bmp", output);
        break;
    case 1:
        OPENCV_IMWRITE("3_SobelX.bmp", output);
        break;
    case 2:
        OPENCV_IMWRITE("4_SobelY.bmp", output);
        break;
    }

    return true;
}

bool border_cpu(const cv::Mat& input)
{
    cv::Mat sobel (input.rows, input.cols, CV_8UC1);

    const int SobelX[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    const int SobelY[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

    clock_t begin = clock();

    for (int row = 1; row < input.rows - 1; row++) {
        for (int col = 1; col < input.cols - 1; col++) {

            int sumX = 0, sumY = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int data = input.ptr(row + i)[col + j];
                    sumX += data * SobelX[(i + 1) * 3 + j + 1];
                    sumY += data * SobelY[(i + 1) * 3 + j + 1];
                }
            }
            sobel.ptr(row)[col] = sqrt((double)(sumX * sumX + sumY * sumY) / 32);
        }
    }

    std::cout << "CPU execution time : " << (float)(clock() - begin) /
        (CLOCKS_PER_SEC / 1000) << "ms\n";

    OPENCV_IMWRITE("6_Sobel_cpu.bmp", sobel);
}

int main(int argc, char *argv[])
{
    vector<cv::Mat> ycbcr_split(3);
    cv::Mat sobel_x;
    cv::Mat sobel_y;
    cv::Mat sobel;

    cout << "OpenCV version : " << CV_VERSION << endl;

    if (!capture(ycbcr_split))
        return EXIT_FAILURE;
    if (!vflip(ycbcr_split[0]))
        return EXIT_FAILURE;
#if 0
    if (!border(ycbcr_split[0], sobel_x, 1))
        return EXIT_FAILURE;
    if (!border(ycbcr_split[0], sobel_y, 2))
        return EXIT_FAILURE;
#endif
    if (!border(ycbcr_split[0], sobel, 0))
        return EXIT_FAILURE;
    if (!border_cpu(ycbcr_split[0]))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
