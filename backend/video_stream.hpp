// video_stream.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <openssl/ssl.h>
#include <string>
#include <thread>

class VideoStream
{
private:
    SSL *ssl;
    cv::VideoCapture camera;
    bool is_streaming;

public:
    VideoStream(SSL *ssl_connection) : ssl(ssl_connection), is_streaming(false)
    {
        camera.open(0);
        if (!camera.isOpened())
        {
            throw std::runtime_error("Failed to open webcam");
        }
    }

    ~VideoStream()
    {
        stop();
        if (camera.isOpened())
        {
            camera.release();
        }
    }

    void start()
    {
        sendHeader();
        streamFrames();
    }

    void stop()
    {
        is_streaming = false;
    }

private:
    void sendHeader()
    {
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";

        if (SSL_write(ssl, header.c_str(), header.size()) <= 0)
        {
            throw std::runtime_error("Failed to send stream header");
        }
    }

    void streamFrames()
    {
        is_streaming = true;
        cv::Mat frame;
        std::vector<uchar> buffer;
        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};

        while (is_streaming)
        {
            if (!camera.read(frame) || frame.empty())
            {
                break;
            }

            cv::imencode(".jpg", frame, buffer, compression_params);

            std::ostringstream frame_header;
            frame_header << "--frame\r\n"
                         << "Content-Type: image/jpeg\r\n"
                         << "Content-Length: " << buffer.size() << "\r\n\r\n";

            std::string header_str = frame_header.str();
            if (SSL_write(ssl, header_str.c_str(), header_str.size()) <= 0 ||
                SSL_write(ssl, buffer.data(), buffer.size()) <= 0 ||
                SSL_write(ssl, "\r\n", 2) <= 0)
            {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
};
