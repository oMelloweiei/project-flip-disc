#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <thread>

#define PORT 8080

const char *cert_file = "/Users/azballkung/Desktop/Training/flipdisc_project/backend/cert.pem";
const char *key_file = "/Users/azballkung/Desktop/Training/flipdisc_project/backend/key.pem";

void stream_video(SSL *ssl)
{
    cv::VideoCapture cap(0); // Open webcam
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open webcam." << std::endl;
        return;
    }

    std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Access-Control-Allow-Origin: *\r\n\r\n";

    SSL_write(ssl, header.c_str(), header.size());

    cv::Mat frame;
    std::vector<uchar> buffer;
    while (true)
    {
        cap >> frame; // Capture frame
        if (frame.empty())
            break;

        // Encode frame as JPEG
        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};
        cv::imencode(".jpg", frame, buffer, compression_params);

        // Write frame to the stream
        std::ostringstream oss;
        oss << "--frame\r\n"
            << "Content-Type: image/jpeg\r\n"
            << "Content-Length: " << buffer.size() << "\r\n\r\n";
        SSL_write(ssl, oss.str().c_str(), oss.str().size());
        SSL_write(ssl, buffer.data(), buffer.size());
        SSL_write(ssl, "\r\n", 2);

        // Small delay for smooth streaming
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}
// Handles each connected client
void handle_client(SSL *ssl, int client_fd)
{
    std::cout << "Client connected. SSL handshake completed." << std::endl;

    char buffer[1024] = {0};
    std::string response_body;
    std::string response_header;

    if (SSL_read(ssl, buffer, sizeof(buffer) - 1) <= 0)
    {
        std::cerr << "SSL_read failed." << std::endl;
        ERR_print_errors_fp(stderr);
        goto cleanup;
    }

    std::cout << "Received request:\n"
              << buffer << std::endl;

    if (strstr(buffer, "OPTIONS /"))
    {
        response_header =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Content-Length: 0\r\n\r\n";

        SSL_write(ssl, response_header.c_str(), response_header.size());
        std::cout << "Handled CORS preflight request (OPTIONS)." << std::endl;
    }
    else if (strstr(buffer, "GET /webcam"))
    {
        stream_video(ssl); // Handle the webcam route
    }
    else if (strstr(buffer, "GET /"))
    {
        response_body = "{\"message\": \"Hello from server!\"}";

        response_header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " +
            std::to_string(response_body.size()) + "\r\n\r\n";

        std::string get_response = response_header + response_body;

        SSL_write(ssl, get_response.c_str(), get_response.size());
        std::cout << "Handled GET request." << std::endl;
    }
    else
    {
        response_body = "{\"error\": \"404 Not Found\"}";

        response_header =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " +
            std::to_string(response_body.size()) + "\r\n\r\n";

        std::string not_found_response = response_header + response_body;

        SSL_write(ssl, not_found_response.c_str(), not_found_response.size());
        std::cout << "Handled invalid request." << std::endl;
    }

cleanup:
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(client_fd);
    std::cout << "Client connection closed." << std::endl;
}

int main()
{
    // Initialize SSL
    SSL_library_init();
    OpenSSL_add_ssl_algorithms();
    SSL_load_error_strings();

    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx)
    {
        ERR_print_errors_fp(stderr);
        return 1;
    }

    // Load certificate and key
    if (SSL_CTX_use_certificate_file(ctx, cert_file, SSL_FILETYPE_PEM) <= 0 ||
        SSL_CTX_use_PrivateKey_file(ctx, key_file, SSL_FILETYPE_PEM) <= 0)
    {
        ERR_print_errors_fp(stderr);
        SSL_CTX_free(ctx);
        return 1;
    }

    if (!SSL_CTX_check_private_key(ctx))
    {
        std::cerr << "Private key does not match the certificate" << std::endl;
        SSL_CTX_free(ctx);
        return 1;
    }

    // Create socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        perror("Socket creation failed");
        SSL_CTX_free(ctx);
        return 1;
    }

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("Bind failed");
        close(sockfd);
        SSL_CTX_free(ctx);
        return 1;
    }

    if (listen(sockfd, 5) < 0)
    {
        perror("Listen failed");
        close(sockfd);
        SSL_CTX_free(ctx);
        return 1;
    }

    std::cout << "Server listening on port " << PORT << std::endl;

    while (true)
    {
        int client_fd = accept(sockfd, nullptr, nullptr);
        if (client_fd < 0)
        {
            perror("Accept failed");
            continue;
        }

        SSL *ssl = SSL_new(ctx);
        SSL_set_fd(ssl, client_fd);
        if (SSL_accept(ssl) <= 0)
        {
            ERR_print_errors_fp(stderr);
            SSL_free(ssl);
            close(client_fd);
            continue;
        }

        // Handle client in a new thread
        std::thread client_thread([ssl, client_fd]()
                                  { handle_client(ssl, client_fd); });
        client_thread.detach();
    }

    // Cleanup
    close(sockfd);
    SSL_CTX_free(ctx);
    return 0;
}