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
#include <unistd.h>
#include <thread>

#define PORT 8080

const char *cert_file = "/Users/azballkung/Desktop/Training/Cplusplus/cert.pem";
const char *key_file = "/Users/azballkung/Desktop/Training/Cplusplus/key.pem";
const std::string static_file_path = "/Users/azballkung/Desktop/Training/Cplusplus/static/build";

// Function to process uploaded image and apply OpenCV operations
void process_image(const std::string &image_path, std::string &response_body)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR); // Read the image

    if (img.empty())
    {
        response_body = "{\"error\": \"Failed to load image.\"}";
        return;
    }

    // Example OpenCV processing: Convert to grayscale
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    // Save the processed image to a temporary file
    std::string output_path = "/tmp/processed_image.jpg";
    cv::imwrite(output_path, gray_img);

    // Build the response body (you could also return the image as a byte stream if needed)
    response_body = "{\"message\": \"Image processed successfully.\", \"processed_image\": \"" + output_path + "\"}";
}

// Read file content
std::string read_file(const std::string &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file)
        return "";
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

// Determine MIME type based on file extension
std::string get_mime_type(const std::string &path)
{
    std::unordered_map<std::string, std::string> mime_types = {
        {".html", "text/html"},
        {".css", "text/css"},
        {".js", "application/javascript"},
        {".jpg", "image/jpeg"},
        {".jpeg", "image/jpeg"},
        {".png", "image/png"},
        {".gif", "image/gif"}};

    auto dot_pos = path.find_last_of('.');
    if (dot_pos != std::string::npos)
    {
        std::string ext = path.substr(dot_pos);
        if (mime_types.count(ext))
            return mime_types[ext];
    }
    return "application/octet-stream"; // Default binary
}

// Handles each connected client
void handle_client(SSL *ssl, int client_fd)
{
    std::cout << "Client connected. SSL handshake completed." << std::endl;

    char buffer[1024] = {0};

    if (SSL_read(ssl, buffer, sizeof(buffer) - 1) <= 0)
    {
        std::cerr << "SSL_read failed." << std::endl;
        ERR_print_errors_fp(stderr);
        SSL_shutdown(ssl);
        SSL_free(ssl);
        close(client_fd);
        return; // Return early if SSL_read fails
    }

    std::cout << "Received request: \n"
              << buffer << std::endl;

    // Check for GET request for static files
    if (strstr(buffer, "GET /"))
    {
        std::string request_line(buffer);
        auto pos = request_line.find(" ");
        auto pos_end = request_line.find(" ", pos + 1);
        std::string requested_path = request_line.substr(pos + 1, pos_end - pos - 1);

        if (requested_path == "/")
            requested_path = "/index.html"; // Default to index.html

        std::string file_path = static_file_path + requested_path;
        std::string file_content = read_file(file_path);

        if (file_content.empty())
        {
            std::string not_found = "HTTP/1.1 404 Not Found\nContent-Type: text/plain\n\n404 Not Found";
            SSL_write(ssl, not_found.c_str(), not_found.size());
        }
        else
        {
            std::string response = "HTTP/1.1 200 OK\nContent-Type: " + get_mime_type(file_path) +
                                   "\nContent-Length: " + std::to_string(file_content.size()) + "\n\n" + file_content;
            SSL_write(ssl, response.c_str(), response.size());
        }
        SSL_shutdown(ssl);
        SSL_free(ssl);
        close(client_fd);
        return;
    }

    // Example: Handling POST request to process an image
    if (strstr(buffer, "POST /process_image"))
    {
        // In a real scenario, you would extract the image from the request body
        // Here we simulate the image file path for simplicity
        std::string image_path = "/path/to/uploaded_image.jpg";

        std::string response_body;
        process_image(image_path, response_body);

        // Prepare the response
        const char *response_header = "HTTP/1.1 200 OK\n"
                                      "Content-Type: application/json\n"
                                      "Access-Control-Allow-Origin: *\n"
                                      "Content-Length: ";

        char response[2048]; // Large buffer to hold response
        snprintf(response, sizeof(response), "%s%ld\n\n%s", response_header, response_body.size(), response_body.c_str());

        // Send the response
        SSL_write(ssl, response, strlen(response));
        SSL_shutdown(ssl);
        SSL_free(ssl);
        close(client_fd);
        std::cout << "Client connection closed." << std::endl;
        return;
    }

    const char *message = "{\"message\": \"Connection is still active\"}";
    char response[1024];

    snprintf(response, sizeof(response),
             "HTTP/1.1 200 OK\n"
             "Content-Type: application/json\n"
             "Access-Control-Allow-Origin: *\n"
             "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\n"
             "Access-Control-Allow-Headers: Content-Type\n"
             "Content-Length: %ld\n\n%s",
             strlen(message), message);

    SSL_write(ssl, response, strlen(response));

    // Cleanup code
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
