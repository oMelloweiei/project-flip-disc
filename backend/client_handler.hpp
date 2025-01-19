// client_handler.hpp
#pragma once

#include "server.hpp"
#include "video_stream.hpp"
#include <sstream>
#include <string>

class ClientHandler : public Server
{
public:
    ClientHandler(int port, const char *cert_file, const char *key_file)
        : Server(port, cert_file, key_file) {}

protected:
    void handleClient(SSL *ssl, int client_fd) override
    {
        std::cout << "Client connected. SSL handshake completed." << std::endl;

        try
        {
            processRequest(ssl);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error handling client: " << e.what() << std::endl;
        }

        cleanup(ssl, client_fd);
    }

private:
    void processRequest(SSL *ssl)
    {
        char buffer[1024] = {0};
        if (SSL_read(ssl, buffer, sizeof(buffer) - 1) <= 0)
        {
            throw std::runtime_error("SSL_read failed");
        }

        std::cout << "Received request:\n"
                  << buffer << std::endl;

        if (strstr(buffer, "OPTIONS /"))
        {
            handleOptions(ssl);
        }
        else if (strstr(buffer, "GET /webcam"))
        {
            VideoStream stream(ssl);
            stream.start();
        }
        else if (strstr(buffer, "GET /"))
        {
            handleGet(ssl);
        }
        else
        {
            handleNotFound(ssl);
        }
    }

    void handleOptions(SSL *ssl)
    {
        std::string response =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Content-Length: 0\r\n\r\n";

        SSL_write(ssl, response.c_str(), response.size());
        std::cout << "Handled CORS preflight request (OPTIONS)." << std::endl;
    }

    void handleGet(SSL *ssl)
    {
        std::string body = "{\"message\": \"Hello from server!\"}";
        sendJsonResponse(ssl, "200 OK", body);
        std::cout << "Handled GET request." << std::endl;
    }

    void handleNotFound(SSL *ssl)
    {
        std::string body = "{\"error\": \"404 Not Found\"}";
        sendJsonResponse(ssl, "404 Not Found", body);
        std::cout << "Handled invalid request." << std::endl;
    }

    void sendJsonResponse(SSL *ssl, const std::string &status, const std::string &body)
    {
        std::string response =
            "HTTP/1.1 " + status + "\r\n"
                                   "Content-Type: application/json\r\n"
                                   "Access-Control-Allow-Origin: *\r\n"
                                   "Content-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" +
            body;

        SSL_write(ssl, response.c_str(), response.size());
    }

    void cleanup(SSL *ssl, int client_fd)
    {
        SSL_shutdown(ssl);
        SSL_free(ssl);
        close(client_fd);
        std::cout << "Client connection closed." << std::endl;
    }
};