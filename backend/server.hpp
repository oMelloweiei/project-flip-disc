// server.hpp
#pragma once

#include <iostream>
#include <thread>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <unistd.h>

class Server
{
private:
    int port;
    int sockfd;
    SSL_CTX *ctx;
    const char *cert_file;
    const char *key_file;

public:
    Server(int port, const char *cert_path, const char *key_path)
        : port(port), sockfd(-1), ctx(nullptr), cert_file(cert_path), key_file(key_path) {}

    ~Server()
    {
        if (sockfd >= 0)
            close(sockfd);
        if (ctx)
            SSL_CTX_free(ctx);
    }

    void start()
    {
        initializeSSL();
        createSocket();
        runServer();
    }

private:
    void initializeSSL()
    {
        SSL_library_init();
        OpenSSL_add_ssl_algorithms();
        SSL_load_error_strings();

        ctx = SSL_CTX_new(TLS_server_method());
        if (!ctx)
        {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("Failed to create SSL context");
        }

        if (SSL_CTX_use_certificate_file(ctx, cert_file, SSL_FILETYPE_PEM) <= 0 ||
            SSL_CTX_use_PrivateKey_file(ctx, key_file, SSL_FILETYPE_PEM) <= 0)
        {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("Failed to load certificate or key");
        }

        if (!SSL_CTX_check_private_key(ctx))
        {
            throw std::runtime_error("Private key does not match the certificate");
        }
    }

    void createSocket()
    {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0)
        {
            throw std::runtime_error("Socket creation failed");
        }

        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        {
            throw std::runtime_error("Bind failed");
        }

        if (listen(sockfd, 5) < 0)
        {
            throw std::runtime_error("Listen failed");
        }

        std::cout << "Server listening on port " << port << std::endl;
    }

    void runServer()
    {
        while (true)
        {
            int client_fd = accept(sockfd, nullptr, nullptr);
            if (client_fd < 0)
            {
                std::cerr << "Accept failed" << std::endl;
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

            std::thread([ssl, client_fd, this]()
                        { handleClient(ssl, client_fd); })
                .detach();
        }
    }

protected:
    virtual void handleClient(SSL *ssl, int client_fd) = 0;
};