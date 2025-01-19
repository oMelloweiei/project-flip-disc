// main.cpp
#include "client_handler.hpp"

int main()
{
    const int PORT = 8080;
    const char *cert_file = "/Users/azballkung/Desktop/Training/flipdisc_project/backend/cert.pem";
    const char *key_file = "/Users/azballkung/Desktop/Training/flipdisc_project/backend/key.pem";

    try
    {
        ClientHandler server(PORT, cert_file, key_file);
        server.start();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}