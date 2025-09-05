#pragma once
#include <chrono>
#include <iostream>
#include <string>

class ALifeTime
{
public:
    ALifeTime(const std::string &msg = "")
    {
        if (msg.size())
            std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    double unit = 0;
    std::string del_msg;
    std::string unit_string;
    ~ALifeTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        if (del_msg.size())
            std::cout << del_msg << std::chrono::duration<double>(end-start).count() << " s";
        else
            std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s";

        if (unit)
            std::cout << " " << unit/std::chrono::duration<double>(end-start).count() << unit_string << "/s" << std::endl;
        else
            std::cout << std::endl;

    }
    double seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end-start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
