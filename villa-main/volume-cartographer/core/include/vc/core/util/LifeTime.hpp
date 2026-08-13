#pragma once
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

class ALifeTime
{
public:
    ALifeTime(const std::string &msg = "")
    {
        if (msg.size())
            std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
        _last_mark = start;
    }
    double unit = 0;
    std::string del_msg;
    std::string unit_string;
    ~ALifeTime()
    {
    }

    void mark(const std::string& label) {
        auto now = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(now - _last_mark).count();
        _marks.push_back({label, seconds});
        _last_mark = now;
    }

    std::string report() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        for(const auto& mark : _marks) {
            ss << mark.first << ": " << mark.second << "s ";
        }
        return ss.str();
    }

    const std::vector<std::pair<std::string, double>>& getMarks() const {
        return _marks;
    }

    double seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end-start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> _last_mark;
    std::vector<std::pair<std::string, double>> _marks;
};
