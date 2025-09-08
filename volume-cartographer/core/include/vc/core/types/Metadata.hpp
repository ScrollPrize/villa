#pragma once


#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include <filesystem>


class Metadata
{

public:
    Metadata() = default;
    explicit Metadata(std::filesystem::path fileLocation);
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    void setPath(const std::filesystem::path& path) { path_ = path; }
    void save() { save(path_); }
    void save(const std::filesystem::path& path);
    [[nodiscard]] bool hasKey(const std::string& key) const { return json_.count(key) > 0; }
    template <typename T>
    T get(const std::string& key) const
    {
        if (json_.find(key) == json_.end()) {
            auto msg = "could not find key '" + key + "' in metadata";
            throw std::runtime_error(msg);
        }
        return json_[key].get<T>();
    }
    template <typename T>
    void set(const std::string& key, T value)
    {
        json_[key] = value;
    }
    void printString() const { std::cout << json_ << std::endl; }
    void printObject() const { std::cout << json_.dump(4) << std::endl; }
    /**@}*/
protected:
    nlohmann::json json_;
    std::filesystem::path path_;
};

