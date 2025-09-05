#pragma once

/** @file */

#include <string>



/**
 * @class IOException
 * @brief IO operation exception
 *
 * Should be thrown when encountering errors reading or writing data.
 *
 * Based on
 * <a href="http://stackoverflow.com/a/8152888">this implementation</a>.
 *
 * @ingroup Types
 */
class IOException : public std::exception
{
public:
    /**@{*/
    /** Constructor */
    explicit IOException(const char* msg) : msg_(msg) {}

    /** @copydoc IOException(const char* msg) */
    explicit IOException(std::string msg) : msg_(std::move(msg)) {}
    /**@}*/

    /** Return exception message */
    const char* what() const noexcept override { return msg_.c_str(); }

protected:
    std::string msg_;
};

