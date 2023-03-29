#ifndef SETTINGS_H
#define SETTINGS_H

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <chrono>

namespace DC_VSLAM
{

/**
 * @brief Logging Function
 * @param com Comment on the value to be printed
 * @param toPrint Value to be printed
 * @param level Level of Importance.
 * DEBUG = 0,
 * INFO = 1,
 * WARNING = 2,
 * ERROR = 3
 */
class Logging
{
    enum Level
    {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        NOCOUT
    };

    Level curLevel {WARNING};

    public:
        template <typename T>
        Logging(const char* com, const T& toPrint, const int level, const char* measurements = "")
        {
            if (curLevel != Level::NOCOUT)
                if ((curLevel <= level) && (level < 4))
                {
                    switch (level)
                    {
                        case Level::DEBUG :
                            std::cout << "[DEBUG] : " << com << " " << toPrint << measurements  << '\n';
                            break;
                        case Level::INFO :
                            std::cout << "[INFO] : " << com << " " << toPrint << measurements  << '\n';
                            break;
                        case Level::WARNING :
                            std::cout << "[WARNING] : " << com << " " << toPrint << measurements  << '\n';
                            break;
                        case Level::ERROR :
                            std::cout << "[ERROR] : " << com << " " << toPrint << measurements << '\n';
                            break;
                        default : 
                            break;
                    }
                }
        }

};

class Timer
{
    std::chrono::_V2::system_clock::time_point start, end;
    std::chrono::duration<float> duration;
    const char* message;
    public:
        Timer(const char* _message = "Timer took");
        ~Timer();
};


class ConfigFile
{
    private:
        const char* configPath;

    public:
        bool badFile = false;
        YAML::Node configNode;
        ConfigFile(const char* config = "config.yaml");
        template<typename T> 
        T getValue(const std::string& first = "", const std::string& second = "", const std::string& third = "")
        {
            if (!third.empty())
                return configNode[first][second][third].as<T>();
            else if (!second.empty())
                return configNode[first][second].as<T>();
            else
                return configNode[first].as<T>();
        }
};

} // namespace DC_VSLAM


#endif // SETTINGS_H