#include "Settings.h"

namespace DC_VSLAM
{

Timer::Timer(const char* _message /* "Timer took" */) : message(_message)
{
    start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;

    float ms = duration.count() * 1000.0f;
    Logging(message,ms,2,"ms");
}

ConfigFile::ConfigFile(const char* config /*"config.yaml"*/) : configPath(config)
{
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.find_last_of("/\\"));
    try
    {
        configNode = YAML::LoadFile(dir_path + "/../config/" + configPath);
    }
    catch(const YAML::BadFile& e)
    {
        std::cerr << "No config file named " << configPath << " in path " << dir_path + "/../config/" + configPath << std::endl;
        badFile = true;
    }
}


} // namespace DC_VSLAM