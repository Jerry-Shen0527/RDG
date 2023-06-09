#pragma once

#ifdef RDG_WITH_OPTIX

#include "./nvrtcConfig.h"

#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#define STRINGIFY(x)  STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR      STRINGIFY(__LINE__)

#if CUDA_NVRTC_ENABLED

#include <nvrtc.h>
// Error check/report helper for users of the C API
#define NVRTC_CHECK_ERROR(func)                                                                  \
    do                                                                                           \
    {                                                                                            \
        nvrtcResult code = func;                                                                 \
        if (code != NVRTC_SUCCESS)                                                               \
            throw std::runtime_error(                                                            \
                "ERROR: " __FILE__ "(" LINE_STR "): " + std::string(nvrtcGetErrorString(code))); \
    } while (0)

#endif

static bool readSourceFile(std::string& str, const std::string& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str());
    if (file.good())
    {
        // Found usable source file
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        str = source_buffer.str();
        return true;
    }
    return false;
}

static void getCuStringFromFile(
    std::string& cu,
    std::string& location,
    const char* sampleDir,
    const char* filename)
{
    std::vector<std::string> source_locations;

    const std::string base_dir = OptiX_DIR;

    // Potential source locations (in priority order)
    if (sampleDir)
        source_locations.push_back(base_dir + '/' + sampleDir + '/' + filename);
    source_locations.push_back(base_dir + "/src/OptiX/" + filename);
    source_locations.push_back(base_dir + "test/src/OptiX/" + filename);

    for (const std::string& loc : source_locations)
    {
        // Try to get source code from file
        if (readSourceFile(cu, loc))
        {
            location = loc;
            return;
        }
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error("Couldn't open source file " + std::string(filename));
}

static std::string g_nvrtcLog;

static void getPtxFromCuString(
    std::string& ptx,
    const char* sample_name,
    const char* cu_source,
    const char* name,
    const char** log_string)
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

    // Gather NVRTC options
    std::vector<const char*> options;

    const std::string base_dir = OptiX_DIR;

    // Set sample dir as the primary include path
    std::string sample_dir;
    if (sample_name)
    {
        sample_dir = std::string("-I") + base_dir + '/' + sample_name;
        options.push_back(sample_dir.c_str());
    }

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char* abs_dirs[] = { OptiX_ABSOLUTE_INCLUDE_DIRS };
    const char* rel_dirs[] = { OptiX_RELATIVE_INCLUDE_DIRS };

    for (const char* dir : abs_dirs)
    {
        include_dirs.push_back(std::string("-I") + dir);
    }
    for (const char* dir : rel_dirs)
    {
        include_dirs.push_back("-I" + base_dir + '/' + dir);
    }
    for (const std::string& dir : include_dirs)
    {
        options.push_back(dir.c_str());
    }

    // Collect NVRTC options
    const char* compiler_options[] = { CUDA_NVRTC_OPTIONS };
    std::copy(
        std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    g_nvrtcLog.resize(log_size);
    if (log_size > 1)
    {
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
        if (log_string)
            *log_string = g_nvrtcLog.c_str();
    }
    if (compileRes != NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

    // Cleanup
    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
}

struct PtxSourceCache
{
    std::map<std::string, std::string*> map;
    ~PtxSourceCache()
    {
        for (std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end();
             ++it)
            delete it->second;
    }
};
static PtxSourceCache g_ptxSourceCache;

inline const char* getPtxString(
    const char* sample,
    const char* shaderDir,
    const char* filename,
    const char** log = nullptr)
{
    if (log)
        *log = NULL;

    std::string *ptx, cu;
    std::string key = std::string(filename) + ";" + (sample ? sample : "");
    std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find(key);

    if (elem == g_ptxSourceCache.map.end())
    {
        ptx = new std::string();
#if CUDA_NVRTC_ENABLED
        std::string location;
        getCuStringFromFile(cu, location, shaderDir, filename);
        getPtxFromCuString(*ptx, sample, cu.c_str(), location.c_str(), log);
#else
        getPtxStringFromFile(*ptx, sample, filename);
#endif
        g_ptxSourceCache.map[key] = ptx;
    }
    else
    {
        ptx = elem->second;
    }

    return ptx->c_str();
}

#endif