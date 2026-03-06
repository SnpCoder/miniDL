#ifndef __MINIDL_LOG_H__
#define __MINIDL_LOG_H__

#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>  // for std::getenv
#include <memory>
#include <string>

namespace miniDL {

class Log {
   public:
    static void Init() {
        if (_logger) return;

        // 1. 获取动态日志级别
        auto log_level = get_log_level_from_env();

        // 2. 创建控制台输出 Sink (带颜色)
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(log_level);
        // 格式: [时间] [级别] [线程] [文件:行号] 消息
        console_sink->set_pattern("%^%Y-%m-%d %H:%M:%S.%e [%L] [%t] [%s:%#] %v%$");

        // 3. 创建文件输出 Sink (直接写在当前目录，避免文件夹不存在导致崩溃)
        auto file_sink =
            std::make_shared<spdlog::sinks::basic_file_sink_mt>("minidl_run.log", true);
        file_sink->set_level(log_level);
        file_sink->set_pattern("%Y-%m-%d %H:%M:%S.%e [%L] [%t] [%s:%#] %v");

        // 4. 组装 Logger
        _logger = std::make_shared<spdlog::logger>(
            "miniDL", spdlog::sinks_init_list{console_sink, file_sink});
        _logger->set_level(log_level);

        // 5. 遇到 Error 级别立即刷新缓冲区 (防止崩溃时日志没写进硬盘)
        _logger->flush_on(spdlog::level::err);

        spdlog::register_logger(_logger);
    }

    // 极致安全的 C++17 单例访问
    inline static std::shared_ptr<spdlog::logger>& GetLogger() {
        if (!_logger) { Init(); }
        return _logger;
    }

   private:
    // 解析环境变量 (完美保留了你的实现逻辑)
    static spdlog::level::level_enum get_log_level_from_env() {
        const char* env_p = std::getenv("MINIDL_LOG_LEVEL");
        if (env_p) {
            std::string level_str(env_p);
            std::transform(level_str.begin(), level_str.end(), level_str.begin(), ::tolower);
            auto level = spdlog::level::from_str(level_str);
            if (level != spdlog::level::off || level_str == "off") { return level; }
        }
        return spdlog::level::warn;  // 默认 WARN
    }

    // C++17 内联静态成员，彻底告别 new 和裸指针
    inline static std::shared_ptr<spdlog::logger> _logger;
};

}  // namespace miniDL

// ============================================================================
// 宏封装 (使用 SPDLOG_LOGGER_XXX 捕获行号)
// ============================================================================
#define MINIDL_TRACE(...) SPDLOG_LOGGER_TRACE(::miniDL::Log::GetLogger(), __VA_ARGS__)
#define MINIDL_DEBUG(...) SPDLOG_LOGGER_DEBUG(::miniDL::Log::GetLogger(), __VA_ARGS__)
#define MINIDL_INFO(...) SPDLOG_LOGGER_INFO(::miniDL::Log::GetLogger(), __VA_ARGS__)
#define MINIDL_WARN(...) SPDLOG_LOGGER_WARN(::miniDL::Log::GetLogger(), __VA_ARGS__)
#define MINIDL_ERROR(...) SPDLOG_LOGGER_ERROR(::miniDL::Log::GetLogger(), __VA_ARGS__)
#define MINIDL_FATAL(...) SPDLOG_LOGGER_CRITICAL(::miniDL::Log::GetLogger(), __VA_ARGS__)

// 保留你的常亮输出功能 (Always print)
#define MINIDL_PRINT(...) fmt::println(__VA_ARGS__)

#endif  // __MINIDL_LOG_H__