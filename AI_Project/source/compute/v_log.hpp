//
// Created by dlwog on 25. 11. 20..
//

#ifndef MYPROJECT_MML_LOG_HPP
#define MYPROJECT_MML_LOG_HPP


#define v_LOG(...)       v_log_internal(v_LOG_LEVEL_NONE , __VA_ARGS__)
#define LOG_INFO(...)  v_log_internal(v_LOG_LEVEL_INFO , __VA_ARGS__)
#define v_LOG_WARN(...)  v_log_internal(v_LOG_LEVEL_WARN , __VA_ARGS__)
#define v_LOG_ERROR(...) v_log_internal(v_LOG_LEVEL_ERROR, __VA_ARGS__)
#define v_LOG_DEBUG(...) v_log_internal(v_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define v_LOG_CONT(...)  v_log_internal(v_LOG_LEVEL_CONT , __VA_ARGS__)
#define v_DEBUG 10

#if (v_DEBUG >= 1)
#define v_PRINT_DEBUG(...) v_LOG_DEBUG(__VA_ARGS__)
#else
#define PRINT_DEBUG(...)
#endif

#if (v_DEBUG >= 5)
#define v_PRINT_DEBUG_5(...) v_LOG_DEBUG(__VA_ARGS__)
#else
#define v_PRINT_DEBUG_5(...)
#endif

#if (v_DEBUG >= 10)
#define v_PRINT_DEBUG_10(...) v_LOG_DEBUG(__VA_ARGS__)
#else
#define v_PRINT_DEBUG_10(...)
#endif


#endif //MYPROJECT_MML_LOG_HPP