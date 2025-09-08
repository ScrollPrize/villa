#pragma once

#include <sys/time.h>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* Types */
typedef int omp_lock_t;
typedef int omp_nest_lock_t;

typedef enum omp_sched_t {
    omp_sched_static = 1,
    omp_sched_dynamic = 2,
    omp_sched_guided = 3,
    omp_sched_auto = 4
} omp_sched_t;

typedef enum omp_proc_bind_t {
    omp_proc_bind_false = 0,
    omp_proc_bind_true = 1,
    omp_proc_bind_master = 2,
    omp_proc_bind_close = 3,
    omp_proc_bind_spread = 4
} omp_proc_bind_t;

/* Thread functions */
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_max_threads(void) { return 1; }
static inline void omp_set_num_threads(int num) { (void)num; }
static inline int omp_get_num_procs(void) { return 1; }
static inline int omp_in_parallel(void) { return 0; }
static inline void omp_set_dynamic(int dynamic) { (void)dynamic; }
static inline int omp_get_dynamic(void) { return 0; }
static inline void omp_set_nested(int nested) { (void)nested; }
static inline int omp_get_nested(void) { return 0; }
static inline int omp_get_thread_limit(void) { return 1; }
static inline void omp_set_max_active_levels(int levels) { (void)levels; }
static inline int omp_get_max_active_levels(void) { return 1; }
static inline int omp_get_level(void) { return 0; }
static inline int omp_get_ancestor_thread_num(int level) { (void)level; return 0; }
static inline int omp_get_team_size(int level) { (void)level; return 1; }
static inline int omp_get_active_level(void) { return 0; }

/* Lock functions */
static inline void omp_init_lock(omp_lock_t* lock) { *lock = 0; }
static inline void omp_destroy_lock(omp_lock_t* lock) { (void)lock; }
static inline void omp_set_lock(omp_lock_t* lock) { *lock = 1; }
static inline void omp_unset_lock(omp_lock_t* lock) { *lock = 0; }
static inline int omp_test_lock(omp_lock_t* lock) {
    if (*lock == 0) {
        *lock = 1;
        return 1;
    }
    return 0;
}

static inline void omp_init_nest_lock(omp_nest_lock_t* lock) { *lock = 0; }
static inline void omp_destroy_nest_lock(omp_nest_lock_t* lock) { (void)lock; }
static inline void omp_set_nest_lock(omp_nest_lock_t* lock) { (*lock)++; }
static inline void omp_unset_nest_lock(omp_nest_lock_t* lock) { if (*lock > 0) (*lock)--; }
static inline int omp_test_nest_lock(omp_nest_lock_t* lock) {
    (*lock)++;
    return *lock;
}

/* Timing functions */
static inline double omp_get_wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static inline double omp_get_wtick(void) {
    return 1.0e-6;
}

/* Schedule functions */
static inline void omp_set_schedule(omp_sched_t kind, int chunk_size) {
    (void)kind;
    (void)chunk_size;
}

static inline void omp_get_schedule(omp_sched_t* kind, int* chunk_size) {
    *kind = omp_sched_static;
    *chunk_size = 0;
}

/* Proc bind functions */
static inline omp_proc_bind_t omp_get_proc_bind(void) { return omp_proc_bind_false; }
static inline int omp_get_num_places(void) { return 0; }
static inline int omp_get_place_num_procs(int place_num) { (void)place_num; return 0; }
static inline void omp_get_place_proc_ids(int place_num, int* ids) { (void)place_num; (void)ids; }
static inline int omp_get_place_num(void) { return -1; }
static inline int omp_get_partition_num_places(void) { return 0; }
static inline void omp_get_partition_place_nums(int* place_nums) { (void)place_nums; }

/* Device functions */
static inline int omp_get_initial_device(void) { return -1; }
static inline int omp_get_num_devices(void) { return 0; }
static inline int omp_get_default_device(void) { return 0; }
static inline void omp_set_default_device(int device_num) { (void)device_num; }
static inline int omp_is_initial_device(void) { return 1; }

/* Cancellation functions */
static inline int omp_get_cancellation(void) { return 0; }
static inline int omp_get_max_task_priority(void) { return 0; }

#ifdef __cplusplus
}
#endif
