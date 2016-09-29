typedef unsigned char uchar;

typedef struct {

} kernel_exec_ctx_t;

typedef struct {

} discovery_ctx_t;

typedef struct {

} scheduling_ctx_t;

#define QUIT 0
#define KERNEL_1 1
#define KERNEL_2 2

typedef struct {
  uchar TYPE;
} task_t;

void discovery_protocol(discovery_ctx_t *);

void run_as_scheduler(discovery_ctx_t *, scheduling_ctx_t *, kernel_exec_ctx_t *, kernel_exec_ctx_t *);

unsigned participating_group_id(discovery_ctx_t *);

task_t get_task_from_scheduler(scheduling_ctx_t *, unsigned);
