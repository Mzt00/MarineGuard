#include "sync.h"
#include "logger.h"

/* ══════════════════════════════════════════════
   1.  PRODUCER – CONSUMER  (semaphores + mutex)
   ══════════════════════════════════════════════ */
#define BUFFER_SIZE 5
static int buffer[BUFFER_SIZE];
static int  buf_in = 0, buf_out = 0;
static sem_t sem_empty, sem_full;
static pthread_mutex_t buf_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile int pc_done = 0;

static void *producer_func(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < 4; i++) {
        int item = id * 100 + i;
        sem_wait(&sem_empty);
        pthread_mutex_lock(&buf_mutex);
        buffer[buf_in] = item;
        buf_in = (buf_in + 1) % BUFFER_SIZE;
        printf("  [Producer %d] Produced item %d\n", id, item);
        fflush(stdout);
        pthread_mutex_unlock(&buf_mutex);
        sem_post(&sem_full);
        usleep(100000);
    }
    return NULL;
}

static void *consumer_func(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < 4; i++) {
        sem_wait(&sem_full);
        pthread_mutex_lock(&buf_mutex);
        int item = buffer[buf_out];
        buf_out = (buf_out + 1) % BUFFER_SIZE;
        printf("  [Consumer %d] Consumed item %d\n", id, item);
        fflush(stdout);
        pthread_mutex_unlock(&buf_mutex);
        sem_post(&sem_empty);
        usleep(150000);
    }
    return NULL;
}

void sync_producer_consumer(void) {
    printf("\n  [Sync] Producer-Consumer Problem\n");
    printf("  ──────────────────────────────────\n");
    sem_init(&sem_empty, 0, BUFFER_SIZE);
    sem_init(&sem_full,  0, 0);

    pthread_t p1, p2, c1;
    int id1 = 1, id2 = 2, cid = 1;
    pthread_create(&p1, NULL, producer_func, &id1);
    pthread_create(&p2, NULL, producer_func, &id2);
    pthread_create(&c1, NULL, consumer_func, &cid);

    pthread_join(p1, NULL);
    pthread_join(p2, NULL);
    pthread_join(c1, NULL);

    sem_destroy(&sem_empty);
    sem_destroy(&sem_full);
    printf("  [Sync] Producer-Consumer complete.\n\n");
    LOG_INFO("Sync: producer-consumer done");
    pc_done = 1;
}

/* ══════════════════════════════════════════════
   2.  READER – WRITER  (read preference)
   ══════════════════════════════════════════════ */
static int shared_data      = 0;
static int reader_count     = 0;
static pthread_mutex_t rw_mutex  = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t cnt_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *reader_func(void *arg) {
    int id = *(int *)arg;
    pthread_mutex_lock(&cnt_mutex);
    reader_count++;
    if (reader_count == 1) pthread_mutex_lock(&rw_mutex);
    pthread_mutex_unlock(&cnt_mutex);

    printf("  [Reader %d] Reading data = %d\n", id, shared_data);
    fflush(stdout);
    usleep(100000);

    pthread_mutex_lock(&cnt_mutex);
    reader_count--;
    if (reader_count == 0) pthread_mutex_unlock(&rw_mutex);
    pthread_mutex_unlock(&cnt_mutex);
    return NULL;
}

static void *writer_func(void *arg) {
    int id = *(int *)arg;
    pthread_mutex_lock(&rw_mutex);
    shared_data += 10;
    printf("  [Writer %d] Wrote data = %d\n", id, shared_data);
    fflush(stdout);
    usleep(100000);
    pthread_mutex_unlock(&rw_mutex);
    return NULL;
}

void sync_reader_writer(void) {
    printf("\n  [Sync] Reader-Writer Problem\n");
    printf("  ──────────────────────────────\n");
    pthread_t r[3], w[2];
    int ids[5] = {1, 2, 3, 1, 2};
    for (int i = 0; i < 3; i++) pthread_create(&r[i], NULL, reader_func, &ids[i]);
    for (int i = 0; i < 2; i++) pthread_create(&w[i], NULL, writer_func, &ids[i]);
    for (int i = 0; i < 3; i++) pthread_join(r[i], NULL);
    for (int i = 0; i < 2; i++) pthread_join(w[i], NULL);
    printf("  [Sync] Reader-Writer complete.\n\n");
    LOG_INFO("Sync: reader-writer done");
}

/* ══════════════════════════════════════════════
   3.  DINING PHILOSOPHERS  (5 philosophers)
   ══════════════════════════════════════════════ */
#define NUM_PHIL 5
static pthread_mutex_t forks[NUM_PHIL];

typedef struct { int id; } PhilArg;

static void *philosopher(void *arg) {
    int id   = ((PhilArg *)arg)->id;
    int left  = id;
    int right = (id + 1) % NUM_PHIL;

    /* Deadlock prevention: odd philosopher picks right fork first */
    if (id % 2 == 0) {
        pthread_mutex_lock(&forks[left]);
        pthread_mutex_lock(&forks[right]);
    } else {
        pthread_mutex_lock(&forks[right]);
        pthread_mutex_lock(&forks[left]);
    }

    printf("  [Philosopher %d] Eating...\n", id);
    fflush(stdout);
    usleep(200000);

    pthread_mutex_unlock(&forks[left]);
    pthread_mutex_unlock(&forks[right]);
    printf("  [Philosopher %d] Finished eating, thinking...\n", id);
    fflush(stdout);
    return NULL;
}

void sync_dining_philosopher(void) {
    printf("\n  [Sync] Dining Philosophers Problem\n");
    printf("  ────────────────────────────────────\n");
    for (int i = 0; i < NUM_PHIL; i++)
        pthread_mutex_init(&forks[i], NULL);

    pthread_t threads[NUM_PHIL];
    PhilArg   args[NUM_PHIL];
    for (int i = 0; i < NUM_PHIL; i++) {
        args[i].id = i;
        pthread_create(&threads[i], NULL, philosopher, &args[i]);
    }
    for (int i = 0; i < NUM_PHIL; i++)
        pthread_join(threads[i], NULL);

    for (int i = 0; i < NUM_PHIL; i++)
        pthread_mutex_destroy(&forks[i]);
    printf("  [Sync] Dining Philosophers complete.\n\n");
    LOG_INFO("Sync: dining philosophers done");
}
