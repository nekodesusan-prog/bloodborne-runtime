/* Kernel semaphores: 32-bit IDs, counted tokens and FIFO waiters.
 * Priority-ordered semaphores are approximated by FIFO order. */
#define _GNU_SOURCE
#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <pthread.h>
#include <errno.h>
#include <time.h>
#define ERR(n) ((int32_t)(UINT32_C(0x80020000)|(n)))
typedef struct Waiter {
    int32_t need, result;
    int done;
    const char *waker; /* BB_WAKE_STATS: the thread that released it */
    pthread_cond_t event;
    struct Waiter *next;
} Waiter;
typedef struct Sema {
    uint32_t id, attr;
    int32_t count, initial, maximum;
    unsigned active;
    int deleted;
    Waiter *first;
    struct Sema *next;
} Sema;
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static Sema *registry;
static uint32_t next_id=1;
static size_t created, deleted, acquired, signaled, timed_out;
static Sema *find(uint32_t id) {
    for (Sema *s=registry;s;s=s->next) if (s->id==id) return s;
    return NULL;
}
static void host_check(int e) {
    if (e) { fprintf(stderr,"STOP: semaphore host error %d\n",e); exit(21); }
}
static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) { perror("clock_gettime"); exit(1); }
    return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec;
}
static ABI int32_t sem_create(uint32_t *out,const char *name,uint32_t attr,
                              int32_t initial,int32_t maximum,const void *options) {
    if (!out || !name || attr>2 || initial<0 || maximum<=0 || initial>maximum) return ERR(22);
    if (options) { fputs("STOP: semaphore options are not implemented\n",stderr); exit(21); }
    Sema *s=calloc(1,sizeof(*s));
    if (!s) return ERR(12);
    pthread_mutex_lock(&lock);
    if (next_id==UINT32_MAX) { pthread_mutex_unlock(&lock); free(s); return ERR(28); }
    s->id=next_id++; s->attr=attr; s->count=s->initial=initial; s->maximum=maximum;
    s->next=registry; registry=s; *out=s->id; ++created;
    printf("Runtime: semaphore id=%u, attr=%u, initial=%d, max=%d\n",s->id,attr,initial,maximum);
    pthread_mutex_unlock(&lock);
    return 0;
}
/* BB_SEMA_SPIN_US=N (0..200, default 0): before the game's main thread sleeps on a semaphore it
 * waits up to N microseconds awake for the count. On an iPad the main thread's semaphore waits in
 * Central Yharnam were about 10,000 per 10 s at 0.02-0.07 ms each (physics and graphics jobs
 * finishing), so going to sleep and being woken was a large part of each. Only the main thread
 * spins: the workers' waits are long and spinning there would only add heat. */
static uint64_t spin_ns(void) {
    static int64_t ns=-1;
    if (ns<0) {
        const char *e=getenv("BB_SEMA_SPIN_US");
        long us=e ? strtol(e,NULL,10) : 0;
        if (us<0) us=0;
        if (us>200) us=200;
        ns=(int64_t)us*1000;
    }
    return (uint64_t)ns;
}
static _Thread_local int is_main=-1;
static size_t spun_waits, spun_won;
static int32_t wait_count(uint32_t id,int32_t need,uint32_t *timeout,int block) {
    pthread_mutex_lock(&lock);
    Sema *s=find(id);
    int32_t result=0;
    /* The short spin: the count read without the lock while spinning (a hint), taken under it. */
    if (s && block && need>0 && s->count<need && !s->deleted && spin_ns() && !(timeout && !*timeout)) {
        if (is_main<0) { const char *n=runtime_thread_name(); is_main=n && !strcmp(n,"main"); }
        if (is_main) {
            ++s->active; ++spun_waits; /* active: the semaphore is not freed while unlocked */
            pthread_mutex_unlock(&lock);
            const uint64_t until=now_ns()+spin_ns();
            while (__atomic_load_n(&s->count,__ATOMIC_RELAXED)<need && !__atomic_load_n(&s->deleted,__ATOMIC_RELAXED) && now_ns()<until)
                for (int i=0;i<32;++i) {
#if defined(__aarch64__)
                    __asm__ volatile("yield");
#elif defined(__x86_64__)
                    __builtin_ia32_pause();
#endif
                }
            pthread_mutex_lock(&lock);
            --s->active;
            if (s->deleted) { if (!s->active) free(s); s=NULL; result=ERR(3); }
            else if (s->count>=need) ++spun_won;
            static uint64_t window;
            const uint64_t now=now_ns();
            if (!window) window=now;
            if (now-window>=10000000000ull) {
                printf("Semaphore spin (%lld us, main thread): %zu waits in 10 s, %.0f%% served without sleeping\n",
                       (long long)(spin_ns()/1000),spun_waits,spun_waits ? 100.0*spun_won/spun_waits : 0.0);
                spun_waits=spun_won=0; window=now;
            }
        }
    }
    if (result) { pthread_mutex_unlock(&lock); return result; }
    if (!s) result=ERR(3);
    else if (need<=0 || need>s->maximum) result=ERR(22);
    else if (s->count>=need) { s->count-=need; ++acquired; runtime_wake_fast(0); }
    else if (!block) { result=ERR(16); runtime_wake_fast(1); }
    else if (timeout && !*timeout) { result=ERR(60); ++timed_out; runtime_wake_fast(2); }
    else {
        /* Priority-ordered semaphores (attr 2) are served FIFO: there is no
           PS4 priority scheduler, host threads run at equal priority. */
        if (getenv("BB_TRACE_SEMA")) fprintf(stderr,"Runtime: blocking wait on semaphore %u (need %d, count %d)\n",id,need,s->count); /* deadlock diagnosis */
        Waiter w={.need=need};
        host_check(bb_cond_init_monotonic(&w.event));
        Waiter **tail=&s->first;
        while (*tail) tail=&(*tail)->next;
        *tail=&w; ++s->active;
        const uint64_t wait_start=runtime_wait_clock();
        uint64_t deadline=timeout ? now_ns()+(uint64_t)*timeout*1000 : 0;
        struct timespec end={.tv_sec=(time_t)(deadline/1000000000),.tv_nsec=(long)(deadline%1000000000)};
        while (!w.done) {
            int e=timeout ? bb_cond_timedwait_monotonic(&w.event,&lock,&end) : pthread_cond_wait(&w.event,&lock);
            if (e==ETIMEDOUT && !w.done) {
                Waiter **p=&s->first;
                while (*p!=&w) p=&(*p)->next;
                *p=w.next; w.result=ERR(60); w.done=1; ++timed_out;
            } else if (e!=ETIMEDOUT) host_check(e);
        }
        const uint64_t waited=runtime_wait_clock()-wait_start;
        runtime_wait_note(2,waited);
        runtime_wake_note(2,waited,w.result ? NULL : w.waker);
        result=w.result;
        if (!result) ++acquired;
        if (timeout) {
            uint64_t now=now_ns();
            *timeout=result ? 0 : (uint32_t)(now>=deadline ? 0 : (deadline-now)/1000);
        }
        host_check(pthread_cond_destroy(&w.event));
        if (!--s->active && s->deleted) free(s);
    }
    pthread_mutex_unlock(&lock);
    return result;
}
static ABI int32_t sem_wait(uint32_t id,int32_t need,uint32_t *timeout) { return wait_count(id,need,timeout,1); }
static ABI int32_t sem_poll(uint32_t id,int32_t need) { return wait_count(id,need,NULL,0); }
static ABI int32_t sem_signal(uint32_t id,int32_t count) {
    pthread_mutex_lock(&lock);
    Sema *s=find(id);
    int32_t result=0;
    if (!s) result=ERR(3);
    else if (count<=0 || count>s->maximum-s->count) result=ERR(22);
    else {
        s->count+=count; ++signaled;
        Waiter **p=&s->first;
        while (*p) {
            Waiter *w=*p;
            if (w->need>s->count) { p=&w->next; continue; }
            s->count-=w->need; *p=w->next; w->done=1;
            if (runtime_wake_stats()) w->waker=runtime_waker_name();
            host_check(pthread_cond_signal(&w->event));
        }
    }
    pthread_mutex_unlock(&lock);
    return result;
}
static int32_t wake_all(Sema *s,int32_t result) {
    int32_t count=0;
    while (s->first) {
        Waiter *w=s->first; s->first=w->next;
        w->result=result; w->done=1; ++count;
        host_check(pthread_cond_signal(&w->event));
    }
    return count;
}
static ABI int32_t sem_cancel(uint32_t id,int32_t count,int32_t *waiters) {
    pthread_mutex_lock(&lock);
    Sema *s=find(id);
    int32_t result=0;
    if (!s) result=ERR(3);
    else if (count>s->maximum) result=ERR(22);
    else {
        int32_t n=wake_all(s,ERR(85));
        if (waiters) *waiters=n;
        s->count=count<0 ? s->initial : count;
    }
    pthread_mutex_unlock(&lock);
    return result;
}
static ABI int32_t sem_delete(uint32_t id) {
    pthread_mutex_lock(&lock);
    Sema **p=&registry;
    while (*p && (*p)->id!=id) p=&(*p)->next;
    if (!*p) { pthread_mutex_unlock(&lock); return ERR(3); }
    Sema *s=*p; *p=s->next; s->deleted=1; ++deleted;
    wake_all(s,ERR(13));
    if (!s->active) free(s);
    pthread_mutex_unlock(&lock);
    return 0;
}
/* Read-only test/diagnostic snapshot, not exposed to guest code. */
unsigned runtime_sema_waiters(uint32_t id) {
    pthread_mutex_lock(&lock);
    Sema *s=find(id); unsigned n=0;
    if (s) for (Waiter *w=s->first;w;w=w->next) ++n;
    pthread_mutex_unlock(&lock);
    return n;
}
uintptr_t runtime_sema_resolve(const char *name) {
    if (!strcmp(name,"188x57JYp0g#p#J")) return (uintptr_t)sem_create;
    if (!strcmp(name,"Zxa0VhQVTsk#p#J")) return (uintptr_t)sem_wait;
    if (!strcmp(name,"4czppHBiriw#p#J")) return (uintptr_t)sem_signal;
    if (!strcmp(name,"12wOHk8ywb0#p#J")) return (uintptr_t)sem_poll;
    if (!strcmp(name,"4DM06U2BNEY#p#J")) return (uintptr_t)sem_cancel;
    if (!strcmp(name,"R1Jvn8bSCW8#p#J")) return (uintptr_t)sem_delete;
    return 0;
}
void runtime_sema_report(void) {
    pthread_mutex_lock(&lock);
    printf("Runtime: semaphores created=%zu, deleted=%zu, acquired=%zu, signals=%zu, timeouts=%zu\n",
           created,deleted,acquired,signaled,timed_out);
    pthread_mutex_unlock(&lock);
}
#else
uintptr_t runtime_sema_resolve(const char *name) { (void)name; return 0; }
void runtime_sema_report(void) { puts("Runtime: Windows semaphore backend not implemented"); }
unsigned runtime_sema_waiters(uint32_t id) { (void)id; return 0; }
#endif
