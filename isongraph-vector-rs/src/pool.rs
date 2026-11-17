//! A small persistent thread pool for searches, folds and index builds.
//!
//! The previous code spawned fresh threads (`std::thread::scope`) for every
//! parallel scan. A [`SearchPool`] keeps its workers alive and parked, so a
//! parallel step costs a wake-up instead of a thread creation.
//!
//! The one primitive is [`SearchPool::run`]: "run this closure on up to `n`
//! participants and return when all of them are done". The calling thread is
//! always participant 0, so a call makes progress even when every worker is busy
//! serving another caller (no deadlock, no queueing behind a long job), and many
//! callers can share one pool. The closure is *work-stealing style*: every
//! participant loops claiming units of work from a counter the closure owns, so
//! the result does not depend on how many participants actually joined.
//!
//! No external dependency (rayon would do the same job; this is about 100 lines
//! and easy to port). It contains the crate's only lifetime-erasing `unsafe`,
//! argued at the `SAFETY` comments.

use std::collections::VecDeque;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;

type Work<'a> = dyn Fn(usize) + Sync + 'a;

/// One `run` call: the closure (lifetime erased) and who is executing it.
struct Job {
    /// Points at the caller's closure. Only dereferenced by a thread that
    /// registered itself in `state.running` while `state.open` was true; the
    /// caller does not return before `running` is back to zero.
    work: *const Work<'static>,
    state: Mutex<JobState>,
    idle: Condvar,
}

struct JobState {
    /// Helpers may still join.
    open: bool,
    /// Participants currently inside the closure (helpers only).
    running: usize,
    /// Participant numbers handed out so far (0 is the caller).
    started: usize,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

// SAFETY: `work` is a pointer to a `Sync` closure and is only used under the
// protocol described on `Job::work`; everything else is Mutex/Condvar.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct Queue {
    tickets: VecDeque<Arc<Job>>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic inside a participant is caught and re-raised by `run`; the
    // bookkeeping itself never panics, so a poisoned lock is still consistent.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A fixed set of parked worker threads shared by the searches of any number of
/// stores. See the module docs.
pub struct SearchPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    threads: usize,
}

impl SearchPool {
    /// A pool that can run `threads` participants at once, the caller of
    /// [`run`](Self::run) included, so it starts `threads - 1` workers. `1` (or
    /// `0`) starts none and every `run` is a plain call.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                tickets: VecDeque::new(),
                shutdown: false,
            }),
            wake: Condvar::new(),
        });
        let workers = (1..threads)
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("isongraph-search-{i}"))
                    .spawn(move || worker_loop(&shared))
                    .expect("could not start a search worker thread")
            })
            .collect();
        Self {
            shared,
            workers,
            threads,
        }
    }

    /// The process-wide pool every store uses unless given its own. Created on
    /// first use with `ISONGRAPH_VECTOR_THREADS` threads, or one per logical
    /// CPU when that is unset.
    pub fn global() -> Arc<SearchPool> {
        static GLOBAL: OnceLock<Arc<SearchPool>> = OnceLock::new();
        GLOBAL
            .get_or_init(|| Arc::new(SearchPool::new(Self::default_threads())))
            .clone()
    }

    /// The size the shared pool has (or will have): `ISONGRAPH_VECTOR_THREADS`
    /// if set to a positive number, else one per logical CPU. Cheap; does not
    /// start the pool.
    pub fn default_threads() -> usize {
        static N: OnceLock<usize> = OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("ISONGRAPH_VECTOR_THREADS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1)
                })
        })
    }

    /// How many participants one `run` can have at most.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `work(i)` on up to `participants` threads (the caller is `i == 0`;
    /// helpers get 1, 2, ...) and return when every one that started is done.
    ///
    /// Helpers join only if a worker is free before the caller finishes, so
    /// `work` must be able to complete the whole job on a single participant
    /// (claim work units from a shared counter). A panic in any participant is
    /// re-raised here after all of them have stopped.
    pub fn run(&self, participants: usize, work: &(dyn Fn(usize) + Sync)) {
        let helpers = participants.min(self.threads).saturating_sub(1);
        if helpers == 0 {
            work(0);
            return;
        }
        // SAFETY: only the lifetime is erased. `work` outlives every use of the
        // pointer because this function does not return (or unwind) until
        // `state.open` is false and `state.running` is zero, and a helper only
        // dereferences the pointer after incrementing `running` while `open`.
        let erased: *const Work<'static> =
            unsafe { std::mem::transmute::<*const Work<'_>, *const Work<'static>>(work) };
        let job = Arc::new(Job {
            work: erased,
            state: Mutex::new(JobState {
                open: true,
                running: 0,
                started: 1,
                panic: None,
            }),
            idle: Condvar::new(),
        });
        {
            let mut q = lock(&self.shared.queue);
            for _ in 0..helpers {
                q.tickets.push_back(job.clone());
            }
        }
        if helpers == 1 {
            self.shared.wake.notify_one();
        } else {
            self.shared.wake.notify_all();
        }

        let mine = catch_unwind(AssertUnwindSafe(|| work(0)));

        let mut state = lock(&job.state);
        state.open = false;
        while state.running > 0 {
            state = job.idle.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        let theirs = state.panic.take();
        drop(state);
        if let Err(payload) = mine {
            resume_unwind(payload);
        }
        if let Some(payload) = theirs {
            resume_unwind(payload);
        }
    }
}

fn worker_loop(shared: &Shared) {
    loop {
        let job = {
            let mut q = lock(&shared.queue);
            loop {
                if let Some(job) = q.tickets.pop_front() {
                    break job;
                }
                if q.shutdown {
                    return;
                }
                q = shared.wake.wait(q).unwrap_or_else(|p| p.into_inner());
            }
        };
        // Join the job only if its caller has not already finished.
        let id = {
            let mut s = lock(&job.state);
            if !s.open {
                continue;
            }
            s.running += 1;
            s.started += 1;
            s.started - 1
        };
        // SAFETY: `running` was incremented while `open`, so the caller is
        // still waiting in `run` and the closure behind `work` is alive.
        let outcome = catch_unwind(AssertUnwindSafe(|| unsafe { (*job.work)(id) }));
        let mut s = lock(&job.state);
        if let Err(payload) = outcome {
            s.panic.get_or_insert(payload);
        }
        s.running -= 1;
        if s.running == 0 {
            job.idle.notify_all();
        }
    }
}

impl Drop for SearchPool {
    fn drop(&mut self) {
        lock(&self.shared.queue).shutdown = true;
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl std::fmt::Debug for SearchPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchPool")
            .field("threads", &self.threads)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_unit_of_work_runs_exactly_once_whoever_joins() {
        let pool = SearchPool::new(6);
        for participants in [1usize, 2, 6, 64] {
            let next = AtomicUsize::new(0);
            let hits: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
            pool.run(participants, &|_id| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= hits.len() {
                    break;
                }
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn many_callers_share_one_pool_without_deadlock() {
        let pool = Arc::new(SearchPool::new(4));
        std::thread::scope(|s| {
            for t in 0..12 {
                let pool = pool.clone();
                s.spawn(move || {
                    for round in 0..50 {
                        let sum = AtomicUsize::new(0);
                        let next = AtomicUsize::new(0);
                        pool.run(4, &|_| loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= 100 {
                                break;
                            }
                            sum.fetch_add(i, Ordering::Relaxed);
                        });
                        assert_eq!(
                            sum.load(Ordering::Relaxed),
                            4950,
                            "caller {t} round {round}"
                        );
                    }
                });
            }
        });
    }

    #[test]
    fn a_panicking_participant_is_reraised_and_the_pool_survives() {
        let pool = SearchPool::new(4);
        let next = AtomicUsize::new(0);
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(4, &|_| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= 200 {
                    break;
                }
                if i == 77 {
                    panic!("boom");
                }
            });
        }));
        assert!(result.is_err());
        let ok = AtomicUsize::new(0);
        pool.run(4, &|_| {
            ok.fetch_add(1, Ordering::Relaxed);
        });
        assert!(ok.load(Ordering::Relaxed) >= 1);
    }

    #[test]
    fn a_single_thread_pool_is_a_plain_call() {
        let pool = SearchPool::new(1);
        let ran = AtomicUsize::new(0);
        pool.run(8, &|id| {
            assert_eq!(id, 0);
            ran.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(ran.load(Ordering::Relaxed), 1);
    }
}
