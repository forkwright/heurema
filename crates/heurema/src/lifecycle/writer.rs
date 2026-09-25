//! The lifecycle writer: one per backend, shared by every lifecycle over it.

use std::fmt;
use std::marker::PhantomData;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};

use crate::HeuremaError;
use crate::error::WriterHeldSnafu;

/// The writer every lifecycle over one backend shares, returned by
/// [`LifecycleBackend::writer`](super::LifecycleBackend::writer). It guards
/// no data: holding it means no other operation that holds it for its whole
/// run, as every lifecycle operation does, is between its head read and its
/// publish.
///
/// [`acquire`](Self::acquire) waits while another thread holds the writer,
/// and refuses, with [`HeuremaError::WriterHeld`], a thread that already
/// holds it: that thread would otherwise wait for itself forever. The
/// returned [`WriterGuard`] releases the writer when it is dropped.
///
/// INVARIANT: a poisoned internal mutex holds consistent state, because
/// every critical section only compares or assigns plain fields and calls
/// nothing that can panic, so its poison is ignored. The guard's `Drop`
/// could not refuse anyway.
///
/// # Panics and leaks
///
/// A panic while a guard is held releases the writer during unwinding. The
/// writer protects no in-memory state, and every backend write is atomic,
/// so nothing is left half done. Leaking a guard (`mem::forget` of a
/// [`Prepared`](crate::Prepared) or [`Staged`](crate::Staged)) keeps the
/// writer held for good.
///
/// # Fairness
///
/// Waiters are not served in FIFO order, as with [`std::sync::Mutex`].
///
/// # Threads and async
///
/// Re-entry is detected per thread, and a [`ThreadId`] is never reused. On
/// a single-threaded async executor, a second task on the same thread is
/// therefore refused rather than deadlocked.
///
/// WARNING: lock order: a thread that holds one backend's writer and takes
/// another's can deadlock against a thread doing the reverse. Take the
/// writers of several backends in one fixed order.
///
/// # Examples
///
/// ```
/// use heurema::{HeuremaError, WriterLock};
///
/// let lock = WriterLock::new();
/// let guard = lock.acquire()?;
/// assert!(matches!(
///     lock.acquire(),
///     Err(HeuremaError::WriterHeld { .. })
/// ));
/// drop(guard);
/// let _again = lock.acquire()?;
/// # Ok::<(), HeuremaError>(())
/// ```
pub struct WriterLock {
    state: Mutex<WriterState>,
    released: Condvar,
}

/// Who holds the writer, and how many threads wait for it.
struct WriterState {
    owner: Option<ThreadId>,
    waiting: usize,
}

impl WriterLock {
    /// A writer no thread holds.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(WriterState {
                owner: None,
                waiting: 0,
            }),
            released: Condvar::new(),
        }
    }

    /// Takes the writer, waiting while another thread holds it.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::WriterHeld`] when the calling thread already holds
    /// it.
    #[track_caller]
    pub fn acquire(&self) -> Result<WriterGuard<'_>, HeuremaError> {
        let current = thread::current().id();
        let mut state = self.state();
        if state.owner == Some(current) {
            return WriterHeldSnafu.fail();
        }
        state.waiting += 1;
        let mut state = self
            .released
            .wait_while(state, |state| state.owner.is_some())
            .unwrap_or_else(PoisonError::into_inner);
        state.waiting -= 1;
        state.owner = Some(current);
        Ok(WriterGuard {
            lock: self,
            _not_send: PhantomData,
        })
    }

    /// The writer's state, locked; see the invariant on [`WriterLock`] for
    /// why poison is ignored.
    fn state(&self) -> MutexGuard<'_, WriterState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// How many threads wait in [`acquire`](Self::acquire).
    fn waiting(&self) -> usize {
        self.state().waiting
    }
}

impl Default for WriterLock {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for WriterLock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let held = self.state().owner.is_some();
        let waiting = self.waiting();
        formatter
            .debug_struct("WriterLock")
            .field("held", &held)
            .field("waiting", &waiting)
            .finish_non_exhaustive()
    }
}

/// Holds a backend's [`WriterLock`] until it is dropped.
///
/// A guard stays on the thread that took it: it is `Sync` but not `Send`.
///
/// ```
/// fn shareable<T: Sync>() {}
/// shareable::<heurema::WriterGuard<'static>>();
/// ```
///
/// This is the example above with only the bound changed:
///
/// ```compile_fail
/// fn shareable<T: Send>() {}
/// shareable::<heurema::WriterGuard<'static>>();
/// ```
#[must_use = "the writer is released as soon as the guard is dropped"]
#[derive(Debug)]
pub struct WriterGuard<'a> {
    lock: &'a WriterLock,
    /// WHY: `!Send` (and `Sync`), like a `MutexGuard`, so the thread
    /// recorded as owner is always the thread holding the guard.
    _not_send: PhantomData<MutexGuard<'static, ()>>,
}

impl Drop for WriterGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.lock.state();
        state.owner = None;
        drop(state);
        self.lock.released.notify_one();
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a writer that cannot be taken fails the test at the call that took it"
)]
mod tests {
    use std::sync::mpsc;

    use super::*;
    use crate::ErrorCategory;

    #[test]
    fn a_free_writer_is_taken_and_released_by_its_guard() {
        let lock = WriterLock::new();
        let guard = lock.acquire().expect("a free writer is taken");
        assert_eq!(lock.state().owner, Some(thread::current().id()));
        drop(guard);
        assert_eq!(lock.state().owner, None, "dropping the guard releases it");
        let _again = lock.acquire().expect("a released writer is taken again");
    }

    #[test]
    fn the_holding_thread_is_refused_instead_of_deadlocking() {
        let lock = WriterLock::new();
        let guard = lock.acquire().expect("a free writer is taken");
        let error = lock.acquire().expect_err("the holding thread is refused");
        assert!(
            matches!(error, HeuremaError::WriterHeld { .. }),
            "{error:?}"
        );
        assert_eq!(error.category(), ErrorCategory::Refused);
        drop(guard);
        let _again = lock.acquire().expect("a released writer is taken again");
    }

    #[test]
    fn another_thread_waits_until_the_writer_is_released() {
        let lock = WriterLock::new();
        let guard = lock.acquire().expect("a free writer is taken");
        let (sender, receiver) = mpsc::channel();
        thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                // WHY moved in: the thread's end, a panic included, then
                // drops the sender and disconnects the channel, which the
                // loop below reports instead of polling forever.
                let sender = sender;
                let taken = lock.acquire().is_ok();
                sender.send(taken).expect("the main thread listens");
            });
            // WHY no sleep: the waiting counter says when the other thread is
            // inside `acquire`, so the test never guesses at timing.
            loop {
                if lock.waiting() == 1 {
                    break;
                }
                match receiver.try_recv() {
                    Ok(taken) => panic!("the other thread returned ({taken}) instead of waiting"),
                    Err(mpsc::TryRecvError::Disconnected) => match waiter.join() {
                        Err(panic) => std::panic::resume_unwind(panic),
                        Ok(()) => panic!("the other thread stopped without a result"),
                    },
                    Err(mpsc::TryRecvError::Empty) => {}
                }
                thread::yield_now();
            }
            assert!(
                receiver.try_recv().is_err(),
                "a waiting thread has not taken the writer"
            );
            drop(guard);
            assert_eq!(receiver.recv(), Ok(true), "released, it is taken");
        });
    }

    #[test]
    fn a_panic_while_holding_the_writer_releases_it() {
        let lock = WriterLock::new();
        let joined = thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _guard = lock.acquire().expect("a free writer is taken");
                    panic!("a writer panics while holding the guard");
                })
                .join()
        });
        assert!(joined.is_err(), "the holding thread panicked");
        let _again = lock
            .acquire()
            .expect("unwinding dropped the guard and released the writer");
    }
}
