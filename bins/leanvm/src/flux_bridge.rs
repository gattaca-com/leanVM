//! `--flux`: publish flux-profiler rings and mirror every `tracing` span
//! as a flux frame, so the existing stage spans and `#[timed]` share one trace.

use std::cell::RefCell;

use flux_profiler::TimerGuard;
use tracing::{Subscriber, span::Id};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

thread_local! {
    static OPEN: RefCell<Vec<TimerGuard>> = const { RefCell::new(Vec::new()) };
}

struct FluxLayer;

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for FluxLayer {
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let guard = TimerGuard::new(span.metadata().name());
            OPEN.with(|open| open.borrow_mut().push(guard));
        }
    }

    fn on_exit(&self, _id: &Id, _ctx: Context<'_, S>) {
        OPEN.with(|open| drop(open.borrow_mut().pop()));
    }
}

/// Seconds to wait after enabling, for the flux collector to attach.
const ATTACH_WAIT_S: u64 = 2;

pub fn init(enabled: bool) -> bool {
    if !enabled {
        return false;
    }
    flux_profiler::enable_profiler("leanvm");
    tracing_subscriber::registry().with(FluxLayer).init();
    eprintln!("flux-profiler enabled, pid {}", std::process::id());
    std::thread::sleep(std::time::Duration::from_secs(ATTACH_WAIT_S));
    true
}
