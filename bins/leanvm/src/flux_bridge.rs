//! `LEANVM_FLUX=1`: publish flux-profiler rings and mirror every `tracing` span
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

pub fn init() -> bool {
    if std::env::var_os("LEANVM_FLUX").is_none() {
        return false;
    }
    flux_profiler::enable_profiler("leanvm");
    tracing_subscriber::registry().with(FluxLayer).init();
    eprintln!("flux-profiler enabled, pid {}", std::process::id());
    let wait = std::env::var("LEANVM_FLUX_WAIT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    std::thread::sleep(std::time::Duration::from_secs(wait));
    true
}
