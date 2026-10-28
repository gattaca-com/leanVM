//! The global allocator of the CLI and the proving benchmarks.
//!
//! A proof allocates gigabytes of short-lived buffers and frees them before it returns.
//!
//! An allocator that returns freed pages to the kernel makes the next proof fault each page in again.
//!
//! With dirty pages that never decay, jemalloc keeps them instead:
//!
//! ```text
//!     proof 1: pages faulted in, then freed  ->  kept mapped in jemalloc's cache
//!     proof 2: pages served from the cache   ->  no fault
//! ```
//!
//! The cost is resident memory: the process holds its peak until it exits.
//!
//! The kernel's transparent huge pages are on request only (`madvise`), so jemalloc is told to ask
//! for them: a pass over a table of hundreds of megabytes then walks 2 MiB pages, and first touch
//! faults once per huge page rather than once per 4 KiB.

pub use tikv_jemallocator::Jemalloc;

/// jemalloc's configuration, which it reads before its first allocation.
///
/// - Dirty pages never decay, so freed memory never returns to the kernel.
/// - Its extents are advised as huge pages.
/// - jemalloc reads a C string pointer here.
/// - A reference to a byte array is that pointer, with no length beside it.
#[unsafe(export_name = "_rjem_malloc_conf")]
static MALLOC_CONF: &[u8; 29] = b"dirty_decay_ms:-1,thp:always\0";
