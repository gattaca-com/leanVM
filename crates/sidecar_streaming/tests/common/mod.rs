#![allow(dead_code)]
use arrival_model::model::{Params, build, generate};
use arrival_model::schedule::ScheduleSet;
use sidecar_streaming::backend::Backend;
use sidecar_streaming::cost::{CostModel, parse_model};
use sidecar_streaming::driver::{RunSpec, run_slot};
use sidecar_streaming::metric::{Metrics, SlotRecord, slot_metrics};
use sidecar_streaming::sim::SimBackend;
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::types::*;
use std::path::Path;

/// A stepped model: padded size doubles at `170 * 2^k` rows.
pub const STEPPED: &str = "form rows\nbackend test\nthreads 8\nisa avx2\nsigma 0\nstep 170\nw0 0\n\
t0.1 30000\nlin.1 550\npad.1 330\nt0.2 35000\nlin.2 600\npad.2 380\nwc.1 150\nwc.2 150\nwk 0\nv.1 6000\nv.2 6000\nbytes.1 300000\nbytes.2 220000\n\
merge.step 400\nmerge.w0 0\nmerge.t0.1 30000\nmerge.lin.1 400\nmerge.pad.1 300\nmerge.t0.2 35000\nmerge.lin.2 450\nmerge.pad.2 350\nmerge.wc.2 200\nmerge.wc.1 200\nmerge.wk 0.02\n";

/// The same without padding: near-linear.
pub const SMOOTH: &str = "form rows\nbackend test\nthreads 8\nisa avx2\nsigma 0\nstep 170\nw0 0\n\
t0.1 30000\nlin.1 500\npad.1 0\nt0.2 31000\nlin.2 520\npad.2 0\nwc.1 100\nwc.2 100\nwk 0\nv.1 6000\nv.2 6000\nbytes.1 300000\nbytes.2 220000\n\
merge.step 400\nmerge.w0 0\nmerge.t0.1 30000\nmerge.lin.1 300\nmerge.pad.1 0\nmerge.t0.2 31000\nmerge.lin.2 320\nmerge.pad.2 0\nmerge.wc.2 150\nmerge.wc.1 150\nmerge.wk 0.02\n";

pub fn model(text: &str) -> Box<dyn CostModel> {
    parse_model(text).unwrap()
}

pub fn sim(text: &str) -> SimBackend {
    SimBackend::new(model(text), Caps::default(), 1, false)
}

pub fn leaf_params() -> Params {
    // The 4 s layout: votes from 567 ms to 1034 ms, a 4 % tail to 1202 ms.
    Params::parse(
        "shape leaf\nslot_ms 4000\nsubnets 1\nagg.per_subnet 1\nvotes.base mix(0.96: uniform(567, 1034), 0.04: uniform(1034, 1202))\n",
        Path::new("."),
    )
}

pub fn merge_params() -> Params {
    Params::parse(
        "shape merge\nslot_ms 4000\nsubnets 16\nagg.per_subnet 3\nagg.base normal(1500, 300)\nagg.invalid 0.05\nagg.missing 0.05\n",
        Path::new("."),
    )
}

pub fn make_set(p: &Params, slots: u32, seed: u64) -> ScheduleSet {
    generate(build(p).unwrap().as_ref(), p, slots, seed)
}

pub fn run(backend: &mut dyn Backend, set: &ScheduleSet, slot: u32, d: Micros, spec: &str, train: Option<&ScheduleSet>) -> (SlotRecord, Metrics) {
    let mut spec = Spec::parse(spec).unwrap();
    if spec.shape() == Shape::Merge {
        spec.default_param("verify", "all_idle");
    }
    let fc = train.map(|t| std::rc::Rc::new(sidecar_streaming::forecast::Forecast::from_schedule_set(t)));
    let mut s = spec.make(fc.as_ref()).unwrap();
    let rs = RunSpec { set, slot, subnet: 0, view: 0, deadline: d, caps: backend.caps(), grace: 0 };
    let rec = run_slot(backend, s.as_mut(), &rs);
    let m = slot_metrics(&rec);
    (rec, m)
}
