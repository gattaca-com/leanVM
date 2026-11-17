//! A forecast of the inputs still to come, built from a training schedule set (SPEC 6.7).
//! Train on one set and evaluate on another with a different seed.

use crate::types::Micros;
use arrival_model::schedule::{Kind, ScheduleSet};
use std::path::Path;

pub struct Forecast {
    /// Pooled arrival times of every run of the training set, sorted.
    times: Vec<Micros>,
    mean_count: f64,
}

impl Forecast {
    pub fn from_set(dir: &Path) -> Result<Forecast, String> {
        Ok(Forecast::from_schedule_set(&ScheduleSet::read(dir)?))
    }

    pub fn from_schedule_set(set: &ScheduleSet) -> Forecast {
        let merge = set.meta.shape == "merge";
        let mut times = Vec::new();
        let mut runs = 0usize;
        for slot in &set.slots {
            if merge {
                runs += 1;
                times.extend(slot.arrivals.iter().filter(|a| a.kind == Kind::Aggregate).map(|a| a.t_us));
            } else {
                for subnet in 0..set.meta.subnets as u8 {
                    for view in 0..set.meta.views as u8 {
                        runs += 1;
                        times.extend(slot.arrivals.iter().filter(|a| a.kind == Kind::Vote && a.subnet == subnet && a.view == view).map(|a| a.t_us));
                    }
                }
            }
        }
        times.sort_unstable();
        Forecast { mean_count: times.len() as f64 / runs.max(1) as f64, times }
    }

    /// Expected times of the inputs still to come, given `seen` arrived by `now`: `k = round(mean count - seen)`
    /// inputs at the `(i + 0.5) / k` quantiles of the pooled CDF conditioned on `t > now`.
    pub fn remaining(&self, now: Micros, seen: u32) -> Vec<Micros> {
        let k = (self.mean_count - seen as f64).round();
        if k < 1.0 {
            return Vec::new();
        }
        let k = k as usize;
        let lo = self.times.partition_point(|&t| t <= now);
        let after = &self.times[lo..];
        if after.is_empty() {
            return Vec::new();
        }
        (0..k).map(|i| after[(((i as f64 + 0.5) / k as f64) * after.len() as f64) as usize]).collect()
    }

    pub fn mean_count(&self) -> f64 {
        self.mean_count
    }
}
