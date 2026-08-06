const RESERVED_CPUS: usize = 4;
const MAX_WORKER_THREADS: usize = 8;

pub(crate) fn thread_budget(available: usize) -> usize {
    available
        .saturating_sub(RESERVED_CPUS)
        .clamp(1, MAX_WORKER_THREADS)
}

pub(crate) fn initialize() {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_budget(available))
        .thread_name(|index| format!("usage-cpu-{index}"))
        .build_global()
        .expect("initialize process CPU worker pool");
}

#[cfg(test)]
mod tests {
    use super::thread_budget;

    #[test]
    fn thread_budget_reserves_four_cores_and_never_exceeds_eight() {
        assert_eq!(thread_budget(1), 1);
        assert_eq!(thread_budget(4), 1);
        assert_eq!(thread_budget(5), 1);
        assert_eq!(thread_budget(8), 4);
        assert_eq!(thread_budget(12), 8);
        assert_eq!(thread_budget(64), 8);
    }
}
