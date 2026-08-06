use super::*;

#[test]
fn hot_snapshot_with_previous_projection_magic_is_rejected() {
    let cache_root = unique_temp_dir("previous-hot-snapshot-magic");
    crate::data::cache::write_hot_snapshot(&cache_root, &42_u64)
        .expect("write current hot snapshot");
    let path = cache_root.join(crate::data::cache::HOT_SNAPSHOT_FILE);
    let mut content = fs::read(&path).expect("read hot snapshot");
    content[..8].copy_from_slice(b"AIUHOT02");
    fs::write(&path, content).expect("write previous hot snapshot");

    let error =
        crate::data::cache::load_hot_snapshot::<u64>(&cache_root).expect_err("reject old snapshot");

    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn parser_revisions_are_owned_per_harness() {
    assert_eq!(crate::data::cache::parser_revision("claude"), 3);
    assert_eq!(crate::data::cache::parser_revision("codex"), 3);
    assert_eq!(crate::data::cache::parser_revision("gemini"), 3);
    assert_eq!(crate::data::cache::parser_revision("kimi"), 4);
    assert_eq!(crate::data::cache::parser_revision("omp"), 4);
    assert_eq!(crate::data::cache::parser_revision("unknown"), 0);
}

#[test]
fn changed_harness_parsers_reparse_active_sources_and_drop_stale_inactive_records() {
    for vendor in ["claude", "codex", "gemini", "kimi", "omp"] {
        let cache_root = unique_temp_dir(&format!("{vendor}-parser-migration"));
        let active = cache_root.join("active.jsonl");
        let retired = cache_root.join("retired.jsonl");
        write_source(&active, "active");
        write_source(&retired, "retired");
        let parse_calls = AtomicUsize::new(0);

        crate::data::cache::refresh_retaining_vendor_cache(
            &cache_root,
            vendor,
            vec![active.clone(), retired.clone()],
            -1,
            |path| {
                parse_calls.fetch_add(1, Ordering::Relaxed);
                let key = if path == active {
                    "old-active"
                } else {
                    "old-retired"
                };
                vec![usage_record(key, "2026-05-01T00:00:00Z", 42)]
            },
        );

        let manifest_path = cache_root.join(crate::data::cache::MANIFEST_FILE);
        let mut manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read old manifest"))
                .expect("parse old manifest");
        manifest["vendors"][vendor]["session_metadata_revision"] = serde_json::json!(1);
        manifest["vendors"][vendor]["files"]
            .as_object_mut()
            .expect("files object")
            .values_mut()
            .for_each(|meta| meta["parser_revision"] = serde_json::json!(1));
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize old manifest"),
        )
        .expect("write old manifest");
        fs::remove_file(&retired).expect("remove retired source");

        crate::data::cache::refresh_retaining_vendor_cache(
            &cache_root,
            vendor,
            vec![active],
            -1,
            |_| {
                parse_calls.fetch_add(1, Ordering::Relaxed);
                vec![usage_record("canonical-active", "2026-05-01T00:00:00Z", 42)]
            },
        );

        let records = crate::data::cache::load_vendor_cached_records(&cache_root, vendor);
        assert_eq!(parse_calls.load(Ordering::Relaxed), 3, "vendor={vendor}");
        assert_eq!(records.len(), 1, "vendor={vendor}");
        assert_eq!(records[0].dedup_key, "canonical-active", "vendor={vendor}");
        assert!(crate::data::cache::vendor_parser_revision_is_current(
            &cache_root,
            vendor
        ));
    }
}
