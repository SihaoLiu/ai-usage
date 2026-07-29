use super::*;

pub(super) fn rebuild_vendor_cache<F>(
    cache_root: &Path,
    vendor: &str,
    manifest_path: &Path,
    mut manifest: CacheManifest,
    active_sources: Vec<CurrentSource>,
    current_fast_tier: i8,
    parse_file: &F,
) -> Vec<PersistedSourceRecord>
where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    let active_records = parse_active_sources(&active_sources, current_fast_tier, parse_file);
    let mut vendor_manifest = VendorManifest {
        session_metadata_revision: parser_revision(vendor),
        ..Default::default()
    };
    let stats = record_stats_by_path(&active_records);

    for source in &active_sources {
        vendor_manifest.files.insert(
            source.key.clone(),
            SourceFileMeta::from_stat(
                &source.stat,
                stats.get(&source.key).copied().unwrap_or_default(),
                parser_revision(vendor),
            ),
        );
    }

    manifest.vendors.insert(vendor.to_string(), vendor_manifest);
    if fs::create_dir_all(cache_root.join(ENTRIES_DIR)).is_err()
        || write_cached_records(&vendor_entries_path(cache_root, vendor), &active_records).is_err()
    {
        return active_records;
    }
    let _ = write_manifest(manifest_path, &manifest);

    active_records
}

pub(super) fn parse_active_sources<F>(
    active_sources: &[CurrentSource],
    current_fast_tier: i8,
    parse_file: &F,
) -> Vec<PersistedSourceRecord>
where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    let per_source: Vec<Vec<PersistedSourceRecord>> = active_sources
        .par_iter()
        .map(|source| {
            parse_file(&source.path)
                .into_iter()
                .map(|record| {
                    PersistedSourceRecord::from_source_record(
                        source.key.clone(),
                        record,
                        current_fast_tier,
                    )
                })
                .collect()
        })
        .collect();
    per_source.into_iter().flatten().collect()
}

pub(super) fn current_sources(source_files: Vec<PathBuf>) -> Vec<CurrentSource> {
    let mut sources = Vec::new();
    let mut occurrences: HashMap<String, usize> = HashMap::new();
    for path in source_files {
        let Some(stat) = stat_source_file(&path) else {
            continue;
        };
        let base_key = source_path_key(&path);
        let occurrence = occurrences.entry(base_key.clone()).or_insert(0);
        let key = if *occurrence == 0 {
            base_key
        } else {
            format!("{}#{}", base_key, occurrence)
        };
        *occurrence += 1;
        sources.push(CurrentSource { key, path, stat });
    }
    sources
}

pub(super) fn stat_source_file(path: &Path) -> Option<SourceFileStat> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let duration = modified
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0));
    Some(SourceFileStat {
        size_bytes: metadata.len(),
        modified_secs: duration.as_secs(),
        modified_nanos: duration.subsec_nanos(),
        #[cfg(unix)]
        changed_secs: metadata.ctime(),
        #[cfg(not(unix))]
        changed_secs: 0,
        #[cfg(unix)]
        changed_nanos: metadata.ctime_nsec(),
        #[cfg(not(unix))]
        changed_nanos: 0,
        #[cfg(unix)]
        device_id: metadata.dev(),
        #[cfg(not(unix))]
        device_id: 0,
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(not(unix))]
        inode: 0,
    })
}

fn source_path_key(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

pub(super) fn vendor_entries_path(cache_root: &Path, vendor: &str) -> PathBuf {
    cache_root
        .join(ENTRIES_DIR)
        .join(format!("{}.bin", safe_file_stem(vendor)))
}

pub(super) fn remote_entries_path(cache_root: &Path, host_id: &str) -> PathBuf {
    cache_root
        .join(REMOTE_DIR)
        .join(format!("{}.bin", safe_file_stem(host_id)))
}

fn safe_file_stem(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn read_manifest(path: &Path) -> CacheManifest {
    let Ok(content) = fs::read_to_string(path) else {
        return CacheManifest::default();
    };
    let Ok(manifest) = serde_json::from_str::<CacheManifest>(&content) else {
        return CacheManifest::default();
    };
    if manifest.version == CACHE_VERSION {
        manifest
    } else {
        CacheManifest::default()
    }
}

pub(super) fn write_manifest(path: &Path, manifest: &CacheManifest) -> io::Result<()> {
    let content = serde_json::to_string_pretty(manifest)?;
    atomic_write(path, content.as_bytes())
}

pub(super) fn read_cached_records(path: &Path) -> io::Result<Vec<PersistedSourceRecord>> {
    #[cfg(test)]
    CACHED_RECORD_READS.set(CACHED_RECORD_READS.get() + 1);
    if let Ok(decoded) = deserialize_framed::<PersistedVendorRecords>(path, ENTRY_FILE_MAGIC) {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported cache entry version",
            ));
        }
        if decoded
            .records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cache entry has negative token count",
            ));
        }
        return Ok(decoded.records);
    }

    if let Ok(decoded) =
        deserialize_framed::<PersistedVendorRecordsBeforeSession>(path, ENTRY_FILE_MAGIC)
    {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported cache entry version",
            ));
        }
        let records: Vec<PersistedSourceRecord> =
            decoded.records.into_iter().map(Into::into).collect();
        if records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cache entry has negative token count",
            ));
        }
        return Ok(records);
    }

    if let Ok(decoded) =
        deserialize_framed::<PersistedVendorRecordsWithFastTier>(path, ENTRY_FILE_MAGIC)
    {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported cache entry version",
            ));
        }
        let records: Vec<PersistedSourceRecord> =
            decoded.records.into_iter().map(Into::into).collect();
        if records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cache entry has negative token count",
            ));
        }
        return Ok(records);
    }

    let decoded: PersistedVendorRecordsV1 = deserialize_framed(path, ENTRY_FILE_MAGIC)?;
    if decoded.format_version != CACHE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported cache entry version",
        ));
    }
    let records: Vec<PersistedSourceRecord> = decoded.records.into_iter().map(Into::into).collect();
    if records
        .iter()
        .any(|record| !record.has_non_negative_token_usage())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache entry has negative token count",
        ));
    }
    Ok(records)
}

pub(super) fn write_cached_records(
    path: &Path,
    records: &[PersistedSourceRecord],
) -> io::Result<()> {
    index::write_records(
        path,
        ENTRY_FILE_MAGIC,
        ENTRY_INDEX_MAGIC,
        CACHE_VERSION,
        records,
    )
}

pub(super) fn read_remote_records(path: &Path) -> io::Result<Vec<PersistedRemoteRecord>> {
    #[cfg(test)]
    REMOTE_RECORD_READS.set(REMOTE_RECORD_READS.get() + 1);
    if let Ok(decoded) = deserialize_framed::<PersistedRemoteRecords>(path, REMOTE_FILE_MAGIC) {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported remote cache entry version",
            ));
        }
        if decoded
            .records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "remote cache entry has negative token count",
            ));
        }
        return Ok(decoded.records);
    }

    if let Ok(decoded) =
        deserialize_framed::<PersistedRemoteRecordsWithFastTier>(path, REMOTE_FILE_MAGIC)
    {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported remote cache entry version",
            ));
        }
        let records: Vec<PersistedRemoteRecord> =
            decoded.records.into_iter().map(Into::into).collect();
        if records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "remote cache entry has negative token count",
            ));
        }
        return Ok(records);
    }

    let decoded: PersistedRemoteRecordsV1 = deserialize_framed(path, REMOTE_FILE_MAGIC)?;
    if decoded.format_version != CACHE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported remote cache entry version",
        ));
    }
    let records: Vec<PersistedRemoteRecord> = decoded.records.into_iter().map(Into::into).collect();
    if records
        .iter()
        .any(|record| !record.has_non_negative_token_usage())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "remote cache entry has negative token count",
        ));
    }
    Ok(records)
}

pub(super) fn write_remote_records(
    path: &Path,
    records: &[PersistedRemoteRecord],
) -> io::Result<()> {
    index::write_records(
        path,
        REMOTE_FILE_MAGIC,
        REMOTE_INDEX_MAGIC,
        CACHE_VERSION,
        records,
    )
}

pub(super) fn read_framed_header(reader: &mut impl Read, magic: &[u8]) -> io::Result<u64> {
    let mut actual_magic = vec![0_u8; magic.len()];
    reader
        .read_exact(&mut actual_magic)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid cache header"))?;
    if actual_magic != magic {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid cache header",
        ));
    }
    let mut checksum_bytes = [0_u8; 8];
    reader
        .read_exact(&mut checksum_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid checksum"))?;
    Ok(u64::from_le_bytes(checksum_bytes))
}

pub(super) fn deserialize_framed<T: DeserializeOwned>(path: &Path, magic: &[u8]) -> io::Result<T> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let stored_checksum = read_framed_header(&mut reader, magic)?;
    let mut reader = FnvReader::new(reader);
    let decoded = bincode::deserialize_from(&mut reader)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    io::copy(&mut reader, &mut io::sink())?;
    if stored_checksum != reader.checksum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache checksum mismatch",
        ));
    }
    Ok(decoded)
}

struct FnvReader<R> {
    inner: R,
    checksum: u64,
}

impl<R> FnvReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            checksum: fnv1a_bytes(0, &[]),
        }
    }
}

impl<R: Read> Read for FnvReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.checksum = fnv1a_bytes(self.checksum, &buffer[..read]);
        Ok(read)
    }
}

pub(super) enum VisitCurrentCachedRecordsError<E> {
    Cache(io::Error),
    Visitor(E),
}

pub(super) fn try_for_each_current_cached_record<E>(
    path: &Path,
    mut visitor: impl FnMut(PersistedSourceRecord) -> Result<(), E>,
) -> Result<(), VisitCurrentCachedRecordsError<E>> {
    let mut reader =
        BufReader::new(fs::File::open(path).map_err(VisitCurrentCachedRecordsError::Cache)?);
    let stored_checksum = read_framed_header(&mut reader, ENTRY_FILE_MAGIC)
        .map_err(VisitCurrentCachedRecordsError::Cache)?;
    let mut reader = FnvReader::new(reader);
    let format_version: u32 = bincode::deserialize_from(&mut reader).map_err(|error| {
        VisitCurrentCachedRecordsError::Cache(io::Error::new(io::ErrorKind::InvalidData, error))
    })?;
    if format_version != CACHE_VERSION {
        return Err(VisitCurrentCachedRecordsError::Cache(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported cache entry version",
        )));
    }
    let record_count: u64 = bincode::deserialize_from(&mut reader).map_err(|error| {
        VisitCurrentCachedRecordsError::Cache(io::Error::new(io::ErrorKind::InvalidData, error))
    })?;
    for _ in 0..record_count {
        let record: PersistedSourceRecord =
            bincode::deserialize_from(&mut reader).map_err(|error| {
                VisitCurrentCachedRecordsError::Cache(io::Error::new(
                    io::ErrorKind::InvalidData,
                    error,
                ))
            })?;
        if !record.has_non_negative_token_usage() {
            return Err(VisitCurrentCachedRecordsError::Cache(io::Error::new(
                io::ErrorKind::InvalidData,
                "cache entry has negative token count",
            )));
        }
        visitor(record).map_err(VisitCurrentCachedRecordsError::Visitor)?;
    }
    io::copy(&mut reader, &mut io::sink()).map_err(VisitCurrentCachedRecordsError::Cache)?;
    if stored_checksum != reader.checksum {
        return Err(VisitCurrentCachedRecordsError::Cache(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache checksum mismatch",
        )));
    }
    Ok(())
}

struct FnvWriter<W> {
    inner: W,
    checksum: u64,
}

impl<W> FnvWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            checksum: fnv1a_bytes(0, &[]),
        }
    }
}

impl<W: Write> Write for FnvWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.checksum = fnv1a_bytes(self.checksum, &bytes[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

pub(super) fn atomic_serialize_framed<T: Serialize + ?Sized>(
    path: &Path,
    magic: &[u8],
    value: &T,
) -> io::Result<()> {
    atomic_write_with(path, |file| {
        file.seek(SeekFrom::Start((magic.len() + 8) as u64))?;
        let checksum = {
            let mut writer = FnvWriter::new(BufWriter::new(&mut *file));
            bincode::serialize_into(&mut writer, value).map_err(io::Error::other)?;
            writer.flush()?;
            writer.checksum
        };
        file.seek(SeekFrom::Start(0))?;
        file.write_all(magic)?;
        file.write_all(&checksum.to_le_bytes())
    })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    atomic_write_with(path, |file| file.write_all(bytes))
}

pub(super) fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&mut fs::File) -> io::Result<()>,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_nanos();
    let tmp_path = path.with_extension(format!("tmp-{}", stamp));
    let result = (|| {
        let mut file = fs::File::create(&tmp_path)?;
        write(&mut file)?;
        file.sync_all()
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(error);
    }
    fs::rename(tmp_path, path)?;
    Ok(())
}
