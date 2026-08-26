use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::time_utils::parse_timestamp;

use super::{CACHE_VERSION, persistence};

const INDEX_VERSION: u32 = 5;
const COMPATIBLE_INDEX_VERSION: u32 = 3;
const SECONDS_PER_DAY: i64 = 86_400;
const HEADER_LEN: u64 = 84;
const BUCKET_METADATA_LEN: u64 = 48;
const RECORD_LOCATION_LEN: u64 = 32;
const CONTEXT_ENTRY_LEN: u64 = 40;
const MAX_BUCKET_BYTES: u64 = 1024 * 1024 * 1024;

pub(super) trait IndexableRecord {
    fn index_timestamp(&self) -> &str;

    fn index_dedup_key(&self) -> Option<&str> {
        None
    }

    fn append_index_context(&self, _digests: &mut Vec<[u8; 32]>) {}

    fn index_duplicate_context(&self) -> Option<[u8; 32]> {
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct IndexHeader {
    version: u32,
    source_checksum: u64,
    source_len: u64,
    record_count: u64,
    bucket_count: u64,
    directory_offset: u64,
    context_offset: u64,
    context_count: u64,
    payload_offset: u64,
    directory_checksum: u64,
    context_checksum: u64,
}

#[derive(Clone, Copy, Debug)]
struct BucketMetadata {
    day: i64,
    offset: u64,
    len: u64,
    checksum: u64,
    record_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct RecordLocation {
    timestamp_seconds: i64,
    offset: u64,
    len: u64,
    checksum: u64,
}

struct EncodedRecords {
    checksum: u64,
    source_len: u64,
    record_count: u64,
    buckets: BTreeMap<i64, Vec<RecordLocation>>,
    context: Vec<[u8; 32]>,
}

struct LoadedIndex {
    header: IndexHeader,
    index_len: u64,
    bytes_read: u64,
}

pub(super) struct IndexedRecords<R> {
    pub(super) records: Vec<R>,
    pub(super) has_records: bool,
}

struct IndexedRecordWriter<'a, F> {
    write: &'a mut F,
    total_checksum: &'a mut u64,
    checksum: u64,
    len: u64,
}

impl<'a, F> IndexedRecordWriter<'a, F> {
    fn new(write: &'a mut F, total_checksum: &'a mut u64) -> Self {
        Self {
            write,
            total_checksum,
            checksum: 0,
            len: 0,
        }
    }
}

impl<F> Write for IndexedRecordWriter<'_, F>
where
    F: FnMut(&[u8]) -> io::Result<()>,
{
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        (self.write)(bytes)?;
        self.len = self.len.checked_add(bytes.len() as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "encoded record is too large")
        })?;
        self.checksum = super::fnv1a_bytes(self.checksum, bytes);
        *self.total_checksum = super::fnv1a_bytes(*self.total_checksum, bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct IndexedRecordReader<R> {
    inner: R,
    checksum: u64,
    bytes_read: u64,
}

impl<R> IndexedRecordReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            checksum: 0,
            bytes_read: 0,
        }
    }
}

impl<R: Read> Read for IndexedRecordReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.bytes_read = self.bytes_read.saturating_add(read as u64);
        self.checksum = super::fnv1a_bytes(self.checksum, &buffer[..read]);
        Ok(read)
    }
}

impl<R> IndexedRecordReader<io::Take<R>> {
    fn remaining(&self) -> u64 {
        self.inner.limit()
    }
}

pub(super) fn write_records<R>(
    path: &Path,
    data_magic: &[u8],
    index_magic: &[u8],
    format_version: u32,
    records: &[R],
) -> io::Result<()>
where
    R: IndexableRecord + Serialize,
{
    let mut encoded = None;
    super::atomic_write_with(path, |file| {
        let header_len = data_magic.len() as u64 + 8;
        file.seek(SeekFrom::Start(header_len))?;
        let result = {
            let mut writer = BufWriter::new(&mut *file);
            let result = encode_records(format_version, records, header_len, |bytes| {
                writer.write_all(bytes)
            })?;
            writer.flush()?;
            result
        };
        file.seek(SeekFrom::Start(0))?;
        file.write_all(data_magic)?;
        file.write_all(&result.checksum.to_le_bytes())?;
        encoded = Some(result);
        Ok(())
    })?;
    write_index(
        path,
        index_magic,
        encoded.expect("record encoding completed"),
    )
}

pub(super) fn ensure<R>(
    path: &Path,
    data_magic: &[u8],
    index_magic: &[u8],
    format_version: u32,
    records: &[R],
) -> io::Result<()>
where
    R: IndexableRecord + Serialize,
{
    if is_current(path, data_magic, index_magic) {
        return Ok(());
    }
    let header_len = data_magic.len() as u64 + 8;
    let encoded = encode_records(format_version, records, header_len, |_| Ok(()))?;
    let (source_checksum, source_len) = source_generation(path, data_magic)?;
    if encoded.checksum != source_checksum || encoded.source_len != source_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache layout does not match current record index format",
        ));
    }
    write_index(path, index_magic, encoded)
}

pub(super) fn is_current(path: &Path, data_magic: &[u8], index_magic: &[u8]) -> bool {
    #[cfg(test)]
    super::INDEX_FULL_VALIDATIONS.set(super::INDEX_FULL_VALIDATIONS.get() + 1);
    let Ok((source_checksum, source_len)) = source_generation(path, data_magic) else {
        return false;
    };
    let Ok(mut reader) = fs::File::open(index_path(path)).map(BufReader::new) else {
        return false;
    };
    let Ok(loaded) = read_header(&mut reader, index_magic) else {
        return false;
    };
    if loaded.header.version != INDEX_VERSION {
        return false;
    }
    validate_full_index(
        &mut reader,
        &loaded,
        source_checksum,
        source_len,
        data_magic.len() as u64 + 8,
    )
    .is_ok()
}

pub(super) fn matches_source_generation(
    path: &Path,
    data_magic: &[u8],
    index_magic: &[u8],
) -> bool {
    let Ok((source_checksum, source_len)) = source_generation(path, data_magic) else {
        return false;
    };
    let Ok(mut reader) = fs::File::open(index_path(path)).map(BufReader::new) else {
        return false;
    };
    let Ok(loaded) = read_header(&mut reader, index_magic) else {
        return false;
    };
    if loaded.header.version != INDEX_VERSION {
        return false;
    }
    if source_format_version(path, data_magic).ok() != Some(CACHE_VERSION) {
        return false;
    }
    validate_header(&loaded, source_checksum, source_len).is_ok()
}

fn source_format_version(path: &Path, data_magic: &[u8]) -> io::Result<u32> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    persistence::read_framed_header(&mut reader, data_magic)?;
    bincode::deserialize_from(&mut reader)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub(super) fn read_range<R>(
    path: &Path,
    data_magic: &[u8],
    index_magic: &[u8],
    start: DateTime<Local>,
    end: DateTime<Local>,
) -> io::Result<IndexedRecords<R>>
where
    R: DeserializeOwned + IndexableRecord,
{
    read_range_indexed(path, data_magic, index_magic, start, end)
}

fn read_range_indexed<R>(
    path: &Path,
    data_magic: &[u8],
    index_magic: &[u8],
    start: DateTime<Local>,
    end: DateTime<Local>,
) -> io::Result<IndexedRecords<R>>
where
    R: DeserializeOwned + IndexableRecord,
{
    if start > end {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cache range starts after it ends",
        ));
    }
    let mut source = BufReader::new(fs::File::open(path)?);
    let source_checksum = super::read_framed_header(&mut source, data_magic)?;
    let source_len = source.get_ref().metadata()?.len();
    let mut index = BufReader::new(fs::File::open(index_path(path))?);
    let mut loaded = read_header(&mut index, index_magic)?;
    validate_header(&loaded, source_checksum, source_len)?;

    let start_seconds = start.timestamp();
    let end_seconds = end.timestamp();
    let start_day = start_seconds.div_euclid(SECONDS_PER_DAY);
    let end_day = end_seconds.div_euclid(SECONDS_PER_DAY);
    let first = lower_bound_bucket(&mut index, &mut loaded, start_day, false)?;
    let last = lower_bound_bucket(&mut index, &mut loaded, end_day, true)?;
    let source_payload_start = data_magic.len() as u64 + 8;
    let mut selected = Vec::new();
    for bucket_index in first..last {
        let metadata = read_bucket_metadata(&mut index, &mut loaded, bucket_index)?;
        let locations = read_bucket(&mut index, &metadata, source_payload_start, source_len)?;
        loaded.bytes_read = loaded.bytes_read.saturating_add(metadata.len);
        selected.extend(locations.into_iter().filter(|location| {
            location.timestamp_seconds >= start_seconds && location.timestamp_seconds <= end_seconds
        }));
    }
    selected.sort_unstable_by_key(|record| record.offset);
    if !selected.windows(2).all(|pair| {
        pair[0]
            .offset
            .checked_add(pair[0].len)
            .is_some_and(|end| end <= pair[1].offset)
    }) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "indexed cache record ranges overlap",
        ));
    }

    let mut records = Vec::with_capacity(selected.len());
    for location in selected {
        source.seek(SeekFrom::Start(location.offset))?;
        let mut reader = IndexedRecordReader::new((&mut source).take(location.len));
        let record: R = bincode::deserialize_from(&mut reader)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        loaded.bytes_read = loaded.bytes_read.saturating_add(location.len);
        if reader.remaining() != 0
            || reader.bytes_read != location.len
            || reader.checksum != location.checksum
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "indexed cache record checksum mismatch",
            ));
        }
        let timestamp = parse_timestamp(record.index_timestamp()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "indexed record has invalid timestamp",
            )
        })?;
        if timestamp.timestamp() != location.timestamp_seconds {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "indexed cache timestamp mismatch",
            ));
        }
        if timestamp >= start && timestamp <= end {
            if let Some(digest) = record.index_duplicate_context()
                && context_contains(&mut index, &mut loaded, &digest)?
            {
                continue;
            }
            records.push(record);
        }
    }

    #[cfg(test)]
    super::INDEXED_CACHE_BYTES_READ.set(
        super::INDEXED_CACHE_BYTES_READ
            .get()
            .saturating_add(loaded.bytes_read),
    );
    Ok(IndexedRecords {
        records,
        has_records: loaded.header.record_count > 0,
    })
}

fn encode_records<R>(
    format_version: u32,
    records: &[R],
    header_len: u64,
    mut write: impl FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<EncodedRecords>
where
    R: IndexableRecord + Serialize,
{
    let mut checksum = 0_u64;
    let mut offset = header_len;
    let mut buckets: BTreeMap<i64, Vec<RecordLocation>> = BTreeMap::new();
    let mut context = Vec::new();
    let mut indexed_keys = HashSet::new();
    let version = bincode::serialize(&format_version).map_err(io::Error::other)?;
    let count = bincode::serialize(&(records.len() as u64)).map_err(io::Error::other)?;
    for bytes in [&version, &count] {
        write(bytes)?;
        checksum = super::fnv1a_bytes(checksum, bytes);
        offset = offset.saturating_add(bytes.len() as u64);
    }

    for record in records {
        let (len, record_checksum) = {
            let mut writer = IndexedRecordWriter::new(&mut write, &mut checksum);
            bincode::serialize_into(&mut writer, record).map_err(io::Error::other)?;
            (writer.len, writer.checksum)
        };
        let owns_key = record
            .index_dedup_key()
            .is_none_or(|key| indexed_keys.insert(key));
        if owns_key && let Some(timestamp) = parse_timestamp(record.index_timestamp()) {
            let timestamp_seconds = timestamp.timestamp();
            buckets
                .entry(timestamp_seconds.div_euclid(SECONDS_PER_DAY))
                .or_default()
                .push(RecordLocation {
                    timestamp_seconds,
                    offset,
                    len,
                    checksum: record_checksum,
                });
        }
        record.append_index_context(&mut context);
        offset = offset.checked_add(len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "encoded cache is too large")
        })?;
    }
    for locations in buckets.values_mut() {
        locations.sort_unstable_by_key(|record| (record.timestamp_seconds, record.offset));
    }
    context.sort_unstable();
    context.dedup();
    Ok(EncodedRecords {
        checksum,
        source_len: offset,
        record_count: records.len() as u64,
        buckets,
        context,
    })
}

fn write_index(data_path: &Path, magic: &[u8], encoded: EncodedRecords) -> io::Result<()> {
    let bucket_count = encoded.buckets.len() as u64;
    let context_count = encoded.context.len() as u64;
    let directory_offset = magic.len() as u64 + 8 + HEADER_LEN;
    let context_offset = checked_offset(
        directory_offset,
        bucket_count,
        BUCKET_METADATA_LEN,
        "record index directory is too large",
    )?;
    let payload_offset = checked_offset(
        context_offset,
        context_count,
        CONTEXT_ENTRY_LEN,
        "record index context is too large",
    )?;

    let mut payload_cursor = payload_offset;
    let mut metadata = Vec::with_capacity(encoded.buckets.len());
    for (day, locations) in &encoded.buckets {
        let len = (locations.len() as u64)
            .checked_mul(RECORD_LOCATION_LEN)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "record index bucket is too large",
                )
            })?;
        if len == 0 || len > MAX_BUCKET_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "record index bucket exceeds size limit",
            ));
        }
        let checksum = locations.iter().fold(0_u64, |checksum, location| {
            super::fnv1a_bytes(checksum, &encode_record_location(location))
        });
        metadata.push(BucketMetadata {
            day: *day,
            offset: payload_cursor,
            len,
            checksum,
            record_count: locations.len() as u64,
        });
        payload_cursor = payload_cursor.checked_add(len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "record index is too large")
        })?;
    }
    let directory_checksum = metadata.iter().fold(0_u64, |checksum, bucket| {
        super::fnv1a_bytes(checksum, &encode_bucket_metadata(bucket))
    });
    let context_checksum = encoded.context.iter().fold(0_u64, |checksum, digest| {
        super::fnv1a_bytes(checksum, digest)
    });
    let header = IndexHeader {
        version: INDEX_VERSION,
        source_checksum: encoded.checksum,
        source_len: encoded.source_len,
        record_count: encoded.record_count,
        bucket_count,
        directory_offset,
        context_offset,
        context_count,
        payload_offset,
        directory_checksum,
        context_checksum,
    };
    let header_bytes = encode_header(&header);
    let index_path = index_path(data_path);
    super::atomic_write_with(&index_path, |file| {
        let mut writer = BufWriter::new(file);
        writer.write_all(magic)?;
        writer.write_all(&super::fnv1a_bytes(0, &header_bytes).to_le_bytes())?;
        writer.write_all(&header_bytes)?;
        for bucket in &metadata {
            writer.write_all(&encode_bucket_metadata(bucket))?;
        }
        for digest in &encoded.context {
            writer.write_all(&encode_context_entry(digest))?;
        }
        for locations in encoded.buckets.values() {
            for location in locations {
                writer.write_all(&encode_record_location(location))?;
            }
        }
        writer.flush()
    })?;
    Ok(())
}

fn read_header(reader: &mut BufReader<fs::File>, magic: &[u8]) -> io::Result<LoadedIndex> {
    let stored_checksum = super::read_framed_header(reader, magic)?;
    let mut bytes = [0_u8; HEADER_LEN as usize];
    reader.read_exact(&mut bytes)?;
    if super::fnv1a_bytes(0, &bytes) != stored_checksum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record index header checksum mismatch",
        ));
    }
    Ok(LoadedIndex {
        header: decode_header(&bytes),
        index_len: reader.get_ref().metadata()?.len(),
        bytes_read: magic.len() as u64 + 8 + HEADER_LEN,
    })
}

fn validate_header(loaded: &LoadedIndex, source_checksum: u64, source_len: u64) -> io::Result<()> {
    let header = loaded.header;
    let expected_context_offset = checked_offset(
        header.directory_offset,
        header.bucket_count,
        BUCKET_METADATA_LEN,
        "invalid record index directory length",
    )?;
    let expected_payload_offset = checked_offset(
        header.context_offset,
        header.context_count,
        CONTEXT_ENTRY_LEN,
        "invalid record index context length",
    )?;
    let valid = matches!(header.version, COMPATIBLE_INDEX_VERSION | INDEX_VERSION)
        && header.source_checksum == source_checksum
        && header.source_len == source_len
        && header.directory_offset == loaded.bytes_read
        && header.context_offset == expected_context_offset
        && header.payload_offset == expected_payload_offset
        && header.payload_offset <= loaded.index_len;
    if !valid {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record index does not match its source",
        ));
    }
    Ok(())
}

fn validate_full_index(
    reader: &mut BufReader<fs::File>,
    loaded: &LoadedIndex,
    source_checksum: u64,
    source_len: u64,
    source_payload_start: u64,
) -> io::Result<()> {
    validate_header(loaded, source_checksum, source_len)?;
    let mut checksum = 0_u64;
    let mut previous: Option<BucketMetadata> = None;
    let mut buckets = Vec::with_capacity(usize::try_from(loaded.header.bucket_count).unwrap_or(0));
    let mut indexed_records = 0_u64;
    reader.seek(SeekFrom::Start(loaded.header.directory_offset))?;
    for _ in 0..loaded.header.bucket_count {
        let mut bytes = [0_u8; BUCKET_METADATA_LEN as usize];
        reader.read_exact(&mut bytes)?;
        checksum = super::fnv1a_bytes(checksum, &bytes);
        let bucket = decode_bucket_metadata(&bytes)?;
        let expected_offset = previous
            .map(|item| item.offset.saturating_add(item.len))
            .unwrap_or(loaded.header.payload_offset);
        if bucket.record_count == 0
            || bucket.len == 0
            || bucket.len > MAX_BUCKET_BYTES
            || bucket.offset != expected_offset
            || bucket
                .offset
                .checked_add(bucket.len)
                .is_none_or(|end| end > loaded.index_len)
            || previous.is_some_and(|item| item.day >= bucket.day)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid record index directory",
            ));
        }
        indexed_records = indexed_records
            .checked_add(bucket.record_count)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "record index overflow"))?;
        previous = Some(bucket);
        buckets.push(bucket);
    }
    if checksum != loaded.header.directory_checksum
        || indexed_records > loaded.header.record_count
        || previous
            .map(|item| item.offset.saturating_add(item.len))
            .unwrap_or(loaded.header.payload_offset)
            != loaded.index_len
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid record index directory checksum",
        ));
    }

    reader.seek(SeekFrom::Start(loaded.header.context_offset))?;
    let mut context_checksum = 0_u64;
    let mut previous_digest = None;
    for _ in 0..loaded.header.context_count {
        let mut bytes = [0_u8; CONTEXT_ENTRY_LEN as usize];
        reader.read_exact(&mut bytes)?;
        let digest = decode_context_entry(&bytes)?;
        context_checksum = super::fnv1a_bytes(context_checksum, &digest);
        if previous_digest.is_some_and(|previous| previous >= digest) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "record index context is not sorted",
            ));
        }
        previous_digest = Some(digest);
    }
    if context_checksum != loaded.header.context_checksum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record index context checksum mismatch",
        ));
    }
    for bucket in buckets {
        read_bucket(reader, &bucket, source_payload_start, source_len)?;
    }
    Ok(())
}

fn lower_bound_bucket(
    reader: &mut BufReader<fs::File>,
    loaded: &mut LoadedIndex,
    day: i64,
    after_equal: bool,
) -> io::Result<u64> {
    let mut low = 0_u64;
    let mut high = loaded.header.bucket_count;
    while low < high {
        let mid = low + (high - low) / 2;
        let bucket = read_bucket_metadata(reader, loaded, mid)?;
        if bucket.day < day || (after_equal && bucket.day == day) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    Ok(low)
}

fn read_bucket_metadata(
    reader: &mut BufReader<fs::File>,
    loaded: &mut LoadedIndex,
    index: u64,
) -> io::Result<BucketMetadata> {
    if index >= loaded.header.bucket_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record index directory position is out of range",
        ));
    }
    let offset = checked_offset(
        loaded.header.directory_offset,
        index,
        BUCKET_METADATA_LEN,
        "record index directory position overflow",
    )?;
    reader.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0_u8; BUCKET_METADATA_LEN as usize];
    reader.read_exact(&mut bytes)?;
    loaded.bytes_read = loaded.bytes_read.saturating_add(BUCKET_METADATA_LEN);
    let metadata = decode_bucket_metadata(&bytes)?;
    if metadata.record_count == 0
        || metadata.len == 0
        || metadata.len > MAX_BUCKET_BYTES
        || metadata.offset < loaded.header.payload_offset
        || metadata
            .offset
            .checked_add(metadata.len)
            .is_none_or(|end| end > loaded.index_len)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid record index bucket metadata",
        ));
    }
    Ok(metadata)
}

fn read_bucket(
    reader: &mut BufReader<fs::File>,
    metadata: &BucketMetadata,
    source_payload_start: u64,
    source_len: u64,
) -> io::Result<Vec<RecordLocation>> {
    if metadata.len != metadata.record_count.saturating_mul(RECORD_LOCATION_LEN) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid record index bucket length",
        ));
    }
    reader.seek(SeekFrom::Start(metadata.offset))?;
    let capacity = usize::try_from(metadata.record_count).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "record index bucket is too large",
        )
    })?;
    let mut locations = Vec::with_capacity(capacity);
    let mut checksum = 0_u64;
    for _ in 0..metadata.record_count {
        let mut bytes = [0_u8; RECORD_LOCATION_LEN as usize];
        reader.read_exact(&mut bytes)?;
        checksum = super::fnv1a_bytes(checksum, &bytes);
        locations.push(decode_record_location(&bytes));
    }
    let valid = checksum == metadata.checksum
        && locations.windows(2).all(|pair| {
            (pair[0].timestamp_seconds, pair[0].offset)
                <= (pair[1].timestamp_seconds, pair[1].offset)
        })
        && locations.iter().all(|record| {
            record.timestamp_seconds.div_euclid(SECONDS_PER_DAY) == metadata.day
                && record.offset >= source_payload_start
                && record
                    .offset
                    .checked_add(record.len)
                    .is_some_and(|end| end <= source_len)
        });
    if !valid {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid record index bucket",
        ));
    }
    Ok(locations)
}

fn context_contains(
    reader: &mut BufReader<fs::File>,
    loaded: &mut LoadedIndex,
    needle: &[u8; 32],
) -> io::Result<bool> {
    let mut low = 0_u64;
    let mut high = loaded.header.context_count;
    while low < high {
        let mid = low + (high - low) / 2;
        let offset = checked_offset(
            loaded.header.context_offset,
            mid,
            CONTEXT_ENTRY_LEN,
            "record index context position overflow",
        )?;
        reader.seek(SeekFrom::Start(offset))?;
        let mut bytes = [0_u8; CONTEXT_ENTRY_LEN as usize];
        reader.read_exact(&mut bytes)?;
        let digest = decode_context_entry(&bytes)?;
        loaded.bytes_read = loaded.bytes_read.saturating_add(CONTEXT_ENTRY_LEN);
        match digest.cmp(needle) {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
            std::cmp::Ordering::Equal => return Ok(true),
        }
    }
    Ok(false)
}

fn source_generation(path: &Path, magic: &[u8]) -> io::Result<(u64, u64)> {
    let mut source = BufReader::new(fs::File::open(path)?);
    let checksum = super::read_framed_header(&mut source, magic)?;
    Ok((checksum, source.get_ref().metadata()?.len()))
}

fn checked_offset(base: u64, count: u64, width: u64, message: &str) -> io::Result<u64> {
    count
        .checked_mul(width)
        .and_then(|len| base.checked_add(len))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, message))
}

fn encode_header(header: &IndexHeader) -> [u8; HEADER_LEN as usize] {
    let mut bytes = [0_u8; HEADER_LEN as usize];
    let mut cursor = 0;
    put_u32(&mut bytes, &mut cursor, header.version);
    for value in [
        header.source_checksum,
        header.source_len,
        header.record_count,
        header.bucket_count,
        header.directory_offset,
        header.context_offset,
        header.context_count,
        header.payload_offset,
        header.directory_checksum,
        header.context_checksum,
    ] {
        put_u64(&mut bytes, &mut cursor, value);
    }
    bytes
}

fn decode_header(bytes: &[u8; HEADER_LEN as usize]) -> IndexHeader {
    let mut cursor = 0;
    IndexHeader {
        version: take_u32(bytes, &mut cursor),
        source_checksum: take_u64(bytes, &mut cursor),
        source_len: take_u64(bytes, &mut cursor),
        record_count: take_u64(bytes, &mut cursor),
        bucket_count: take_u64(bytes, &mut cursor),
        directory_offset: take_u64(bytes, &mut cursor),
        context_offset: take_u64(bytes, &mut cursor),
        context_count: take_u64(bytes, &mut cursor),
        payload_offset: take_u64(bytes, &mut cursor),
        directory_checksum: take_u64(bytes, &mut cursor),
        context_checksum: take_u64(bytes, &mut cursor),
    }
}

fn encode_bucket_metadata(metadata: &BucketMetadata) -> [u8; BUCKET_METADATA_LEN as usize] {
    let mut bytes = [0_u8; BUCKET_METADATA_LEN as usize];
    let mut cursor = 0;
    put_i64(&mut bytes, &mut cursor, metadata.day);
    for value in [
        metadata.offset,
        metadata.len,
        metadata.checksum,
        metadata.record_count,
    ] {
        put_u64(&mut bytes, &mut cursor, value);
    }
    let checksum = super::fnv1a_bytes(0, &bytes[..cursor]);
    put_u64(&mut bytes, &mut cursor, checksum);
    bytes
}

fn decode_bucket_metadata(
    bytes: &[u8; BUCKET_METADATA_LEN as usize],
) -> io::Result<BucketMetadata> {
    let expected_checksum = super::fnv1a_bytes(0, &bytes[..40]);
    let mut cursor = 0;
    let metadata = BucketMetadata {
        day: take_i64(bytes, &mut cursor),
        offset: take_u64(bytes, &mut cursor),
        len: take_u64(bytes, &mut cursor),
        checksum: take_u64(bytes, &mut cursor),
        record_count: take_u64(bytes, &mut cursor),
    };
    if take_u64(bytes, &mut cursor) != expected_checksum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record index bucket metadata checksum mismatch",
        ));
    }
    Ok(metadata)
}

fn encode_record_location(location: &RecordLocation) -> [u8; RECORD_LOCATION_LEN as usize] {
    let mut bytes = [0_u8; RECORD_LOCATION_LEN as usize];
    let mut cursor = 0;
    put_i64(&mut bytes, &mut cursor, location.timestamp_seconds);
    for value in [location.offset, location.len, location.checksum] {
        put_u64(&mut bytes, &mut cursor, value);
    }
    bytes
}

fn decode_record_location(bytes: &[u8; RECORD_LOCATION_LEN as usize]) -> RecordLocation {
    let mut cursor = 0;
    RecordLocation {
        timestamp_seconds: take_i64(bytes, &mut cursor),
        offset: take_u64(bytes, &mut cursor),
        len: take_u64(bytes, &mut cursor),
        checksum: take_u64(bytes, &mut cursor),
    }
}

fn encode_context_entry(digest: &[u8; 32]) -> [u8; CONTEXT_ENTRY_LEN as usize] {
    let mut bytes = [0_u8; CONTEXT_ENTRY_LEN as usize];
    bytes[..32].copy_from_slice(digest);
    bytes[32..].copy_from_slice(&super::fnv1a_bytes(0, digest).to_le_bytes());
    bytes
}

fn decode_context_entry(bytes: &[u8; CONTEXT_ENTRY_LEN as usize]) -> io::Result<[u8; 32]> {
    let mut digest = [0_u8; 32];
    digest.copy_from_slice(&bytes[..32]);
    let mut checksum = [0_u8; 8];
    checksum.copy_from_slice(&bytes[32..]);
    if u64::from_le_bytes(checksum) != super::fnv1a_bytes(0, &digest) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record index context entry checksum mismatch",
        ));
    }
    Ok(digest)
}

fn put_u32(bytes: &mut [u8], cursor: &mut usize, value: u32) {
    bytes[*cursor..*cursor + 4].copy_from_slice(&value.to_le_bytes());
    *cursor += 4;
}

fn put_u64(bytes: &mut [u8], cursor: &mut usize, value: u64) {
    bytes[*cursor..*cursor + 8].copy_from_slice(&value.to_le_bytes());
    *cursor += 8;
}

fn put_i64(bytes: &mut [u8], cursor: &mut usize, value: i64) {
    bytes[*cursor..*cursor + 8].copy_from_slice(&value.to_le_bytes());
    *cursor += 8;
}

fn take_u32(bytes: &[u8], cursor: &mut usize) -> u32 {
    let mut value = [0_u8; 4];
    value.copy_from_slice(&bytes[*cursor..*cursor + 4]);
    *cursor += 4;
    u32::from_le_bytes(value)
}

fn take_u64(bytes: &[u8], cursor: &mut usize) -> u64 {
    let mut value = [0_u8; 8];
    value.copy_from_slice(&bytes[*cursor..*cursor + 8]);
    *cursor += 8;
    u64::from_le_bytes(value)
}

fn take_i64(bytes: &[u8], cursor: &mut usize) -> i64 {
    let mut value = [0_u8; 8];
    value.copy_from_slice(&bytes[*cursor..*cursor + 8]);
    *cursor += 8;
    i64::from_le_bytes(value)
}

fn index_path(path: &Path) -> PathBuf {
    path.with_extension("idx")
}
