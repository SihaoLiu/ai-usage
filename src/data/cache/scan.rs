use std::io;
use std::path::Path;

use super::{
    CACHE_VERSION, CachedUsageRecord, ENTRY_FILE_MAGIC, ENTRY_INDEX_MAGIC, index, persistence,
    read_cached_records, vendor_entries_path,
};

pub(crate) enum VisitCachedRecordsError<E> {
    Cache(io::Error),
    Visitor(E),
}

pub(crate) fn try_for_each_vendor_persisted_record<E>(
    cache_root: &Path,
    vendor: &str,
    mut visitor: impl FnMut(CachedUsageRecord) -> Result<(), E>,
) -> Result<(), VisitCachedRecordsError<E>> {
    let path = vendor_entries_path(cache_root, vendor);
    if !path.exists() {
        return Ok(());
    }
    if index::matches_source_generation(&path, ENTRY_FILE_MAGIC, ENTRY_INDEX_MAGIC) {
        let streamed = persistence::try_for_each_current_cached_record(&path, |record| {
            visitor(record.into_cached_usage_record(vendor))
        });
        return match streamed {
            Ok(()) => Ok(()),
            Err(persistence::VisitCurrentCachedRecordsError::Cache(error)) => {
                Err(VisitCachedRecordsError::Cache(error))
            }
            Err(persistence::VisitCurrentCachedRecordsError::Visitor(error)) => {
                Err(VisitCachedRecordsError::Visitor(error))
            }
        };
    }

    let records = read_cached_records(&path).map_err(VisitCachedRecordsError::Cache)?;
    let _ = index::ensure(
        &path,
        ENTRY_FILE_MAGIC,
        ENTRY_INDEX_MAGIC,
        CACHE_VERSION,
        &records,
    );
    for record in records {
        visitor(record.into_cached_usage_record(vendor))
            .map_err(VisitCachedRecordsError::Visitor)?;
    }
    Ok(())
}
