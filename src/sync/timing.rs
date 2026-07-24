pub(crate) fn stable_hash(identity: &str, discriminator: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in identity
        .as_bytes()
        .iter()
        .copied()
        .chain(discriminator.to_be_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
