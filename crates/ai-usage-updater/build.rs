fn main() {
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=AI_USAGE_BUILD_TARGET={target}");
    println!("cargo:rerun-if-env-changed=TARGET");
}
