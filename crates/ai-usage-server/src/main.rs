use std::path::PathBuf;
use ai_usage_server::{AppState, ServerConfig, build_app, spawn_auto_update_worker};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/etc/ai-usage-server/config.yaml"));
    let config = ServerConfig::load_from_path(&config_path)?;
    let listen = config.listen.clone();
    let auto_update = config.auto_update.clone();

    tracing_subscriber::fmt()
        .with_env_filter(config.log_level.clone())
        .init();

    let state = AppState::new(config)?;
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!("listening on {}", listen);
    spawn_auto_update_worker(auto_update);
    axum::serve(listener, build_app(state)).await?;
    Ok(())
}
