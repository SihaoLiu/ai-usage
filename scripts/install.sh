#!/bin/sh
set -eu

REPOSITORY="${AI_USAGE_REPOSITORY:-SihaoLiu/ai-usage}"
INSTALL_DIR="${AI_USAGE_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${AI_USAGE_VERSION:-latest}"
BINARY_NAME="ai-usage"

fail() {
    printf '%s\n' "$1" >&2
    exit 1
}

release_target() {
    system="$(uname -s)"
    machine="$(uname -m)"

    case "$system" in
        Linux)
            case "$machine" in
                x86_64 | amd64) printf '%s\n' 'x86_64-linux-musl' ;;
                aarch64 | arm64) printf '%s\n' 'aarch64-linux-musl' ;;
                *) fail "Unsupported Linux architecture: $machine" ;;
            esac
            ;;
        Darwin)
            case "$machine" in
                arm64 | aarch64) printf '%s\n' 'aarch64-apple-darwin' ;;
                *) fail "Unsupported macOS architecture: $machine" ;;
            esac
            ;;
        *) fail "Unsupported operating system: $system" ;;
    esac
}

download() {
    url="$1"
    destination="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$destination" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$destination" "$url"
    else
        fail 'Install requires curl or wget.'
    fi
}

target="$(release_target)"
asset="$BINARY_NAME-$target"
case "$VERSION" in
    latest) url="https://github.com/$REPOSITORY/releases/latest/download/$asset" ;;
    v*) url="https://github.com/$REPOSITORY/releases/download/$VERSION/$asset" ;;
    *) fail 'AI_USAGE_VERSION must be latest or a v-prefixed release tag.' ;;
esac

temporary="$(mktemp "${TMPDIR:-/tmp}/$BINARY_NAME.XXXXXX")"
trap 'rm -f "$temporary"' EXIT HUP INT TERM

printf 'Downloading %s...\n' "$asset"
download "$url" "$temporary"
chmod +x "$temporary"
mkdir -p "$INSTALL_DIR"
destination="$INSTALL_DIR/$BINARY_NAME"
mv "$temporary" "$destination"

printf 'Installed %s\n' "$destination"
case ":${PATH:-}:" in
    *":$INSTALL_DIR:"*) ;;
    *) printf 'Add %s to PATH to run ai-usage from future shells.\n' "$INSTALL_DIR" >&2 ;;
esac

printf 'Starting ai-usage...\n'
exec "$destination" "$@"
