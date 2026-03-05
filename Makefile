BIN := vibe-usage
MUSL_TARGET := x86_64-unknown-linux-musl
INSTALL_DIR ?= $(HOME)/.local/bin

.PHONY: all build release install clean

all: build release

build:
	RUSTFLAGS="-C target-cpu=native" cargo build --release
	@echo "Built: target/release/$(BIN)"

release:
	cargo build --release --target $(MUSL_TARGET)
	@echo "Built: target/$(MUSL_TARGET)/release/$(BIN)"

install: build
	@if [ ! -d "$(INSTALL_DIR)" ]; then echo "Error: $(INSTALL_DIR) does not exist" >&2; exit 1; fi
	@if [ ! -w "$(INSTALL_DIR)" ]; then echo "Error: $(INSTALL_DIR) is not writable" >&2; exit 1; fi
	cp target/release/$(BIN) $(INSTALL_DIR)/$(BIN)
	@echo "Installed: $(INSTALL_DIR)/$(BIN)"

clean:
	cargo clean
