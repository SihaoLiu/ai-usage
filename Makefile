BIN := vibe-usage
MUSL_TARGET := x86_64-unknown-linux-musl
DEMO := docs/assets/ai-usage.gif

.PHONY: all build demo release clean

all: build release

build:
	RUSTFLAGS="-C target-cpu=native" cargo build --release
	@echo "Built: target/release/$(BIN)"

demo: build
	docs/assets/render-demo.py --output $(DEMO)
	@echo "Updated: $(DEMO)"

release: demo
	cargo build --release --target $(MUSL_TARGET)
	@echo "Built: target/$(MUSL_TARGET)/release/$(BIN)"

clean:
	cargo clean
