SCHEMA := schema/rocket_executable_def.fbs
GENERATED_DIR ?= generated
FLATCC ?= iree-flatcc-cli
FLATC ?= flatc

.PHONY: generate generate-c generate-rust validate clean

generate: generate-c generate-rust

generate-c:
	mkdir -p "$(GENERATED_DIR)/c"
	"$(FLATCC)" --reader --builder --verifier --common \
	  -o "$(GENERATED_DIR)/c" "$(SCHEMA)"

generate-rust:
	mkdir -p "$(GENERATED_DIR)/rust"
	"$(FLATC)" --rust -o "$(GENERATED_DIR)/rust" "$(SCHEMA)"

validate:
	rm -rf build/validate
	mkdir -p build/validate
	"$(FLATCC)" --reader --builder --verifier --common \
	  -o build/validate "$(SCHEMA)"

clean:
	rm -rf build generated

