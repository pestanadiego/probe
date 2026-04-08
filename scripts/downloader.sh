#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_DIR="$ROOT_DIR/corpus"
EVAL_DIR="$CORPUS_DIR/eval"
DATASHEETS_DIR="$CORPUS_DIR/datasheets/pdf"
OVERWRITE=0

for arg in "$@"; do
	case "$arg" in
		--overwrite)
			OVERWRITE=1
			;;
		*)
			printf 'unknown argument: %s\nusage: %s [--overwrite]\n' "$arg" "$0" >&2
			exit 1
			;;
	esac
done

log() {
	printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

section() {
	printf '\n[%s] ===== %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

reuse_or_download() {
	local file_path="$1"
	shift

	if [[ -f "$file_path" && "$OVERWRITE" -eq 0 ]]; then
		log "using existing file: $file_path"
		return 0
	fi

	if [[ -f "$file_path" && "$OVERWRITE" -eq 1 ]]; then
		log "overwrite enabled, replacing existing file: $file_path"
		rm -f "$file_path"
	fi

	if ! "$@"; then
		return 1
	fi

	log "downloaded file: $file_path"
	return 0
}

verify_pdf() {
	local file_path="$1"

	if [[ -f "$file_path" ]] && [[ $(stat -c%s "$file_path") -gt 102400 ]]; then
		log "verified pdf: $file_path"
	else
		log "warning: $file_path is missing or unexpectedly small"
	fi
}

download_techqa() {
	local target_dir="$EVAL_DIR/techqa"
	local raw_dir="$target_dir/raw"
	local archive_path="$raw_dir/TechQA.tar.gz"

	section "TechQA"
	log "preparing directories"
	mkdir -p "$target_dir" "$raw_dir"

	if ! command -v hf >/dev/null 2>&1; then
		log "installing huggingface_hub cli"
		pip install -U "huggingface_hub[cli]"
	fi

	if ! reuse_or_download "$archive_path" hf download PrimeQA/TechQA TechQA.tar.gz --repo-type dataset --local-dir "$raw_dir"; then
		log "error: TechQA download failed, continuing"
		return 1
	fi

	log "extracting TechQA into $target_dir"
	if ! tar -xzf "$archive_path" -C "$target_dir"; then
		log "error: TechQA extraction failed, continuing"
		return 1
	fi

	log "TechQA download complete"
	log "archive: $archive_path"
	find "$target_dir" -maxdepth 2 -mindepth 1 | sort
}

download_emqap() {
	local target_dir="$EVAL_DIR/emqap"
	local repo_url="https://github.com/abhi1nandy2/EMNLP-2021-Findings/archive/refs/heads/main.zip"
	local archive_path="$target_dir/emqap_repo.zip"

	section "EMQAP"
	log "preparing directories"
	mkdir -p "$target_dir"

	if ! reuse_or_download "$archive_path" curl -fL "$repo_url" -o "$archive_path"; then
		log "error: EMQAP download failed, continuing"
		return 1
	fi

	log "extracting EMQAP data"
	if ! unzip -q "$archive_path" -d "$target_dir"; then
		log "error: EMQAP extraction failed, continuing"
		return 1
	fi

	if [[ -d "$target_dir/EMNLP-2021-Findings-main/data" ]]; then
		log "moving extracted data into $target_dir"
		mv "$target_dir/EMNLP-2021-Findings-main/data"/* "$target_dir/"
	else
		log "warning: expected EMQAP data folder was not found"
	fi

	log "cleaning temporary files"
	rm -rf "$target_dir/EMNLP-2021-Findings-main"
	log "keeping archive for reuse: $archive_path"

	log "EMQAP download complete"
	find "$target_dir" -maxdepth 2 -mindepth 1 | sort
}

download_manuals() {
	section "Hardware Manuals"
	log "preparing directories"
	mkdir -p "$DATASHEETS_DIR"

	if ! reuse_or_download "$DATASHEETS_DIR/ESP32_Technical_Reference.pdf" curl -fL "https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf" -o "$DATASHEETS_DIR/ESP32_Technical_Reference.pdf"; then
		log "warning: continuing after ESP32 manual issue"
	fi

	if ! reuse_or_download "$DATASHEETS_DIR/STM32F4_Reference_Manual.pdf" curl -fL "https://users.ece.utexas.edu/~valvano/EE345L/Labs/Fall2011/CortexM4_TRM_r0p1.pdf" -o "$DATASHEETS_DIR/STM32F4_Reference_Manual.pdf"; then
		log "warning: continuing after STM32 manual issue"
	fi

	if ! reuse_or_download "$DATASHEETS_DIR/ARM_Cortex-M4_User_Guide.pdf" curl -fL "https://eng.auburn.edu/~nelson/courses/elec5260_6260/slides/ARM%20STM32F476%20Interrupts.pdf" -o "$DATASHEETS_DIR/ARM_Cortex-M4_User_Guide.pdf"; then
		log "warning: continuing after ARM manual issue"
	fi

	section "Hardware Manual Verification"
	verify_pdf "$DATASHEETS_DIR/ESP32_Technical_Reference.pdf"
	verify_pdf "$DATASHEETS_DIR/STM32F4_Reference_Manual.pdf"
	verify_pdf "$DATASHEETS_DIR/ARM_Cortex-M4_User_Guide.pdf"

	log "manuals saved to $DATASHEETS_DIR"
	find "$DATASHEETS_DIR" -maxdepth 1 -mindepth 1 | sort
}

main() {
	section "Starting Combined Download Process"
	log "root directory: $ROOT_DIR"
	log "eval directory: $EVAL_DIR"
	log "datasheets directory: $DATASHEETS_DIR"
	if [[ "$OVERWRITE" -eq 1 ]]; then
		log "overwrite mode enabled"
	else
		log "overwrite mode disabled; existing files will be reused"
	fi

	download_manuals
	if ! download_emqap; then
		log "continuing after EMQAP failure"
	fi
	if ! download_techqa; then
		log "continuing after TechQA failure"
	fi

	section "All Downloads Complete"
	log "data is ready under corpus/eval and corpus/datasheets/pdf"
}

main "$@"
