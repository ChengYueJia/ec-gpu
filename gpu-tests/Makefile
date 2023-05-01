help: ## Display this help screen
	@grep -h \
		-E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test_all: ## Run Super Circuit benchmarks
	@cargo test --no-default-features --features cuda

fft_bench: ## Run Super Circuit benchmarks
	@cargo bench --bench fft --no-default-features --features cuda

msm_bench: ## Run Super Circuit benchmarks
	@cargo bench --bench msm --no-default-features --features cuda

bench_all: fft_bench msm_bench ## Run All Circuit benchmarks

.PHONY: test_all bench_all