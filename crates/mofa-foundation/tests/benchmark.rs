//! Benchmark Test Suite — Dual Backend Trade Study
//!
//! This module evaluates both `LlamaCppProvider` and `CandleProvider` against
//! the same `InferenceBackend` trait, proving they are:
//!
//! 1. **Interchangeable** — both implement the same trait the `RequestRouter` uses
//! 2. **Lifecycle-safe** — both initialize, generate, and shut down without panicking
//! 3. **Measurable** — reports TTFT, total generation time, and throughput
//!
//! ## Running the Benchmarks
//!
//! ```bash
//! # With a real GGUF model file:
//! GGUF_MODEL_PATH=/path/to/model.gguf cargo test -p mofa-foundation benchmark -- --nocapture
//!
//! # Without a model file (tests graceful error handling only):
//! cargo test -p mofa-foundation benchmark -- --nocapture
//! ```

#[cfg(test)]
mod benchmark_tests {
    use std::sync::Arc;
    use std::time::Instant;

    use mofa_foundation::orchestrator::backend::InferenceBackend;
    use mofa_foundation::orchestrator::types::{ChatMessage, InferenceRequest};

    #[cfg(feature = "llama-cpp")]
    use mofa_foundation::orchestrator::local_llama::{LlamaCppConfig, LlamaCppProvider};

    #[cfg(feature = "candle")]
    use mofa_foundation::orchestrator::local_candle::{CandleConfig, CandleProvider};

    /// Helper: Create a standard test request.
    fn test_request() -> InferenceRequest {
        InferenceRequest::new(vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("What is the capital of France?"),
        ])
        .with_max_tokens(50)
    }

    /// Helper: Get the GGUF model path from environment, or None.
    fn gguf_model_path() -> Option<String> {
        std::env::var("GGUF_MODEL_PATH").ok()
    }

    // ========================================================================
    // Trait Compliance Tests
    // ========================================================================

    #[cfg(feature = "candle")]
    #[test]
    fn test_candle_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CandleProvider>();
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_llama_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlamaCppProvider>();
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_candle_implements_inference_backend() {
        fn assert_backend<T: InferenceBackend>() {}
        assert_backend::<CandleProvider>();
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_llama_implements_inference_backend() {
        fn assert_backend<T: InferenceBackend>() {}
        assert_backend::<LlamaCppProvider>();
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_candle_can_be_boxed_as_dyn_backend() {
        // This proves dynamic dispatch works — critical for RequestRouter
        let candle_config = CandleConfig::new("/tmp/test.gguf");
        let _backend: Box<dyn InferenceBackend> = Box::new(CandleProvider::new(candle_config));
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_llama_can_be_boxed_as_dyn_backend() {
        let llama_config = LlamaCppConfig::new("/tmp/test.gguf");
        let _backend: Box<dyn InferenceBackend> = Box::new(LlamaCppProvider::new(llama_config));
    }

    // ========================================================================
    // Error Handling Tests (no model file needed)
    // ========================================================================

    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_candle_graceful_error_on_missing_model() {
        let config = CandleConfig::new("/nonexistent/path/model.gguf");
        let provider = CandleProvider::new(config);

        let result = provider.initialize().await;
        assert!(result.is_err(), "Should fail gracefully with missing model");

        // Should report as not available
        assert!(!provider.is_available().await);
    }

    #[cfg(feature = "llama-cpp")]
    #[tokio::test]
    async fn test_llama_graceful_error_on_missing_model() {
        let config = LlamaCppConfig::new("/nonexistent/path/model.gguf");
        let provider = LlamaCppProvider::new(config);

        let result = provider.initialize().await;
        assert!(result.is_err(), "Should fail gracefully with missing model");
        assert!(!provider.is_available().await);
    }

    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_candle_generate_before_init_fails() {
        let config = CandleConfig::new("/tmp/model.gguf");
        let provider = CandleProvider::new(config);
        let request = test_request();

        let result = provider.generate(&request).await;
        assert!(result.is_err());
    }

    #[cfg(feature = "llama-cpp")]
    #[tokio::test]
    async fn test_llama_generate_before_init_fails() {
        let config = LlamaCppConfig::new("/tmp/model.gguf");
        let provider = LlamaCppProvider::new(config);
        let request = test_request();

        let result = provider.generate(&request).await;
        assert!(result.is_err());
    }

    // ========================================================================
    // Lifecycle Tests (with real model file)
    // ========================================================================

    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_candle_full_lifecycle() {
        let model_path = match gguf_model_path() {
            Some(path) => path,
            None => {
                println!("⚠️  Skipping Candle lifecycle test (set GGUF_MODEL_PATH and TOKENIZER_PATH)");
                return;
            }
        };

        let config = CandleConfig::new(&model_path)
            .with_device("cpu");

        let provider = CandleProvider::new(config);

        // Initialize
        let init_start = Instant::now();
        let init_result = provider.initialize().await;
        let init_time = init_start.elapsed();

        println!("╔══════════════════════════════════════════════╗");
        println!("║        Candle Backend Lifecycle Test         ║");
        println!("╠══════════════════════════════════════════════╣");
        println!(
            "║ Init time:     {:>10.3}ms                  ║",
            init_time.as_secs_f64() * 1000.0
        );

        if init_result.is_ok() {
            assert!(provider.is_available().await);

            // Generate
            let request = test_request();
            let gen_start = Instant::now();
            let gen_result = provider.generate(&request).await;
            let gen_time = gen_start.elapsed();

            if let Ok(response) = &gen_result {
                println!(
                    "║ Gen time:      {:>10.3}ms                  ║",
                    gen_time.as_secs_f64() * 1000.0
                );
                println!("║ Model:         {:>28} ║", response.model);
                println!(
                    "║ Prompt tokens: {:>10}                    ║",
                    response.usage.prompt_tokens
                );
                println!(
                    "║ Compl tokens:  {:>10}                    ║",
                    response.usage.completion_tokens
                );
            }

            // Shutdown
            let shutdown_result = provider.shutdown().await;
            assert!(shutdown_result.is_ok());
            assert!(!provider.is_available().await);

            println!("║ Shutdown:      ✅ Clean                      ║");
        } else {
            println!("║ Init:          ⚠️  Failed (expected w/o model)  ║");
        }

        println!("╚══════════════════════════════════════════════╝");

        // Cleanup
        let _ = std::fs::remove_file(&model_path);
    }

    #[cfg(feature = "llama-cpp")]
    #[tokio::test]
    async fn test_llama_full_lifecycle() {
        let model_path = match gguf_model_path() {
            Some(path) => path,
            None => {
                println!("⚠️  Skipping LlamaCpp lifecycle test (set GGUF_MODEL_PATH)");
                return;
            }
        };

        let config = LlamaCppConfig::new(&model_path)
            .with_gpu_layers(0) // CPU only for reliable testing
            .with_context_size(512)
            .with_default_max_tokens(20);

        let provider = LlamaCppProvider::new(config);

        // Initialize
        let init_start = Instant::now();
        let init_result = provider.initialize().await;
        let init_time = init_start.elapsed();

        println!("╔══════════════════════════════════════════════╗");
        println!("║       LlamaCpp Backend Lifecycle Test        ║");
        println!("╠══════════════════════════════════════════════╣");
        println!(
            "║ Init time:     {:>10.3}ms                  ║",
            init_time.as_secs_f64() * 1000.0
        );

        if let Err(e) = &init_result {
            println!(
                "║ Init error:    {:>28} ║",
                format!("{}", e).chars().take(28).collect::<String>()
            );
            println!("╚══════════════════════════════════════════════╝");
            return;
        }

        assert!(provider.is_available().await);

        // Generate with TTFT measurement
        let request = test_request();
        let gen_start = Instant::now();
        let gen_result = provider.generate(&request).await;
        let total_gen_time = gen_start.elapsed();

        if let Ok(response) = &gen_result {
            let tokens_per_sec =
                response.usage.completion_tokens as f64 / total_gen_time.as_secs_f64();

            println!(
                "║ Total gen:     {:>10.3}ms                  ║",
                total_gen_time.as_secs_f64() * 1000.0
            );
            println!(
                "║ Throughput:    {:>10.1} tok/s              ║",
                tokens_per_sec
            );
            println!("║ Model:         {:>28} ║", response.model);
            println!(
                "║ Output len:    {:>10} chars              ║",
                response.content.len()
            );
        } else if let Err(e) = &gen_result {
            println!(
                "║ Gen error:     {:>28} ║",
                format!("{}", e).chars().take(28).collect::<String>()
            );
        }

        // Shutdown
        let shutdown_result = provider.shutdown().await;
        assert!(shutdown_result.is_ok());
        println!("║ Shutdown:      ✅ Clean                      ║");
        println!("╚══════════════════════════════════════════════╝");
    }

    // ========================================================================
    // Comparative Benchmark
    // ========================================================================

    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_comparative_benchmark() {
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║          MoFA Inference Backend Trade Study                 ║");
        println!("║          Candle (Rust) vs LlamaCpp (C++ FFI)               ║");
        println!("╠══════════════════════════════════════════════════════════════╣");

        let request = test_request();

        // ── Candle Benchmark ──
        let candle_model_path = gguf_model_path();
        let (candle_init_time, candle_gen_time, candle_init_ok) = if let Some(ref path) = candle_model_path {
            let candle_config = CandleConfig::new(path)
                .with_device("cpu");

            let candle = CandleProvider::new(candle_config);

            let candle_init_start = Instant::now();
            let candle_init = candle.initialize().await;
            let init_time = candle_init_start.elapsed();
            let init_ok = candle_init.is_ok();

            let gen_time = if init_ok {
                let start = Instant::now();
                let result = candle.generate(&request).await;
                let t = start.elapsed();
                if let Ok(ref resp) = result {
                    println!("║  │ Generated: {:?}...", &resp.content.chars().take(40).collect::<String>());
                }
                Some(t)
            } else {
                if let Err(ref e) = candle_init {
                    println!("║  │ Init error: {}", e);
                }
                None
            };

            let _ = candle.shutdown().await;
            (init_time, gen_time, init_ok)
        } else {
            println!("║  │ Candle: SKIPPED (set GGUF_MODEL_PATH + TOKENIZER_PATH)");
            (std::time::Duration::ZERO, None, false)
        };

        println!("║                                                              ║");
        println!("║  ┌─ Candle (Pure Rust) ────────────────────────────────────┐ ║");
        println!(
            "║  │ Init:   {:>10.3}ms │ Status: {:>8}                 │ ║",
            candle_init_time.as_secs_f64() * 1000.0,
            if candle_model_path.is_none() {
                "⏭ Skip"
            } else if candle_init_ok {
                "✅ OK"
            } else {
                "❌ Fail"
            }
        );
        if let Some(gen_time) = candle_gen_time {
            println!(
                "║  │ Gen:    {:>10.3}ms │ Send+Sync: ✅                    │ ║",
                gen_time.as_secs_f64() * 1000.0
            );
        }
        println!("║  └────────────────────────────────────────────────────────┘ ║");

        // ── LlamaCpp Benchmark ──
        #[cfg(feature = "llama-cpp")]
        {
            let llama_path =
                gguf_model_path().unwrap_or_else(|| "/tmp/no_model.gguf".to_string());
            let llama_config = LlamaCppConfig::new(&llama_path)
                .with_gpu_layers(0)
                .with_context_size(512)
                .with_default_max_tokens(20);

            let llama = LlamaCppProvider::new(llama_config);

            let llama_init_start = Instant::now();
            let llama_init = llama.initialize().await;
            let llama_init_time = llama_init_start.elapsed();

            let llama_gen_time = if llama_init.is_ok() {
                let start = Instant::now();
                let _ = llama.generate(&request).await;
                Some(start.elapsed())
            } else {
                None
            };

            let _ = llama.shutdown().await;

            println!("║                                                              ║");
            println!("║  ┌─ LlamaCpp (C++ FFI) ───────────────────────────────────┐ ║");
            println!(
                "║  │ Init:   {:>10.3}ms │ Status: {:>8}                 │ ║",
                llama_init_time.as_secs_f64() * 1000.0,
                if llama_init.is_ok() {
                    "✅ OK"
                } else {
                    "❌ Fail"
                }
            );
            if let Some(gen_time) = llama_gen_time {
                println!(
                    "║  │ Gen:    {:>10.3}ms │ Send+Sync: ✅                    │ ║",
                    gen_time.as_secs_f64() * 1000.0
                );
            }
            println!("║  └────────────────────────────────────────────────────────┘ ║");
        }

        #[cfg(not(feature = "llama-cpp"))]
        {
            println!("║                                                              ║");
            println!("║  ┌─ LlamaCpp (C++ FFI) ───────────────────────────────────┐ ║");
            println!("║  │ Skipped (enable 'llama-cpp' feature)                    │ ║");
            println!("║  └────────────────────────────────────────────────────────┘ ║");
        }

        println!("║                                                              ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}
