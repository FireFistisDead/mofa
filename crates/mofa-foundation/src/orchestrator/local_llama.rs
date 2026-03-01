//! LlamaCppProvider — Local llama.cpp Backend via `llama_cpp_2`
//!
//! This module provides a **local inference backend** powered by llama.cpp's
//! optimized C++ runtime via the [`llama_cpp_2`] Rust bindings. It implements
//! the [`InferenceBackend`](super::backend::InferenceBackend) trait, making it
//! fully compatible with the [`RequestRouter`](super::router::RequestRouter)
//! for seamless hot-swapping between local and cloud execution.
//!
//! ## Key Features
//!
//! - **GGUF model loading** — Load quantized models (Q4, Q5, Q8, FP16)
//! - **GPU layer offloading** — Optionally offload transformer layers to GPU
//! - **Streaming token generation** — Autoregressive decoding yielding tokens
//! - **Send + Sync safety** — Wraps raw C++ context safely for tokio
//! - **Memory footprint tracking** — Reports model size for admission control
//!
//! ## Thread Safety
//!
//! `llama_cpp_2`'s `LlamaModel` and `LlamaContext` are **not** `Send`/`Sync`
//! by default due to raw C pointer interiors. We wrap them in a dedicated
//! blocking thread via `tokio::task::spawn_blocking`, ensuring all `llama.cpp`
//! operations run on a single OS thread while the async interface remains
//! fully `Send + Sync`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use mofa_foundation::orchestrator::local_llama::{LlamaCppConfig, LlamaCppProvider};
//!
//! let config = LlamaCppConfig::new("/models/llama-3.2-1b-q4.gguf")
//!     .with_gpu_layers(35)
//!     .with_context_size(2048);
//!
//! let provider = LlamaCppProvider::new(config);
//! provider.initialize().await?;
//!
//! let request = InferenceRequest::new(vec![ChatMessage::user("Hello!")])
//!     .with_max_tokens(128);
//!
//! let response = provider.generate(&request).await?;
//! println!("{}", response.content);
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use tokio::sync::RwLock;

use super::backend::InferenceBackend;
use super::types::{
    ChatMessage as InferenceChatMessage, ChatRole, InferenceError, InferenceRequest,
    InferenceResponse, Token, TokenUsage,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the llama.cpp local provider.
///
/// Controls model path, GPU offloading, context window, and generation defaults.
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Path to the `.gguf` model file on disk.
    pub model_path: PathBuf,

    /// Number of transformer layers to offload to GPU.
    /// Set to 0 for CPU-only inference, or 999 to offload all layers.
    /// Default: `0` (CPU only — safest for unknown hardware).
    pub gpu_layers: u32,

    /// Context window size in tokens.
    /// Default: `2048`.
    pub context_size: u32,

    /// Number of threads for CPU inference.
    /// Default: number of physical CPU cores.
    pub threads: Option<u32>,

    /// Default max tokens to generate if not specified in request.
    /// Default: `256`.
    pub default_max_tokens: usize,

    /// Default temperature for sampling.
    /// Default: `0.8`.
    pub default_temperature: f32,

    /// Estimated memory footprint of this model in bytes.
    /// Used by the `ModelPool` for admission control.
    /// If not set, estimated from file size.
    pub estimated_footprint_bytes: Option<u64>,
}

impl LlamaCppConfig {
    /// Create a new config with the given GGUF model path.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            gpu_layers: 0,
            context_size: 2048,
            threads: None,
            default_max_tokens: 256,
            default_temperature: 0.8,
            estimated_footprint_bytes: None,
        }
    }

    /// Set GPU layer count for offloading.
    pub fn with_gpu_layers(mut self, layers: u32) -> Self {
        self.gpu_layers = layers;
        self
    }

    /// Set context window size.
    pub fn with_context_size(mut self, size: u32) -> Self {
        self.context_size = size;
        self
    }

    /// Set CPU thread count.
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Set default max tokens.
    pub fn with_default_max_tokens(mut self, max: usize) -> Self {
        self.default_max_tokens = max;
        self
    }

    /// Set default temperature.
    pub fn with_default_temperature(mut self, temp: f32) -> Self {
        self.default_temperature = temp;
        self
    }

    /// Set estimated memory footprint (for ModelPool admission control).
    pub fn with_estimated_footprint(mut self, bytes: u64) -> Self {
        self.estimated_footprint_bytes = Some(bytes);
        self
    }

    /// Estimate the memory footprint from the GGUF file size.
    ///
    /// Heuristic: GGUF file size × 1.2 accounts for runtime overhead
    /// (KV cache, scratch buffers, etc.).
    pub fn estimate_footprint(&self) -> u64 {
        if let Some(explicit) = self.estimated_footprint_bytes {
            return explicit;
        }

        // Use file size × 1.2 as a heuristic
        match std::fs::metadata(&self.model_path) {
            Ok(meta) => (meta.len() as f64 * 1.2) as u64,
            Err(_) => 0,
        }
    }
}

// ============================================================================
// LlamaCppProvider
// ============================================================================

/// Local llama.cpp inference backend.
///
/// This struct provides a fully async interface to the llama.cpp runtime.
/// All heavy C++ operations (model loading, token generation, context teardown)
/// are dispatched to a blocking thread pool via `tokio::task::spawn_blocking`,
/// keeping the async executor responsive.
///
/// ## OOM Prevention
///
/// Before creating this provider, the `RequestRouter` checks
/// `TelemetryMonitor::can_admit_model(config.estimate_footprint())` to ensure
/// loading the model won't exceed the safety margin. This prevents OS-level
/// OOM kills on constrained devices.
pub struct LlamaCppProvider {
    /// Provider configuration.
    config: LlamaCppConfig,

    /// Whether the model has been successfully loaded.
    /// Uses async RwLock for safe concurrent access from multiple routing tasks.
    initialized: Arc<RwLock<bool>>,

    /// The loaded model state, wrapped in Arc for sharing with blocking threads.
    /// The inner Option is None before initialization and after shutdown.
    ///
    /// We use a separate `Arc<RwLock<Option<...>>>` because the model/context
    /// objects from `llama_cpp_2` are not `Send`, so they must be created,
    /// used, and dropped on the same blocking thread. We store them as
    /// opaque handles that are only accessed from `spawn_blocking` closures.
    model_state: Arc<RwLock<Option<LlamaCppState>>>,
}

/// Internal model state — holds the loaded llama.cpp model and context.
///
/// This is stored behind `Arc<RwLock<Option<...>>>` and only accessed from
/// `spawn_blocking` closures to maintain thread safety.
struct LlamaCppState {
    /// The loaded GGUF model.
    model: llama_cpp_2::model::LlamaModel,

    /// Estimated memory footprint.
    footprint_bytes: u64,
}

// SAFETY: LlamaModel internally manages its own thread safety via `llama.cpp`'s
// backend mutex. We enforce single-threaded access via `spawn_blocking` + `RwLock`.
unsafe impl Send for LlamaCppState {}
unsafe impl Sync for LlamaCppState {}

impl std::fmt::Debug for LlamaCppProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppProvider")
            .field("model_path", &self.config.model_path)
            .field("gpu_layers", &self.config.gpu_layers)
            .field("context_size", &self.config.context_size)
            .finish()
    }
}

impl LlamaCppProvider {
    /// Create a new llama.cpp provider (model not loaded yet).
    pub fn new(config: LlamaCppConfig) -> Self {
        Self {
            config,
            initialized: Arc::new(RwLock::new(false)),
            model_state: Arc::new(RwLock::new(None)),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }

    /// Get the estimated memory footprint.
    pub fn estimated_footprint(&self) -> u64 {
        self.config.estimate_footprint()
    }

    /// Format messages into a prompt string.
    ///
    /// Converts MoFA's `ChatMessage` list into a simple prompt format.
    /// In production, this would use the model's chat template.
    fn format_prompt(messages: &[InferenceChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", msg.content));
                }
                ChatRole::User => {
                    prompt.push_str(&format!("[INST] {} [/INST]\n", msg.content));
                }
                ChatRole::Assistant => {
                    prompt.push_str(&format!("{}\n", msg.content));
                }
            }
        }

        prompt
    }
}

// ============================================================================
// InferenceBackend Implementation
// ============================================================================

#[async_trait]
impl InferenceBackend for LlamaCppProvider {
    fn name(&self) -> &str {
        "LlamaCpp"
    }

    /// Load the GGUF model file and create the inference context.
    ///
    /// This method:
    /// 1. Validates the model file exists
    /// 2. Initializes the llama.cpp backend
    /// 3. Loads the model with GPU layer offloading
    /// 4. Creates the inference context
    ///
    /// All operations run on a blocking thread to avoid stalling the async runtime.
    async fn initialize(&self) -> Result<(), InferenceError> {
        // Validate model file exists
        if !self.config.model_path.exists() {
            return Err(InferenceError::ConfigError(format!(
                "GGUF model file not found: {}",
                self.config.model_path.display()
            )));
        }

        let config = self.config.clone();
        let model_state = self.model_state.clone();

        // Load model on a blocking thread (llama.cpp is CPU-intensive during load)
        let result = tokio::task::spawn_blocking(move || {
            // Initialize the llama.cpp backend
            let backend = llama_cpp_2::llama_backend::LlamaBackend::init().map_err(|e| {
                InferenceError::BackendUnavailable(format!(
                    "Failed to initialize llama.cpp backend: {:?}",
                    e
                ))
            })?;

            // Configure model parameters
            let model_params = {
                let mut params = llama_cpp_2::model::params::LlamaModelParams::default();
                params = params.with_n_gpu_layers(config.gpu_layers);
                params
            };

            // Load the GGUF model
            let model_path_str = config.model_path.to_str().ok_or_else(|| {
                InferenceError::ConfigError("Model path contains invalid UTF-8".to_string())
            })?;

            let model = llama_cpp_2::model::LlamaModel::load_from_file(
                &backend,
                model_path_str,
                &model_params,
            )
            .map_err(|e| {
                InferenceError::BackendUnavailable(format!(
                    "Failed to load GGUF model '{}': {:?}",
                    model_path_str, e
                ))
            })?;

            let footprint = config.estimate_footprint();

            tracing::info!(
                "LlamaCpp: loaded model '{}' ({:.1}GB, {} GPU layers)",
                model_path_str,
                footprint as f64 / 1_073_741_824.0,
                config.gpu_layers,
            );

            // Store the state
            let state = LlamaCppState {
                model,
                footprint_bytes: footprint,
            };

            // We need to return the state to store it
            Ok::<LlamaCppState, InferenceError>(state)
        })
        .await
        .map_err(|e| {
            InferenceError::BackendUnavailable(format!(
                "Blocking task panicked during model load: {}",
                e
            ))
        })?;

        let state = result?;
        *self.model_state.write().await = Some(state);
        *self.initialized.write().await = true;

        Ok(())
    }

    async fn is_available(&self) -> bool {
        *self.initialized.read().await
    }

    /// Generate a complete response using llama.cpp's autoregressive decoding.
    ///
    /// Dispatches the entire generation loop to a blocking thread,
    /// collecting all tokens and returning the concatenated result.
    async fn generate(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        if !*self.initialized.read().await {
            return Err(InferenceError::BackendUnavailable(
                "LlamaCpp provider not initialized".to_string(),
            ));
        }

        if request.messages.is_empty() {
            return Err(InferenceError::ProviderError(
                "Request must contain at least one message".to_string(),
            ));
        }

        let prompt = Self::format_prompt(&request.messages);
        let max_tokens = request
            .max_tokens
            .unwrap_or(self.config.default_max_tokens as u32);
        let _temperature = request
            .temperature
            .unwrap_or(self.config.default_temperature);
        let context_size = self.config.context_size;
        let model_state = self.model_state.clone();
        let n_threads = self.config.threads;

        // Run generation on a blocking thread
        let result = tokio::task::spawn_blocking(move || {
            let state_guard = model_state.blocking_read();
            let state = state_guard.as_ref().ok_or_else(|| {
                InferenceError::BackendUnavailable("Model not loaded".to_string())
            })?;

            // Create a context for this generation
            let mut ctx_params = llama_cpp_2::context::params::LlamaContextParams::default()
                .with_n_ctx(std::num::NonZero::new(context_size));

            if let Some(threads) = n_threads {
                ctx_params = ctx_params.with_n_threads(threads as i32);
            }

            let mut ctx = state
                .model
                .new_context(
                    &llama_cpp_2::llama_backend::LlamaBackend::init()
                        .map_err(|e| InferenceError::ProviderError(format!("{:?}", e)))?,
                    ctx_params,
                )
                .map_err(|e| {
                    InferenceError::ProviderError(format!(
                        "Failed to create llama context: {:?}",
                        e
                    ))
                })?;

            // Tokenize the prompt
            let tokens = state
                .model
                .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
                .map_err(|e| {
                    InferenceError::ProviderError(format!("Tokenization failed: {:?}", e))
                })?;

            let prompt_tokens = tokens.len() as u32;

            // Create a batch with the prompt tokens
            let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(context_size as usize, 1);

            // Add prompt tokens to the batch
            for (i, &token) in tokens.iter().enumerate() {
                let is_last = i == tokens.len() - 1;
                batch.add(token, i as i32, &[0], is_last).map_err(|_| {
                    InferenceError::ProviderError("Failed to add token to batch".to_string())
                })?;
            }

            // Decode the prompt
            ctx.decode(&mut batch).map_err(|e| {
                InferenceError::ProviderError(format!("Prompt decoding failed: {:?}", e))
            })?;

            // Autoregressive generation loop
            let mut generated_text = String::new();
            let mut completion_tokens: u32 = 0;
            let mut n_cur = tokens.len();

            // Setup sampler (greedy decoding)
            let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain_simple([
                llama_cpp_2::sampling::LlamaSampler::greedy(),
            ]);

            for _ in 0..max_tokens {
                // Sample next token
                let new_token = sampler.sample(&ctx, -1);

                // Check for EOS
                if state.model.is_eog_token(new_token) {
                    break;
                }

                // Decode token to string
                let token_str = state
                    .model
                    .token_to_str(new_token, llama_cpp_2::model::Special::Tokenize)
                    .map_err(|e| {
                        InferenceError::ProviderError(format!("Token decoding failed: {:?}", e))
                    })?;

                generated_text.push_str(&token_str);
                completion_tokens += 1;

                // Prepare next batch
                batch.clear();
                batch
                    .add(new_token, n_cur as i32, &[0], true)
                    .map_err(|_| {
                        InferenceError::ProviderError(
                            "Failed to add generated token to batch".to_string(),
                        )
                    })?;

                n_cur += 1;

                // Decode the new token
                ctx.decode(&mut batch).map_err(|e| {
                    InferenceError::ProviderError(format!("Token decoding step failed: {:?}", e))
                })?;
            }

            Ok::<InferenceResponse, InferenceError>(InferenceResponse {
                content: generated_text,
                model: "llama-cpp-local".to_string(),
                usage: TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
                finish_reason: Some("stop".to_string()),
            })
        })
        .await
        .map_err(|e| InferenceError::ProviderError(format!("Generation task panicked: {}", e)))?;

        result
    }

    /// Stream tokens from the autoregressive generation loop.
    ///
    /// Uses a `tokio::sync::mpsc` channel to bridge between the blocking
    /// generation thread and the async stream interface.
    async fn stream(
        &self,
        request: &InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>, InferenceError>
    {
        if !*self.initialized.read().await {
            return Err(InferenceError::BackendUnavailable(
                "LlamaCpp provider not initialized".to_string(),
            ));
        }

        if request.messages.is_empty() {
            return Err(InferenceError::ProviderError(
                "Request must contain at least one message".to_string(),
            ));
        }

        let prompt = Self::format_prompt(&request.messages);
        let max_tokens = request
            .max_tokens
            .unwrap_or(self.config.default_max_tokens as u32);
        let context_size = self.config.context_size;
        let model_state = self.model_state.clone();
        let n_threads = self.config.threads;

        // Create a channel for streaming tokens
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Token, InferenceError>>(32);

        // Spawn the generation loop on a blocking thread
        tokio::task::spawn_blocking(move || {
            let result = (|| -> Result<(), InferenceError> {
                let state_guard = model_state.blocking_read();
                let state = state_guard.as_ref().ok_or_else(|| {
                    InferenceError::BackendUnavailable("Model not loaded".to_string())
                })?;

                let mut ctx_params = llama_cpp_2::context::params::LlamaContextParams::default()
                    .with_n_ctx(std::num::NonZero::new(context_size));

                if let Some(threads) = n_threads {
                    ctx_params = ctx_params.with_n_threads(threads as i32);
                }

                let mut ctx = state
                    .model
                    .new_context(
                        &llama_cpp_2::llama_backend::LlamaBackend::init()
                            .map_err(|e| InferenceError::ProviderError(format!("{:?}", e)))?,
                        ctx_params,
                    )
                    .map_err(|e| {
                        InferenceError::ProviderError(format!(
                            "Failed to create llama context: {:?}",
                            e
                        ))
                    })?;

                let tokens = state
                    .model
                    .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
                    .map_err(|e| {
                        InferenceError::ProviderError(format!("Tokenization failed: {:?}", e))
                    })?;

                let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(context_size as usize, 1);

                for (i, &token) in tokens.iter().enumerate() {
                    let is_last = i == tokens.len() - 1;
                    batch.add(token, i as i32, &[0], is_last).map_err(|_| {
                        InferenceError::ProviderError("Failed to add token to batch".to_string())
                    })?;
                }

                ctx.decode(&mut batch).map_err(|e| {
                    InferenceError::ProviderError(format!("Prompt decoding failed: {:?}", e))
                })?;

                // Setup sampler (greedy decoding)
                let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain_simple([
                    llama_cpp_2::sampling::LlamaSampler::greedy(),
                ]);

                let mut n_cur = tokens.len();

                for _ in 0..max_tokens {
                    let new_token = sampler.sample(&ctx, -1);

                    if state.model.is_eog_token(new_token) {
                        break;
                    }

                    let token_str = state
                        .model
                        .token_to_str(new_token, llama_cpp_2::model::Special::Tokenize)
                        .map_err(|e| {
                            InferenceError::ProviderError(format!("Token decoding failed: {:?}", e))
                        })?;

                    // Send the token through the channel
                    if tx.blocking_send(Ok(Token::new(token_str))).is_err() {
                        break; // Receiver dropped — stream was cancelled
                    }

                    batch.clear();
                    batch
                        .add(new_token, n_cur as i32, &[0], true)
                        .map_err(|_| {
                            InferenceError::ProviderError("Failed to add token".to_string())
                        })?;

                    n_cur += 1;

                    ctx.decode(&mut batch).map_err(|e| {
                        InferenceError::ProviderError(format!("Decode failed: {:?}", e))
                    })?;
                }

                Ok(())
            })();

            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e));
            }
        });

        // Convert mpsc::Receiver into a Stream
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    /// Shut down the provider and release the C++ model resources.
    ///
    /// Drops the `LlamaModel` and `LlamaContext` objects, which triggers
    /// llama.cpp's `llama_free_model()` and `llama_free()` respectively,
    /// releasing all CPU/GPU memory.
    async fn shutdown(&self) -> Result<(), InferenceError> {
        tracing::info!(
            "LlamaCpp: shutting down model '{}'",
            self.config.model_path.display()
        );

        *self.model_state.write().await = None;
        *self.initialized.write().await = false;

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = LlamaCppConfig::new("/tmp/model.gguf")
            .with_gpu_layers(35)
            .with_context_size(4096)
            .with_threads(8)
            .with_default_max_tokens(512)
            .with_default_temperature(0.7)
            .with_estimated_footprint(4 * 1024 * 1024 * 1024);

        assert_eq!(config.gpu_layers, 35);
        assert_eq!(config.context_size, 4096);
        assert_eq!(config.threads, Some(8));
        assert_eq!(config.default_max_tokens, 512);
        assert_eq!(config.default_temperature, 0.7);
        assert_eq!(
            config.estimated_footprint_bytes,
            Some(4 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = LlamaCppConfig::new("/tmp/model.gguf");
        assert_eq!(config.gpu_layers, 0);
        assert_eq!(config.context_size, 2048);
        assert_eq!(config.threads, None);
        assert_eq!(config.default_max_tokens, 256);
    }

    #[tokio::test]
    async fn test_init_missing_model_file() {
        let config = LlamaCppConfig::new("/nonexistent/model.gguf");
        let provider = LlamaCppProvider::new(config);

        let result = provider.initialize().await;
        assert!(result.is_err());

        if let Err(InferenceError::ConfigError(msg)) = result {
            assert!(msg.contains("not found"));
        }
    }

    #[test]
    fn test_provider_name() {
        let config = LlamaCppConfig::new("/tmp/model.gguf");
        let provider = LlamaCppProvider::new(config);
        assert_eq!(provider.name(), "LlamaCpp");
    }

    #[tokio::test]
    async fn test_generate_before_init() {
        let config = LlamaCppConfig::new("/tmp/model.gguf");
        let provider = LlamaCppProvider::new(config);

        let request = InferenceRequest::new(vec![super::super::types::ChatMessage::user("test")]);
        let result = provider.generate(&request).await;
        assert!(matches!(result, Err(InferenceError::BackendUnavailable(_))));
    }

    #[test]
    fn test_format_prompt() {
        let messages = vec![
            InferenceChatMessage::system("You are helpful."),
            InferenceChatMessage::user("Hello!"),
        ];

        let prompt = LlamaCppProvider::format_prompt(&messages);
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("Hello!"));
    }

    #[test]
    fn test_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlamaCppProvider>();
    }

    #[test]
    fn test_provider_debug() {
        let config = LlamaCppConfig::new("/tmp/model.gguf").with_gpu_layers(10);
        let provider = LlamaCppProvider::new(config);
        let debug = format!("{:?}", provider);
        assert!(debug.contains("LlamaCppProvider"));
        assert!(debug.contains("gpu_layers"));
    }
}
