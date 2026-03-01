//! CandleProvider - Local Candle Backend for Native Rust Inference
//!
//! Loads quantized GGUF models and performs real autoregressive text generation
//! using Hugging Face's Candle framework (pure Rust, no C++ FFI).

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use futures::Stream;
use std::pin::Pin;
use tokenizers::Tokenizer;
use tokio::sync::RwLock;

use super::backend::InferenceBackend;
use super::types::{
    ChatMessage as InferenceChatMessage, ChatRole, InferenceError, InferenceRequest,
    InferenceResponse, Token, TokenUsage,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Candle local provider.
#[derive(Debug, Clone)]
pub struct CandleConfig {
    /// Path to the GGUF model weights file.
    pub model_path: PathBuf,
    /// Path to the tokenizer (`tokenizer.json`).
    pub tokenizer_path: Option<PathBuf>,
    /// Preferred device: `"cuda"`, `"metal"`, or `"cpu"`. Default: `"auto"`.
    pub device: String,
    /// CUDA device ordinal. Default: `0`.
    pub cuda_device_id: usize,
    /// Default max tokens to generate. Default: `256`.
    pub default_max_tokens: usize,
    /// Default sampling temperature. Default: `0.8`.
    pub default_temperature: f64,
    /// Top-p (nucleus sampling). Default: `0.9`.
    pub default_top_p: f64,
    /// Repeat penalty. Default: `1.1`.
    pub repeat_penalty: f32,
    /// Tokens for repeat penalty window. Default: `64`.
    pub repeat_last_n: usize,
    /// Random seed (0 = random). Default: `299792458`.
    pub seed: u64,
    /// Estimated memory footprint in bytes.
    pub estimated_footprint_bytes: Option<u64>,
}

impl CandleConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: None,
            device: "auto".to_string(),
            cuda_device_id: 0,
            default_max_tokens: 256,
            default_temperature: 0.8,
            default_top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 299792458,
            estimated_footprint_bytes: None,
        }
    }

    pub fn with_tokenizer(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    pub fn with_device(mut self, device: &str) -> Self {
        self.device = device.to_string();
        self
    }

    pub fn with_cuda_device(mut self, id: usize) -> Self {
        self.cuda_device_id = id;
        self
    }

    pub fn with_default_max_tokens(mut self, max: usize) -> Self {
        self.default_max_tokens = max;
        self
    }

    pub fn with_default_temperature(mut self, temp: f64) -> Self {
        self.default_temperature = temp;
        self
    }

    pub fn with_estimated_footprint(mut self, bytes: u64) -> Self {
        self.estimated_footprint_bytes = Some(bytes);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = penalty;
        self
    }

    pub fn estimate_footprint(&self) -> u64 {
        if let Some(explicit) = self.estimated_footprint_bytes {
            return explicit;
        }
        match std::fs::metadata(&self.model_path) {
            Ok(meta) => (meta.len() as f64 * 1.3) as u64,
            Err(_) => 0,
        }
    }
}

// ============================================================================
// CandleDeviceInfo
// ============================================================================

#[derive(Debug, Clone)]
pub struct CandleDeviceInfo {
    pub name: String,
    pub is_gpu: bool,
}

// ============================================================================
// CandleModelState
// ============================================================================

/// Internal state: loaded quantized model, tokenizer, and device.
struct CandleModelState {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    device_info: CandleDeviceInfo,
    footprint_bytes: u64,
    eos_token_id: Option<u32>,
    model_name: String,
    ready: bool,
}

unsafe impl Send for CandleModelState {}
unsafe impl Sync for CandleModelState {}

// ============================================================================
// CandleProvider
// ============================================================================

/// Local Candle inference backend with real GGUF model loading.
pub struct CandleProvider {
    config: CandleConfig,
    initialized: Arc<RwLock<bool>>,
    model_state: Arc<RwLock<Option<CandleModelState>>>,
}

impl std::fmt::Debug for CandleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleProvider")
            .field("model_path", &self.config.model_path)
            .field("device", &self.config.device)
            .finish()
    }
}

impl CandleProvider {
    pub fn new(config: CandleConfig) -> Self {
        Self {
            config,
            initialized: Arc::new(RwLock::new(false)),
            model_state: Arc::new(RwLock::new(None)),
        }
    }

    pub fn config(&self) -> &CandleConfig {
        &self.config
    }

    pub fn estimated_footprint(&self) -> u64 {
        self.config.estimate_footprint()
    }

    fn select_device(&self) -> Result<(Device, CandleDeviceInfo), InferenceError> {
        let device_pref = self.config.device.to_lowercase();

        if device_pref == "cpu" {
            return Ok((Device::Cpu, CandleDeviceInfo { name: "CPU".to_string(), is_gpu: false }));
        }

        if device_pref == "cuda" || device_pref == "auto" {
            match Device::new_cuda(self.config.cuda_device_id) {
                Ok(device) => {
                    tracing::info!("Candle: using CUDA device {}", self.config.cuda_device_id);
                    return Ok((device, CandleDeviceInfo {
                        name: format!("CUDA:{}", self.config.cuda_device_id),
                        is_gpu: true,
                    }));
                }
                Err(e) => {
                    if device_pref == "cuda" {
                        tracing::warn!("Candle: CUDA unavailable: {}", e);
                    }
                }
            }
        }

        if device_pref == "metal" || device_pref == "auto" {
            match Device::new_metal(0) {
                Ok(device) => {
                    tracing::info!("Candle: using Metal device");
                    return Ok((device, CandleDeviceInfo {
                        name: "Metal:0".to_string(),
                        is_gpu: true,
                    }));
                }
                Err(e) => {
                    if device_pref == "metal" {
                        tracing::warn!("Candle: Metal unavailable: {}", e);
                    }
                }
            }
        }

        tracing::info!("Candle: using CPU (no GPU available)");
        Ok((Device::Cpu, CandleDeviceInfo { name: "CPU".to_string(), is_gpu: false }))
    }

    fn resolve_tokenizer_path(&self) -> Result<PathBuf, InferenceError> {
        if let Some(ref path) = self.config.tokenizer_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }
        if let Some(parent) = self.config.model_path.parent() {
            let sibling = parent.join("tokenizer.json");
            if sibling.exists() {
                return Ok(sibling);
            }
        }
        if let Ok(env_path) = std::env::var("TOKENIZER_PATH") {
            let p = PathBuf::from(&env_path);
            if p.exists() {
                return Ok(p);
            }
        }
        Err(InferenceError::ConfigError(
            "Tokenizer not found. Place tokenizer.json next to the GGUF model or set TOKENIZER_PATH env var.".to_string(),
        ))
    }

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

    fn find_eos_token(tokenizer: &Tokenizer) -> Option<u32> {
        let candidates = ["</s>", "<|end|>", "<|eot_id|>", "<|endoftext|>"];
        for candidate in &candidates {
            if let Some(id) = tokenizer.token_to_id(candidate) {
                return Some(id);
            }
        }
        // Fallback: token ID 2 is EOS in many Llama models
        Some(2)
    }
}

// ============================================================================
// InferenceBackend Implementation
// ============================================================================

#[async_trait]
impl InferenceBackend for CandleProvider {
    fn name(&self) -> &str {
        "Candle"
    }

    /// Initialize: load GGUF model weights and tokenizer.
    async fn initialize(&self) -> Result<(), InferenceError> {
        if !self.config.model_path.exists() {
            return Err(InferenceError::ConfigError(format!(
                "Model path not found: {}",
                self.config.model_path.display()
            )));
        }

        let config = self.config.clone();
        let tokenizer_path = self.resolve_tokenizer_path()?;

        let result = tokio::task::spawn_blocking(move || {
            // Device selection
            let provider = CandleProvider::new(config.clone());
            let (device, device_info) = provider.select_device()?;

            tracing::info!(
                "Candle: loading GGUF model '{}' on {}",
                config.model_path.display(),
                device_info.name
            );

            // Load GGUF file
            let mut file = std::fs::File::open(&config.model_path).map_err(|e| {
                InferenceError::ConfigError(format!("Failed to open GGUF file: {}", e))
            })?;

            let gguf_content = gguf_file::Content::read(&mut file).map_err(|e| {
                InferenceError::ProviderError(format!("Failed to parse GGUF file: {}", e))
            })?;

            // Extract model name from GGUF metadata
            let model_name = gguf_content
                .metadata
                .get("general.name")
                .and_then(|v| match v {
                    gguf_file::Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    config
                        .model_path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                });

            // Load quantized model weights
            let model = ModelWeights::from_gguf(gguf_content, &mut file, &device).map_err(|e| {
                InferenceError::ProviderError(format!("Failed to load model weights: {}", e))
            })?;

            // Load tokenizer
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                InferenceError::ConfigError(format!(
                    "Failed to load tokenizer from '{}': {}",
                    tokenizer_path.display(),
                    e
                ))
            })?;

            let eos_token_id = CandleProvider::find_eos_token(&tokenizer);
            let footprint = config.estimate_footprint();

            tracing::info!(
                "Candle: loaded '{}' on {} (footprint ~{:.1}MB, EOS token: {:?})",
                model_name,
                device_info.name,
                footprint as f64 / 1_048_576.0,
                eos_token_id
            );

            Ok::<CandleModelState, InferenceError>(CandleModelState {
                model,
                tokenizer,
                device,
                device_info,
                footprint_bytes: footprint,
                eos_token_id,
                model_name,
                ready: true,
            })
        })
        .await
        .map_err(|e| {
            InferenceError::BackendUnavailable(format!(
                "Blocking task panicked during Candle init: {}",
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

    /// Generate text using real autoregressive inference.
    async fn generate(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        if !*self.initialized.read().await {
            return Err(InferenceError::BackendUnavailable(
                "Candle provider not initialized".to_string(),
            ));
        }

        if request.messages.is_empty() {
            return Err(InferenceError::ProviderError(
                "Request must contain at least one message".to_string(),
            ));
        }

        let prompt = Self::format_prompt(&request.messages);
        let max_tokens = request.max_tokens.unwrap_or(self.config.default_max_tokens as u32);
        let temperature = request
            .temperature
            .map(|t| t as f64)
            .unwrap_or(self.config.default_temperature);
        let top_p = request
            .top_p
            .map(|p| p as f64)
            .unwrap_or(self.config.default_top_p);
        let repeat_penalty = self.config.repeat_penalty;
        let repeat_last_n = self.config.repeat_last_n;
        let seed = self.config.seed;
        let model_state = self.model_state.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut state_guard = model_state.blocking_write();
            let state = state_guard.as_mut().ok_or_else(|| {
                InferenceError::BackendUnavailable("Model not loaded".to_string())
            })?;

            if !state.ready {
                return Err(InferenceError::BackendUnavailable(
                    "Candle model not ready".to_string(),
                ));
            }

            // Tokenize the prompt
            let encoding = state.tokenizer.encode(prompt.as_str(), true).map_err(|e| {
                InferenceError::ProviderError(format!("Tokenization failed: {}", e))
            })?;
            let prompt_tokens = encoding.get_ids().to_vec();
            let prompt_token_count = prompt_tokens.len() as u32;

            // Set up logits processor for sampling
            let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_p));

            // Autoregressive generation loop
            let mut all_tokens = prompt_tokens.clone();
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut next_token = *prompt_tokens.last().unwrap_or(&0);

            for index in 0..max_tokens {
                let input = Tensor::new(&[next_token], &state.device).map_err(|e| {
                    InferenceError::ProviderError(format!("Tensor creation failed: {}", e))
                })?;

                let logits = state.model.forward(&input, prompt_tokens.len() + index as usize).map_err(|e| {
                    InferenceError::ProviderError(format!("Forward pass failed: {}", e))
                })?;

                let logits = logits.squeeze(0).map_err(|e| {
                    InferenceError::ProviderError(format!("Logits squeeze failed: {}", e))
                })?;

                // Apply repeat penalty
                let logits = if repeat_penalty != 1.0 && !all_tokens.is_empty() {
                    let start = all_tokens.len().saturating_sub(repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        repeat_penalty,
                        &all_tokens[start..],
                    )
                    .map_err(|e| {
                        InferenceError::ProviderError(format!("Repeat penalty failed: {}", e))
                    })?
                } else {
                    logits
                };

                // Sample next token
                next_token = logits_processor.sample(&logits).map_err(|e| {
                    InferenceError::ProviderError(format!("Sampling failed: {}", e))
                })?;

                // Check for EOS
                if let Some(eos_id) = state.eos_token_id {
                    if next_token == eos_id {
                        break;
                    }
                }

                all_tokens.push(next_token);
                generated_tokens.push(next_token);
            }

            // Decode generated tokens
            let generated_text = state
                .tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| {
                    InferenceError::ProviderError(format!("Decoding failed: {}", e))
                })?;

            let completion_tokens = generated_tokens.len() as u32;
            let model_name = format!("candle-{}", state.model_name);

            Ok::<InferenceResponse, InferenceError>(InferenceResponse {
                content: generated_text,
                model: model_name,
                usage: TokenUsage {
                    prompt_tokens: prompt_token_count,
                    completion_tokens,
                    total_tokens: prompt_token_count + completion_tokens,
                },
                finish_reason: Some(if completion_tokens < max_tokens {
                    "stop".to_string()
                } else {
                    "length".to_string()
                }),
            })
        })
        .await
        .map_err(|e| InferenceError::ProviderError(format!(
            "Generation task panicked: {}", e
        )))?;

        result
    }

    /// Stream tokens from real autoregressive generation.
    async fn stream(
        &self,
        request: &InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>, InferenceError>
    {
        if !*self.initialized.read().await {
            return Err(InferenceError::BackendUnavailable(
                "Candle provider not initialized".to_string(),
            ));
        }

        if request.messages.is_empty() {
            return Err(InferenceError::ProviderError(
                "Request must contain at least one message".to_string(),
            ));
        }

        let prompt = Self::format_prompt(&request.messages);
        let max_tokens = request.max_tokens.unwrap_or(self.config.default_max_tokens as u32);
        let temperature = request
            .temperature
            .map(|t| t as f64)
            .unwrap_or(self.config.default_temperature);
        let top_p = request
            .top_p
            .map(|p| p as f64)
            .unwrap_or(self.config.default_top_p);
        let repeat_penalty = self.config.repeat_penalty;
        let repeat_last_n = self.config.repeat_last_n;
        let seed = self.config.seed;
        let model_state = self.model_state.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Token, InferenceError>>(32);

        tokio::task::spawn_blocking(move || {
            let result = (|| -> Result<(), InferenceError> {
                let mut state_guard = model_state.blocking_write();
                let state = state_guard.as_mut().ok_or_else(|| {
                    InferenceError::BackendUnavailable("Model not loaded".to_string())
                })?;

                if !state.ready {
                    return Err(InferenceError::BackendUnavailable(
                        "Candle model not ready".to_string(),
                    ));
                }

                let encoding = state.tokenizer.encode(prompt.as_str(), true).map_err(|e| {
                    InferenceError::ProviderError(format!("Tokenization failed: {}", e))
                })?;
                let prompt_tokens = encoding.get_ids().to_vec();

                let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_p));
                let mut all_tokens = prompt_tokens.clone();
                let mut next_token = *prompt_tokens.last().unwrap_or(&0);

                for index in 0..max_tokens {
                    let input = Tensor::new(&[next_token], &state.device).map_err(|e| {
                        InferenceError::ProviderError(format!("Tensor creation failed: {}", e))
                    })?;

                    let logits = state.model.forward(&input, prompt_tokens.len() + index as usize).map_err(|e| {
                        InferenceError::ProviderError(format!("Forward pass failed: {}", e))
                    })?;

                    let logits = logits.squeeze(0).map_err(|e| {
                        InferenceError::ProviderError(format!("Logits squeeze failed: {}", e))
                    })?;

                    let logits = if repeat_penalty != 1.0 && !all_tokens.is_empty() {
                        let start = all_tokens.len().saturating_sub(repeat_last_n);
                        candle_transformers::utils::apply_repeat_penalty(
                            &logits,
                            repeat_penalty,
                            &all_tokens[start..],
                        )
                        .map_err(|e| {
                            InferenceError::ProviderError(format!("Repeat penalty failed: {}", e))
                        })?
                    } else {
                        logits
                    };

                    next_token = logits_processor.sample(&logits).map_err(|e| {
                        InferenceError::ProviderError(format!("Sampling failed: {}", e))
                    })?;

                    if let Some(eos_id) = state.eos_token_id {
                        if next_token == eos_id {
                            break;
                        }
                    }

                    all_tokens.push(next_token);

                    // Decode this single token and stream it
                    let token_text = state.tokenizer.decode(&[next_token], true).unwrap_or_default();
                    if tx.blocking_send(Ok(Token::new(token_text))).is_err() {
                        break; // Receiver dropped
                    }
                }

                Ok(())
            })();

            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e));
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    /// Release model weights and free memory.
    async fn shutdown(&self) -> Result<(), InferenceError> {
        tracing::info!(
            "Candle: shutting down model '{}'",
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
        let config = CandleConfig::new("/tmp/model.gguf")
            .with_device("cuda")
            .with_cuda_device(1)
            .with_default_max_tokens(512)
            .with_default_temperature(0.7)
            .with_estimated_footprint(4 * 1024 * 1024 * 1024);

        assert_eq!(config.device, "cuda");
        assert_eq!(config.cuda_device_id, 1);
        assert_eq!(config.default_max_tokens, 512);
        assert_eq!(
            config.estimated_footprint_bytes,
            Some(4 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = CandleConfig::new("/tmp/model.gguf");
        assert_eq!(config.device, "auto");
        assert_eq!(config.cuda_device_id, 0);
        assert_eq!(config.default_max_tokens, 256);
        assert_eq!(config.default_top_p, 0.9);
    }

    #[tokio::test]
    async fn test_init_missing_model_file() {
        let config = CandleConfig::new("/nonexistent/model.gguf");
        let provider = CandleProvider::new(config);

        let result = provider.initialize().await;
        assert!(result.is_err());

        if let Err(InferenceError::ConfigError(msg)) = result {
            assert!(msg.contains("not found"));
        }
    }

    #[test]
    fn test_provider_name() {
        let config = CandleConfig::new("/tmp/model.gguf");
        let provider = CandleProvider::new(config);
        assert_eq!(provider.name(), "Candle");
    }

    #[tokio::test]
    async fn test_generate_before_init() {
        let config = CandleConfig::new("/tmp/model.gguf");
        let provider = CandleProvider::new(config);

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

        let prompt = CandleProvider::format_prompt(&messages);
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("[INST]"));
    }

    #[test]
    fn test_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CandleProvider>();
    }

    #[test]
    fn test_provider_debug() {
        let config = CandleConfig::new("/tmp/model.gguf").with_device("cpu");
        let provider = CandleProvider::new(config);
        let debug = format!("{:?}", provider);
        assert!(debug.contains("CandleProvider"));
    }

    #[test]
    fn test_device_info() {
        let info = CandleDeviceInfo {
            name: "CUDA:0".to_string(),
            is_gpu: true,
        };
        assert!(info.is_gpu);
        assert_eq!(info.name, "CUDA:0");
    }
}
