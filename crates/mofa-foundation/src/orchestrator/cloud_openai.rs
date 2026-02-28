//! Cloud OpenAI Provider — `InferenceBackend` Adapter
//!
//! This module implements the [`InferenceBackend`] trait using the `async-openai`
//! crate as the cloud fallback provider. It serves as the **primary remote inference
//! adapter** in the MoFA orchestration layer, connecting to any OpenAI-compatible
//! API (OpenAI, Azure OpenAI, vLLM, Ollama, LocalAI, etc.).
//!
//! ## MoFA Architecture Role
//!
//! In the microkernel design, this provider occupies the **cloud fallback** position:
//!
//! ```text
//!   Request → [Orchestrator] → Local backend available? ── Yes → Local (MLX/Candle)
//!                                     │
//!                                     No
//!                                     │
//!                                     ▼
//!                            CloudOpenAIProvider  ← this module
//! ```
//!
//! When local GPU resources are unavailable or overloaded, the orchestrator
//! transparently routes requests through this cloud adapter.
//!
//! ## Key Design Decisions
//!
//! ### HTTP/2 Connection Pooling
//!
//! The `async-openai` client is built on `reqwest`, which maintains a persistent
//! HTTP/2 connection pool by default. We create the client **once** during
//! `CloudOpenAIProvider::new()` and reuse it across all requests, ensuring:
//!
//! - **Connection reuse**: No TLS handshake per request
//! - **Multiplexing**: Multiple concurrent requests over a single TCP connection
//! - **Keep-alive**: Connections are kept warm for rapid successive calls
//!
//! ### Retry Strategy
//!
//! Transient failures (HTTP 429 rate limits, 500/502/503 server errors) are
//! automatically retried with **exponential backoff and jitter**:
//!
//! ```text
//! Attempt 1: immediate
//! Attempt 2: base_delay × 2^0 + jitter  (e.g., ~1s)
//! Attempt 3: base_delay × 2^1 + jitter  (e.g., ~2s)
//! Attempt 4: base_delay × 2^2 + jitter  (e.g., ~4s)
//! ```
//!
//! The jitter prevents thundering-herd effects when multiple agents retry
//! simultaneously after a provider outage.
//!
//! ### Thread Safety
//!
//! The provider uses `tokio::sync::RwLock<bool>` for the initialization flag,
//! ensuring safe concurrent access from multiple tokio tasks. The `async-openai`
//! `Client` itself is `Clone + Send + Sync`, so it can be shared freely.

use std::pin::Pin;
use std::sync::Arc;

use async_openai::{
    Client,
    config::OpenAIConfig as AsyncOpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use tokio::sync::RwLock;

use super::backend::InferenceBackend;
use super::types::{
    ChatRole, InferenceError, InferenceRequest, InferenceResponse, Token, TokenUsage,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Cloud OpenAI inference backend.
///
/// This struct captures all parameters needed to connect to an OpenAI-compatible
/// API. It supports both the official OpenAI API and self-hosted alternatives
/// (vLLM, Ollama, LocalAI) via the `base_url` field.
///
/// # Example
///
/// ```rust
/// use mofa_foundation::orchestrator::cloud_openai::CloudOpenAIConfig;
///
/// // Official OpenAI
/// let config = CloudOpenAIConfig::new("sk-...")
///     .with_default_model("gpt-4o");
///
/// // Self-hosted vLLM
/// let config = CloudOpenAIConfig::new("dummy-key")
///     .with_base_url("http://localhost:8000/v1")
///     .with_default_model("llama-3.1-8b");
/// ```
#[derive(Debug, Clone)]
pub struct CloudOpenAIConfig {
    /// OpenAI API key (or compatible API key).
    pub api_key: String,

    /// Base URL for the API.
    /// Default: `"https://api.openai.com/v1"` (set by async-openai).
    /// Override for self-hosted services.
    pub base_url: Option<String>,

    /// Default model to use when `InferenceRequest.model` is `None`.
    pub default_model: String,

    /// Optional organization ID for OpenAI multi-org accounts.
    pub org_id: Option<String>,

    /// Maximum number of retry attempts for transient failures.
    /// Default: `3`.
    pub max_retries: u32,

    /// Base delay between retries in milliseconds.
    /// Actual delay = `base_delay_ms × 2^attempt + random_jitter`.
    /// Default: `1000` (1 second).
    pub retry_base_delay_ms: u64,
}

impl CloudOpenAIConfig {
    /// Create a new configuration with the given API key.
    ///
    /// Uses `"gpt-4o-mini"` as the default model (cost-effective, fast).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            default_model: "gpt-4o-mini".to_string(),
            org_id: None,
            max_retries: 3,
            retry_base_delay_ms: 1000,
        }
    }

    /// Create a configuration from the `OPENAI_API_KEY` environment variable.
    ///
    /// # Panics
    /// Panics if `OPENAI_API_KEY` is not set. For fallible creation, use
    /// `CloudOpenAIConfig::try_from_env()`.
    pub fn from_env() -> Self {
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
        Self::new(api_key)
    }

    /// Try to create a configuration from the `OPENAI_API_KEY` environment variable.
    ///
    /// Returns `Err(InferenceError::ConfigError)` if the env var is not set.
    pub fn try_from_env() -> Result<Self, InferenceError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            InferenceError::ConfigError("OPENAI_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(api_key))
    }

    /// Set the base URL for a self-hosted or alternative API.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set the organization ID.
    pub fn with_org_id(mut self, org_id: impl Into<String>) -> Self {
        self.org_id = Some(org_id.into());
        self
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the base retry delay in milliseconds.
    pub fn with_retry_base_delay_ms(mut self, ms: u64) -> Self {
        self.retry_base_delay_ms = ms;
        self
    }
}

// ============================================================================
// CloudOpenAIProvider
// ============================================================================

/// Cloud OpenAI inference backend — the primary cloud fallback provider.
///
/// This struct wraps the `async-openai` [`Client`] and implements
/// [`InferenceBackend`], providing:
///
/// - **Persistent HTTP/2 connection pooling** via the underlying `reqwest` client
/// - **Automatic retry with exponential backoff** for HTTP 429/5xx errors
/// - **Streaming token generation** through OpenAI's SSE streaming API
/// - **Normalized I/O**: Converts between MoFA's `InferenceRequest`/`InferenceResponse`
///   and OpenAI's native request/response types
///
/// ## Thread Safety
///
/// This provider is `Send + Sync` and can be shared across tokio tasks via `Arc`.
/// The `async-openai` `Client` is `Clone` and internally reference-counted,
/// so cloning the provider is cheap.
pub struct CloudOpenAIProvider {
    /// The async-openai client, pre-configured with API key, base URL, and org ID.
    /// Created once and reused for all requests (connection pooling).
    client: Client<AsyncOpenAIConfig>,

    /// Provider configuration.
    config: CloudOpenAIConfig,

    /// Whether the provider has been initialized.
    /// Uses async RwLock for safe concurrent access from multiple tasks.
    initialized: Arc<RwLock<bool>>,
}

impl std::fmt::Debug for CloudOpenAIProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudOpenAIProvider")
            .field("config", &self.config)
            .field("name", &"CloudOpenAI")
            .finish()
    }
}

impl CloudOpenAIProvider {
    /// Create a new Cloud OpenAI provider with the given configuration.
    ///
    /// The `async-openai` client is created immediately with persistent HTTP/2
    /// connection pooling enabled. No API calls are made until `initialize()`
    /// or `generate()`/`stream()` is called.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use mofa_foundation::orchestrator::cloud_openai::{CloudOpenAIConfig, CloudOpenAIProvider};
    ///
    /// let config = CloudOpenAIConfig::new("sk-...")
    ///     .with_default_model("gpt-4o");
    ///
    /// let provider = CloudOpenAIProvider::new(config);
    /// ```
    pub fn new(config: CloudOpenAIConfig) -> Self {
        // Build the async-openai config
        let mut openai_config = AsyncOpenAIConfig::new().with_api_key(&config.api_key);

        if let Some(ref base_url) = config.base_url {
            openai_config = openai_config.with_api_base(base_url);
        }

        if let Some(ref org_id) = config.org_id {
            openai_config = openai_config.with_org_id(org_id);
        }

        // Create the client — this establishes the HTTP/2 connection pool
        // via reqwest's default connector (supports keep-alive, multiplexing)
        let client = Client::with_config(openai_config);

        Self {
            client,
            config,
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Resolve the model to use for a request.
    ///
    /// Uses the request's model if specified, otherwise falls back to the
    /// provider's default model.
    fn resolve_model(&self, request: &InferenceRequest) -> String {
        request
            .model
            .clone()
            .unwrap_or_else(|| self.config.default_model.clone())
    }

    /// Convert MoFA chat messages to async-openai request messages.
    ///
    /// Maps our provider-agnostic `ChatRole` to OpenAI's specific message types.
    fn convert_messages(
        &self,
        messages: &[super::types::ChatMessage],
    ) -> Result<Vec<ChatCompletionRequestMessage>, InferenceError> {
        messages
            .iter()
            .map(|msg| self.convert_single_message(msg))
            .collect()
    }

    /// Convert a single MoFA message to an async-openai message.
    fn convert_single_message(
        &self,
        msg: &super::types::ChatMessage,
    ) -> Result<ChatCompletionRequestMessage, InferenceError> {
        match msg.role {
            ChatRole::System => {
                let m = ChatCompletionRequestSystemMessageArgs::default()
                    .content(msg.content.as_str())
                    .build()
                    .map_err(|e| {
                        InferenceError::ProviderError(format!(
                            "Failed to build system message: {}",
                            e
                        ))
                    })?;
                Ok(ChatCompletionRequestMessage::System(m))
            }
            ChatRole::User => {
                let m = ChatCompletionRequestUserMessageArgs::default()
                    .content(msg.content.as_str())
                    .build()
                    .map_err(|e| {
                        InferenceError::ProviderError(format!(
                            "Failed to build user message: {}",
                            e
                        ))
                    })?;
                Ok(ChatCompletionRequestMessage::User(m))
            }
            ChatRole::Assistant => {
                let m = ChatCompletionRequestAssistantMessageArgs::default()
                    .content(msg.content.as_str())
                    .build()
                    .map_err(|e| {
                        InferenceError::ProviderError(format!(
                            "Failed to build assistant message: {}",
                            e
                        ))
                    })?;
                Ok(ChatCompletionRequestMessage::Assistant(m))
            }
        }
    }

    /// Execute a request with retry logic for transient failures.
    ///
    /// Retries on:
    /// - HTTP 429 (rate limited) — applies exponential backoff
    /// - HTTP 500/502/503 (server error) — may be transient
    ///
    /// Does NOT retry on:
    /// - HTTP 400 (bad request) — client error, fix the request
    /// - HTTP 401/403 (auth error) — fix credentials
    /// - HTTP 404 (not found) — wrong model or endpoint
    async fn execute_with_retry(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let model = self.resolve_model(request);
        let openai_messages = self.convert_messages(&request.messages)?;

        let mut last_error: Option<InferenceError> = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff with jitter
                let base_delay = self.config.retry_base_delay_ms * 2_u64.pow(attempt - 1);
                let jitter = rand::random::<u64>() % (base_delay / 2 + 1);
                let delay = std::time::Duration::from_millis(base_delay + jitter);

                tracing::warn!(
                    "CloudOpenAI: retry attempt {}/{} after {:?} for model '{}'",
                    attempt,
                    self.config.max_retries,
                    delay,
                    model
                );

                tokio::time::sleep(delay).await;
            }

            // Build the OpenAI request
            let mut req_builder = CreateChatCompletionRequestArgs::default();
            req_builder.model(&model).messages(openai_messages.clone());

            if let Some(temp) = request.temperature {
                req_builder.temperature(temp);
            }
            if let Some(max_tokens) = request.max_tokens {
                req_builder.max_tokens(max_tokens as u32);
            }
            if let Some(top_p) = request.top_p {
                req_builder.top_p(top_p);
            }
            if let Some(ref stops) = request.stop_sequences {
                req_builder.stop(stops.clone());
            }

            let openai_request = req_builder.build().map_err(|e| {
                InferenceError::ProviderError(format!("Failed to build OpenAI request: {}", e))
            })?;

            // Execute the API call
            match self.client.chat().create(openai_request).await {
                Ok(response) => {
                    // Extract the first choice's content
                    let content = response
                        .choices
                        .first()
                        .and_then(|c| c.message.content.as_ref())
                        .cloned()
                        .unwrap_or_default();

                    let finish_reason = response
                        .choices
                        .first()
                        .and_then(|c| c.finish_reason.as_ref())
                        .map(|r| format!("{:?}", r).to_lowercase());

                    let usage = response
                        .usage
                        .map(|u| TokenUsage {
                            prompt_tokens: u.prompt_tokens,
                            completion_tokens: u.completion_tokens,
                            total_tokens: u.total_tokens,
                        })
                        .unwrap_or_default();

                    return Ok(InferenceResponse {
                        content,
                        model: response.model,
                        usage,
                        finish_reason,
                    });
                }
                Err(e) => {
                    let error = self.classify_error(&e);

                    // Only retry on retryable errors
                    if !self.is_retryable(&error) {
                        return Err(error);
                    }

                    tracing::warn!(
                        "CloudOpenAI: transient error on attempt {}: {}",
                        attempt + 1,
                        e
                    );
                    last_error = Some(error);
                }
            }
        }

        // All retries exhausted
        Err(last_error.unwrap_or_else(|| {
            InferenceError::ProviderError("All retry attempts exhausted".to_string())
        }))
    }

    /// Classify an async-openai error into our unified error type.
    ///
    /// This enables the retry logic to make intelligent decisions about
    /// which errors are transient (retryable) vs permanent (fail-fast).
    fn classify_error(&self, error: &async_openai::error::OpenAIError) -> InferenceError {
        let error_string = error.to_string();

        // Check for rate limiting (HTTP 429)
        if error_string.contains("429")
            || error_string.to_lowercase().contains("rate limit")
            || error_string.to_lowercase().contains("too many requests")
        {
            return InferenceError::RateLimited(error_string);
        }

        // Check for server errors (HTTP 5xx)
        if error_string.contains("500")
            || error_string.contains("502")
            || error_string.contains("503")
            || error_string.contains("504")
        {
            return InferenceError::ProviderError(format!("Server error: {}", error_string));
        }

        // Check for auth errors
        if error_string.contains("401") || error_string.contains("403") {
            return InferenceError::ConfigError(format!("Authentication error: {}", error_string));
        }

        // Check for timeout
        if error_string.to_lowercase().contains("timeout") {
            return InferenceError::Timeout(error_string);
        }

        // Default: generic provider error
        InferenceError::ProviderError(error_string)
    }

    /// Determine if an error is retryable.
    ///
    /// Only transient errors (rate limits, server errors, timeouts) are retried.
    /// Configuration errors and client errors are NOT retried.
    fn is_retryable(&self, error: &InferenceError) -> bool {
        matches!(
            error,
            InferenceError::RateLimited(_)
                | InferenceError::Timeout(_)
                | InferenceError::StreamError(_)
        ) || matches!(error, InferenceError::ProviderError(msg) if msg.starts_with("Server error"))
    }
}

// ============================================================================
// InferenceBackend Implementation
// ============================================================================

#[async_trait]
impl InferenceBackend for CloudOpenAIProvider {
    fn name(&self) -> &str {
        "CloudOpenAI"
    }

    /// Initialize the provider by validating configuration.
    ///
    /// For cloud backends, initialization is lightweight — we just verify
    /// that the API key is non-empty and mark ourselves as ready. The
    /// actual HTTP connection is established lazily on the first request
    /// (thanks to reqwest's connection pooling).
    async fn initialize(&self) -> Result<(), InferenceError> {
        if self.config.api_key.is_empty() {
            return Err(InferenceError::ConfigError("API key is empty".to_string()));
        }

        tracing::info!(
            "CloudOpenAI: initialized with model='{}', base_url={:?}",
            self.config.default_model,
            self.config.base_url.as_deref().unwrap_or("default"),
        );

        let mut init = self.initialized.write().await;
        *init = true;

        Ok(())
    }

    /// Check if the provider is initialized and ready.
    ///
    /// This does NOT make an API call — it only checks the internal state.
    /// For a full health check (e.g., listing models), that would be a
    /// separate method in a future phase.
    async fn is_available(&self) -> bool {
        *self.initialized.read().await
    }

    /// Generate a complete response (non-streaming).
    ///
    /// This method:
    /// 1. Converts MoFA messages to OpenAI format
    /// 2. Sends the request with retry logic
    /// 3. Extracts the response content and usage
    /// 4. Returns a normalized [`InferenceResponse`]
    async fn generate(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        if !*self.initialized.read().await {
            return Err(InferenceError::BackendUnavailable(
                "CloudOpenAI provider not initialized — call initialize() first".to_string(),
            ));
        }

        if request.messages.is_empty() {
            return Err(InferenceError::ProviderError(
                "Request must contain at least one message".to_string(),
            ));
        }

        tracing::debug!(
            "CloudOpenAI: generating response for model='{}' with {} messages",
            self.resolve_model(request),
            request.messages.len()
        );

        self.execute_with_retry(request).await
    }

    /// Stream tokens from the OpenAI streaming API.
    ///
    /// This method:
    /// 1. Converts MoFA messages to OpenAI format
    /// 2. Creates a streaming request via the OpenAI SSE endpoint
    /// 3. Wraps the `ChatCompletionResponseStream` in our `Pin<Box<dyn Stream>>`
    /// 4. Maps each SSE chunk to a `Token` value
    ///
    /// The returned stream is `Send` and can be polled from any tokio task.
    async fn stream(
        &self,
        request: &InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>, InferenceError>
    {
        if !*self.initialized.read().await {
            return Err(InferenceError::BackendUnavailable(
                "CloudOpenAI provider not initialized — call initialize() first".to_string(),
            ));
        }

        if request.messages.is_empty() {
            return Err(InferenceError::ProviderError(
                "Request must contain at least one message".to_string(),
            ));
        }

        let model = self.resolve_model(request);
        let openai_messages = self.convert_messages(&request.messages)?;

        // Build the streaming request
        let mut req_builder = CreateChatCompletionRequestArgs::default();
        req_builder.model(&model).messages(openai_messages);

        if let Some(temp) = request.temperature {
            req_builder.temperature(temp);
        }
        if let Some(max_tokens) = request.max_tokens {
            req_builder.max_tokens(max_tokens as u32);
        }
        if let Some(top_p) = request.top_p {
            req_builder.top_p(top_p);
        }
        if let Some(ref stops) = request.stop_sequences {
            req_builder.stop(stops.clone());
        }

        let openai_request = req_builder.build().map_err(|e| {
            InferenceError::ProviderError(format!("Failed to build streaming request: {}", e))
        })?;

        tracing::debug!(
            "CloudOpenAI: starting stream for model='{}' with {} messages",
            model,
            request.messages.len()
        );

        // Create the SSE stream
        let openai_stream = self
            .client
            .chat()
            .create_stream(openai_request)
            .await
            .map_err(|e| {
                let classified = self.classify_error(&e);
                tracing::error!("CloudOpenAI: stream creation failed: {}", e);
                classified
            })?;

        // Map the OpenAI stream chunks to our Token type
        let token_stream = openai_stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    // Extract delta content from the first choice
                    let text = chunk
                        .choices
                        .first()
                        .and_then(|c| c.delta.content.as_ref())
                        .cloned()
                        .unwrap_or_default();

                    Ok(Token::new(text))
                }
                Err(e) => {
                    tracing::warn!("CloudOpenAI: stream chunk error: {}", e);
                    Err(InferenceError::StreamError(format!(
                        "Stream chunk error: {}",
                        e
                    )))
                }
            }
        });

        Ok(Box::pin(token_stream))
    }

    /// Shut down the provider.
    ///
    /// For cloud backends, this marks the provider as unavailable.
    /// The underlying `reqwest` connection pool is dropped when the
    /// provider itself is dropped (RAII).
    async fn shutdown(&self) -> Result<(), InferenceError> {
        tracing::info!("CloudOpenAI: shutting down provider");

        let mut init = self.initialized.write().await;
        *init = false;

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
        let config = CloudOpenAIConfig::new("test-key-123")
            .with_default_model("gpt-4o")
            .with_base_url("https://custom.api.com/v1")
            .with_org_id("org-123")
            .with_max_retries(5)
            .with_retry_base_delay_ms(500);

        assert_eq!(config.api_key, "test-key-123");
        assert_eq!(config.default_model, "gpt-4o");
        assert_eq!(
            config.base_url.as_deref(),
            Some("https://custom.api.com/v1")
        );
        assert_eq!(config.org_id.as_deref(), Some("org-123"));
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.retry_base_delay_ms, 500);
    }

    #[test]
    fn test_config_defaults() {
        let config = CloudOpenAIConfig::new("key");
        assert_eq!(config.default_model, "gpt-4o-mini");
        assert!(config.base_url.is_none());
        assert!(config.org_id.is_none());
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay_ms, 1000);
    }

    #[test]
    fn test_provider_creation() {
        let config = CloudOpenAIConfig::new("test-key");
        let provider = CloudOpenAIProvider::new(config);
        assert_eq!(provider.name(), "CloudOpenAI");
    }

    #[test]
    fn test_model_resolution_with_request_model() {
        let config = CloudOpenAIConfig::new("key").with_default_model("gpt-4o-mini");
        let provider = CloudOpenAIProvider::new(config);

        let request = InferenceRequest::new(vec![super::super::types::ChatMessage::user("test")])
            .with_model("gpt-4o");

        assert_eq!(provider.resolve_model(&request), "gpt-4o");
    }

    #[test]
    fn test_model_resolution_fallback_to_default() {
        let config = CloudOpenAIConfig::new("key").with_default_model("gpt-4o-mini");
        let provider = CloudOpenAIProvider::new(config);

        let request = InferenceRequest::new(vec![super::super::types::ChatMessage::user("test")]);

        assert_eq!(provider.resolve_model(&request), "gpt-4o-mini");
    }

    #[test]
    fn test_error_classification_rate_limit() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);

        let error = InferenceError::RateLimited("429 Too Many Requests".into());
        assert!(provider.is_retryable(&error));
    }

    #[test]
    fn test_error_classification_server_error() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);

        let error = InferenceError::ProviderError("Server error: 500".into());
        assert!(provider.is_retryable(&error));
    }

    #[test]
    fn test_error_classification_config_not_retryable() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);

        let error = InferenceError::ConfigError("missing key".into());
        assert!(!provider.is_retryable(&error));
    }

    #[test]
    fn test_error_classification_timeout_retryable() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);

        let error = InferenceError::Timeout("request timed out".into());
        assert!(provider.is_retryable(&error));
    }

    #[tokio::test]
    async fn test_initialize_with_empty_key() {
        let config = CloudOpenAIConfig::new("");
        let provider = CloudOpenAIProvider::new(config);

        let result = provider.initialize().await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InferenceError::ConfigError(_)
        ));
    }

    #[tokio::test]
    async fn test_initialize_success() {
        let config = CloudOpenAIConfig::new("valid-key");
        let provider = CloudOpenAIProvider::new(config);

        assert!(!provider.is_available().await);

        let result = provider.initialize().await;
        assert!(result.is_ok());
        assert!(provider.is_available().await);
    }

    #[tokio::test]
    async fn test_generate_before_init() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);

        let request = InferenceRequest::new(vec![super::super::types::ChatMessage::user("test")]);

        let result = provider.generate(&request).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InferenceError::BackendUnavailable(_)
        ));
    }

    #[tokio::test]
    async fn test_generate_empty_messages() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);
        provider.initialize().await.unwrap();

        let request = InferenceRequest::new(vec![]);
        let result = provider.generate(&request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_shutdown() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);

        provider.initialize().await.unwrap();
        assert!(provider.is_available().await);

        provider.shutdown().await.unwrap();
        assert!(!provider.is_available().await);
    }

    #[test]
    fn test_message_conversion() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);

        let messages = vec![
            super::super::types::ChatMessage::system("You are helpful."),
            super::super::types::ChatMessage::user("Hello"),
            super::super::types::ChatMessage::assistant("Hi there!"),
        ];

        let converted = provider.convert_messages(&messages);
        assert!(converted.is_ok());
        assert_eq!(converted.unwrap().len(), 3);
    }

    #[test]
    fn test_try_from_env_missing_key() {
        // Ensure env var is not set for this test
        // SAFETY: This is a single-threaded test; removing the env var is safe here.
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let result = CloudOpenAIConfig::try_from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_provider_debug_format() {
        let config = CloudOpenAIConfig::new("key");
        let provider = CloudOpenAIProvider::new(config);
        let debug = format!("{:?}", provider);
        assert!(debug.contains("CloudOpenAIProvider"));
    }
}
