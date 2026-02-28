//! Unified Inference Backend Trait
//!
//! This module defines the **`InferenceBackend`** trait — the central abstraction
//! of the MoFA inference orchestration layer. It provides a single, object-safe
//! interface that every inference provider (cloud API, local GPU, edge accelerator)
//! must implement.
//!
//! ## MoFA Microkernel Alignment
//!
//! The MoFA architecture follows a microkernel design where the foundation layer
//! defines **stable trait boundaries** that isolate higher layers from implementation
//! details. `InferenceBackend` is the "system call" interface for inference:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    MoFA Agent / Workflow Layer                   │
//! │         (uses InferenceRequest / InferenceResponse only)        │
//! └────────────────────────────┬─────────────────────────────────────┘
//!                              │  Box<dyn InferenceBackend>
//! ┌────────────────────────────┴─────────────────────────────────────┐
//! │                    InferenceBackend Trait                         │
//! │    ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
//! │    │ CloudOpenAI  │  │  LocalMLX    │  │ LinuxCandleBackend│    │
//! │    │  (Phase 1)   │  │  (Phase 2)   │  │    (Phase 3)      │    │
//! │    └──────────────┘  └──────────────┘  └───────────────────┘    │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Object Safety & Concurrency
//!
//! The trait is designed to be **object-safe** so backends can be dynamically
//! dispatched via `Box<dyn InferenceBackend>`. This enables:
//!
//! - Runtime backend selection (e.g., prefer local GPU, fallback to cloud)
//! - Backend swapping without recompilation
//! - The future `ModelPool` to manage a heterogeneous collection of backends
//!
//! All methods are `async` (via `#[async_trait]`) and the trait requires
//! `Send + Sync`, ensuring safe use across tokio tasks and threads.
//!
//! ## Streaming Design
//!
//! The `stream()` method returns a `Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>`.
//! This is the standard Rust pattern for dynamically-dispatched async streams:
//!
//! - **`Pin<Box<...>>`**: Required because the stream is a self-referential future
//! - **`dyn Stream`**: Object-safe erasure of the concrete stream type
//! - **`+ Send`**: Allows the stream to be polled from any tokio worker thread
//!
//! Consumers drive the stream with `StreamExt::next()` from the `futures` crate.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use super::types::{InferenceError, InferenceRequest, InferenceResponse, Token};

/// The core inference backend trait for the MoFA orchestration layer.
///
/// Every inference provider — whether a cloud API adapter (OpenAI, Anthropic),
/// a local runtime (MLX on macOS, Candle on Linux), or an edge accelerator —
/// implements this trait to plug into the unified orchestrator.
///
/// ## Lifecycle
///
/// ```text
/// ┌────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
/// │  Construct  │ ──▶ │  initialize() │ ──▶ │  generate()  │ ──▶ │ shutdown │
/// │  (new)      │     │  (load/conn)  │     │  / stream()  │     │  ()      │
/// └────────────┘     └──────────────┘     └──────────────┘     └──────────┘
/// ```
///
/// 1. **Construction**: Backend is created with its configuration (API keys, model paths)
/// 2. **Initialization**: Asynchronously load weights, establish connections, warm caches
/// 3. **Generation**: Process inference requests (streaming or non-streaming)
/// 4. **Shutdown**: Gracefully release resources (GPU memory, HTTP connections)
///
/// ## Object Safety
///
/// This trait is object-safe: `Box<dyn InferenceBackend>` is a valid type.
/// The orchestrator uses this to maintain a dynamic collection of backends
/// and select the best one at runtime based on availability and load.
///
/// ## Thread Safety
///
/// The `Send + Sync` bounds ensure that a backend can be shared across
/// tokio tasks via `Arc<dyn InferenceBackend>`. Implementations must
/// use interior mutability (e.g., `RwLock`, `Mutex`) if they need to
/// mutate internal state during inference.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use mofa_foundation::orchestrator::{
///     InferenceBackend, InferenceRequest, ChatMessage,
/// };
///
/// async fn run_inference(backend: Arc<dyn InferenceBackend>) {
///     backend.initialize().await.expect("Backend init failed");
///
///     if backend.is_available().await {
///         let request = InferenceRequest::new(vec![
///             ChatMessage::user("Explain quantum computing"),
///         ]);
///         let response = backend.generate(&request).await.unwrap();
///         println!("Response: {}", response.content);
///     }
///
///     backend.shutdown().await.expect("Shutdown failed");
/// }
/// ```
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Returns the human-readable name of this backend.
    ///
    /// Used for logging, telemetry, and display. Examples:
    /// - `"CloudOpenAI"`
    /// - `"LocalMLX-AppleSilicon"`
    /// - `"LinuxCandle-CUDA"`
    fn name(&self) -> &str;

    /// Asynchronously initialize the backend.
    ///
    /// This is called once after construction. Implementations should:
    /// - **Cloud backends**: Validate API keys, establish HTTP/2 connections, warm DNS
    /// - **Local backends**: Load model weights into GPU/CPU memory, initialize tokenizers
    ///
    /// Returns `Ok(())` on success, or an [`InferenceError`] if initialization fails.
    async fn initialize(&self) -> Result<(), InferenceError>;

    /// Check if the backend is ready to serve inference requests.
    ///
    /// This is a lightweight health check — it should NOT perform a full inference.
    /// Typical checks include:
    /// - Is the model loaded in memory?
    /// - Is the API endpoint reachable?
    /// - Is the GPU device healthy?
    ///
    /// Returns `true` if the backend can accept requests, `false` otherwise.
    async fn is_available(&self) -> bool;

    /// Execute a non-streaming inference request.
    ///
    /// Sends the full request to the backend and waits for the complete response.
    /// This is the simplest interface and is suitable for batch processing,
    /// background tasks, and scenarios where streaming is not needed.
    ///
    /// The backend is responsible for:
    /// - Converting [`InferenceRequest`] to its provider-specific format
    /// - Executing the inference (API call or local forward pass)
    /// - Converting the provider response back to [`InferenceResponse`]
    async fn generate(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError>;

    /// Execute a streaming inference request.
    ///
    /// Returns a pinned, boxed, `Send` async stream that yields individual
    /// [`Token`] values as they are generated. This enables:
    ///
    /// - **Low latency**: First token arrives before full generation completes
    /// - **Progressive rendering**: Chat UIs can display tokens as they arrive
    /// - **Backpressure**: The consumer controls polling rate
    ///
    /// ## Stream Semantics
    ///
    /// - The stream yields `Ok(Token)` for each generated token
    /// - The stream yields `Err(InferenceError)` if an error occurs mid-stream
    /// - The stream terminates naturally when generation is complete
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = backend.stream(&request).await?;
    /// while let Some(result) = stream.next().await {
    ///     match result {
    ///         Ok(token) => print!("{}", token.text),
    ///         Err(e) => eprintln!("Stream error: {}", e),
    ///     }
    /// }
    /// ```
    async fn stream(
        &self,
        request: &InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>, InferenceError>;

    /// Gracefully shut down the backend and release all resources.
    ///
    /// Implementations should:
    /// - **Cloud backends**: Close HTTP connection pools, flush pending requests
    /// - **Local backends**: Unload model weights, free GPU memory, drop tokenizers
    ///
    /// After `shutdown()`, `is_available()` should return `false` and
    /// `generate()`/`stream()` should return `BackendUnavailable`.
    async fn shutdown(&self) -> Result<(), InferenceError>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time verification that `InferenceBackend` is object-safe.
    ///
    /// If this function compiles, the trait can be used as `dyn InferenceBackend`,
    /// which is essential for dynamic dispatch in the orchestrator.
    #[allow(dead_code)]
    fn assert_object_safe(_: &dyn InferenceBackend) {}

    /// Compile-time verification that `Box<dyn InferenceBackend>` is Send + Sync.
    ///
    /// This ensures backends can be shared across tokio tasks via `Arc`.
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_trait_bounds() {
        // If this compiles, the trait satisfies Send + Sync for dynamic dispatch
        assert_send_sync::<Box<dyn InferenceBackend>>();
    }
}
