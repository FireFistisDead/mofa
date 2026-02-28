//! ModelPool — Thread-Safe Model Registry with LRU Eviction
//!
//! This module implements the **concurrency-safe model pool** for the MoFA
//! inference orchestrator. It manages the lifecycle of locally-loaded
//! [`InferenceBackend`](super::backend::InferenceBackend) instances and provides
//! **LRU (Least Recently Used) eviction** to reclaim memory on constrained devices.
//!
//! ## Why LRU Eviction is Critical for Edge Devices
//!
//! Consider a 16GB device running two models:
//!
//! ```text
//! Model A (Llama-3.1-8B, FP16): ~16GB
//! Model B (Whisper-small, FP32):  ~1GB
//!
//! Total loaded: 17GB → exceeds physical RAM → OOM crash!
//! ```
//!
//! The `ModelPool` prevents this by:
//!
//! 1. **Tracking last-access timestamps** for every loaded model
//! 2. **Consulting the [`TelemetryMonitor`](super::telemetry::TelemetryMonitor)**
//!    before admitting new models
//! 3. **Evicting the least-recently-used model** when memory is insufficient,
//!    calling its `shutdown()` method for graceful teardown before removal
//!
//! ```text
//! ┌────────────────────────────────────────┐
//! │             ModelPool                  │
//! │  ┌──────────────────────────────────┐  │
//! │  │ RwLock<HashMap<String, Instance>>│  │
//! │  │                                  │  │
//! │  │  "llama-8b" → ModelInstance {     │  │
//! │  │    backend: Box<dyn Backend>,     │  │
//! │  │    last_used: Instant::now(),     │  │
//! │  │    footprint: 14GB,              │  │
//! │  │  }                               │  │
//! │  │                                  │  │
//! │  │  "whisper" → ModelInstance {      │  │
//! │  │    backend: Box<dyn Backend>,     │  │
//! │  │    last_used: 30min ago,         │  │ ← LRU candidate
//! │  │    footprint: 1GB,               │  │
//! │  │  }                               │  │
//! │  └──────────────────────────────────┘  │
//! └────────────────────────────────────────┘
//! ```
//!
//! ## Thread Safety
//!
//! The pool uses `tokio::sync::RwLock` to allow concurrent read access
//! (e.g., multiple inference requests routing to different models) while
//! ensuring exclusive write access during model loading and eviction.
//!
//! All public methods are `async` and safe to call from multiple tokio tasks.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio::time::Instant;

use super::backend::InferenceBackend;
use super::telemetry::TelemetryMonitor;
use super::types::{InferenceError, InferenceRequest, InferenceResponse};

// ============================================================================
// ModelInstance
// ============================================================================

/// A loaded model instance in the pool with lifecycle metadata.
///
/// Each instance tracks its last access time (for LRU eviction) and
/// estimated memory footprint (for admission control decisions).
pub struct ModelInstance {
    /// The backend implementation (e.g., `CloudOpenAIProvider`, future local backends).
    pub backend: Box<dyn InferenceBackend>,

    /// When this model was last used for inference.
    /// Updated on every `generate()` or `stream()` call.
    pub last_used: Instant,

    /// Estimated memory footprint of this model in bytes.
    ///
    /// For local models, this is typically calculated as:
    /// `parameters × bytes_per_param` (e.g., 7B × 2 bytes for FP16 = 14GB).
    ///
    /// This value is provided at registration time and used by the eviction
    /// logic to estimate how much RAM will be freed if this model is evicted.
    pub estimated_footprint_bytes: u64,
}

impl std::fmt::Debug for ModelInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelInstance")
            .field("backend", &self.backend.name())
            .field("last_used", &self.last_used)
            .field(
                "estimated_footprint_mb",
                &(self.estimated_footprint_bytes / (1024 * 1024)),
            )
            .finish()
    }
}

// ============================================================================
// EvictionResult
// ============================================================================

/// Result of an eviction attempt.
#[derive(Debug)]
pub struct EvictionResult {
    /// The model ID that was evicted.
    pub model_id: String,
    /// The estimated memory freed in bytes.
    pub freed_bytes: u64,
    /// The name of the backend that was evicted.
    pub backend_name: String,
}

// ============================================================================
// ModelPool
// ============================================================================

/// Thread-safe model pool with LRU eviction for OOM prevention.
///
/// The pool manages locally-loaded [`InferenceBackend`] instances and
/// coordinates with the [`TelemetryMonitor`] to ensure that model loading
/// never exceeds the device's physical memory capacity.
///
/// ## Usage
///
/// ```rust,ignore
/// use mofa_foundation::orchestrator::pool::ModelPool;
/// use mofa_foundation::orchestrator::telemetry::TelemetryMonitor;
/// use std::sync::Arc;
///
/// let telemetry = Arc::new(TelemetryMonitor::new());
/// let pool = ModelPool::new(telemetry);
///
/// // Register a model
/// pool.register("llama-8b", backend, 14 * 1024 * 1024 * 1024).await?;
///
/// // Generate inference
/// let response = pool.generate("llama-8b", &request).await?;
///
/// // Evict LRU model if memory is tight
/// if let Some(evicted) = pool.evict_lru().await? {
///     println!("Evicted {} to free {}MB", evicted.model_id, evicted.freed_bytes / 1_048_576);
/// }
/// ```
pub struct ModelPool {
    /// The model registry — maps model IDs to their loaded instances.
    ///
    /// Uses `RwLock` to allow concurrent reads (multiple inference requests)
    /// with exclusive writes (model loading/eviction).
    models: Arc<RwLock<HashMap<String, ModelInstance>>>,

    /// The telemetry monitor for memory admission control.
    telemetry: Arc<TelemetryMonitor>,
}

impl std::fmt::Debug for ModelPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelPool")
            .field("telemetry", &self.telemetry)
            .finish()
    }
}

impl ModelPool {
    /// Create a new empty model pool with the given telemetry monitor.
    pub fn new(telemetry: Arc<TelemetryMonitor>) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            telemetry: telemetry.clone(),
        }
    }

    /// Get a reference to the telemetry monitor.
    pub fn telemetry(&self) -> &TelemetryMonitor {
        &self.telemetry
    }

    /// Get the number of currently loaded models.
    pub async fn model_count(&self) -> usize {
        self.models.read().await.len()
    }

    /// Check if a model with the given ID is loaded in the pool.
    pub async fn has_model(&self, model_id: &str) -> bool {
        self.models.read().await.contains_key(model_id)
    }

    /// List all currently loaded model IDs.
    pub async fn list_models(&self) -> Vec<String> {
        self.models.read().await.keys().cloned().collect()
    }

    /// Get the total estimated memory footprint of all loaded models.
    pub async fn total_footprint_bytes(&self) -> u64 {
        self.models
            .read()
            .await
            .values()
            .map(|i| i.estimated_footprint_bytes)
            .sum()
    }

    // ========================================================================
    // Model Registration
    // ========================================================================

    /// Register (load) a new model into the pool.
    ///
    /// This method:
    /// 1. Checks if the model is already loaded (returns error if so)
    /// 2. Calls `backend.initialize()` to set up the backend
    /// 3. Inserts the initialized backend into the registry
    ///
    /// **Memory admission is NOT checked here** — the caller (typically the
    /// `RequestRouter`) should check `TelemetryMonitor::can_admit_model()`
    /// before calling this method. This separation of concerns allows the
    /// router to implement different policies (e.g., evict-then-retry).
    ///
    /// # Arguments
    /// * `model_id` — Unique identifier for this model instance
    /// * `backend` — The backend implementing [`InferenceBackend`]
    /// * `estimated_footprint_bytes` — Estimated memory consumption
    pub async fn register(
        &self,
        model_id: &str,
        backend: Box<dyn InferenceBackend>,
        estimated_footprint_bytes: u64,
    ) -> Result<(), InferenceError> {
        // Check for duplicate registration
        if self.has_model(model_id).await {
            return Err(InferenceError::ProviderError(format!(
                "Model '{}' is already loaded in the pool",
                model_id
            )));
        }

        // Initialize the backend
        backend.initialize().await.map_err(|e| {
            InferenceError::BackendUnavailable(format!(
                "Failed to initialize backend for model '{}': {}",
                model_id, e
            ))
        })?;

        tracing::info!(
            "ModelPool: registered model '{}' (backend='{}', footprint={:.1}GB)",
            model_id,
            backend.name(),
            estimated_footprint_bytes as f64 / 1_073_741_824.0,
        );

        // Insert into the registry
        let instance = ModelInstance {
            backend,
            last_used: Instant::now(),
            estimated_footprint_bytes,
        };

        self.models
            .write()
            .await
            .insert(model_id.to_string(), instance);

        Ok(())
    }

    // ========================================================================
    // Model Removal
    // ========================================================================

    /// Remove a specific model from the pool, shutting it down gracefully.
    ///
    /// Returns the estimated memory that was freed.
    pub async fn remove(&self, model_id: &str) -> Result<u64, InferenceError> {
        let mut models = self.models.write().await;

        let instance = models.remove(model_id).ok_or_else(|| {
            InferenceError::BackendUnavailable(format!("Model '{}' not found in pool", model_id))
        })?;

        let freed = instance.estimated_footprint_bytes;

        // Graceful shutdown — allows the backend to release resources (GPU mem, file handles)
        if let Err(e) = instance.backend.shutdown().await {
            tracing::warn!(
                "ModelPool: shutdown warning for '{}': {} (model was still removed)",
                model_id,
                e
            );
        }

        tracing::info!(
            "ModelPool: removed model '{}' (freed ~{:.1}GB)",
            model_id,
            freed as f64 / 1_073_741_824.0,
        );

        Ok(freed)
    }

    // ========================================================================
    // LRU Eviction
    // ========================================================================

    /// Evict the **Least Recently Used** model from the pool.
    ///
    /// ## LRU Algorithm
    ///
    /// 1. Scan all loaded models for the one with the **oldest `last_used` timestamp**
    /// 2. Gracefully shut down the model's backend (release GPU memory, file handles)
    /// 3. Remove it from the registry
    /// 4. Return the freed memory estimate
    ///
    /// ## Why LRU?
    ///
    /// LRU is the optimal eviction strategy for inference workloads because:
    /// - **Temporal locality**: Models used recently are likely to be used again soon
    /// - **Simplicity**: O(n) scan is acceptable for small model pools (typically 2-5 models)
    /// - **Predictability**: Users can reason about which model will be evicted
    ///
    /// ## OOM Prevention
    ///
    /// The eviction loop in the `RequestRouter` calls this method repeatedly
    /// until `TelemetryMonitor::can_admit_model()` returns `true` or the pool
    /// is empty. This guarantees that memory is freed before attempting to load
    /// a new model.
    ///
    /// # Returns
    /// - `Ok(Some(EvictionResult))` — a model was evicted
    /// - `Ok(None)` — the pool is empty, nothing to evict
    /// - `Err(InferenceError)` — shutdown failed (model was still removed)
    pub async fn evict_lru(&self) -> Result<Option<EvictionResult>, InferenceError> {
        // Find the LRU model under a read lock first
        let lru_id = {
            let models = self.models.read().await;

            if models.is_empty() {
                return Ok(None);
            }

            models
                .iter()
                .min_by_key(|(_, instance)| instance.last_used)
                .map(|(id, _)| id.clone())
        };

        let lru_id = match lru_id {
            Some(id) => id,
            None => return Ok(None),
        };

        // Remove and shut down
        let mut models = self.models.write().await;
        let instance = match models.remove(&lru_id) {
            Some(inst) => inst,
            None => return Ok(None), // Race condition: another task evicted it
        };

        let freed = instance.estimated_footprint_bytes;
        let backend_name = instance.backend.name().to_string();

        tracing::info!(
            "ModelPool: LRU evicting '{}' (backend='{}', idle for {:?}, freeing ~{:.1}GB)",
            lru_id,
            backend_name,
            instance.last_used.elapsed(),
            freed as f64 / 1_073_741_824.0,
        );

        // Graceful shutdown
        if let Err(e) = instance.backend.shutdown().await {
            tracing::warn!(
                "ModelPool: shutdown warning during LRU eviction of '{}': {}",
                lru_id,
                e
            );
        }

        Ok(Some(EvictionResult {
            model_id: lru_id,
            freed_bytes: freed,
            backend_name,
        }))
    }

    /// Evict models until at least `required_bytes` of headroom is available,
    /// or the pool is empty.
    ///
    /// Returns the total bytes freed across all evictions.
    pub async fn evict_until_headroom(&self, required_bytes: u64) -> Result<u64, InferenceError> {
        let mut total_freed: u64 = 0;

        loop {
            // Check if we have enough headroom now
            if self.telemetry.can_admit_model(required_bytes).await {
                tracing::info!(
                    "ModelPool: sufficient headroom after freeing ~{:.1}GB",
                    total_freed as f64 / 1_073_741_824.0,
                );
                return Ok(total_freed);
            }

            // Try to evict the LRU model
            match self.evict_lru().await? {
                Some(result) => {
                    total_freed += result.freed_bytes;
                }
                None => {
                    // Pool is empty, can't free any more
                    tracing::warn!(
                        "ModelPool: exhausted all models but still insufficient headroom \
                         (freed ~{:.1}GB total, need {:.1}GB)",
                        total_freed as f64 / 1_073_741_824.0,
                        required_bytes as f64 / 1_073_741_824.0,
                    );
                    return Ok(total_freed);
                }
            }
        }
    }

    // ========================================================================
    // Inference Operations
    // ========================================================================

    /// Generate inference using a specific model in the pool.
    ///
    /// Updates the model's `last_used` timestamp on success.
    pub async fn generate(
        &self,
        model_id: &str,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        // We need a write lock to update last_used
        let mut models = self.models.write().await;

        let instance = models.get_mut(model_id).ok_or_else(|| {
            InferenceError::BackendUnavailable(format!("Model '{}' not found in pool", model_id))
        })?;

        let response = instance.backend.generate(request).await?;

        // Touch the LRU timestamp
        instance.last_used = Instant::now();

        Ok(response)
    }

    /// Shut down all models and clear the pool.
    ///
    /// This is the graceful teardown path — called during application shutdown
    /// to ensure all backends release their resources cleanly.
    pub async fn shutdown_all(&self) -> Vec<(String, Result<(), InferenceError>)> {
        let mut models = self.models.write().await;
        let mut results = Vec::new();

        let model_ids: Vec<String> = models.keys().cloned().collect();

        for model_id in model_ids {
            if let Some(instance) = models.remove(&model_id) {
                let result = instance.backend.shutdown().await;
                results.push((model_id, result));
            }
        }

        tracing::info!("ModelPool: shut down {} models", results.len());

        results
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::backend::InferenceBackend;
    use crate::orchestrator::types::{
        InferenceError, InferenceRequest, InferenceResponse, Token, TokenUsage,
    };
    use async_trait::async_trait;
    use futures::Stream;
    use std::pin::Pin;

    /// A mock backend for testing the pool without real inference.
    struct MockBackend {
        name: String,
    }

    impl MockBackend {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl InferenceBackend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }

        async fn initialize(&self) -> Result<(), InferenceError> {
            Ok(())
        }

        async fn is_available(&self) -> bool {
            true
        }

        async fn generate(
            &self,
            _request: &InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
            Ok(InferenceResponse {
                content: format!("Response from {}", self.name),
                model: self.name.clone(),
                usage: TokenUsage::default(),
                finish_reason: Some("stop".to_string()),
            })
        }

        async fn stream(
            &self,
            _request: &InferenceRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>, InferenceError>
        {
            Err(InferenceError::ProviderError(
                "Streaming not implemented in mock".to_string(),
            ))
        }

        async fn shutdown(&self) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    fn create_pool() -> ModelPool {
        let telemetry = Arc::new(TelemetryMonitor::with_safety_margin(0));
        ModelPool::new(telemetry)
    }

    #[tokio::test]
    async fn test_empty_pool() {
        let pool = create_pool();
        assert_eq!(pool.model_count().await, 0);
        assert!(pool.list_models().await.is_empty());
        assert_eq!(pool.total_footprint_bytes().await, 0);
    }

    #[tokio::test]
    async fn test_register_model() {
        let pool = create_pool();
        let backend = Box::new(MockBackend::new("test-model"));

        pool.register("test-model", backend, 1024).await.unwrap();

        assert_eq!(pool.model_count().await, 1);
        assert!(pool.has_model("test-model").await);
        assert_eq!(pool.total_footprint_bytes().await, 1024);
    }

    #[tokio::test]
    async fn test_duplicate_registration_fails() {
        let pool = create_pool();
        pool.register("m1", Box::new(MockBackend::new("m1")), 1024)
            .await
            .unwrap();

        let result = pool
            .register("m1", Box::new(MockBackend::new("m1-dupe")), 2048)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_remove_model() {
        let pool = create_pool();
        pool.register("m1", Box::new(MockBackend::new("m1")), 1024)
            .await
            .unwrap();

        let freed = pool.remove("m1").await.unwrap();
        assert_eq!(freed, 1024);
        assert_eq!(pool.model_count().await, 0);
    }

    #[tokio::test]
    async fn test_remove_nonexistent_fails() {
        let pool = create_pool();
        let result = pool.remove("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_lru_eviction_order() {
        let pool = create_pool();

        // Register two models — "old" is registered first (older last_used)
        pool.register("old", Box::new(MockBackend::new("old")), 1024)
            .await
            .unwrap();

        // Small delay to ensure different timestamps
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        pool.register("new", Box::new(MockBackend::new("new")), 2048)
            .await
            .unwrap();

        // LRU eviction should evict "old" (earlier timestamp)
        let evicted = pool.evict_lru().await.unwrap().unwrap();
        assert_eq!(evicted.model_id, "old");
        assert_eq!(evicted.freed_bytes, 1024);

        // Only "new" should remain
        assert_eq!(pool.model_count().await, 1);
        assert!(pool.has_model("new").await);
    }

    #[tokio::test]
    async fn test_lru_eviction_empty_pool() {
        let pool = create_pool();
        let result = pool.evict_lru().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_generate_through_pool() {
        let pool = create_pool();
        pool.register("m1", Box::new(MockBackend::new("m1")), 1024)
            .await
            .unwrap();

        let request =
            InferenceRequest::new(vec![crate::orchestrator::types::ChatMessage::user("hello")]);
        let response = pool.generate("m1", &request).await.unwrap();

        assert_eq!(response.content, "Response from m1");
    }

    #[tokio::test]
    async fn test_generate_nonexistent_model() {
        let pool = create_pool();
        let request =
            InferenceRequest::new(vec![crate::orchestrator::types::ChatMessage::user("hello")]);

        let result = pool.generate("missing", &request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_shutdown_all() {
        let pool = create_pool();
        pool.register("m1", Box::new(MockBackend::new("m1")), 1024)
            .await
            .unwrap();
        pool.register("m2", Box::new(MockBackend::new("m2")), 2048)
            .await
            .unwrap();

        let results = pool.shutdown_all().await;
        assert_eq!(results.len(), 2);
        assert_eq!(pool.model_count().await, 0);
    }

    #[test]
    fn test_pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ModelPool>();
    }
}
