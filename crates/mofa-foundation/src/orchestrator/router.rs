//! RequestRouter — Smart Inference Routing with Admission Control
//!
//! This module implements the **policy-driven request router** for the MoFA
//! inference orchestrator. It is the "brain" that decides whether an inference
//! request should be executed locally (via the [`ModelPool`]) or remotely
//! (via the [`CloudOpenAIProvider`]).
//!
//! ## Routing Decision Matrix
//!
//! The router uses the following decision tree to prevent OOM crashes while
//! maximizing local execution (lower latency, no API costs):
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                     InferenceRequest arrives                     │
//! └────────────────────────┬─────────────────────────────────────────┘
//!                          │
//!                          ▼
//!                ┌───────────────────┐
//!                │ Is model loaded   │
//!                │ in ModelPool?     │
//!                └──────┬────────────┘
//!                  YES  │         NO
//!          ┌────────────┘         │
//!          ▼                      ▼
//!   ┌──────────────┐    ┌────────────────────┐
//!   │ Route to     │    │ Can TelemetryMonitor│
//!   │ local pool   │    │ admit model?        │
//!   └──────────────┘    └──────┬──────────────┘
//!                         YES  │         NO
//!                ┌─────────────┘         │
//!                ▼                       ▼
//!     ┌──────────────────┐    ┌────────────────────┐
//!     │ Load model,      │    │ Evict LRU models   │
//!     │ route locally    │    │ from pool           │
//!     └──────────────────┘    └──────┬──────────────┘
//!                                    │
//!                               ┌────┴────┐
//!                               │ Enough  │
//!                               │ RAM now?│
//!                               └──┬──────┘
//!                             YES  │    NO
//!                    ┌─────────────┘    │
//!                    ▼                  ▼
//!         ┌──────────────────┐  ┌────────────────┐
//!         │ Load model,      │  │ FALLBACK to    │
//!         │ route locally    │  │ CloudOpenAI    │
//!         └──────────────────┘  └────────────────┘
//! ```
//!
//! ## Why This Prevents OOM Crashes
//!
//! The key insight is that the router **never loads a model without first
//! confirming that sufficient RAM is available**. The sequence is:
//!
//! 1. **Check RAM** → `TelemetryMonitor::can_admit_model(footprint)`
//! 2. **Evict if needed** → `ModelPool::evict_until_headroom(footprint)`
//! 3. **Check RAM again** → If still insufficient, fall back to cloud
//! 4. **Only then load** → The model is loaded into the pool
//!
//! This guarantees that loading a model will never push the system below
//! the safety margin, preventing the Linux OOM killer, macOS memory pressure
//! termination, or Windows disk thrashing.
//!
//! ## Thread Safety
//!
//! The `RequestRouter` is `Send + Sync` and can be shared across tokio tasks
//! via `Arc`. All routing operations are fully `async`.

use std::sync::Arc;

use super::backend::InferenceBackend;
use super::cloud_openai::CloudOpenAIProvider;
use super::pool::ModelPool;
use super::telemetry::TelemetryMonitor;
use super::types::{InferenceError, InferenceRequest, InferenceResponse};

// ============================================================================
// RoutingPolicy
// ============================================================================

/// The strategy the router uses to choose between local and cloud backends.
///
/// Different policies optimize for different objectives:
///
/// - **`LocalFirst`**: Minimize API costs, maximize privacy (data stays on device)
/// - **`CloudFirst`**: Minimize latency for models not loaded locally
/// - **`LocalOnly`**: Never use cloud — fail if local execution is impossible
/// - **`CloudOnly`**: Never load local models — always use cloud API
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingPolicy {
    /// Try local execution first; if insufficient RAM after eviction, fall back to cloud.
    /// This is the recommended default for edge devices.
    LocalFirstWithCloudFallback,

    /// Always use the cloud provider. Useful for testing or when local hardware
    /// is too constrained for any models.
    CloudOnly,

    /// Always use local models. Fails with `BackendUnavailable` if the model
    /// cannot be loaded (e.g., insufficient RAM even after eviction).
    LocalOnly,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        Self::LocalFirstWithCloudFallback
    }
}

// ============================================================================
// RoutingDecision
// ============================================================================

/// The outcome of a routing decision, for logging and telemetry.
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Request was routed to a local model in the pool.
    Local {
        model_id: String,
        /// Whether the model was already loaded or had to be freshly loaded.
        was_cached: bool,
    },
    /// Request was routed to the cloud fallback.
    CloudFallback {
        /// Why the request fell back to cloud.
        reason: String,
    },
}

// ============================================================================
// RequestRouter
// ============================================================================

/// Smart inference router with admission control and cloud fallback.
///
/// The router is the primary entry point for inference requests in the MoFA
/// orchestrator. It consults hardware telemetry, manages model loading/eviction,
/// and ensures requests are served without risking OOM crashes.
///
/// ## Usage
///
/// ```rust,ignore
/// use mofa_foundation::orchestrator::router::{RequestRouter, RoutingPolicy};
/// use mofa_foundation::orchestrator::pool::ModelPool;
/// use mofa_foundation::orchestrator::telemetry::TelemetryMonitor;
/// use mofa_foundation::orchestrator::cloud_openai::{CloudOpenAIConfig, CloudOpenAIProvider};
/// use std::sync::Arc;
///
/// let telemetry = Arc::new(TelemetryMonitor::new());
/// let pool = Arc::new(ModelPool::new(telemetry.clone()));
/// let cloud = Arc::new(CloudOpenAIProvider::new(CloudOpenAIConfig::new("sk-...")));
///
/// let router = RequestRouter::new(pool, cloud, telemetry, RoutingPolicy::LocalFirstWithCloudFallback);
///
/// // Route a request — the router decides local vs cloud automatically
/// let response = router.route_request(&request).await?;
/// ```
pub struct RequestRouter {
    /// The local model pool.
    pool: Arc<ModelPool>,

    /// The cloud fallback provider.
    cloud: Arc<CloudOpenAIProvider>,

    /// The telemetry monitor for admission control.
    telemetry: Arc<TelemetryMonitor>,

    /// The active routing policy.
    policy: RoutingPolicy,
}

impl std::fmt::Debug for RequestRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestRouter")
            .field("policy", &self.policy)
            .field("pool", &self.pool)
            .field("telemetry", &self.telemetry)
            .finish()
    }
}

impl RequestRouter {
    /// Create a new request router.
    ///
    /// # Arguments
    /// * `pool` — The local model pool for on-device inference
    /// * `cloud` — The cloud fallback provider
    /// * `telemetry` — The telemetry monitor for RAM admission control
    /// * `policy` — The routing policy to apply
    pub fn new(
        pool: Arc<ModelPool>,
        cloud: Arc<CloudOpenAIProvider>,
        telemetry: Arc<TelemetryMonitor>,
        policy: RoutingPolicy,
    ) -> Self {
        tracing::info!("RequestRouter: initialized with policy={:?}", policy);

        Self {
            pool,
            cloud,
            telemetry,
            policy,
        }
    }

    /// Get the current routing policy.
    pub fn policy(&self) -> &RoutingPolicy {
        &self.policy
    }

    /// Change the routing policy at runtime.
    pub fn set_policy(&mut self, policy: RoutingPolicy) {
        tracing::info!("RequestRouter: policy changed to {:?}", policy);
        self.policy = policy;
    }

    /// Get a reference to the model pool.
    pub fn pool(&self) -> &ModelPool {
        &self.pool
    }

    /// Get a reference to the cloud provider.
    pub fn cloud(&self) -> &CloudOpenAIProvider {
        &self.cloud
    }

    // ========================================================================
    // Core Routing Logic
    // ========================================================================

    /// Route an inference request according to the active policy.
    ///
    /// This is the main entry point for inference. The router:
    ///
    /// 1. Checks the routing policy
    /// 2. For `LocalFirstWithCloudFallback`:
    ///    - Checks if the model is already loaded → route locally
    ///    - If not loaded, checks RAM → tries to load locally
    ///    - If RAM insufficient, evicts LRU models
    ///    - If still insufficient, falls back to cloud
    /// 3. For `CloudOnly`: Always routes to cloud
    /// 4. For `LocalOnly`: Only routes locally, fails if impossible
    ///
    /// # Arguments
    /// * `request` — The inference request to route
    ///
    /// # Returns
    /// The inference response and the routing decision for telemetry
    pub async fn route_request(
        &self,
        request: &InferenceRequest,
    ) -> Result<(InferenceResponse, RoutingDecision), InferenceError> {
        match self.policy {
            RoutingPolicy::CloudOnly => self.route_cloud(request).await,
            RoutingPolicy::LocalOnly => self.route_local_only(request).await,
            RoutingPolicy::LocalFirstWithCloudFallback => self.route_local_first(request).await,
        }
    }

    /// Route directly to the cloud provider.
    async fn route_cloud(
        &self,
        request: &InferenceRequest,
    ) -> Result<(InferenceResponse, RoutingDecision), InferenceError> {
        tracing::debug!("RequestRouter: routing to cloud (policy=CloudOnly)");

        let response = self.cloud.generate(request).await?;

        Ok((
            response,
            RoutingDecision::CloudFallback {
                reason: "CloudOnly policy".to_string(),
            },
        ))
    }

    /// Route only to local backends; fail if no local model is available.
    async fn route_local_only(
        &self,
        request: &InferenceRequest,
    ) -> Result<(InferenceResponse, RoutingDecision), InferenceError> {
        let model_id = request.model.as_deref().unwrap_or("default");

        if self.pool.has_model(model_id).await {
            let response = self.pool.generate(model_id, request).await?;
            return Ok((
                response,
                RoutingDecision::Local {
                    model_id: model_id.to_string(),
                    was_cached: true,
                },
            ));
        }

        Err(InferenceError::BackendUnavailable(format!(
            "Model '{}' not loaded and policy is LocalOnly (no cloud fallback)",
            model_id
        )))
    }

    /// The primary routing path: try local first, fall back to cloud.
    ///
    /// ## OOM Prevention Sequence
    ///
    /// 1. **Model already loaded?** → Route to it directly (no RAM pressure)
    /// 2. **Not loaded → Check RAM** → `TelemetryMonitor::can_admit_model()`
    /// 3. **Insufficient RAM → Evict LRU** → `ModelPool::evict_until_headroom()`
    /// 4. **Still insufficient → Cloud fallback** → `CloudOpenAIProvider::generate()`
    ///
    /// At no point is a model loaded without confirmed RAM headroom,
    /// which is why this sequence is OOM-safe.
    async fn route_local_first(
        &self,
        request: &InferenceRequest,
    ) -> Result<(InferenceResponse, RoutingDecision), InferenceError> {
        let model_id = request.model.as_deref().unwrap_or("default");

        // ── Step 1: Model already loaded? Route directly ──
        if self.pool.has_model(model_id).await {
            tracing::debug!(
                "RequestRouter: model '{}' cached in pool, routing locally",
                model_id
            );

            match self.pool.generate(model_id, request).await {
                Ok(response) => {
                    return Ok((
                        response,
                        RoutingDecision::Local {
                            model_id: model_id.to_string(),
                            was_cached: true,
                        },
                    ));
                }
                Err(e) => {
                    tracing::warn!(
                        "RequestRouter: local generation failed for '{}': {}, falling back to cloud",
                        model_id,
                        e
                    );
                    // Fall through to cloud fallback
                }
            }
        }

        // ── Step 2–4: Model not loaded or failed → Cloud fallback ──
        //
        // In Phase 2, we don't yet have concrete local backends (that's Phase 3).
        // The pool currently only holds manually-registered backends.
        // When a model isn't in the pool, we fall back to the cloud provider.
        //
        // In Phase 3 (LocalMLXProvider, llama.cpp), this section will:
        //   1. Estimate model footprint from model metadata
        //   2. Call `telemetry.can_admit_model(footprint)`
        //   3. If NO: call `pool.evict_until_headroom(footprint)`
        //   4. If STILL NO: fall back to cloud (below)
        //   5. If YES: instantiate the local backend and register it

        tracing::info!(
            "RequestRouter: model '{}' not in local pool, falling back to cloud",
            model_id
        );

        let response = self.cloud.generate(request).await.map_err(|e| {
            tracing::error!(
                "RequestRouter: cloud fallback also failed for '{}': {}",
                model_id,
                e
            );
            e
        })?;

        Ok((
            response,
            RoutingDecision::CloudFallback {
                reason: format!("Model '{}' not loaded locally, routed to cloud", model_id),
            },
        ))
    }

    // ========================================================================
    // Admission Control (for Phase 3 local model loading)
    // ========================================================================

    /// Attempt to make room for a model with the given memory footprint.
    ///
    /// This method implements the full admission control sequence:
    ///
    /// 1. Check if the model can be admitted with current memory
    /// 2. If not, evict LRU models until sufficient headroom exists
    /// 3. Return whether admission is now possible
    ///
    /// ## OOM Prevention
    ///
    /// This method is the **critical safety gate** that prevents OOM crashes.
    /// By requiring successful admission before any model loading occurs,
    /// we guarantee that the system will never exceed its physical memory
    /// capacity. If eviction cannot free enough RAM, the caller must either
    /// give up or fall back to cloud inference.
    ///
    /// # Arguments
    /// * `estimated_footprint_bytes` — The estimated memory the model needs
    ///
    /// # Returns
    /// `true` if enough RAM is now available (after potential eviction),
    /// `false` if even complete pool eviction cannot free enough memory.
    pub async fn try_admit(&self, estimated_footprint_bytes: u64) -> Result<bool, InferenceError> {
        // Fast path: enough RAM already
        if self
            .telemetry
            .can_admit_model(estimated_footprint_bytes)
            .await
        {
            tracing::debug!(
                "RequestRouter: admission granted for {:.1}GB model (sufficient headroom)",
                estimated_footprint_bytes as f64 / 1_073_741_824.0,
            );
            return Ok(true);
        }

        // Slow path: evict LRU models to make room
        tracing::info!(
            "RequestRouter: insufficient RAM for {:.1}GB model, triggering LRU eviction",
            estimated_footprint_bytes as f64 / 1_073_741_824.0,
        );

        self.pool
            .evict_until_headroom(estimated_footprint_bytes)
            .await?;

        // Check again after eviction
        let can_admit = self
            .telemetry
            .can_admit_model(estimated_footprint_bytes)
            .await;

        if !can_admit {
            tracing::warn!(
                "RequestRouter: eviction exhausted but still insufficient RAM \
                 for {:.1}GB model — cloud fallback required",
                estimated_footprint_bytes as f64 / 1_073_741_824.0,
            );
        }

        Ok(can_admit)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::cloud_openai::{CloudOpenAIConfig, CloudOpenAIProvider};
    use crate::orchestrator::pool::ModelPool;
    use crate::orchestrator::telemetry::TelemetryMonitor;

    fn create_router(policy: RoutingPolicy) -> RequestRouter {
        let telemetry = Arc::new(TelemetryMonitor::new());
        let pool = Arc::new(ModelPool::new(telemetry.clone()));
        let cloud_config = CloudOpenAIConfig::new("test-key");
        let cloud = Arc::new(CloudOpenAIProvider::new(cloud_config));

        RequestRouter::new(pool, cloud, telemetry, policy)
    }

    #[test]
    fn test_default_policy() {
        let policy = RoutingPolicy::default();
        assert_eq!(policy, RoutingPolicy::LocalFirstWithCloudFallback);
    }

    #[test]
    fn test_router_creation() {
        let router = create_router(RoutingPolicy::LocalFirstWithCloudFallback);
        assert_eq!(*router.policy(), RoutingPolicy::LocalFirstWithCloudFallback);
    }

    #[test]
    fn test_policy_change() {
        let mut router = create_router(RoutingPolicy::CloudOnly);
        assert_eq!(*router.policy(), RoutingPolicy::CloudOnly);

        router.set_policy(RoutingPolicy::LocalOnly);
        assert_eq!(*router.policy(), RoutingPolicy::LocalOnly);
    }

    #[tokio::test]
    async fn test_local_only_no_model_fails() {
        let router = create_router(RoutingPolicy::LocalOnly);
        let request =
            InferenceRequest::new(vec![crate::orchestrator::types::ChatMessage::user("hello")])
                .with_model("nonexistent");

        let result = router.route_request(&request).await;
        assert!(result.is_err());

        if let Err(InferenceError::BackendUnavailable(msg)) = result {
            assert!(msg.contains("not loaded"));
        } else {
            panic!("Expected BackendUnavailable error");
        }
    }

    #[tokio::test]
    async fn test_try_admit_tiny_model() {
        let router = create_router(RoutingPolicy::LocalFirstWithCloudFallback);
        // 1 byte should always be admittable
        let can_admit = router.try_admit(1).await.unwrap();
        assert!(can_admit);
    }

    #[tokio::test]
    async fn test_try_admit_huge_model_denied() {
        let router = create_router(RoutingPolicy::LocalFirstWithCloudFallback);
        // 1 exabyte should never be admittable
        let exabyte = 1024 * 1024 * 1024 * 1024 * 1024;
        let can_admit = router.try_admit(exabyte).await.unwrap();
        assert!(!can_admit);
    }

    #[test]
    fn test_router_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RequestRouter>();
    }

    #[test]
    fn test_routing_decision_debug() {
        let decision = RoutingDecision::Local {
            model_id: "llama-8b".to_string(),
            was_cached: true,
        };
        let debug = format!("{:?}", decision);
        assert!(debug.contains("llama-8b"));

        let fallback = RoutingDecision::CloudFallback {
            reason: "insufficient RAM".to_string(),
        };
        let debug = format!("{:?}", fallback);
        assert!(debug.contains("insufficient RAM"));
    }

    #[test]
    fn test_router_debug() {
        let router = create_router(RoutingPolicy::CloudOnly);
        let debug = format!("{:?}", router);
        assert!(debug.contains("RequestRouter"));
        assert!(debug.contains("CloudOnly"));
    }
}
