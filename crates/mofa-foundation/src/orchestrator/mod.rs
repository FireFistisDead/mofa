//! Model Orchestrator Module
//!
//! This module provides the complete inference orchestration stack for MoFA:
//!
//! ## Phase 1: Unified Inference Abstraction
//!
//! Provider-agnostic inference layer bridging local and cloud backends:
//!
//! - **[`InferenceBackend`]** — Object-safe async trait (streaming, init, teardown)
//! - **[`CloudOpenAIProvider`]** — Cloud fallback with HTTP/2 pooling + retry
//! - Core types: [`InferenceRequest`], [`InferenceResponse`], [`Token`]
//!
//! ## Phase 2: Telemetry, ModelPool & Policy Router
//!
//! Smart orchestration brain that prevents OOM crashes on edge devices:
//!
//! - **[`TelemetryMonitor`]** — Cross-platform RAM monitoring with safety margins
//! - **[`ModelPool`]** — Thread-safe model registry with LRU eviction
//! - **[`RequestRouter`]** — Policy-driven routing (local-first, cloud-only, etc.)
//!
//! ## Edge Model Orchestrator (Legacy GSoC 2026)
//!
//! On-device model lifecycle management:
//!
//! - Model type routing, LRU scheduling, memory pressure awareness
//! - Dynamic precision degradation, multi-model pipeline chaining
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │                   MoFA Agent / Workflow                       │
//! └────────────────────────┬──────────────────────────────────────┘
//!                          │ InferenceRequest
//!                          ▼
//! ┌───────────────────────────────────────────────────────────────┐
//! │                    RequestRouter                              │
//! │  policy: LocalFirstWithCloudFallback                         │
//! │  ┌─────────────────┐      ┌─────────────────────────────┐   │
//! │  │ TelemetryMonitor│─────▶│ can_admit_model(footprint)? │   │
//! │  └─────────────────┘      └──────────┬──────────────────┘   │
//! │                              YES ┌───┴───┐ NO               │
//! │                                  ▼       ▼                  │
//! │                          ┌──────────┐ ┌──────────────┐      │
//! │                          │ModelPool │ │ Evict LRU    │      │
//! │                          │(local)   │ │ then retry   │      │
//! │                          └──────────┘ └──────┬───────┘      │
//! │                                         still NO?           │
//! │                                              ▼              │
//! │                                    ┌─────────────────┐      │
//! │                                    │ CloudOpenAI     │      │
//! │                                    │ (fallback)      │      │
//! │                                    └─────────────────┘      │
//! └───────────────────────────────────────────────────────────────┘
//! ```

// ---------------------------------------------------------------------------
// Phase 1: Unified Inference Abstraction
// ---------------------------------------------------------------------------

/// Core types: InferenceRequest, InferenceResponse, ChatMessage, Token, etc.
pub mod types;

/// The `InferenceBackend` trait — object-safe async interface for all backends.
pub mod backend;

/// Cloud OpenAI provider — the primary cloud fallback backend.
pub mod cloud_openai;

// ---------------------------------------------------------------------------
// Phase 2: Telemetry, ModelPool & Policy Router
// ---------------------------------------------------------------------------

/// Cross-platform hardware telemetry monitor for memory admission control.
pub mod telemetry;

/// Thread-safe model registry with LRU eviction for OOM prevention.
pub mod pool;

/// Policy-driven request router with admission control and cloud fallback.
pub mod router;

// ---------------------------------------------------------------------------
// Phase 3: Dual Local Backends (Trade Study)
// ---------------------------------------------------------------------------

/// Local Candle backend — pure-Rust inference via Hugging Face Candle.
/// Supports CUDA, Metal, and CPU with automatic fallback.
/// Requires the `candle` feature flag.
#[cfg(feature = "candle")]
pub mod local_candle;

/// Local llama.cpp backend — optimized C++ inference via `llama_cpp_2` FFI.
/// Requires the `llama-cpp` feature flag.
#[cfg(feature = "llama-cpp")]
pub mod local_llama;

// ---------------------------------------------------------------------------
// Edge Model Orchestrator (legacy)
// ---------------------------------------------------------------------------

pub mod traits;

#[cfg(all(target_os = "linux", feature = "linux-candle"))]
pub mod linux_candle;

#[cfg(all(target_os = "linux", feature = "linux-candle"))]
pub mod pipeline;

// ── Re-exports: Edge Orchestrator ──

pub use traits::{
    DegradationLevel, ModelOrchestrator, ModelProvider, ModelProviderConfig, ModelType,
    OrchestratorError, OrchestratorResult, PoolStatistics,
};

// ── Re-exports: Phase 1 (Inference Abstraction) ──

pub use types::{
    ChatMessage as InferenceChatMessage, ChatRole, InferenceError, InferenceRequest,
    InferenceResponse, InferenceResult, Token, TokenUsage,
};

pub use backend::InferenceBackend;

pub use cloud_openai::{CloudOpenAIConfig, CloudOpenAIProvider};

// ── Re-exports: Phase 2 (Telemetry, Pool, Router) ──

pub use pool::{EvictionResult, ModelInstance, ModelPool};
pub use router::{RequestRouter, RoutingDecision, RoutingPolicy};
pub use telemetry::{TelemetryMonitor, TelemetrySnapshot};

// ── Re-exports: Phase 3 (Local Backends) ──

#[cfg(feature = "candle")]
pub use local_candle::{CandleConfig, CandleDeviceInfo, CandleProvider};

#[cfg(feature = "llama-cpp")]
pub use local_llama::{LlamaCppConfig, LlamaCppProvider};

// ── Re-exports: Linux Candle (feature-gated) ──

#[cfg(all(target_os = "linux", feature = "linux-candle"))]
pub use linux_candle::{LinuxCandleProvider, ModelPool as CandleModelPool};

#[cfg(all(target_os = "linux", feature = "linux-candle"))]
pub use pipeline::{InferencePipeline, PipelineBuilder, PipelineOutput, PipelineStage};
