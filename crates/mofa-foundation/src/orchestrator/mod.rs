//! Model Orchestrator Module
//!
//! This module provides two complementary orchestration subsystems:
//!
//! ## 1. Unified Inference Orchestrator (Phase 1)
//!
//! A provider-agnostic inference abstraction that bridges local runtimes
//! (macOS Apple Silicon, Linux CUDA) with cloud APIs via a single trait:
//!
//! - **[`InferenceBackend`]** — Object-safe trait with streaming, init, and teardown
//! - **[`CloudOpenAIProvider`]** — Cloud fallback using `async-openai` with HTTP/2 pooling
//! - Core types: [`InferenceRequest`], [`InferenceResponse`], [`Token`]
//!
//! ## 2. Edge Model Orchestrator (GSoC 2026 — Idea 3)
//!
//! On-device model lifecycle management for edge inference:
//!
//! - Model type routing (ASR / LLM / TTS / Embedding)
//! - Model lifecycle management (load/unload with idle timeout)
//! - Smart scheduling with LRU eviction
//! - Memory pressure awareness and dynamic precision degradation
//! - Multi-model pipeline chaining (ASR → LLM → TTS)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              MoFA Agent / Workflow               │
//! └────────────────────┬────────────────────────────┘
//!                      │ InferenceRequest
//!                      ▼
//! ┌─────────────────────────────────────────────────┐
//! │          Box<dyn InferenceBackend>               │
//! │  ┌──────────────┐  ┌────────────┐  ┌─────────┐ │
//! │  │CloudOpenAI   │  │ LocalMLX   │  │ Candle  │ │
//! │  │ (fallback)   │  │ (Phase 2)  │  │(Linux)  │ │
//! │  └──────────────┘  └────────────┘  └─────────┘ │
//! └─────────────────────────────────────────────────┘
//! ```

// ---------------------------------------------------------------------------
// Unified Inference Orchestrator (Phase 1)
// ---------------------------------------------------------------------------

/// Core types: InferenceRequest, InferenceResponse, ChatMessage, Token, etc.
pub mod types;

/// The `InferenceBackend` trait — object-safe async interface for all backends.
pub mod backend;

/// Cloud OpenAI provider — the primary cloud fallback backend.
pub mod cloud_openai;

// ---------------------------------------------------------------------------
// Edge Model Orchestrator (existing)
// ---------------------------------------------------------------------------

pub mod traits;

#[cfg(all(target_os = "linux", feature = "linux-candle"))]
pub mod linux_candle;

#[cfg(all(target_os = "linux", feature = "linux-candle"))]
pub mod pipeline;

// Re-export core traits and types (Edge Orchestrator)
pub use traits::{
    DegradationLevel, ModelOrchestrator, ModelProvider, ModelProviderConfig, ModelType,
    OrchestratorError, OrchestratorResult, PoolStatistics,
};

// Re-export Unified Inference types
pub use types::{
    ChatMessage as InferenceChatMessage, ChatRole, InferenceError, InferenceRequest,
    InferenceResponse, InferenceResult, Token, TokenUsage,
};

// Re-export the InferenceBackend trait
pub use backend::InferenceBackend;

// Re-export the Cloud OpenAI provider
pub use cloud_openai::{CloudOpenAIConfig, CloudOpenAIProvider};

// Re-export Linux implementation when available
#[cfg(all(target_os = "linux", feature = "linux-candle"))]
pub use linux_candle::{LinuxCandleProvider, ModelPool};

// Re-export pipeline types when available
#[cfg(all(target_os = "linux", feature = "linux-candle"))]
pub use pipeline::{InferencePipeline, PipelineBuilder, PipelineOutput, PipelineStage};
