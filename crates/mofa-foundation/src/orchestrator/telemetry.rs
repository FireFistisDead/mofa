//! Telemetry Monitor — Cross-Platform Hardware Admission Control
//!
//! This module provides **real-time hardware telemetry** for the MoFA inference
//! orchestrator. Its primary purpose is to prevent **Out-Of-Memory (OOM) crashes**
//! on constrained edge devices (e.g., 16GB laptops) by enforcing memory admission
//! control before loading models.
//!
//! ## OOM Prevention Strategy
//!
//! Loading a 7B parameter model in FP16 consumes ~14GB of RAM. On a 16GB device,
//! loading such a model without checking available memory would leave only ~2GB
//! for the OS, window manager, and other processes — triggering the Linux OOM killer
//! or macOS memory pressure termination.
//!
//! The `TelemetryMonitor` solves this by:
//!
//! 1. **Querying real-time available RAM** via `sysinfo` (cross-platform)
//! 2. **Reserving a safety margin** (default: 2GB) for the OS and system processes
//! 3. **Admission gating** — `can_admit_model()` returns `false` if loading the
//!    model would breach the safety margin, giving the orchestrator a chance to
//!    evict idle models or fall back to cloud inference
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Total System RAM (16GB)                      │
//! ├──────────────────────┬──────────────────────┬──────────────────┤
//! │   OS + System (2GB)  │  Loaded Models (12GB)│   Free (2GB)     │
//! │   ← safety margin →  │                      │ ← must keep ≥0  │
//! └──────────────────────┴──────────────────────┴──────────────────┘
//!
//! can_admit_model(4GB) → false (would leave 0GB free, below safety margin)
//! can_admit_model(1GB) → true  (would leave 1GB free, above 0)
//! ```
//!
//! ## Thread Safety
//!
//! `TelemetryMonitor` is `Send + Sync` — all mutable state is behind interior
//! mutability primitives. The `sysinfo` refresh is run on a blocking thread
//! via `tokio::task::spawn_blocking` to avoid stalling the async runtime.

use std::sync::atomic::{AtomicU64, Ordering};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

// ============================================================================
// Constants
// ============================================================================

/// Default OS safety margin: 2GB
///
/// This is the minimum amount of RAM that must remain free for the OS,
/// window manager, system services, and non-MoFA processes. Loading a
/// model that would push available memory below this threshold is denied.
///
/// ## Rationale
///
/// - **Linux**: The OOM killer activates when memory is critically low.
///   Keeping 2GB free prevents the kernel from killing MoFA or other
///   critical processes.
/// - **macOS**: Memory pressure triggers aggressive page compression
///   and swap, causing severe performance degradation before any process
///   is terminated. 2GB keeps the system responsive.
/// - **Windows**: Low memory triggers aggressive page file usage,
///   causing disk thrashing and application freezes.
const DEFAULT_SAFETY_MARGIN_BYTES: u64 = 2 * 1024 * 1024 * 1024; // 2 GB

// ============================================================================
// TelemetrySnapshot
// ============================================================================

/// A point-in-time snapshot of system hardware telemetry.
///
/// Produced by [`TelemetryMonitor::snapshot()`] for logging, metrics export,
/// and the orchestrator's admission control decisions.
#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    /// Total physical RAM on the system (bytes).
    pub total_memory_bytes: u64,
    /// Currently available (free + reclaimable) RAM (bytes).
    pub available_memory_bytes: u64,
    /// Currently used RAM (bytes).
    pub used_memory_bytes: u64,
    /// The configured safety margin (bytes).
    pub safety_margin_bytes: u64,
    /// Effective headroom = available - safety_margin. May be zero if negative.
    pub headroom_bytes: u64,
}

// ============================================================================
// TelemetryMonitor
// ============================================================================

/// Cross-platform hardware telemetry monitor for memory admission control.
///
/// The monitor queries system memory via `sysinfo` and provides a simple
/// admission gate: "can this model fit in RAM without risking an OOM crash?"
///
/// ## Usage
///
/// ```rust,ignore
/// use mofa_foundation::orchestrator::telemetry::TelemetryMonitor;
///
/// let monitor = TelemetryMonitor::new();
///
/// // Check if a 4GB model can be loaded safely
/// let estimated_model_size = 4 * 1024 * 1024 * 1024; // 4 GB
/// if monitor.can_admit_model(estimated_model_size).await {
///     println!("Safe to load model");
/// } else {
///     println!("Insufficient RAM — trigger eviction or cloud fallback");
/// }
/// ```
pub struct TelemetryMonitor {
    /// The OS safety margin in bytes. RAM below this threshold is reserved
    /// for the operating system and must never be consumed by models.
    safety_margin_bytes: AtomicU64,
}

impl std::fmt::Debug for TelemetryMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryMonitor")
            .field(
                "safety_margin_bytes",
                &self.safety_margin_bytes.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for TelemetryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetryMonitor {
    /// Create a new monitor with the default 2GB safety margin.
    pub fn new() -> Self {
        Self {
            safety_margin_bytes: AtomicU64::new(DEFAULT_SAFETY_MARGIN_BYTES),
        }
    }

    /// Create a new monitor with a custom safety margin.
    ///
    /// # Arguments
    /// * `safety_margin_bytes` — Minimum free RAM to preserve for the OS (bytes)
    pub fn with_safety_margin(safety_margin_bytes: u64) -> Self {
        Self {
            safety_margin_bytes: AtomicU64::new(safety_margin_bytes),
        }
    }

    /// Update the safety margin at runtime.
    ///
    /// This is useful for dynamically adjusting the margin based on workload
    /// (e.g., tighter margin during batch processing, looser during interactive use).
    pub fn set_safety_margin(&self, bytes: u64) {
        self.safety_margin_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Get the currently configured safety margin in bytes.
    pub fn safety_margin(&self) -> u64 {
        self.safety_margin_bytes.load(Ordering::Relaxed)
    }

    /// Query the current available system RAM in bytes.
    ///
    /// This runs `sysinfo` on a **blocking thread** via `tokio::task::spawn_blocking`
    /// to avoid stalling the async runtime — `sysinfo` reads from `/proc/meminfo`
    /// (Linux), `vm_stat` (macOS), or `GlobalMemoryStatusEx` (Windows), which are
    /// fast but technically blocking syscalls.
    ///
    /// Returns `(total_bytes, available_bytes)`.
    pub async fn query_memory(&self) -> (u64, u64) {
        tokio::task::spawn_blocking(|| {
            let mut sys = System::new_with_specifics(
                RefreshKind::new().with_memory(MemoryRefreshKind::everything()),
            );
            sys.refresh_memory();
            (sys.total_memory(), sys.available_memory())
        })
        .await
        .unwrap_or((0, 0))
    }

    /// Get the currently available system RAM in bytes (convenience method).
    pub async fn available_memory_bytes(&self) -> u64 {
        let (_, available) = self.query_memory().await;
        available
    }

    /// Determine if a model with the given estimated memory footprint can be
    /// safely loaded without risking an OOM crash.
    ///
    /// ## Admission Logic
    ///
    /// ```text
    /// available_ram - safety_margin = headroom
    /// headroom >= estimated_footprint → ADMIT
    /// headroom <  estimated_footprint → DENY
    /// ```
    ///
    /// ## Why This Prevents OOM
    ///
    /// By subtracting the safety margin from available memory before comparing
    /// against the model footprint, we guarantee that loading the model will
    /// never push the system below the minimum free RAM threshold. This gives
    /// the OS kernel, window manager, and other processes a guaranteed memory
    /// buffer, preventing the OOM killer from activating.
    ///
    /// # Arguments
    /// * `estimated_footprint_bytes` — Estimated memory consumption of the model
    ///
    /// # Returns
    /// `true` if the model can safely fit, `false` if admission should be denied
    pub async fn can_admit_model(&self, estimated_footprint_bytes: u64) -> bool {
        let available = self.available_memory_bytes().await;
        let margin = self.safety_margin();

        let headroom = available.saturating_sub(margin);

        tracing::debug!(
            "TelemetryMonitor: available={:.1}GB, margin={:.1}GB, headroom={:.1}GB, model={:.1}GB → {}",
            available as f64 / 1_073_741_824.0,
            margin as f64 / 1_073_741_824.0,
            headroom as f64 / 1_073_741_824.0,
            estimated_footprint_bytes as f64 / 1_073_741_824.0,
            if headroom >= estimated_footprint_bytes {
                "ADMIT"
            } else {
                "DENY"
            },
        );

        headroom >= estimated_footprint_bytes
    }

    /// Produce a telemetry snapshot for logging and metrics.
    pub async fn snapshot(&self) -> TelemetrySnapshot {
        let (total, available) = self.query_memory().await;
        let margin = self.safety_margin();
        let headroom = available.saturating_sub(margin);

        TelemetrySnapshot {
            total_memory_bytes: total,
            available_memory_bytes: available,
            used_memory_bytes: total.saturating_sub(available),
            safety_margin_bytes: margin,
            headroom_bytes: headroom,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_safety_margin() {
        let monitor = TelemetryMonitor::new();
        assert_eq!(monitor.safety_margin(), DEFAULT_SAFETY_MARGIN_BYTES);
        assert_eq!(monitor.safety_margin(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_custom_safety_margin() {
        let monitor = TelemetryMonitor::with_safety_margin(4 * 1024 * 1024 * 1024);
        assert_eq!(monitor.safety_margin(), 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_set_safety_margin() {
        let monitor = TelemetryMonitor::new();
        monitor.set_safety_margin(1024);
        assert_eq!(monitor.safety_margin(), 1024);
    }

    #[tokio::test]
    async fn test_query_memory_returns_nonzero() {
        let monitor = TelemetryMonitor::new();
        let (total, available) = monitor.query_memory().await;

        // On any real system, total and available should be > 0
        assert!(total > 0, "Total memory should be > 0");
        assert!(available > 0, "Available memory should be > 0");
        assert!(available <= total, "Available should be <= total");
    }

    #[tokio::test]
    async fn test_can_admit_tiny_model() {
        let monitor = TelemetryMonitor::with_safety_margin(0);
        // 1 byte model should always fit
        assert!(monitor.can_admit_model(1).await);
    }

    #[tokio::test]
    async fn test_cannot_admit_impossibly_large_model() {
        let monitor = TelemetryMonitor::new();
        // 1 exabyte model should never fit
        let exabyte = 1024 * 1024 * 1024 * 1024 * 1024;
        assert!(!monitor.can_admit_model(exabyte).await);
    }

    #[tokio::test]
    async fn test_snapshot_consistency() {
        let monitor = TelemetryMonitor::new();
        let snap = monitor.snapshot().await;

        assert!(snap.total_memory_bytes > 0);
        assert!(snap.available_memory_bytes <= snap.total_memory_bytes);
        assert_eq!(snap.safety_margin_bytes, DEFAULT_SAFETY_MARGIN_BYTES);
        assert_eq!(
            snap.headroom_bytes,
            snap.available_memory_bytes
                .saturating_sub(snap.safety_margin_bytes)
        );
    }

    #[test]
    fn test_monitor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TelemetryMonitor>();
    }

    #[test]
    fn test_monitor_debug() {
        let monitor = TelemetryMonitor::new();
        let debug = format!("{:?}", monitor);
        assert!(debug.contains("TelemetryMonitor"));
    }
}
