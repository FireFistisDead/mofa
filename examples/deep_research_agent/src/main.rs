//! Deep Research Agent Example
//!
//! Demonstrates chaining Search, Scrape, and FileSystem tools to simulate a
//! deep-research workflow, while deliberately commenting on architectural
//! limitations in the current MoFA ecosystem.
//!
//! # How to run
//!
//! ```bash
//! # Set your LLM API key (OpenAI-compatible endpoint)
//! export OPENAI_API_KEY=your-api-key
//! # Optional: custom base URL (e.g. Ollama)
//! export OPENAI_BASE_URL=http://localhost:11434/v1
//!
//! cd examples
//! cargo run -p deep_research_agent
//! ```

use async_trait::async_trait;
use mofa_sdk::llm::{openai_from_env, LLMAgentBuilder};
use mofa_sdk::react::{ReActAgent, ReActTool};
use serde_json::Value;
use std::sync::Arc;
use tracing::info;

// =============================================================================
// PAIN POINT #1: DuckSearchTool and WebScrapperTool are COMMENTED OUT in
// mofa-plugins (see crates/mofa-plugins/src/tools/duck_search.rs and
// web_scrapper.rs). They implemented an older `Tool` trait that no longer
// exists in the public API. Nobody migrated them to the current `ToolExecutor`
// or `ReActTool` interfaces, so they are dead code.
//
// This means every example that needs search or scraping must re-implement
// the tools from scratch, duplicating effort across the ecosystem.
// =============================================================================

// =============================================================================
// PAIN POINT #2: The plugin-layer `ToolExecutor` trait (mofa_plugins) and the
// react-layer `ReActTool` trait (mofa_foundation::react) are entirely separate
// type hierarchies. FileSystemTool and HttpRequestTool implement `ToolExecutor`,
// but ReActAgent expects `ReActTool`. There is no built-in adapter between
// the two, so we must write manual wrapper structs for every tool.
//
// A unified `Tool` abstraction that works across *all* agent patterns would
// eliminate this friction.
// =============================================================================

// ---------------------------------------------------------------------------
// Tool 1: DuckDuckGo Search (mock – the real implementation is commented out)
// ---------------------------------------------------------------------------

/// Mock search tool standing in for the abandoned `DuckDuckGoSearchResults`.
///
/// PAIN POINT #3: Because the real tool is disabled, we fall back to a mock
/// that returns canned data. In production this would need `reqwest` + HTML
/// parsing (scraper crate) — exactly the deps that are already in
/// mofa-plugins' Cargo.toml but gated behind commented-out code.
struct DuckSearchTool;

#[async_trait]
impl ReActTool for DuckSearchTool {
    fn name(&self) -> &str {
        "duck_search"
    }

    fn description(&self) -> &str {
        "Search the web using DuckDuckGo. Input should be a search query string. \
         Returns a JSON array of {title, link, snippet} objects."
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }))
    }

    async fn execute(&self, input: &str) -> Result<String, String> {
        let query = if let Ok(json) = serde_json::from_str::<Value>(input) {
            json.get("query")
                .and_then(|v| v.as_str())
                .unwrap_or(input)
                .to_owned()
        } else {
            input.to_owned()
        };

        info!("DuckSearchTool: searching for '{}'", query);

        // PAIN POINT #4: This is a mock. The real DuckDuckGoSearchResults in
        // mofa-plugins relied on reqwest + scraper to parse duckduckgo.com/html.
        // That code is entirely commented out; until it is resurrected behind
        // a feature flag and adapted to `ReActTool`, every consumer must
        // roll their own HTTP-based search integration.
        let mock_results = serde_json::json!([
            {
                "title": "Solid-State Batteries 2026: Breakthroughs and Challenges",
                "link": "https://example.com/solid-state-batteries-2026",
                "snippet": "Recent advances in sulfide-based solid electrolytes have \
                            pushed energy densities beyond 500 Wh/kg in lab prototypes."
            },
            {
                "title": "Toyota and Samsung SDI Announce Joint Solid-State Venture",
                "link": "https://example.com/toyota-samsung-solid-state",
                "snippet": "The partnership targets mass production of solid-state cells \
                            for EVs by late 2027, leveraging oxide-based electrolytes."
            },
            {
                "title": "QuantumScape Reports Record Cycle Life in 2026 Tests",
                "link": "https://example.com/quantumscape-2026",
                "snippet": "QuantumScape's lithium-metal cells achieved 1,000+ cycles \
                            at 80% capacity retention, a milestone for solid-state tech."
            },
            {
                "title": "Challenges Remain: Manufacturing Scale for Solid Electrolytes",
                "link": "https://example.com/solid-state-manufacturing",
                "snippet": "Despite lab success, scaling thin-film sulfide electrolytes \
                            to gigafactory volumes remains the primary bottleneck."
            }
        ]);

        Ok(serde_json::to_string_pretty(&mock_results).unwrap())
    }
}

// ---------------------------------------------------------------------------
// Tool 2: Web Scraper (mock – the real implementation is commented out)
// ---------------------------------------------------------------------------

/// Mock web scraper standing in for the abandoned `WebScrapper`.
///
/// PAIN POINT #5: The real WebScrapper returned the *entire* text content of
/// a web page with no truncation, summarisation, or chunking. In a deep-
/// research workflow this easily produces 50-100 KB of raw text per URL,
/// which blows up the LLM context window in a single tool call.
///
/// There is no built-in integration between WebScrapper and the
/// ContextCompressor (mofa_foundation::context_compression) — the agent
/// must manually orchestrate compression, which defeats the purpose of
/// autonomous tool chaining.
struct WebScrapperTool;

#[async_trait]
impl ReActTool for WebScrapperTool {
    fn name(&self) -> &str {
        "web_scraper"
    }

    fn description(&self) -> &str {
        "Scrape the text content of a web page given its URL. \
         WARNING: output can be very large and may exceed context limits. \
         Prefer using this on specific article pages, not landing pages."
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to scrape"
                }
            },
            "required": ["url"]
        }))
    }

    async fn execute(&self, input: &str) -> Result<String, String> {
        let url = if let Ok(json) = serde_json::from_str::<Value>(input) {
            json.get("url")
                .and_then(|v| v.as_str())
                .unwrap_or(input)
                .to_owned()
        } else {
            input.to_owned()
        };

        info!("WebScrapperTool: scraping '{}'", url);

        // PAIN POINT #6: Even in production, the old WebScrapper just returned
        // raw text. There is no pipeline to:
        //   1. Strip boilerplate / nav / ads (readability extraction)
        //   2. Chunk the text into context-window-safe segments
        //   3. Embed + retrieve only relevant chunks (RAG)
        //   4. Auto-compress via ContextCompressor before returning
        //
        // Each of these capabilities exists *somewhere* in MoFA (RAG pipeline,
        // context compression) but they are not wired together as a composable
        // tool chain. The developer must glue them manually.
        let mock_content = format!(
            "[Scraped content from {url}]\n\n\
             Solid-State Battery Technology: 2026 Update\n\n\
             Solid-state batteries replace the liquid electrolyte in conventional \
             lithium-ion cells with a solid material — typically a ceramic oxide, \
             sulfide glass, or polymer composite. The key advantages are higher \
             energy density (potentially 2-3x), improved safety (no flammable \
             liquid), and longer cycle life.\n\n\
             Key developments in 2026:\n\
             - Samsung SDI demonstrated a 900 Wh/L pouch cell using argyrodite \
               sulfide electrolyte with a lithium-metal anode.\n\
             - Toyota's pilot line in Primearth achieved 80% yield on 50 Ah \
               prismatic cells using oxide-sulfide hybrid electrolyte.\n\
             - QuantumScape published peer-reviewed data showing 1,200 cycles \
               at C/3 rate with less than 20% capacity fade.\n\
             - CATL announced a 'condensed matter' battery with semi-solid \
               electrolyte targeting commercial aviation.\n\
             - Solid Power shipped B-sample cells to BMW for vehicle-level \
               testing, using sulfide-based electrolyte.\n\n\
             Remaining challenges:\n\
             - Interfacial resistance between solid electrolyte and electrodes\n\
             - Dendrite penetration at high current densities\n\
             - Manufacturing cost: current estimates are 3-5x conventional Li-ion\n\
             - Scalability of thin-film electrolyte deposition techniques\n\n\
             [End of scraped content — {word_count} words]\n\n\
             NOTE: In production, this would be tens of thousands of words of \
             raw HTML-stripped text with no summarisation.",
            word_count = 180
        );

        Ok(mock_content)
    }
}

// ---------------------------------------------------------------------------
// Tool 3: FileSystem (adapter wrapping the real mofa-plugins FileSystemTool)
// ---------------------------------------------------------------------------

/// Adapter that wraps `mofa_plugins::tools::FileSystemTool` (which implements
/// `ToolExecutor`) into the `ReActTool` interface required by `ReActAgent`.
///
/// PAIN POINT #7: This adapter should not need to exist. The framework should
/// provide a generic `ToolExecutor → ReActTool` bridge, or better yet, unify
/// the two traits. Every new tool from mofa-plugins requires a hand-written
/// wrapper to be usable in a ReAct agent.
struct FileSystemReActTool {
    inner: mofa_plugins::tools::FileSystemTool,
}

impl FileSystemReActTool {
    fn new(allowed_paths: Vec<String>) -> Self {
        Self {
            inner: mofa_plugins::tools::FileSystemTool::new(allowed_paths),
        }
    }
}

#[async_trait]
impl ReActTool for FileSystemReActTool {
    fn name(&self) -> &str {
        "filesystem"
    }

    fn description(&self) -> &str {
        "Perform file system operations: read, write, list, exists, delete, mkdir. \
         Input must be a JSON object with 'operation' (read|write|list|exists|delete|mkdir), \
         'path', and optionally 'content' (for write)."
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "exists", "delete", "mkdir"],
                    "description": "File operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)"
                }
            },
            "required": ["operation", "path"]
        }))
    }

    async fn execute(&self, input: &str) -> Result<String, String> {
        // PAIN POINT #8: We must manually parse the string input back into
        // serde_json::Value because ReActTool::execute takes &str, while
        // ToolExecutor::execute takes serde_json::Value. This lossy
        // serialisation round-trip is wasteful and error-prone.
        let args: Value = serde_json::from_str(input).map_err(|e| {
            format!(
                "FileSystemReActTool: failed to parse input as JSON: {e}. \
                 Input was: {input}"
            )
        })?;

        use mofa_plugins::ToolExecutor;
        match self.inner.execute(args).await {
            Ok(result) => Ok(serde_json::to_string_pretty(&result)
                .unwrap_or_else(|_| result.to_string())),
            Err(e) => Err(format!("FileSystemTool error: {e}")),
        }
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("=== MoFA Deep Research Agent ===");
    info!("This example highlights tool-integration pain points in the current architecture.\n");

    // -------------------------------------------------------------------------
    // 1. Create the LLM provider from environment variables
    // -------------------------------------------------------------------------
    let provider = Arc::new(openai_from_env()?);

    let llm_agent = Arc::new(
        LLMAgentBuilder::new()
            .with_id("deep-research-llm")
            .with_provider(provider)
            .with_system_prompt(SYSTEM_PROMPT)
            .build(),
    );

    // -------------------------------------------------------------------------
    // 2. Prepare the output directory for the FileSystem tool
    // -------------------------------------------------------------------------
    let output_dir = std::path::Path::new("./research_outputs");
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
    }
    let canonical_output = std::fs::canonicalize(output_dir)?;
    info!("Research outputs will be saved to: {}", canonical_output.display());

    // -------------------------------------------------------------------------
    // 3. Instantiate tools
    // -------------------------------------------------------------------------
    // PAIN POINT #9: We have to create three separate structs, two of which
    // are mocks for abandoned code, and one is a hand-written adapter. In an
    // ideal framework, this would be:
    //
    //   let tools = builtin_tools!["duck_search", "web_scraper", "filesystem"];
    //
    // or at minimum:
    //
    //   let fs = FileSystemTool::new(...).into_react_tool();
    //
    let search_tool = Arc::new(DuckSearchTool);
    let scrape_tool = Arc::new(WebScrapperTool);
    let fs_tool = Arc::new(FileSystemReActTool::new(vec![
        canonical_output.to_string_lossy().to_string(),
    ]));

    // -------------------------------------------------------------------------
    // 4. Build the ReAct agent
    // -------------------------------------------------------------------------
    let agent = ReActAgent::builder()
        .with_llm(llm_agent)
        .with_tool(search_tool)
        .with_tool(scrape_tool)
        .with_tool(fs_tool)
        .with_max_iterations(15)
        .with_temperature(0.3)
        .with_verbose(true)
        // PAIN POINT #10: The system prompt below must manually describe the
        // entire workflow because there is no declarative workflow DSL that
        // chains tools. The agent is expected to figure out the pipeline
        // (search → scrape → synthesise → save) purely from natural language.
        // If the LLM hallucinates a different order, the research fails.
        .with_system_prompt(SYSTEM_PROMPT)
        .build()?;

    // -------------------------------------------------------------------------
    // 5. Execute the research query
    // -------------------------------------------------------------------------
    let query = "Conduct a deep research report on the Advancements in \
                 Solid State Batteries 2026. \
                 Search for recent developments, scrape the most relevant pages \
                 for details, synthesise the findings into a structured report, \
                 and save the final report to disk as 'solid_state_batteries_2026.md'.";

    info!("Submitting query:\n{}\n", query);

    let result = agent.run(query).await?;

    // -------------------------------------------------------------------------
    // 6. Print results
    // -------------------------------------------------------------------------
    println!("\n{}", "=".repeat(80));
    println!("DEEP RESEARCH AGENT — FINAL RESULT");
    println!("{}\n", "=".repeat(80));

    if result.success {
        println!("Status : SUCCESS");
    } else {
        println!("Status : FAILED");
    }
    println!("Iterations : {}", result.iterations);
    println!("Duration : {} ms", result.duration_ms);
    println!("\n--- Answer ---\n");
    println!("{}", result.answer);

    println!("\n--- Steps Trace ---\n");
    for step in &result.steps {
        println!(
            "[Step {}] {:?}: {}",
            step.step_number, step.step_type, step.content
        );
    }

    // PAIN POINT #11 (summary): In a production deep-research agent you would
    // also need:
    //   - Automatic context-window management (the scraped content can easily
    //     exceed the model's limit; the ContextCompressor exists but is not
    //     integrated into the tool pipeline).
    //   - RAG-based retrieval so the agent only sees relevant chunks instead
    //     of full page dumps.
    //   - A declarative workflow graph (search → scrape → compress → synthesise
    //     → save) instead of hoping the LLM follows the prompt instructions.
    //   - Retry / fallback logic when a scrape fails (currently no built-in
    //     tool-level retry in ReActAgent).
    //
    // All of these capabilities exist as separate modules in MoFA but are not
    // wired together into a cohesive "research pipeline" abstraction.

    Ok(())
}

// =============================================================================
// System Prompt
// =============================================================================

/// Detailed system prompt that tells the agent the exact workflow to follow.
const SYSTEM_PROMPT: &str = r#"You are a Deep Research Agent. Your job is to produce a comprehensive, well-structured research report by following this exact workflow:

## Workflow

1. **SEARCH**: Use the `duck_search` tool to find relevant sources for the research topic. Issue 1-3 targeted search queries to cover different facets of the topic.

2. **SCRAPE**: For each promising search result, use the `web_scraper` tool to retrieve the full content of the page. Focus on the 2-3 most relevant URLs from the search results rather than scraping everything.

3. **SYNTHESISE**: After gathering enough information from search and scrape steps, synthesise the findings into a structured Markdown report with:
   - An executive summary
   - Key findings organised by theme
   - Technical details where relevant
   - A "Challenges & Open Questions" section
   - A list of sources with URLs

4. **SAVE**: Use the `filesystem` tool with operation "write" to save the final report to disk. The path should be a filename like "solid_state_batteries_2026.md" (it will be saved under the allowed research_outputs directory).

## Important Rules
- Always think step-by-step before acting.
- Do NOT try to scrape more than 3 URLs — context limits are real.
- When writing the final report, be concise but thorough.
- After saving the file, provide the final report text as your Final Answer.

## Available Tools
- `duck_search`: Search the web. Input: {"query": "your search terms"}
- `web_scraper`: Scrape a URL. Input: {"url": "https://example.com/page"}
- `filesystem`: File operations. Input: {"operation": "write", "path": "report.md", "content": "..."}
"#;
