use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::TryStreamExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub prompt: Option<String>,
    pub completion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    pub modality: Option<String>,
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub output_modalities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterRawModel {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub context_length: Option<u64>,
    pub created: Option<i64>,
    pub architecture: Option<Architecture>,
    pub pricing: Option<ModelPricing>,
    #[serde(default)]
    pub supported_parameters: Vec<String>,
    pub reasoning: Option<ReasoningInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningInfo {
    #[serde(default)]
    pub mandatory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterCatalogResponse {
    pub data: Vec<OpenRouterRawModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedModel {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub context_length: u64,
    pub created: i64,
    pub prompt_price: f64,
    pub completion_price: f64,
    pub is_free: bool,
    pub output_modalities: Vec<String>,
    pub input_modalities: Vec<String>,
    pub top_weekly_rank: usize,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub reasoning_mandatory: bool,
}

#[derive(Debug, Default)]
pub struct AppState {
    pub models: RwLock<Vec<ProcessedModel>>,
    pub http_client: reqwest::Client,
    pub api_key: Option<String>,
    /// model_id -> instant until which the model is skipped (recent 429/5xx)
    pub cooldown: Mutex<HashMap<String, Instant>>,
}

const COOLDOWN_SECS: u64 = 60;

fn is_cooling(state: &AppState, model: &str) -> bool {
    let mut map = state.cooldown.lock().unwrap();
    match map.get(model) {
        Some(until) if *until > Instant::now() => true,
        Some(_) => {
            map.remove(model);
            false
        }
        None => false,
    }
}

fn mark_cooldown(state: &AppState, model: &str, secs: u64) {
    let now = Instant::now();
    let mut map = state.cooldown.lock().unwrap();
    if map.len() > 64 {
        map.retain(|_, until| *until > now);
    }
    map.insert(model.to_string(), now + Duration::from_secs(secs));
}

/// Input modalities the client actually sent (image/audio/file/video parts).
fn required_input_modalities(payload: &serde_json::Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) else {
        return out;
    };

    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            let kind = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let modality = match kind {
                "image_url" | "input_image" | "image" => Some("image"),
                "input_audio" | "audio" => Some("audio"),
                "input_file" | "file" => Some("file"),
                "video_url" | "input_video" | "video" => Some("video"),
                _ => None,
            };
            if let Some(md) = modality {
                if !out.iter().any(|x| x == md) {
                    out.push(md.to_string());
                }
            }
        }
    }

    out
}

/// Keep only candidates that accept every modality in `required`. When none of
/// the route's candidates qualify (e.g. `free:coding` + image), fall back to
/// catalog models that do, ranked by weekly popularity.
fn filter_by_modality(
    candidates: Vec<String>,
    modality_map: &HashMap<String, Vec<String>>,
    required: &[String],
    catalog_fallback: &[String],
) -> Vec<String> {
    let fitting: Vec<String> = candidates
        .iter()
        .filter(|m| match modality_map.get(*m) {
            Some(mods) => required.iter().all(|r| mods.contains(r)),
            None => true,
        })
        .cloned()
        .collect();

    if !fitting.is_empty() {
        if fitting.len() < candidates.len() {
            info!(
                "Modality filter ({}): {} of {} candidates skipped",
                required.join("+"),
                candidates.len() - fitting.len(),
                candidates.len()
            );
        }
        return fitting;
    }

    if !catalog_fallback.is_empty() {
        warn!(
            "No route candidate accepts {}; using catalog models that do",
            required.join("+")
        );
        return catalog_fallback.to_vec();
    }

    warn!(
        "No model accepts {}; forwarding with the original candidates",
        required.join("+")
    );
    candidates
}

/// Rough prompt size estimate: JSON payload of `messages` at ~4 chars per token.
fn estimate_prompt_tokens(payload: &serde_json::Value) -> u64 {
    let chars = payload
        .get("messages")
        .and_then(|m| serde_json::to_string(m).ok())
        .map(|s| s.len())
        .unwrap_or(0);
    (chars as u64 / 4) + 16
}

fn payload_output_budget(payload: &serde_json::Value) -> u64 {
    payload
        .get("max_tokens")
        .or_else(|| payload.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1024)
}

/// Keep only candidates with a context window big enough for this prompt.
/// Unknown context (0) or unknown model passes. If nothing fits, keep the
/// largest-context candidate so the request still has a chance.
fn filter_by_context(
    candidates: Vec<String>,
    context_map: &HashMap<String, u64>,
    needed: u64,
) -> Vec<String> {
    let fitting: Vec<String> = candidates
        .iter()
        .filter(|m| match context_map.get(*m) {
            Some(&ctx) => ctx == 0 || ctx >= needed,
            None => true,
        })
        .cloned()
        .collect();

    if !fitting.is_empty() {
        if fitting.len() < candidates.len() {
            info!(
                "Context filter: {} of {} candidates too small for ~{} tokens",
                candidates.len() - fitting.len(),
                candidates.len(),
                needed
            );
        }
        return fitting;
    }

    let largest = candidates
        .iter()
        .max_by_key(|m| context_map.get(*m).copied().unwrap_or(0))
        .cloned();

    warn!(
        "No candidate fits ~{} tokens; falling back to largest context window",
        needed
    );

    largest.into_iter().collect()
}

/// True when the response carries usable assistant text. Unknown shapes pass.
fn has_text_content(v: &serde_json::Value) -> bool {
    let Some(msg) = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
    else {
        return true;
    };

    match msg.get("content") {
        Some(serde_json::Value::String(s)) => !s.trim().is_empty(),
        Some(serde_json::Value::Array(a)) => !a.is_empty(),
        Some(serde_json::Value::Null) | None => false,
        _ => true,
    }
}

/// Drop candidates that are cooling down. If every candidate is cooling down we
/// return the original list, so a stale cooldown never causes a hard outage.
fn filter_cooling(state: &AppState, candidates: Vec<String>) -> Vec<String> {
    let ready: Vec<String> = candidates
        .iter()
        .filter(|m| !is_cooling(state, m))
        .cloned()
        .collect();

    if ready.is_empty() {
        return candidates;
    }
    if ready.len() < candidates.len() {
        info!(
            "Cooldown active: {} of {} candidates skipped",
            candidates.len() - ready.len(),
            candidates.len()
        );
    }
    ready
}

#[derive(Debug, Deserialize)]
pub struct FilterQuery {
    pub variant: Option<String>,
    pub output: Option<String>,
    pub sort: Option<String>,
    pub max_price: Option<f64>,
    pub tools: Option<bool>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse {
    pub total: usize,
    pub count: usize,
    pub filters: ResponseFilters,
    pub data: Vec<ProcessedModel>,
}

#[derive(Debug, Serialize)]
pub struct ResponseFilters {
    pub variant: String,
    pub output: String,
    pub sort: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIModelItem {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIModelsResponse {
    pub object: String,
    pub data: Vec<OpenAIModelItem>,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "openrouter_free_finder=info,tower_http=info".into()),
        )
        .init();

    let api_key = std::env::var("OPENROUTER_API_KEY").ok();
    if api_key.is_some() {
        info!("🔑 OPENROUTER_API_KEY detected in .env for local fallback.");
    } else {
        info!("ℹ️ Client Authorization header will be inherited directly.");
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("Failed to build HTTP client");

    let state = Arc::new(AppState {
        models: RwLock::new(Vec::new()),
        http_client: client,
        api_key,
        cooldown: Mutex::new(HashMap::new()),
    });

    info!("Performing initial OpenRouter catalog & ranking fetch...");
    if let Err(e) = sync_models(state.clone()).await {
        error!("Initial sync failed: {}", e);
    } else {
        info!("Initial sync complete.");
    }

    // Background poller
    let poller_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(900));
        loop {
            interval.tick().await;
            if let Err(e) = sync_models(poller_state.clone()).await {
                error!("Periodic sync error: {}", e);
            }
        }
    });

    // Support BOTH /v1/models and /models for universal compatibility with OmniRoute & standard OpenAI SDKs
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(get_openai_models))
        .route("/models", get(get_openai_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/api/models", get(get_rich_models))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3033".to_string());
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    info!("🚀 Server running on http://0.0.0.0:{}", port);
    info!("👉 OmniRoute Base URL: http://192.168.1.200:{}", port);
    info!("👉 OmniRoute Base URL (alt): http://192.168.1.200:{}/v1", port);

    axum::serve(listener, app).await.expect("Server error");
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

fn filter_and_sort_models(all: &[ProcessedModel], query: &FilterQuery) -> Vec<ProcessedModel> {
    let variant = query.variant.as_deref().unwrap_or("free");
    let output_filter = query.output.as_deref().unwrap_or("text");
    let sort = query.sort.as_deref().unwrap_or("top-weekly");

    let mut filtered: Vec<ProcessedModel> = all
        .iter()
        .filter(|m| {
            match variant {
                "free" => {
                    if !m.is_free {
                        return false;
                    }
                }
                "paid" => {
                    if m.is_free {
                        return false;
                    }
                }
                _ => {}
            }

            if !output_filter.is_empty() && output_filter != "all" {
                if !m.output_modalities.iter().any(|o| o == output_filter) {
                    return false;
                }
            }

            if let Some(max_p) = query.max_price {
                if m.prompt_price > max_p {
                    return false;
                }
            }

            if let Some(req_tools) = query.tools {
                if req_tools && !m.supports_tools {
                    return false;
                }
            }

            true
        })
        .cloned()
        .collect();

    match sort {
        "top-weekly" => {
            filtered.sort_by(|a, b| a.top_weekly_rank.cmp(&b.top_weekly_rank));
        }
        "context" => {
            filtered.sort_by(|a, b| b.context_length.cmp(&a.context_length));
        }
        "cheapest" => {
            filtered.sort_by(|a, b| {
                let p_a = if a.prompt_price < 0.0 { f64::MAX } else { a.prompt_price };
                let p_b = if b.prompt_price < 0.0 { f64::MAX } else { b.prompt_price };
                p_a.partial_cmp(&p_b).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        "newest" => {
            filtered.sort_by(|a, b| b.created.cmp(&a.created));
        }
        _ => {}
    }

    filtered
}

async fn get_openai_models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FilterQuery>,
) -> Result<Json<OpenAIModelsResponse>, StatusCode> {
    let all = state.models.read().await;
    let mut list = filter_and_sort_models(&all, &query);

    let limit = query.limit.unwrap_or(20);
    if list.len() > limit {
        list.truncate(limit);
    }

    let mut items: Vec<OpenAIModelItem> = Vec::with_capacity(list.len() + 8);

    let aliases = [
        "auto-free",
        "free:top-1",
        "free:top-2",
        "free:top-3",
        "free:context",
        "free:tools",
        "free:coding",
        "cheapest:text",
    ];

    for alias in aliases {
        items.push(OpenAIModelItem {
            id: alias.to_string(),
            object: "model".to_string(),
            created: 1793669930,
            owned_by: "smart-proxy".to_string(),
        });
    }

    for m in list {
        let owner = m.id.split('/').next().unwrap_or("openrouter").to_string();
        items.push(OpenAIModelItem {
            id: m.id,
            object: "model".to_string(),
            created: m.created,
            owned_by: owner,
        });
    }

    Ok(Json(OpenAIModelsResponse {
        object: "list".to_string(),
        data: items,
    }))
}

async fn get_rich_models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FilterQuery>,
) -> Result<Json<ApiResponse>, StatusCode> {
    let variant = query.variant.clone().unwrap_or_else(|| "free".to_string());
    let output_filter = query.output.clone().unwrap_or_else(|| "text".to_string());
    let sort = query.sort.clone().unwrap_or_else(|| "top-weekly".to_string());
    let limit = query.limit.unwrap_or(20);

    let all = state.models.read().await;
    let mut filtered = filter_and_sort_models(&all, &query);

    let total_matches = filtered.len();
    if filtered.len() > limit {
        filtered.truncate(limit);
    }

    Ok(Json(ApiResponse {
        total: total_matches,
        count: filtered.len(),
        filters: ResponseFilters {
            variant,
            output: output_filter,
            sort,
        },
        data: filtered,
    }))
}

fn resolve_target_candidates(requested: &str, all_models: &[ProcessedModel]) -> Vec<String> {
    match requested {
        "auto-free" | "free:top-1" => {
            vec![
                "nvidia/nemotron-3-ultra-550b-a55b:free".to_string(),
                "nvidia/nemotron-3.5-super-vl:free".to_string(),
                "cohere/north-mini-code:free".to_string(),
                "stealth/crouton-alpha".to_string(),
            ]
        }
        "free:top-2" => {
            vec![
                "nvidia/nemotron-3.5-super-vl:free".to_string(),
                "nvidia/nemotron-3-ultra-550b-a55b:free".to_string(),
                "cohere/north-mini-code:free".to_string(),
            ]
        }
        "free:top-3" => {
            let q = FilterQuery {
                variant: Some("free".to_string()),
                output: Some("text".to_string()),
                sort: Some("top-weekly".to_string()),
                max_price: None,
                tools: None,
                limit: Some(5),
            };
            let list = filter_and_sort_models(all_models, &q);
            let mut res: Vec<String> = list.into_iter().skip(2).map(|m| m.id).collect();
            if res.is_empty() {
                res.push("nvidia/nemotron-3.5-super-vl:free".to_string());
            }
            res
        }
        "free:context" => {
            let q = FilterQuery {
                variant: Some("free".to_string()),
                output: Some("text".to_string()),
                sort: Some("context".to_string()),
                max_price: None,
                tools: None,
                limit: Some(3),
            };
            let list = filter_and_sort_models(all_models, &q);
            let mut res: Vec<String> = list.into_iter().map(|m| m.id).collect();
            if res.is_empty() {
                res.push("stealth/crouton-alpha".to_string());
            }
            res
        }
        "free:tools" => {
            let q = FilterQuery {
                variant: Some("free".to_string()),
                output: Some("text".to_string()),
                sort: Some("top-weekly".to_string()),
                max_price: None,
                tools: Some(true),
                limit: Some(3),
            };
            let list = filter_and_sort_models(all_models, &q);
            let mut res: Vec<String> = list.into_iter().map(|m| m.id).collect();
            if res.is_empty() {
                res.push("stealth/crouton-alpha".to_string());
            }
            res
        }
        "free:coding" | "auto-free-coding" => {
            let mut res: Vec<String> = all_models
                .iter()
                .filter(|m| m.is_free && (m.id.contains("code") || m.id.contains("flint")))
                .map(|m| m.id.clone())
                .collect();
            if res.is_empty() {
                res.push("cohere/north-mini-code:free".to_string());
            }
            res
        }
        "cheapest:text" => {
            let q = FilterQuery {
                variant: Some("paid".to_string()),
                output: Some("text".to_string()),
                sort: Some("cheapest".to_string()),
                max_price: None,
                tools: None,
                limit: Some(3),
            };
            let list = filter_and_sort_models(all_models, &q);
            let mut res: Vec<String> = list.into_iter().map(|m| m.id).collect();
            if res.is_empty() {
                res.push("deepseek/deepseek-v4.1-flash-1017".to_string());
            }
            res
        }
        custom => vec![custom.to_string()],
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut payload): Json<serde_json::Value>,
) -> Result<Response, (StatusCode, String)> {
    let auth_header = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| state.api_key.as_ref().map(|k| format!("Bearer {}", k)))
        .ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                "Missing Authorization header. Please pass your OpenRouter API key.".to_string(),
            )
        })?;

    let mut requested_model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("auto-free")
        .to_string();

    // Strip prefix if OmniRoute sends prefix prepended (e.g. "openrouter-free/auto-free" -> "auto-free")
    if let Some(stripped) = requested_model.strip_prefix("openrouter-free/") {
        requested_model = stripped.to_string();
    }

    let all_models = state.models.read().await;
    let candidates = resolve_target_candidates(&requested_model, &all_models);

    // A tiny output budget is fully consumed by reasoning on reasoning-capable
    // models, leaving `content: null`. Turn reasoning off for those requests.
    let small_budget = matches!(
        payload
            .get("max_tokens")
            .or_else(|| payload.get("max_completion_tokens"))
            .and_then(|v| v.as_u64()),
        Some(n) if n <= 128
    ) && payload.get("reasoning").is_none();

    let reasoning_capable: std::collections::HashSet<String> = if small_budget {
        all_models
            .iter()
            .filter(|m| m.supports_reasoning && !m.reasoning_mandatory)
            .map(|m| m.id.clone())
            .collect()
    } else {
        Default::default()
    };

    // Skip candidates whose context window cannot fit this prompt.
    let needed = estimate_prompt_tokens(&payload) + payload_output_budget(&payload);
    let context_map: HashMap<String, u64> = if needed > 4096 {
        all_models
            .iter()
            .map(|m| (m.id.clone(), m.context_length))
            .collect()
    } else {
        Default::default()
    };

    // Multimodal requests must land on a model that accepts those modalities.
    let required_modalities = required_input_modalities(&payload);
    let modality_map: HashMap<String, Vec<String>> = if required_modalities.is_empty() {
        Default::default()
    } else {
        all_models
            .iter()
            .map(|m| (m.id.clone(), m.input_modalities.clone()))
            .collect()
    };
    let modality_fallback: Vec<String> = if required_modalities.is_empty() {
        Vec::new()
    } else {
        let mut fitting: Vec<&ProcessedModel> = all_models
            .iter()
            .filter(|m| {
                m.is_free
                    && m.output_modalities.iter().any(|o| o == "text")
                    && required_modalities
                        .iter()
                        .all(|r| m.input_modalities.contains(r))
            })
            .collect();
        fitting.sort_by_key(|m| m.top_weekly_rank);
        fitting.into_iter().map(|m| m.id.clone()).collect()
    };
    drop(all_models);

    if small_budget {
        info!("Small max_tokens budget: reasoning will be disabled per model support");
    }
    if !required_modalities.is_empty() {
        info!(
            "Multimodal request detected: {}",
            required_modalities.join("+")
        );
    }

    let candidates = if modality_map.is_empty() {
        candidates
    } else {
        filter_by_modality(
            candidates,
            &modality_map,
            &required_modalities,
            &modality_fallback,
        )
    };

    let candidates = if context_map.is_empty() {
        candidates
    } else {
        filter_by_context(candidates, &context_map, needed)
    };

    // Skip models that recently returned 429/5xx. If every candidate is cooling
    // down, the full list is used so a stale cooldown never causes an outage.
    let candidates = filter_cooling(&state, candidates);

    let openrouter_url = "https://openrouter.ai/api/v1/chat/completions";

    let mut last_response = None;
    let mut injected_reasoning = false;

    for (attempt, target_model) in candidates.iter().enumerate() {
        if attempt > 0 {
            warn!("⚠️ Fallback attempt #{}: switching to model '{}'", attempt, target_model);
        } else {
            info!("Routing: '{}' -> '{}'", requested_model, target_model);
        }

        if let Some(obj) = payload.as_object_mut() {
            obj.insert("model".to_string(), serde_json::Value::String(target_model.clone()));

            if small_budget && reasoning_capable.contains(target_model) {
                obj.insert(
                    "reasoning".to_string(),
                    serde_json::json!({ "effort": "none" }),
                );
                injected_reasoning = true;
            } else if injected_reasoning {
                obj.remove("reasoning");
            }
        }

        let upstream_res = match state
            .http_client
            .post(openrouter_url)
            .header(header::AUTHORIZATION, &auth_header)
            .header(header::CONTENT_TYPE, "application/json")
            .header("HTTP-Referer", "http://localhost:3033")
            .header("X-Title", "OpenRouter Free Finder")
            .json(&payload)
            .send()
            .await
        {
            Ok(res) => res,
            Err(e) => {
                error!("Connection error with '{}': {}", target_model, e);
                mark_cooldown(&state, target_model, 30);
                continue;
            }
        };

        let status = upstream_res.status();

        // Context overflow: the prompt does not fit this model. Try a bigger one.
        if status == StatusCode::BAD_REQUEST && attempt + 1 < candidates.len() {
            let bytes = match upstream_res.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    error!("Body read error from '{}': {}", target_model, e);
                    continue;
                }
            };
            let lower = String::from_utf8_lossy(&bytes).to_lowercase();
            let context_overflow = lower.contains("context length")
                || lower.contains("context_length")
                || lower.contains("too many tokens")
                || lower.contains("maximum context");

            let modality_mismatch = !required_modalities.is_empty()
                && (lower.contains("does not support")
                    || lower.contains("modality")
                    || lower.contains("image_url")
                    || lower.contains("image input")
                    || lower.contains("invalid content")
                    || lower.contains("unsupported content"));

            if context_overflow || modality_mismatch {
                warn!(
                    "Model '{}' rejected the request ({}). Trying next candidate...",
                    target_model,
                    if context_overflow { "context too long" } else { "modality unsupported" }
                );
                continue;
            }

            return Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(bytes))
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }

        let is_throttled = status == StatusCode::TOO_MANY_REQUESTS
            || status == StatusCode::SERVICE_UNAVAILABLE
            || status == StatusCode::BAD_GATEWAY;

        if is_throttled {
            mark_cooldown(&state, target_model, COOLDOWN_SECS);
        }

        if is_throttled && attempt + 1 < candidates.len() {
            warn!(
                "Model '{}' returned {}. Cooling down {}s, trying next candidate...",
                target_model, status, COOLDOWN_SECS
            );
            last_response = Some(upstream_res);
            continue;
        }

        // Non-streaming: a 200 with no text content (all tokens spent on reasoning)
        // is a silent failure for OpenAI clients. Treat it as a failed attempt.
        let streaming = payload
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !streaming {
            let mut response_builder = Response::builder().status(status);

            for (k, v) in upstream_res.headers().iter() {
                if k == header::CONTENT_TYPE
                    || k == header::CACHE_CONTROL
                    || k.as_str().starts_with("x-")
                {
                    if let (Ok(hname), Ok(hval)) = (
                        HeaderName::from_bytes(k.as_str().as_bytes()),
                        HeaderValue::from_bytes(v.as_bytes()),
                    ) {
                        response_builder = response_builder.header(hname, hval);
                    }
                }
            }

            let bytes = match upstream_res.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    error!("Body read error from '{}': {}", target_model, e);
                    continue;
                }
            };

            let empty = serde_json::from_slice::<serde_json::Value>(&bytes)
                .map(|v| !has_text_content(&v))
                .unwrap_or(false);

            if empty && attempt + 1 < candidates.len() {
                warn!(
                    "Model '{}' returned 200 with empty content. Cooling down 30s, trying next candidate...",
                    target_model
                );
                mark_cooldown(&state, target_model, 30);
                continue;
            }

            return response_builder
                .body(Body::from(bytes))
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }

        let mut response_builder = Response::builder().status(status);

        for (k, v) in upstream_res.headers().iter() {
            if k == header::CONTENT_TYPE || k == header::CACHE_CONTROL || k.as_str().starts_with("x-") {
                if let (Ok(hname), Ok(hval)) = (
                    HeaderName::from_bytes(k.as_str().as_bytes()),
                    HeaderValue::from_bytes(v.as_bytes()),
                ) {
                    response_builder = response_builder.header(hname, hval);
                }
            }
        }

        let stream = upstream_res.bytes_stream().map_err(std::io::Error::other);
        let body = Body::from_stream(stream);

        return response_builder
            .body(body)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
    }

    if let Some(res) = last_response {
        let status = res.status();
        let response_builder = Response::builder().status(status);
        let stream = res.bytes_stream().map_err(std::io::Error::other);
        return response_builder
            .body(Body::from_stream(stream))
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
    }

    Err((
        StatusCode::BAD_GATEWAY,
        "All candidate models failed to respond.".to_string(),
    ))
}

async fn sync_models(state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let catalog_res = state
        .http_client
        .get("https://openrouter.ai/api/v1/models?sort=top-weekly&output_modalities=text")
        .send()
        .await?
        .json::<OpenRouterCatalogResponse>()
        .await?;

    let mut processed: Vec<ProcessedModel> = Vec::with_capacity(catalog_res.data.len());

    for (rank, raw) in catalog_res.data.into_iter().enumerate() {
        let prompt_p: f64 = raw
            .pricing
            .as_ref()
            .and_then(|p| p.prompt.as_deref())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        let completion_p: f64 = raw
            .pricing
            .as_ref()
            .and_then(|p| p.completion.as_deref())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        let is_free = raw.id.ends_with(":free")
            || (prompt_p == 0.0 && completion_p == 0.0 && !raw.id.starts_with("openrouter/"));

        let (in_mod, out_mod) = match raw.architecture {
            Some(arch) => (arch.input_modalities, arch.output_modalities),
            None => (vec!["text".to_string()], vec!["text".to_string()]),
        };

        let supports_tools = raw.supported_parameters.iter().any(|param| param == "tools");
        let supports_reasoning = raw
            .supported_parameters
            .iter()
            .any(|param| param == "reasoning");
        let reasoning_mandatory = raw
            .reasoning
            .as_ref()
            .map(|r| r.mandatory)
            .unwrap_or(false);

        processed.push(ProcessedModel {
            id: raw.id,
            name: raw.name,
            description: raw.description,
            context_length: raw.context_length.unwrap_or(0),
            created: raw.created.unwrap_or(0),
            prompt_price: prompt_p,
            completion_price: completion_p,
            is_free,
            output_modalities: out_mod,
            input_modalities: in_mod,
            top_weekly_rank: rank + 1,
            supports_tools,
            supports_reasoning,
            reasoning_mandatory,
        });
    }

    info!("Total processed models in memory: {}", processed.len());

    let mut lock = state.models.write().await;
    *lock = processed;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with(cooldown: &[&str]) -> AppState {
        let s = AppState::default();
        for m in cooldown {
            mark_cooldown(&s, m, 60);
        }
        s
    }

    fn candidates() -> Vec<String> {
        vec!["a:free".into(), "b:free".into(), "c:free".into()]
    }

    #[test]
    fn keeps_all_when_nothing_cooling() {
        let s = state_with(&[]);
        assert_eq!(filter_cooling(&s, candidates()).len(), 3);
    }

    #[test]
    fn skips_cooling_model() {
        let s = state_with(&["a:free"]);
        let out = filter_cooling(&s, candidates());
        assert_eq!(out, vec!["b:free".to_string(), "c:free".to_string()]);
    }

    #[test]
    fn falls_back_to_full_list_when_all_cooling() {
        let s = state_with(&["a:free", "b:free", "c:free"]);
        assert_eq!(filter_cooling(&s, candidates()).len(), 3);
    }

    #[test]
    fn expired_cooldown_is_forgotten() {
        let s = AppState::default();
        s.cooldown.lock().unwrap().insert(
            "a:free".to_string(),
            Instant::now() - Duration::from_secs(1),
        );
        assert!(!is_cooling(&s, "a:free"));
    }

    #[test]
    fn detects_empty_content() {
        let null = serde_json::json!({"choices":[{"message":{"content":null}}]});
        let blank = serde_json::json!({"choices":[{"message":{"content":"  "}}]});
        let ok = serde_json::json!({"choices":[{"message":{"content":"Hi"}}]});
        let parts = serde_json::json!({"choices":[{"message":{"content":[{"type":"text","text":"x"}]}}]});
        assert!(!has_text_content(&null));
        assert!(!has_text_content(&blank));
        assert!(has_text_content(&ok));
        assert!(has_text_content(&parts));
        assert!(has_text_content(&serde_json::json!({"error":"nope"})));
    }
}



#[cfg(test)]
mod context_tests {
    use super::*;
    use std::collections::HashMap;

    fn map(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn estimates_prompt_size_from_messages() {
        let p = serde_json::json!({"messages":[{"role":"user","content":"a".repeat(400)}]});
        let est = estimate_prompt_tokens(&p);
        assert!(est >= 100 && est <= 140, "got {}", est);
    }

    #[test]
    fn drops_candidates_too_small() {
        let c = vec!["small".to_string(), "big".to_string()];
        let m = map(&[("small", 8_000), ("big", 200_000)]);
        assert_eq!(filter_by_context(c, &m, 50_000), vec!["big".to_string()]);
    }

    #[test]
    fn keeps_all_when_everything_fits() {
        let c = vec!["a".to_string(), "b".to_string()];
        let m = map(&[("a", 128_000), ("b", 200_000)]);
        assert_eq!(filter_by_context(c, &m, 10_000).len(), 2);
    }

    #[test]
    fn picks_largest_when_nothing_fits() {
        let c = vec!["a".to_string(), "b".to_string()];
        let m = map(&[("a", 8_000), ("b", 32_000)]);
        assert_eq!(filter_by_context(c, &m, 500_000), vec!["b".to_string()]);
    }

    #[test]
    fn unknown_context_passes() {
        let c = vec!["unknown".to_string()];
        let m = map(&[]);
        assert_eq!(filter_by_context(c, &m, 999_999), vec!["unknown".to_string()]);
    }
}

#[cfg(test)]
mod modality_tests {
    use super::*;
    use std::collections::HashMap;

    fn map(pairs: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    #[test]
    fn detects_image_parts() {
        let p = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"text","text":"what is this?"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,xx"}}
        ]}]});
        assert_eq!(required_input_modalities(&p), vec!["image".to_string()]);
    }

    #[test]
    fn detects_audio_and_dedups() {
        let p = serde_json::json!({"messages":[
            {"role":"user","content":[{"type":"input_audio","input_audio":{"data":"xx"}}]},
            {"role":"user","content":[{"type":"audio","audio":{"data":"yy"}}]}
        ]});
        assert_eq!(required_input_modalities(&p), vec!["audio".to_string()]);
    }

    #[test]
    fn plain_text_needs_nothing() {
        let p = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(required_input_modalities(&p).is_empty());
    }

    #[test]
    fn keeps_only_vision_capable_candidates() {
        let c = vec!["text-only".to_string(), "vision".to_string()];
        let m = map(&[("text-only", &["text"]), ("vision", &["text", "image"])]);
        let req = vec!["image".to_string()];
        assert_eq!(
            filter_by_modality(c, &m, &req, &[]),
            vec!["vision".to_string()]
        );
    }

    #[test]
    fn falls_back_to_catalog_when_route_has_no_vision_model() {
        let c = vec!["cohere/north-mini-code:free".to_string()];
        let m = map(&[("cohere/north-mini-code:free", &["text"])]);
        let fb = vec!["stealth/crouton-alpha".to_string()];
        let req = vec!["image".to_string()];
        assert_eq!(filter_by_modality(c, &m, &req, &fb), fb);
    }

    #[test]
    fn unknown_model_passes_through() {
        let c = vec!["not-in-catalog".to_string()];
        let req = vec!["image".to_string()];
        assert_eq!(filter_by_modality(c, &map(&[]), &req, &[]), vec!["not-in-catalog".to_string()]);
    }
}
