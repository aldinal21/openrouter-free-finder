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
use std::{sync::Arc, time::Duration};
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
}

#[derive(Debug, Default)]
pub struct AppState {
    pub models: RwLock<Vec<ProcessedModel>>,
    pub http_client: reqwest::Client,
    pub api_key: Option<String>,
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
            let q = FilterQuery {
                variant: Some("free".to_string()),
                output: Some("text".to_string()),
                sort: Some("top-weekly".to_string()),
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
        "free:top-2" => {
            let q = FilterQuery {
                variant: Some("free".to_string()),
                output: Some("text".to_string()),
                sort: Some("top-weekly".to_string()),
                max_price: None,
                tools: None,
                limit: Some(4),
            };
            let list = filter_and_sort_models(all_models, &q);
            let mut res: Vec<String> = list.into_iter().skip(1).map(|m| m.id).collect();
            if res.is_empty() {
                res.push("nvidia/nemotron-3-ultra-550b-a55b:free".to_string());
            }
            res
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
    drop(all_models);

    let openrouter_url = "https://openrouter.ai/api/v1/chat/completions";

    let mut last_response = None;

    for (attempt, target_model) in candidates.iter().enumerate() {
        if attempt > 0 {
            warn!("⚠️ Fallback attempt #{}: switching to model '{}'", attempt, target_model);
        } else {
            info!("Routing: '{}' -> '{}'", requested_model, target_model);
        }

        if let Some(obj) = payload.as_object_mut() {
            obj.insert("model".to_string(), serde_json::Value::String(target_model.clone()));
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
                continue;
            }
        };

        let status = upstream_res.status();

        if (status == StatusCode::TOO_MANY_REQUESTS
            || status == StatusCode::SERVICE_UNAVAILABLE
            || status == StatusCode::BAD_GATEWAY)
            && attempt + 1 < candidates.len()
        {
            warn!("Model '{}' returned status {}. Trying next fallback candidate...", target_model, status);
            last_response = Some(upstream_res);
            continue;
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
        });
    }

    info!("Total processed models in memory: {}", processed.len());

    let mut lock = state.models.write().await;
    *lock = processed;

    Ok(())
}
