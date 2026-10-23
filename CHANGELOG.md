## v0.2.0 (2026-10-22)

### Feat

- **chat**: passed through reasoning_content from thinking models
- **health**: added a plutus check reporting whether the forecast model loads
- added a dev server command with reload
- **embeddings**: passed dimensions and encoding_format through to the backend
- **experiments**: logged a last-value baseline beside the plutus test loss
- **logging**: returned OpenAI-shaped JSON with the request id for unhandled errors
- **logging**: accepted and echoed X-Request-ID on every response
- **responses**: rejected previous_response_id with a 400 instead of ignoring it
- **health**: returned 503 from /health when unhealthy and added a Docker HEALTHCHECK
- **images**: added support for image generation via swarmui
- **responses**: added support for llama.cpp as the llm backend
- **responses**: added support for the response creation api
- added structured completion output support
- added github ci and addressed linting errors
- restructured slices and added bruno collection
- added health check endpoint
- added token usage to the completions endpoint
- **ouranos-ml-10**: Chat Completions
- **ouranos-ml-8**: Added embeddings endpoint
- **ouranos-ml-9**: Added list models feature

### Fix

- **chat**: opened the upstream stream before the response starts
- **responses**: dropped unmodeled output items instead of failing validation
- **images**: matched SwarmUI model-not-found errors by substring
- **images**: omitted size when no width or height is known
- **experiments**: hardened the genetic tuner and plutus outputs
- **experiments**: made the plutus symbol limit opt-in
- **llm**: bounded LLM client timeout and disabled silent retries
- **plutus**: returned 422 for wrong sequence lengths
- **chat**: rejected empty messages in chat completions with 422
- **plutus**: mapped predictions back through feature_fields
- **health**: logged health check failures once per outage without tracebacks
- **responses**: mirror json_schema text format as response_format for llama.cpp
- pin setup-uv to v7 (v10 tag does not exist)
- upgrade dependencies and Actions to resolve CI lint and Node 20 deprecation
- readded middleware linting suppression

### Refactor

- **experiments**: made run abstract and typed the time series dataset
- **chat**: defaulted Usage to zero tokens instead of rebuilding it inline
- **health**: merged the status enums and gave every check one async signature
- audited codebase and various improvements
- configured opentelemetry traces and metrics
- **logging**: added structured console and loki logging
- moved inference logic into shared location
- added unit tests
- updated default settings to ollama port
- made llm provider more generic
- added common ai assistant files to the gitignore
- **ouranos-ml-12**: converted the lmstudio client to the async version
- **ouranos-ml-3**: Setup uv, ruff, and ty

### Perf

- **experiments**: cached genetic tuner fitness per param set
