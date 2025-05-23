# System Patterns

## Architecture

- Clean Architecture pattern with layers:
  - Domain (core business logic)
  - Application (use case implementation)
  - Infrastructure (external integrations)
  - API (interface layer)

## Key Components

1. API Routes - Handle HTTP request/response
2. Application Queries - Business logic
3. Domain Models - Core data structures
4. Infrastructure - Model integrations

## Data Flow

Request -> API -> Application -> Domain -> Infrastructure -> Model -> Response

## Design Patterns

- Repository pattern for data access
- Factory pattern for model instantiation
- Strategy pattern for different model backends
