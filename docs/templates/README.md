# Templates

Templates for issues and pull requests, shared with `ouranos-pantheon`.

| Template | Use for |
|----------|---------|
| [issues/feature.md](issues/feature.md) | New capability or behaviour change |
| [issues/bug.md](issues/bug.md) | Something broken |
| [issues/tech-debt.md](issues/tech-debt.md) | Cleanup with no user-facing change |
| [pull-request.md](pull-request.md) | Every pull request |

## Conventions

- **Issue titles:** `<Area> | <Title>`, where Area is a domain (`Chat`, `Responses`, `Embeddings`, `Images`,
  `Models`, `Plutus`, `Health`), `Experiments`, or `Service` for cross-cutting work.
- **Issue labels:** `bug` / `enhancement` where it applies, a priority (`P1`–`P3`), and a category (`ops`, `dx`,
  `observability`, `performance`, `documentation`) where one fits.
- **PR titles:** Conventional Commits, `<type>(<scope>): <summary>`.
- **Keep it short.** A few bullets per section; skip anything the diff or code already says. Older issues and PRs
  are too verbose to use as examples.
