# Contributing

Thanks for your interest in improving this project.

## Development

```bash
cargo build
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

Smoke-test the binary:

```bash
cargo build --release
echo '10 20 30 40' | ./target/release/german_tank
```

## Pull requests

1. Keep changes focused and documented in `CHANGELOG.md` when user-visible.
2. Ensure CI passes (format, clippy, release build, stdin smoke test).
3. This project depends on [Fugue](https://fugue.run) (`fugue-ppl`); prefer Fugue idioms from that project’s docs and examples when extending the model or inference.
