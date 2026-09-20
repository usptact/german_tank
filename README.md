# german_tank

Bayesian inference for the [German tank problem](https://en.wikipedia.org/wiki/German_tank_problem) in Rust, using the [Fugue](https://fugue.run) probabilistic programming library (`fugue-ppl`).

Observed tank serial numbers are read from stdin; posterior summaries for the total population size `N` (and capture rate `λ`) are written to stdout.

## Model

```text
N   ~ DiscreteUniform(max(y), 10_000)
λ   ~ HalfNormal(σ = 10)          # |Normal(0, 10)| in Fugue
K   ~ Poisson(N · λ)              # observed capture count
yᵢ  ~ DiscreteUniform(1, N)       # observed serial numbers
```

Inference uses Fugue’s adaptive Metropolis–Hastings (`adaptive_mcmc_chain`).

## Requirements

- Rust 1.87+ (edition 2024; matches Fugue’s MSRV)

## Build

```bash
cargo build --release
```

## Usage

Pipe whitespace- or comma-separated serial numbers (`≥ 1`) on stdin:

```bash
echo '364 925 822 403 282 877 59 441 133 595 351 585 385 864 981 632 276 194 798 127' \
  | ./target/release/german_tank
```

Or from a file:

```bash
./target/release/german_tank < serials.txt
```

Example stdout:

```text
k=20
max_y=981
mvue=1029.0
samples=20000
warmup=5000
N_mean=1033.9 N_sd=56.4 N_median=1016 N_ci95=[982, 1192]
lam_mean=0.020230 lam_sd=0.004547 lam_median=0.019842 lam_ci95=[0.012314, 0.029992]
```

Invalid or empty input prints an error on stderr and exits with status `1`.

## Acknowledgments

This project is built on **[Fugue](https://fugue.run)** (`[fugue-ppl](https://crates.io/crates/fugue-ppl)`), a type-safe monadic probabilistic programming library for Rust by [Alex Nodeland](https://github.com/alexnodeland).

- Documentation: [fugue.run](https://fugue.run)
- Source: [github.com/alexnodeland/fugue](https://github.com/alexnodeland/fugue)
- API docs: [docs.rs/fugue-ppl](https://docs.rs/fugue-ppl)

Fugue is used here under its MIT license; see that project for copyright and license terms.

## License

This project is licensed under the [MIT License](LICENSE).
