//! German Tank problem — Bayesian inference with Fugue.
//!
//! Reads observed tank serial numbers from stdin (whitespace- or
//! comma-separated integers), runs MCMC, and writes posterior summaries
//! to stdout.
//!
//! Model:
//!
//! ```text
//! N   ~ DiscreteUniform(max(y), 10_000)
//! λ   ~ HalfNormal(σ = 10)          // |Normal(0, 10)| in Fugue
//! K   ~ Poisson(N · λ)              // observed capture count
//! yᵢ  ~ DiscreteUniform(1, N)       // observed serial numbers
//! ```
//!
//! Example:
//!
//! ```text
//! echo '364 925 822 403 282 877 59 441' | cargo run --release
//! ```

use fugue::inference::diagnostics::extract_i64_values;
use fugue::inference::mh::adaptive_mcmc_chain;
use fugue::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::{self, Read};
use std::process::ExitCode;

/// Upper bound on the prior for `N`.
const N_PRIOR_UPPER: i64 = 10_000;
/// HalfNormal scale for the capture-rate prior (`λ ~ HalfNormal(sd=10)`).
const LAM_SD: f64 = 10.0;
const N_SAMPLES: usize = 20_000;
const N_WARMUP: usize = 5_000;
const SEED: u64 = 42;

fn german_tank_model(y: &[i64]) -> Model<(i64, f64)> {
    let max_y = *y.iter().max().expect("at least one observation");
    let k = y.len() as u64;
    let y = y.to_vec();

    prob!(
        let n <- sample(
            addr!("N"),
            DiscreteUniform::new(max_y, N_PRIOR_UPPER).unwrap()
        );

        // HalfNormal(sd): |Z| for Z ~ Normal(0, sd). Fugue has no HalfNormal;
        // the absolute value of a zero-mean Normal is distributionally identical.
        let lam_raw <- sample(addr!("lam"), Normal::new(0.0, LAM_SD).unwrap());
        let lam = lam_raw.abs().max(1e-12);

        observe(addr!("nobs"), Poisson::new(n as f64 * lam).unwrap(), k);

        let _serials <- plate!(i in 0..y.len() => {
            observe(
                addr!("y", i),
                DiscreteUniform::new(1, n).unwrap(),
                y[i],
            )
        });

        pure((n, lam))
    )
}

fn read_serials_from_stdin() -> Result<Vec<i64>, String> {
    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("failed to read stdin: {e}"))?;

    let mut y = Vec::new();
    for token in buf.split(|c: char| c.is_whitespace() || c == ',') {
        if token.is_empty() {
            continue;
        }
        let v: i64 = token
            .parse()
            .map_err(|_| format!("invalid serial number: {token:?}"))?;
        if v < 1 {
            return Err(format!("serial numbers must be >= 1, got {v}"));
        }
        y.push(v);
    }

    if y.is_empty() {
        return Err("expected at least one serial number on stdin".into());
    }

    let max_y = *y.iter().max().unwrap();
    if max_y > N_PRIOR_UPPER {
        return Err(format!(
            "max(y)={max_y} exceeds prior upper bound {N_PRIOR_UPPER}"
        ));
    }

    Ok(y)
}

fn quantile_sorted(sorted: &[i64], q: f64) -> i64 {
    let n = sorted.len();
    if n == 0 {
        return 0;
    }
    let idx = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
    sorted[idx]
}

fn quantile_sorted_f64(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let idx = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
    sorted[idx]
}

fn mean_sd_i64(values: &[i64]) -> (f64, f64) {
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    (mean, var.sqrt())
}

fn mean_sd_f64(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    (mean, var.sqrt())
}

fn main() -> ExitCode {
    let y = match read_serials_from_stdin() {
        Ok(y) => y,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let k = y.len();
    let max_y = *y.iter().max().unwrap();
    let mvue = (max_y as f64) * (1.0 + 1.0 / k as f64) - 1.0;

    let mut rng = StdRng::seed_from_u64(SEED);
    let y_owned = y.clone();
    let samples = adaptive_mcmc_chain(
        &mut rng,
        move || german_tank_model(&y_owned),
        N_SAMPLES,
        N_WARMUP,
    );

    let traces: Vec<Trace> = samples.iter().map(|(_, t)| t.clone()).collect();
    let n_samples: Vec<i64> = extract_i64_values(&traces, &addr!("N"));
    let lam_samples: Vec<f64> = samples.iter().map(|((_, lam), _)| *lam).collect();

    if n_samples.is_empty() || lam_samples.is_empty() {
        eprintln!("error: MCMC produced no posterior samples");
        return ExitCode::FAILURE;
    }

    let mut sorted_n = n_samples.clone();
    sorted_n.sort_unstable();
    let (n_mean, n_sd) = mean_sd_i64(&n_samples);
    let n_median = quantile_sorted(&sorted_n, 0.50);
    let n_lo = quantile_sorted(&sorted_n, 0.025);
    let n_hi = quantile_sorted(&sorted_n, 0.975);

    let mut sorted_lam = lam_samples.clone();
    sorted_lam.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (lam_mean, lam_sd) = mean_sd_f64(&lam_samples);
    let lam_median = quantile_sorted_f64(&sorted_lam, 0.50);
    let lam_lo = quantile_sorted_f64(&sorted_lam, 0.025);
    let lam_hi = quantile_sorted_f64(&sorted_lam, 0.975);

    println!("k={}", k);
    println!("max_y={}", max_y);
    println!("mvue={:.1}", mvue);
    println!("samples={}", N_SAMPLES);
    println!("warmup={}", N_WARMUP);
    println!(
        "N_mean={:.1} N_sd={:.1} N_median={} N_ci95=[{}, {}]",
        n_mean, n_sd, n_median, n_lo, n_hi
    );
    println!(
        "lam_mean={:.6} lam_sd={:.6} lam_median={:.6} lam_ci95=[{:.6}, {:.6}]",
        lam_mean, lam_sd, lam_median, lam_lo, lam_hi
    );

    ExitCode::SUCCESS
}
