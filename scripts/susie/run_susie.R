#!/usr/bin/env Rscript

# run_susie.R - Command-line wrapper for SuSiE fine-mapping
#
# Usage:
#   Rscript run_susie.R --input <file> --input-type <sumstats|individual> \
#                       --ld <file|compute> --output <dir> [options]
#
# Outputs JSON to stdout with results paths and summary

suppressPackageStartupMessages({
  library(optparse)
  library(susieR)
  library(data.table)
  library(jsonlite)
})

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL,
              help = "Input file (summary stats CSV or PLINK prefix)"),
  make_option(c("-t", "--input-type"), type = "character", default = "sumstats",
              help = "Input type: 'sumstats' or 'individual' [default: %default]"),
  make_option(c("--ld"), type = "character", default = NULL,
              help = "LD matrix file (.npy, .txt, .rds) or 'compute'"),
  make_option(c("--ld-ref"), type = "character", default = NULL,
              help = "Reference panel for LD computation (e.g., 1000G_EUR)"),
  make_option(c("-o", "--output"), type = "character", default = "output",
              help = "Output directory [default: %default]"),
  make_option(c("-L", "--L"), type = "integer", default = 10,
              help = "Maximum number of causal variants [default: %default]"),
  make_option(c("-c", "--coverage"), type = "double", default = 0.95,
              help = "Credible set coverage [default: %default]"),
  make_option(c("--min-abs-corr"), type = "double", default = 0.5,
              help = "Minimum absolute correlation for CS [default: %default]"),
  make_option(c("-n", "--n"), type = "integer", default = NULL,
              help = "Sample size (required for summary stats)"),
  make_option(c("-v", "--verbose"), action = "store_true", default = FALSE,
              help = "Print detailed progress")
)

parser <- OptionParser(usage = "%prog [options]", option_list = option_list)
args <- parse_args(parser)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

log_msg <- function(msg, verbose = args$verbose) {
  if (verbose) {
    message(sprintf("[%s] %s", Sys.time(), msg))
  }
}

output_json <- function(result) {
  cat(toJSON(result, auto_unbox = TRUE, pretty = TRUE))
  cat("\n")
}

error_exit <- function(msg) {
  result <- list(
    success = FALSE,
    error = msg,
    files = list()
  )
  output_json(result)
  quit(status = 1)
}

load_ld_matrix <- function(ld_path) {
  log_msg(sprintf("Loading LD matrix from: %s", ld_path))

  ext <- tools::file_ext(ld_path)

  if (ext == "rds") {
    R <- readRDS(ld_path)
  } else if (ext == "npy") {
    # Requires reticulate for .npy files
    if (!requireNamespace("reticulate", quietly = TRUE)) {
      error_exit("Package 'reticulate' required to read .npy files")
    }
    np <- reticulate::import("numpy")
    R <- np$load(ld_path)
  } else {
    # Assume whitespace-delimited text
    R <- as.matrix(fread(ld_path, header = FALSE))
  }

  # Validate LD matrix
  if (nrow(R) != ncol(R)) {
    error_exit(sprintf("LD matrix is not square: %d x %d", nrow(R), ncol(R)))
  }

  return(R)
}

load_summary_stats <- function(input_path) {
  log_msg(sprintf("Loading summary statistics from: %s", input_path))

  df <- fread(input_path)

  # Standardize column names (case-insensitive)
  names(df) <- toupper(names(df))

  # Check for required columns
  required <- c("SNP", "CHR", "BP")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    error_exit(sprintf("Missing required columns: %s", paste(missing, collapse = ", ")))
  }

  # Check for effect size columns
  has_beta_se <- all(c("BETA", "SE") %in% names(df))
  has_z <- "Z" %in% names(df)

  if (!has_beta_se && !has_z) {
    error_exit("Summary stats must have either BETA+SE or Z columns")
  }

  # Calculate Z if not present
  if (!has_z && has_beta_se) {
    df$Z <- df$BETA / df$SE
  }

  return(df)
}

run_susie_sumstats <- function(df, R, n, L, coverage, min_abs_corr) {
  log_msg(sprintf("Running SuSiE RSS with L=%d, coverage=%.2f", L, coverage))

  # Validate dimensions
  if (nrow(df) != nrow(R)) {
    error_exit(sprintf(
      "Dimension mismatch: %d variants in summary stats, %d in LD matrix",
      nrow(df), nrow(R)
    ))
  }

  # Run SuSiE RSS
  fit <- susie_rss(
    z = df$Z,
    R = R,
    n = n,
    L = L,
    coverage = coverage,
    min_abs_corr = min_abs_corr,
    verbose = args$verbose
  )

  return(fit)
}

run_susie_individual <- function(input_prefix, L, coverage, min_abs_corr) {
  log_msg(sprintf("Running SuSiE with individual-level data: %s", input_prefix))

  # This would require reading PLINK files
  # For now, placeholder - would use snpStats or similar
  error_exit("Individual-level data input not yet implemented. Use summary statistics.")
}

extract_results <- function(fit, df) {
  # Extract PIPs
  pip <- susie_get_pip(fit)

  # Extract credible sets
  cs <- fit$sets$cs
  cs_coverage <- fit$sets$coverage

  # Build results data frame
  results <- data.frame(
    SNP = df$SNP,
    CHR = df$CHR,
    BP = df$BP,
    PIP = pip,
    CS = NA_integer_,
    CS_COVERAGE = NA_real_,
    stringsAsFactors = FALSE
  )

  # Add alleles if present
  if ("A1" %in% names(df)) results$A1 <- df$A1
  if ("A2" %in% names(df)) results$A2 <- df$A2
  if ("BETA" %in% names(df)) results$BETA <- df$BETA
  if ("SE" %in% names(df)) results$SE <- df$SE
  if ("P" %in% names(df)) results$P <- df$P

  # Assign credible set membership
  if (!is.null(cs)) {
    for (i in seq_along(cs)) {
      idx <- cs[[i]]
      results$CS[idx] <- i
      results$CS_COVERAGE[idx] <- cs_coverage[i]
    }
  }

  # Sort by PIP descending
  results <- results[order(-results$PIP), ]

  return(results)
}

build_summary <- function(fit, results) {
  cs <- fit$sets$cs

  credible_sets <- list()
  if (!is.null(cs)) {
    for (i in seq_along(cs)) {
      idx <- cs[[i]]
      cs_results <- results[results$CS == i & !is.na(results$CS), ]
      lead_idx <- which.max(cs_results$PIP)

      credible_sets[[i]] <- list(
        cs_id = i,
        coverage = fit$sets$coverage[i],
        n_variants = length(idx),
        lead_variant = cs_results$SNP[lead_idx],
        lead_pip = cs_results$PIP[lead_idx]
      )
    }
  }

  summary <- list(
    n_variants = nrow(results),
    n_credible_sets = length(cs),
    credible_sets = credible_sets,
    parameters = list(
      L = args$L,
      coverage = args$coverage,
      min_abs_corr = args$`min-abs-corr`
    ),
    converged = fit$converged
  )

  return(summary)
}

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

main <- function() {
  # Validate required arguments
  if (is.null(args$input)) {
    error_exit("--input is required")
  }

  if (is.null(args$ld) && args$`input-type` == "sumstats") {
    error_exit("--ld is required for summary statistics input")
  }

  if (is.null(args$n) && args$`input-type` == "sumstats") {
    error_exit("--n (sample size) is required for summary statistics input")
  }

  # Create output directory
  dir.create(args$output, showWarnings = FALSE, recursive = TRUE)

  # Run based on input type
  if (args$`input-type` == "sumstats") {
    # Load data
    df <- load_summary_stats(args$input)
    R <- load_ld_matrix(args$ld)

    # Run SuSiE
    fit <- run_susie_sumstats(
      df = df,
      R = R,
      n = args$n,
      L = args$L,
      coverage = args$coverage,
      min_abs_corr = args$`min-abs-corr`
    )
  } else if (args$`input-type` == "individual") {
    fit <- run_susie_individual(
      input_prefix = args$input,
      L = args$L,
      coverage = args$coverage,
      min_abs_corr = args$`min-abs-corr`
    )
    df <- NULL  # Would be loaded in the function
  } else {
    error_exit(sprintf("Unknown input type: %s", args$`input-type`))
  }

  # Extract and save results
  log_msg("Extracting results...")
  results <- extract_results(fit, df)
  summary <- build_summary(fit, results)

  # Define output paths
  results_csv <- file.path(args$output, "susie_results.csv")
  summary_json <- file.path(args$output, "susie_summary.json")
  fit_rds <- file.path(args$output, "susie_fit.rds")

  # Write outputs
  log_msg(sprintf("Writing results to: %s", results_csv))
  fwrite(results, results_csv)

  log_msg(sprintf("Writing summary to: %s", summary_json))
  write(toJSON(summary, auto_unbox = TRUE, pretty = TRUE), summary_json)

  log_msg(sprintf("Writing fit object to: %s", fit_rds))
  saveRDS(fit, fit_rds)

  # Build lead variants list
  lead_variants <- character()
  if (!is.null(fit$sets$cs)) {
    for (i in seq_along(fit$sets$cs)) {
      cs_results <- results[results$CS == i & !is.na(results$CS), ]
      lead_variants <- c(lead_variants, cs_results$SNP[which.max(cs_results$PIP)])
    }
  }

  # Output final JSON
  output <- list(
    success = TRUE,
    n_credible_sets = length(fit$sets$cs),
    lead_variants = lead_variants,
    files = list(
      results_csv = results_csv,
      summary_json = summary_json,
      rds = fit_rds
    ),
    warnings = if (length(fit$sets$cs) == 0) list("No credible sets identified") else list()
  )

  output_json(output)
}

# Run main
tryCatch(
  main(),
  error = function(e) {
    error_exit(conditionMessage(e))
  }
)
