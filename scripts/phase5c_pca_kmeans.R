#Composer through a Cursor IDE guided the development of this script

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(purrr)
  library(ggplot2)
  library(cluster)
  library(reticulate)
})


# Configuration

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)

if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
  script_dir <- dirname(script_path)
} else {
  script_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

if (basename(script_dir) == "scripts") {
  base_dir <- dirname(script_dir)
} else if (dir.exists(file.path(script_dir, "scripts"))) {
  base_dir <- script_dir
} else {
  stop("Could not determine project root. Run from project root or scripts/ directory.")
}

#output directory
results_dir <- file.path(base_dir, "data", "05_results")
tag <- Sys.getenv("STREAM_C_TAG", unset = "v14")
out_dir <- file.path(base_dir, "data", "06_evaluation", "stream_c_r", tag)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

#input files
n_eval <- as.integer(Sys.getenv("STREAM_C_N_EVAL", unset = "250"))
baseline_path <- file.path(results_dir, sprintf("baseline_results_%d_local_ollama_%s.jsonl", n_eval, tag))
hybrid_path <- file.path(results_dir, sprintf("hybrid_rag_results_%d_local_ollama_%s.jsonl", n_eval, tag))

cat("Stream C tag:", tag, "\n")
cat("Baseline input:", baseline_path, "\n")
cat("Hybrid input:", hybrid_path, "\n")
cat("Output dir:", out_dir, "\n")

#embedding model
embedding_model <- "sentence-transformers/all-mpnet-base-v2"

#k-means parameters
k_min <- 2
k_max <- 8
strat_col <- Sys.getenv("STREAM_C_STRAT_COL", unset = "category")
expected_segments <- as.integer(Sys.getenv("STREAM_C_EXPECTED_SEGMENTS", unset = "4"))
selected_k <- as.integer(Sys.getenv("STREAM_C_SELECTED_K", unset = "6"))

if (!file.exists(baseline_path) || !file.exists(hybrid_path)) {
  stop(sprintf("Missing baseline or hybrid %s JSONL files.", tag))
}

python_bin <- file.path(base_dir, "venv311", "Scripts", "python.exe")
if (file.exists(python_bin)) {
  use_python(python_bin, required = TRUE)
}
if (!py_module_available("sentence_transformers")) {
  stop(
    paste(
      "Python module 'sentence_transformers' is missing in the selected environment.",
      "Install with: .\\venv311\\Scripts\\python.exe -m pip install sentence-transformers"
    )
  )
}


# Helpers
read_jsonl <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  lines <- lines[nzchar(trimws(lines))]
  bind_rows(lapply(lines, fromJSON))
}

extract_answer_only <- function(txt) {
  if (is.na(txt) || !nzchar(txt)) return("")
  lines <- unlist(strsplit(txt, "\n"))
  lines <- trimws(lines)
  lines <- lines[nzchar(lines)]
  ans_line <- lines[str_detect(tolower(lines), "^answer:")]
  if (length(ans_line) > 0) {
    return(trimws(sub("^Answer:\\s*", "", ans_line[1], ignore.case = TRUE)))
  }
  txt
}

local_embed <- function(texts, model = "sentence-transformers/all-mpnet-base-v2", batch_size = 32L, max_chars = 24000L) {
  st <- import("sentence_transformers")
  model_obj <- st$SentenceTransformer(model)

  # Keep text sizes bounded for memory/performance.
  texts <- vapply(texts, function(x) {
    if (is.na(x) || !nzchar(x)) {
      return(" ")
    }
    x <- as.character(x)
    if (nchar(x) > max_chars) {
      substr(x, 1L, max_chars)
    } else {
      x
    }
  }, character(1), USE.NAMES = FALSE)

  emb <- model_obj$encode(
    as.list(texts),
    batch_size = as.integer(batch_size),
    show_progress_bar = TRUE,
    convert_to_numpy = TRUE,
    normalize_embeddings = FALSE
  )
  emb_r <- py_to_r(emb)
  if (is.null(dim(emb_r))) {
    matrix(emb_r, nrow = 1)
  } else {
    emb_r
  }
}

cosine_rowwise <- function(a, b) {
  # a and b are matrices with same rows
  num <- rowSums(a * b)
  den <- sqrt(rowSums(a * a)) * sqrt(rowSums(b * b))
  ifelse(den == 0, 0, num / den)
}

# -----------------------------
# Load and prepare data
# -----------------------------
baseline <- read_jsonl(baseline_path) |>
  mutate(dataset = "baseline")

hybrid <- read_jsonl(hybrid_path) |>
  mutate(dataset = "hybrid_rag")

if (nrow(baseline) != n_eval || nrow(hybrid) != n_eval) {
  stop(
    sprintf(
      "Expected %d rows each; got baseline=%d hybrid=%d",
      n_eval, nrow(baseline), nrow(hybrid)
    )
  )
}
b_ids <- sort(baseline$id)
h_ids <- sort(hybrid$id)
if (!identical(b_ids, h_ids)) {
  stop("Baseline and hybrid id sets differ.")
}
cat(sprintf("Loaded %d baseline + %d hybrid rows (ids aligned).\n", nrow(baseline), nrow(hybrid)))

ctx_map <- hybrid |>
  select(id, retrieved_context) |>
  distinct(id, .keep_all = TRUE)

baseline <- baseline |>
  left_join(ctx_map, by = "id", suffix = c("", "_hyb")) |>
  mutate(reference_context = coalesce(retrieved_context_hyb, "")) |>
  select(-retrieved_context_hyb)

hybrid <- hybrid |>
  mutate(reference_context = retrieved_context)

df <- bind_rows(baseline, hybrid) |>
  transmute(
    dataset,
    id,
    category,
    prompt = as.character(prompt),
    ai_response = as.character(ai_response),
    answer_text = map_chr(as.character(ai_response), extract_answer_only),
    reference_context = as.character(reference_context)
  )

if (!(strat_col %in% names(df))) {
  warning(
    paste0(
      "Requested stratification column '", strat_col,
      "' not found. Falling back to 'category'."
    )
  )
  strat_col <- "category"
}


# Embeddings

cat("Embedding answers, prompts, and contexts locally via sentence-transformers...\n")
answer_emb <- local_embed(df$answer_text, model = embedding_model)
question_emb <- local_embed(df$prompt, model = embedding_model)
context_emb <- local_embed(df$reference_context, model = embedding_model)

df$sim_answer_question <- cosine_rowwise(answer_emb, question_emb)
df$sim_answer_context <- cosine_rowwise(answer_emb, context_emb)
df$drift_q <- 1 - df$sim_answer_question
df$drift_ctx <- 1 - df$sim_answer_context


# PCA

pca <- prcomp(answer_emb, center = TRUE, scale. = TRUE)
df$pc1 <- pca$x[, 1]
df$pc2 <- pca$x[, 2]
df$pc3 <- pca$x[, 3]

explained <- (pca$sdev ^ 2) / sum(pca$sdev ^ 2)
explained_tbl <- tibble(
  pc = paste0("PC", seq_along(explained)),
  explained_variance = explained
)


# KMeans selection

k_tbl <- map_dfr(k_min:k_max, function(k) {
  set.seed(42)
  km <- kmeans(answer_emb, centers = k, nstart = 20)
  sil <- silhouette(km$cluster, dist(answer_emb))
  tibble(
    k = k,
    silhouette = mean(sil[, "sil_width"]),
    tot_withinss = km$tot.withinss
  )
})

# Keep diagnostics visual-only; selected_k is a manual choice (default k=6).

if (is.na(selected_k) || !(selected_k %in% k_tbl$k)) {
  warning(
    paste0(
      "Selected k=", selected_k, " is invalid for tested range [",
      min(k_tbl$k), ", ", max(k_tbl$k), "]. Falling back to k=6."
    )
  )
  selected_k <- 6L
}
if (!(selected_k %in% k_tbl$k)) {
  selected_k <- as.integer(k_tbl$k[which.min(abs(k_tbl$k - selected_k))])
}

best_k <- selected_k

k_tbl <- k_tbl |>
  mutate(
    selected_for_analysis = k == best_k
  )

set.seed(42)
km_final <- kmeans(answer_emb, centers = best_k, nstart = 20)
df$cluster <- km_final$cluster - 1

# Normalize cluster numbering to 0-5 for consistent reporting and visualization
segment_count <- df |>
  distinct(.data[[strat_col]]) |>
  nrow()
if (!is.na(expected_segments) && expected_segments > 0 && segment_count != expected_segments) {
  warning(
    paste0(
      "Observed ", segment_count, " segments in '", strat_col,
      "' but expected ", expected_segments, "."
    )
  )
}


# Paired statistics baseline vs hybrid
paired <- df |>
  select(id, dataset, drift_q, drift_ctx) |>
  pivot_wider(names_from = dataset, values_from = c(drift_q, drift_ctx))

diff_q <- paired$drift_q_hybrid_rag - paired$drift_q_baseline
diff_ctx <- paired$drift_ctx_hybrid_rag - paired$drift_ctx_baseline

stats_tbl <- bind_rows(
  tibble(
    metric = "drift_q",
    mean_baseline = mean(paired$drift_q_baseline, na.rm = TRUE),
    mean_hybrid = mean(paired$drift_q_hybrid_rag, na.rm = TRUE),
    mean_diff_h_minus_b = mean(diff_q, na.rm = TRUE),
    shapiro_p = shapiro.test(diff_q)$p.value,
    ttest_p = t.test(paired$drift_q_hybrid_rag, paired$drift_q_baseline, paired = TRUE)$p.value,
    wilcoxon_p = wilcox.test(paired$drift_q_hybrid_rag, paired$drift_q_baseline, paired = TRUE, exact = FALSE)$p.value
  ),
  tibble(
    metric = "drift_ctx",
    mean_baseline = mean(paired$drift_ctx_baseline, na.rm = TRUE),
    mean_hybrid = mean(paired$drift_ctx_hybrid_rag, na.rm = TRUE),
    mean_diff_h_minus_b = mean(diff_ctx, na.rm = TRUE),
    shapiro_p = shapiro.test(diff_ctx)$p.value,
    ttest_p = t.test(paired$drift_ctx_hybrid_rag, paired$drift_ctx_baseline, paired = TRUE)$p.value,
    wilcoxon_p = wilcox.test(paired$drift_ctx_hybrid_rag, paired$drift_ctx_baseline, paired = TRUE, exact = FALSE)$p.value
  )
)


# Summaries

overall_summary <- df |>
  group_by(dataset) |>
  summarise(
    rows = n(),
    mean_drift_q = mean(drift_q, na.rm = TRUE),
    mean_drift_ctx = mean(drift_ctx, na.rm = TRUE),
    mean_sim_q = mean(sim_answer_question, na.rm = TRUE),
    mean_sim_ctx = mean(sim_answer_context, na.rm = TRUE),
    .groups = "drop"
  )

cluster_mix <- df |>
  count(cluster, dataset) |>
  group_by(cluster) |>
  mutate(cluster_pct = 100 * n / sum(n)) |>
  ungroup()


# Plots

plot_dataset <- ggplot(df, aes(x = pc1, y = pc2, color = dataset)) +
  geom_point(alpha = 0.8, size = 2) +
  labs(
    title = "Stream C PCA by Dataset",
    x = "PC1",
    y = "PC2"
  ) +
  theme_minimal(base_size = 12)

plot_cluster <- ggplot(df, aes(x = pc1, y = pc2, color = as.factor(cluster))) +
  geom_point(alpha = 0.8, size = 2) +
  scale_color_discrete(
    name = "Cluster",
    labels = c("0", "1", "2", "3", "4", "5")
  ) +
  labs(
    title = paste0("Stream C PCA with KMeans Clusters (k=", best_k, ")"),
    x = "PC1",
    y = "PC2"
  ) +
  theme_minimal(base_size = 12)

plot_k <- ggplot(k_tbl, aes(x = k, y = silhouette)) +
  geom_line() +
  geom_point(size = 2) +
  scale_x_continuous(breaks = k_min:k_max) +
  labs(
    title = "K selection via silhouette",
    x = "k",
    y = "Mean silhouette"
  ) +
  theme_minimal(base_size = 12)

plot_elbow <- ggplot(k_tbl, aes(x = k, y = tot_withinss)) +
  geom_line() +
  geom_point(size = 2) +
  scale_x_continuous(breaks = k_min:k_max) +
  labs(
    title = "K selection via elbow",
    x = "k",
    y = "Total within-cluster sum of squares"
  ) +
  theme_minimal(base_size = 12)

plot_category_pca <- ggplot(df, aes(x = pc1, y = pc2, color = category, shape = dataset)) +
  geom_point(alpha = 0.75, size = 2) +
  labs(
    title = "Stream C PCA by Category",
    x = "PC1",
    y = "PC2",
    color = "Category",
    shape = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  guides(
    color = guide_legend(order = 1, nrow = 1),
    shape = guide_legend(order = 2, nrow = 1)
  ) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.spacing.y = grid::unit(4, "pt")
  )

# Stratified drift comparison by category (hybrid - baseline).
cat_means <- df |>
  group_by(category, dataset) |>
  summarise(
    mean_drift_q = mean(drift_q, na.rm = TRUE),
    mean_drift_ctx = mean(drift_ctx, na.rm = TRUE),
    .groups = "drop"
  )

cat_delta <- cat_means |>
  pivot_wider(
    names_from = dataset,
    values_from = c(mean_drift_q, mean_drift_ctx)
  ) |>
  mutate(
    delta_drift_q = mean_drift_q_hybrid_rag - mean_drift_q_baseline,
    delta_drift_ctx = mean_drift_ctx_hybrid_rag - mean_drift_ctx_baseline
  ) |>
  select(category, delta_drift_q, delta_drift_ctx) |>
  pivot_longer(
    cols = c(delta_drift_q, delta_drift_ctx),
    names_to = "metric",
    values_to = "delta"
  )

plot_category_delta <- ggplot(cat_delta, aes(x = category, y = delta, fill = metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
  labs(
    title = "Category-Level Drift Delta (Hybrid - Baseline)",
    x = NULL,
    y = "Delta (lower is better)",
    fill = "Metric"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Distribution-level comparison to complement means.
dist_long <- df |>
  select(dataset, drift_q, drift_ctx) |>
  pivot_longer(
    cols = c(drift_q, drift_ctx),
    names_to = "metric",
    values_to = "value"
  )

#plot drift distribution by dataset
plot_drift_distribution <- ggplot(dist_long, aes(x = dataset, y = value, fill = dataset)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  facet_wrap(~metric, nrow = 1, scales = "free_y") +
  labs(
    title = "Drift Distribution by Dataset",
    x = NULL,
    y = "Drift"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

# Cluster composition as stacked percentage bars for discussion.
plot_cluster_mix <- ggplot(cluster_mix, aes(x = factor(cluster), y = cluster_pct, fill = dataset)) +
  geom_col(position = "stack", width = 0.7) +
  scale_x_discrete(labels = c("0", "1", "2", "3", "4", "5")) +
  labs(
    title = "Cluster Composition by Dataset",
    x = "Cluster",
    y = "Share within cluster (%)",
    fill = "Dataset"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "stream_c_pca_by_dataset.png"), plot_dataset, width = 8, height = 6, dpi = 300)
ggsave(file.path(out_dir, "stream_c_pca_by_cluster.png"), plot_cluster, width = 8, height = 6, dpi = 300)
ggsave(file.path(out_dir, "stream_c_pca_by_category.png"), plot_category_pca, width = 10, height = 7, dpi = 300)
ggsave(file.path(out_dir, "stream_c_k_selection_silhouette.png"), plot_k, width = 7, height = 4.5, dpi = 300)
ggsave(file.path(out_dir, "stream_c_k_selection_elbow.png"), plot_elbow, width = 7, height = 4.5, dpi = 300)
ggsave(file.path(out_dir, "stream_c_category_drift_delta.png"), plot_category_delta, width = 9, height = 4.8, dpi = 300)
ggsave(file.path(out_dir, "stream_c_drift_distribution.png"), plot_drift_distribution, width = 8.5, height = 4.8, dpi = 300)
ggsave(file.path(out_dir, "stream_c_cluster_composition_stacked.png"), plot_cluster_mix, width = 8, height = 4.8, dpi = 300)


# Save outputs
write_csv(df, file.path(out_dir, "stream_c_per_row_scores.csv"))
write_csv(explained_tbl, file.path(out_dir, "stream_c_pca_explained_variance.csv"))
write_csv(k_tbl, file.path(out_dir, "stream_c_k_selection.csv"))
write_csv(overall_summary, file.path(out_dir, "stream_c_overall_summary.csv"))
write_csv(cluster_mix, file.path(out_dir, "stream_c_cluster_mix.csv"))
write_csv(cat_delta, file.path(out_dir, "stream_c_category_drift_delta.csv"))
write_csv(stats_tbl, file.path(out_dir, "stream_c_paired_stats.csv"))

print(overall_summary)
print(stats_tbl)
