# Majority-class baseline for binary classification data
#
# Usage:
#   Rscript baseline_metrics.R [train_csv] [test_csv] [target_column] [positive_class]
#
# Defaults are the Home Credit filenames and TARGET column. The test file is
# inspected only: it is evaluated when it includes the target column; otherwise
# the script reports that test-set accuracy and AUC are unavailable.
#
# Contents
# 1. Majority-class baseline
#    - read_target_column(), roc_auc(), evaluate_majority_baseline()
# 2. Cross-validated model comparison
#    - benchmark_models_cv(): logistic regression, random forest, and XGBoost
#    - optional upsampling, downsampling, and SMOTE
# 3. Randomized hyperparameter tuning
#    - tune_best_model_random_search(): 5,000-row subsample and 3-fold CV
# 4. Command-line baseline runner
#    - run_baseline_cli()

read_target_column <- function(path, target_column) {
  header <- names(utils::read.csv(
    path,
    nrows = 0,
    check.names = FALSE
  ))

  if (!target_column %in% header) {
    return(NULL)
  }

  # Read only the target column so this remains practical for wide CSV files.
  column_types <- rep("NULL", length(header))
  names(column_types) <- header
  column_types[target_column] <- "character"

  data <- utils::read.csv(
    path,
    colClasses = column_types,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  data[[target_column]]
}

roc_auc <- function(actual, score, positive_class) {
  is_positive <- actual == positive_class
  n_positive <- sum(is_positive)
  n_negative <- sum(!is_positive)

  if (n_positive == 0 || n_negative == 0) {
    return(NA_real_)
  }

  # Mann-Whitney form of ROC AUC; average ranks handle ties correctly.
  ranks <- rank(score, ties.method = "average")
  numerator <- sum(ranks[is_positive]) - as.numeric(n_positive) * (n_positive + 1) / 2
  denominator <- as.numeric(n_positive) * as.numeric(n_negative)
  numerator / denominator
}

evaluate_majority_baseline <- function(target, positive_class = NULL) {
  target <- trimws(as.character(target))
  target <- target[!is.na(target) & target != ""]
  class_counts <- sort(table(target), decreasing = TRUE)

  if (length(class_counts) != 2) {
    stop("The target must contain exactly two non-missing classes.")
  }

  majority_class <- names(class_counts)[1]
  if (is.null(positive_class)) {
    positive_class <- if ("1" %in% names(class_counts)) "1" else names(class_counts)[2]
  }
  if (!positive_class %in% names(class_counts)) {
    stop("positive_class is not present in the target column.")
  }

  # Every observation receives the same positive-class probability, so ROC AUC
  # must be 0.5. Calculating it explicitly makes the result reusable.
  constant_score <- rep(unname(class_counts[positive_class]) / sum(class_counts), length(target))
  baseline_accuracy <- mean(target == majority_class)
  baseline_auc <- roc_auc(target, constant_score, positive_class)

  list(
    observations = length(target),
    class_counts = class_counts,
    majority_class = majority_class,
    positive_class = positive_class,
    accuracy = baseline_accuracy,
    auc = baseline_auc
  )
}

print_baseline_results <- function(results, dataset_name) {
  cat("\n", dataset_name, " majority-class baseline\n", sep = "")
  cat("Observations: ", format(results$observations, big.mark = ","), "\n", sep = "")
  cat("Class counts:\n")
  print(results$class_counts)
  cat("Predicted class: ", results$majority_class, "\n", sep = "")
  cat("Accuracy: ", sprintf("%.4f%%", 100 * results$accuracy), "\n", sep = "")
  cat("ROC AUC: ", sprintf("%.3f", results$auc), "\n", sep = "")
}

# Cross-validated model comparison -----------------------------------------
#
# Install the required packages once, if necessary:
# install.packages(c("dials", "dplyr", "recipes", "rsample", "parsnip",
#                    "themis", "tune", "workflows", "yardstick", "glmnet",
#                    "ranger", "xgboost"))
#
# Notebook example:
# home_credit <- utils::read.csv("application_train.csv", check.names = FALSE)
# benchmark <- benchmark_models_cv(home_credit, target = "TARGET",
#                                  id_columns = "SK_ID_CURR", v = 5,
#                                  imbalance_methods = c("none", "upsample",
#                                                        "downsample", "smote"))
# benchmark$summary
# tuning <- tune_best_model_random_search(
#   home_credit, target = "TARGET", id_columns = "SK_ID_CURR",
#   benchmark_summary = benchmark$summary, tuning_rows = 5000, v = 3,
#   random_candidates = 20
# )
# tuning$best_parameters
# test_data <- utils::read.csv("application_test.csv", check.names = FALSE)
# submission <- create_kaggle_submission(
#   tuning$final_fit, test_data, id_column = "SK_ID_CURR",
#   output_path = "submission.csv"
# )
# submit_kaggle_submission("submission.csv", message = "Random-search tuned model")

require_packages <- function(packages, task) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Install these packages before ", task, ": ",
      paste(missing, collapse = ", "), "."
    )
  }
}

require_benchmark_packages <- function(imbalance_methods) {
  packages <- c("dplyr", "recipes", "rsample", "parsnip", "workflows",
                "yardstick", "ranger", "xgboost")
  if (any(imbalance_methods != "none")) {
    packages <- c(packages, "themis")
  }
  require_packages(packages, "running the cross-validation benchmark")
}

require_tuning_packages <- function(model_name, imbalance_method) {
  packages <- c("dials", "recipes", "rsample", "parsnip", "tune",
                "workflows", "yardstick")
  if (imbalance_method != "none") {
    packages <- c(packages, "themis")
  }
  if (startsWith(model_name, "Logistic regression")) {
    packages <- c(packages, "glmnet")
  } else if (model_name == "Random forest") {
    packages <- c(packages, "ranger")
  } else if (model_name == "XGBoost") {
    packages <- c(packages, "xgboost")
  }
  require_packages(packages, "tuning the selected model")
}

prepare_binary_data <- function(data, target, positive_class = "1", id_columns = character()) {
  if (!target %in% names(data)) {
    stop("Target column '", target, "' is not present.")
  }

  keep <- !is.na(data[[target]]) & as.character(data[[target]]) != ""
  data <- data[keep, , drop = FALSE]
  values <- as.character(data[[target]])
  classes <- sort(unique(values))

  if (length(classes) != 2) {
    stop("Cross-validation requires exactly two non-missing target classes.")
  }
  if (!positive_class %in% classes) {
    stop("positive_class must match one of: ", paste(classes, collapse = ", "))
  }

  negative_class <- setdiff(classes, positive_class)
  data[[target]] <- factor(values, levels = c(negative_class, positive_class))

  missing_ids <- setdiff(id_columns, names(data))
  if (length(missing_ids) > 0) {
    warning("Ignoring ID columns not found in the data: ", paste(missing_ids, collapse = ", "))
  }
  list(data = data, id_columns = intersect(id_columns, names(data)))
}

default_predictor_sets <- function(data, target, id_columns = character()) {
  available <- setdiff(names(data), c(target, id_columns))
  numeric_only <- available[vapply(data[available], is.numeric, logical(1))]

  if (length(numeric_only) == 0) {
    stop("No numeric predictors were found for the numeric-only logistic model.")
  }
  list(
    numeric_only = numeric_only,
    all_predictors = available
  )
}

add_class_balance_step <- function(recipe, method) {
  # These steps are baked only when a workflow is fitted on a CV analysis fold.
  # The assessment fold keeps the real default-rate distribution.
  switch(
    method,
    none = recipe,
    upsample = themis::step_upsample(recipe, recipes::all_outcomes(), over_ratio = 1),
    downsample = themis::step_downsample(recipe, recipes::all_outcomes(), under_ratio = 1),
    smote = themis::step_smote(recipe, recipes::all_outcomes(), over_ratio = 1, neighbors = 5),
    stop("Unknown imbalance method: ", method)
  )
}

make_model_recipe <- function(training_data, target, predictors, imbalance_method = "none") {
  if (length(predictors) == 0) {
    stop("At least one predictor is required.")
  }

  formula <- stats::reformulate(predictors, response = target)
  recipe <- recipes::recipe(formula, data = training_data) |>
    recipes::step_string2factor(recipes::all_nominal_predictors()) |>
    recipes::step_unknown(recipes::all_nominal_predictors()) |>
    recipes::step_novel(recipes::all_nominal_predictors()) |>
    recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01) |>
    recipes::step_impute_mode(recipes::all_nominal_predictors()) |>
    recipes::step_dummy(recipes::all_nominal_predictors()) |>
    recipes::step_impute_median(recipes::all_numeric_predictors()) |>
    recipes::step_zv(recipes::all_predictors()) |>
    recipes::step_normalize(recipes::all_numeric_predictors())

  add_class_balance_step(recipe, imbalance_method)
}

make_model_candidates <- function() {
  list(
    list(
      name = "Logistic regression: numeric predictors",
      predictor_set = "numeric_only",
      spec = parsnip::logistic_reg() |>
        parsnip::set_engine("glm") |>
        parsnip::set_mode("classification")
    ),
    list(
      name = "Logistic regression: all predictors",
      predictor_set = "all_predictors",
      spec = parsnip::logistic_reg() |>
        parsnip::set_engine("glm") |>
        parsnip::set_mode("classification")
    ),
    list(
      name = "Random forest",
      predictor_set = "all_predictors",
      # Leaving mtry unspecified uses ranger's predictor-count-aware default.
      spec = parsnip::rand_forest(trees = 750, min_n = 5) |>
        parsnip::set_engine("ranger", importance = "impurity", num.threads = 1) |>
        parsnip::set_mode("classification")
    ),
    list(
      name = "XGBoost",
      predictor_set = "all_predictors",
      spec = parsnip::boost_tree(
        trees = 800, tree_depth = 6, learn_rate = 0.05,
        loss_reduction = 0, sample_size = 0.80
      ) |>
        parsnip::set_engine(
          "xgboost", objective = "binary:logistic", eval_metric = "auc", nthread = 1
        ) |>
        parsnip::set_mode("classification")
    )
  )
}

benchmark_models_cv <- function(
    data, target, positive_class = "1", id_columns = character(),
    predictor_sets = NULL, imbalance_methods = c("none", "upsample", "downsample", "smote"),
    v = 5, repeats = 1, seed = 2026) {
  prepared <- prepare_binary_data(data, target, positive_class, id_columns)
  data <- prepared$data

  if (is.null(predictor_sets)) {
    predictor_sets <- default_predictor_sets(data, target, prepared$id_columns)
  }
  required_sets <- c("numeric_only", "all_predictors")
  absent_sets <- setdiff(required_sets, names(predictor_sets))
  if (length(absent_sets) > 0) {
    stop("predictor_sets must include: ", paste(absent_sets, collapse = ", "))
  }
  valid_imbalance_methods <- c("none", "upsample", "downsample", "smote")
  unknown_methods <- setdiff(imbalance_methods, valid_imbalance_methods)
  if (length(unknown_methods) > 0) {
    stop("Unknown imbalance_methods: ", paste(unknown_methods, collapse = ", "))
  }
  require_benchmark_packages(imbalance_methods)

  # rsample uses an unquoted column name for strata; this temporary column lets
  # the function accept an arbitrary target-column name supplied as a string.
  data$.cv_strata_target <- data[[target]]
  set.seed(seed)
  folds <- rsample::vfold_cv(data, v = v, repeats = repeats, strata = .cv_strata_target)
  candidates <- make_model_candidates()
  probability_column <- paste0(".pred_", positive_class)
  fold_results <- list()
  result_index <- 1

  for (candidate in candidates) {
    predictors <- intersect(predictor_sets[[candidate$predictor_set]], names(data))

    if (length(predictors) == 0) {
      warning("Skipping ", candidate$name, ": no listed predictors are in the data.")
      next
    }

    for (imbalance_method in imbalance_methods) {
      for (fold_number in seq_along(folds$splits)) {
        split <- folds$splits[[fold_number]]
        analysis_data <- rsample::analysis(split)
        assessment_data <- rsample::assessment(split)

        fold_auc <- tryCatch({
          recipe <- make_model_recipe(analysis_data, target, predictors, imbalance_method)
          fitted_workflow <- workflows::workflow() |>
            workflows::add_recipe(recipe) |>
            workflows::add_model(candidate$spec) |>
            workflows::fit(data = analysis_data)
          probabilities <- predict(fitted_workflow, assessment_data, type = "prob")

          if (!probability_column %in% names(probabilities)) {
            stop("The model did not return probability column '", probability_column, "'.")
          }
          yardstick::roc_auc_vec(
            truth = assessment_data[[target]],
            estimate = probabilities[[probability_column]],
            event_level = "second"
          )
        }, error = function(error) {
          warning(candidate$name, " (", imbalance_method, ") failed on fold ",
                  fold_number, ": ", error$message)
          NA_real_
        })

        fold_results[[result_index]] <- data.frame(
          model = candidate$name,
          predictor_set = candidate$predictor_set,
          imbalance_method = imbalance_method,
          fold = fold_number,
          auc = fold_auc,
          stringsAsFactors = FALSE
        )
        result_index <- result_index + 1
      }
    }
  }

  fold_auc <- dplyr::bind_rows(fold_results)
  auc_summary <- fold_auc |>
    dplyr::group_by(.data$model, .data$predictor_set, .data$imbalance_method) |>
    dplyr::summarise(
      successful_folds = sum(!is.na(.data$auc)),
      mean_auc = mean(.data$auc, na.rm = TRUE),
      sd_auc = stats::sd(.data$auc, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::arrange(dplyr::desc(.data$mean_auc))

  list(summary = auc_summary, fold_auc = fold_auc, folds = folds)
}

# Randomized hyperparameter tuning ------------------------------------------

stratified_subsample <- function(data, target, n, seed = 2026) {
  if (n >= nrow(data)) {
    return(data)
  }

  set.seed(seed)
  target_values <- data[[target]]
  class_counts <- table(target_values)
  requested <- floor(n * class_counts / sum(class_counts))
  requested[requested == 0] <- 1
  # Allocate any remaining rows to the largest class to reach the requested n.
  requested[which.max(class_counts)] <- requested[which.max(class_counts)] +
    (n - sum(requested))

  sampled_rows <- unlist(lapply(names(class_counts), function(class_name) {
    rows <- which(as.character(target_values) == class_name)
    sample(rows, size = min(length(rows), requested[class_name]))
  }), use.names = FALSE)

  data[sampled_rows, , drop = FALSE]
}

choose_tuning_target <- function(benchmark_summary, model_name = NULL, imbalance_method = NULL) {
  supported_models <- c(
    "Logistic regression: numeric predictors",
    "Logistic regression: all predictors",
    "Random forest", "XGBoost"
  )
  choices <- benchmark_summary[benchmark_summary$model %in% supported_models, , drop = FALSE]

  if (!is.null(model_name)) {
    choices <- choices[choices$model == model_name, , drop = FALSE]
  }
  if (!is.null(imbalance_method)) {
    choices <- choices[choices$imbalance_method == imbalance_method, , drop = FALSE]
  }
  choices <- choices[!is.na(choices$mean_auc), , drop = FALSE]
  if (nrow(choices) == 0) {
    stop("No successful benchmark result matched the requested model/imbalance method.")
  }

  choices[which.max(choices$mean_auc), , drop = FALSE]
}

make_tunable_specification <- function(model_name, predictor_count) {
  if (startsWith(model_name, "Logistic regression")) {
    specification <- parsnip::logistic_reg(
      penalty = tune::tune(), mixture = tune::tune()
    ) |>
      parsnip::set_engine("glmnet") |>
      parsnip::set_mode("classification")

    parameters <- dials::parameters(specification) |>
      dials::update(
        penalty = dials::penalty(range = c(-8, 0)),
        mixture = dials::mixture(range = c(0, 1))
      )
  } else if (model_name == "XGBoost") {
    specification <- parsnip::boost_tree(
      trees = tune::tune(), tree_depth = tune::tune(), learn_rate = tune::tune(),
      min_n = tune::tune(), loss_reduction = tune::tune(), sample_size = tune::tune()
    ) |>
      parsnip::set_engine(
        "xgboost", objective = "binary:logistic", eval_metric = "auc", nthread = 1
      ) |>
      parsnip::set_mode("classification")

    parameters <- dials::parameters(specification) |>
      dials::update(
        trees = dials::trees(range = c(300L, 1500L)),
        tree_depth = dials::tree_depth(range = c(2L, 10L)),
        learn_rate = dials::learn_rate(range = c(-4, -1)),
        min_n = dials::min_n(range = c(2L, 40L)),
        loss_reduction = dials::loss_reduction(range = c(-10, 1)),
        sample_size = dials::sample_prop(range = c(0.50, 1.00))
      )
  } else if (model_name == "Random forest") {
    specification <- parsnip::rand_forest(
      trees = 1000, mtry = tune::tune(), min_n = tune::tune()
    ) |>
      parsnip::set_engine("ranger", importance = "impurity", num.threads = 1) |>
      parsnip::set_mode("classification")

    parameters <- dials::parameters(specification) |>
      dials::update(
        mtry = dials::mtry(range = c(1L, max(2L, min(100L, predictor_count)))),
        min_n = dials::min_n(range = c(2L, 40L))
      )
  } else {
    stop("Randomized tuning currently supports logistic regression, Random forest, and XGBoost.")
  }

  list(specification = specification, parameters = parameters)
}

tune_best_model_random_search <- function(
    data, target, benchmark_summary, positive_class = "1", id_columns = character(),
    model_name = NULL, imbalance_method = NULL, predictor_sets = NULL,
    tuning_rows = 5000, v = 3, random_candidates = 20, seed = 2026) {
  prepared <- prepare_binary_data(data, target, positive_class, id_columns)
  full_training_data <- prepared$data

  if (is.null(predictor_sets)) {
    predictor_sets <- default_predictor_sets(full_training_data, target, prepared$id_columns)
  }
  winner <- choose_tuning_target(benchmark_summary, model_name, imbalance_method)
  require_tuning_packages(winner$model, winner$imbalance_method)
  predictors <- predictor_sets[[winner$predictor_set]]
  predictors <- intersect(predictors, names(full_training_data))
  if (length(predictors) == 0) {
    stop("The selected model has no usable predictors.")
  }

  tuning_data <- stratified_subsample(full_training_data, target, tuning_rows, seed)
  # This is a separate, inexpensive 3-fold CV used only to select parameters.
  tuning_data$.cv_strata_target <- tuning_data[[target]]
  set.seed(seed)
  tuning_folds <- rsample::vfold_cv(tuning_data, v = v, strata = .cv_strata_target)

  tuned <- make_tunable_specification(winner$model, length(predictors))
  tuning_recipe <- make_model_recipe(
    tuning_data, target, predictors, winner$imbalance_method
  )
  tuning_workflow <- workflows::workflow() |>
    workflows::add_recipe(tuning_recipe) |>
    workflows::add_model(tuned$specification)

  set.seed(seed)
  random_grid <- dials::grid_random(tuned$parameters, size = random_candidates)
  tuning_results <- tune::tune_grid(
    tuning_workflow,
    resamples = tuning_folds,
    grid = random_grid,
    metrics = yardstick::metric_set(yardstick::roc_auc),
    control = tune::control_grid(save_pred = FALSE, verbose = TRUE, allow_par = FALSE)
  )
  best_parameters <- tune::select_best(tuning_results, metric = "roc_auc")
  final_workflow <- workflows::finalize_workflow(tuning_workflow, best_parameters)

  # The selected recipe includes the chosen class-balance step and is fitted on
  # all labeled training rows. It is ready to score application_test or new data.
  final_fit <- workflows::fit(final_workflow, data = full_training_data)

  list(
    selected_benchmark_row = winner,
    tuning_rows = nrow(tuning_data),
    best_parameters = best_parameters,
    tuning_metrics = tune::collect_metrics(tuning_results),
    final_workflow = final_workflow,
    final_fit = final_fit
  )
}


run_baseline_cli <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  train_path <- if (length(args) >= 1) args[1] else "application_train.csv"
  test_path <- if (length(args) >= 2) args[2] else "application_test.csv"
  target_column <- if (length(args) >= 3) args[3] else "TARGET"
  positive_class <- if (length(args) >= 4) args[4] else NULL

  train_target <- read_target_column(train_path, target_column)
  if (is.null(train_target)) {
    stop("Target column '", target_column, "' was not found in ", train_path, ".")
  }

  train_results <- evaluate_majority_baseline(train_target, positive_class)
  print_baseline_results(train_results, "Training data")

  if (file.exists(test_path)) {
    test_target <- read_target_column(test_path, target_column)
    if (is.null(test_target)) {
      cat("\nTest data do not contain '", target_column,
          "'; accuracy and ROC AUC cannot be calculated without labels.\n", sep = "")
    } else {
      test_results <- evaluate_majority_baseline(test_target, train_results$positive_class)
      print_baseline_results(test_results, "Test data")
    }
  }
}

# Run the lightweight majority-class baseline only when called as a script.
if (sys.nframe() == 0) {
  run_baseline_cli()
}
