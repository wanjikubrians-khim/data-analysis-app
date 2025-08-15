#!/usr/bin/env Rscript
# R Analytics Engine for Advanced Statistical Analysis
# Provides comprehensive statistical analysis, modeling, and specialized R features

# Load required libraries with error handling
required_packages <- c(
  "jsonlite", "dplyr", "ggplot2", "plotly", "corrplot", 
  "VIM", "mice", "car", "lmtest", "tseries", "forecast", 
  "cluster", "factoextra", "psych", "GPArotation", "MASS",
  "survival", "randomForest", "rpart", "e1071", "caret",
  "nortest", "moments", "bcp", "changepoint", "pracma"
)

# Function to install and load packages
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing package:", pkg, "\n")
      install.packages(pkg, dependencies = TRUE, quiet = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# Install and load required packages
suppressWarnings(suppressMessages(install_and_load(required_packages)))

# Main R Analytics Class
RAnalyzer <- setRefClass("RAnalyzer",
  fields = list(
    data = "data.frame",
    numeric_columns = "character",
    categorical_columns = "character",
    results = "list"
  ),
  
  methods = list(
    # Load data from CSV file
    load_data = function(file_path) {
      tryCatch({
        data <<- read.csv(file_path, stringsAsFactors = FALSE)
        identify_column_types()
        
        return(list(
          success = TRUE,
          message = paste("Data loaded successfully. Shape:", nrow(data), "x", ncol(data)),
          shape = c(nrow(data), ncol(data)),
          columns = names(data),
          numeric_columns = numeric_columns,
          categorical_columns = categorical_columns
        ))
      }, error = function(e) {
        return(list(
          success = FALSE,
          message = paste("Error loading data:", e$message)
        ))
      })
    },
    
    # Identify column types
    identify_column_types = function() {
      numeric_columns <<- names(data)[sapply(data, is.numeric)]
      categorical_columns <<- names(data)[sapply(data, function(x) is.character(x) || is.factor(x))]
    },
    
    # Advanced statistical summary
    advanced_statistics = function() {
      if (nrow(data) == 0) {
        return(list(error = "No data loaded"))
      }
      
      result <- list(
        dataset_info = list(
          total_rows = nrow(data),
          total_columns = ncol(data),
          memory_usage = paste(object.size(data), "bytes"),
          missing_values = sum(is.na(data)),
          duplicate_rows = sum(duplicated(data))
        ),
        numeric_analysis = list(),
        categorical_analysis = list(),
        advanced_tests = list()
      )
      
      # Numeric analysis with R-specific statistics
      if (length(numeric_columns) > 0) {
        for (col in numeric_columns) {
          col_data <- data[[col]][!is.na(data[[col]])]
          if (length(col_data) > 0) {
            
            # Basic descriptive statistics
            desc_stats <- summary(col_data)
            
            # Advanced R statistics
            result$numeric_analysis[[col]] <- list(
              count = length(col_data),
              mean = mean(col_data),
              median = median(col_data),
              sd = sd(col_data),
              var = var(col_data),
              min = min(col_data),
              max = max(col_data),
              q1 = quantile(col_data, 0.25),
              q3 = quantile(col_data, 0.75),
              iqr = IQR(col_data),
              range = diff(range(col_data)),
              skewness = moments::skewness(col_data),
              kurtosis = moments::kurtosis(col_data),
              cv = sd(col_data) / mean(col_data),
              mad = mad(col_data),
              trimmed_mean = mean(col_data, trim = 0.1),
              geometric_mean = exp(mean(log(col_data[col_data > 0]))),
              harmonic_mean = 1 / mean(1 / col_data[col_data != 0]),
              missing_count = sum(is.na(data[[col]]))
            )
          }
        }
      }
      
      # Categorical analysis
      if (length(categorical_columns) > 0) {
        for (col in categorical_columns) {
          col_data <- data[[col]][!is.na(data[[col]])]
          freq_table <- table(col_data)
          
          result$categorical_analysis[[col]] <- list(
            unique_values = length(unique(col_data)),
            most_frequent = names(freq_table)[which.max(freq_table)],
            most_frequent_count = max(freq_table),
            missing_count = sum(is.na(data[[col]])),
            entropy = -sum((freq_table/sum(freq_table)) * log2(freq_table/sum(freq_table))),
            gini_simpson = 1 - sum((freq_table/sum(freq_table))^2),
            value_counts = as.list(head(sort(freq_table, decreasing = TRUE), 10))
          )
        }
      }
      
      return(result)
    },
    
    # Advanced correlation analysis with R-specific methods
    advanced_correlation = function() {
      if (length(numeric_columns) < 2) {
        return(list(error = "Insufficient numeric data for correlation analysis"))
      }
      
      numeric_data <- data[numeric_columns]
      
      # Multiple correlation methods
      pearson_cor <- cor(numeric_data, use = "complete.obs", method = "pearson")
      spearman_cor <- cor(numeric_data, use = "complete.obs", method = "spearman")
      kendall_cor <- cor(numeric_data, use = "complete.obs", method = "kendall")
      
      # Partial correlations
      partial_cor <- tryCatch({
        psych::partial.r(numeric_data)
      }, error = function(e) NULL)
      
      # Correlation tests
      cor_tests <- list()
      for (i in 1:(length(numeric_columns)-1)) {
        for (j in (i+1):length(numeric_columns)) {
          col1 <- numeric_columns[i]
          col2 <- numeric_columns[j]
          
          # Pearson test
          pearson_test <- cor.test(numeric_data[[col1]], numeric_data[[col2]], method = "pearson")
          
          # Spearman test
          spearman_test <- cor.test(numeric_data[[col1]], numeric_data[[col2]], method = "spearman")
          
          cor_tests[[paste(col1, "vs", col2)]] <- list(
            pearson = list(
              correlation = pearson_test$estimate,
              p_value = pearson_test$p.value,
              significant = pearson_test$p.value < 0.05,
              confidence_interval = pearson_test$conf.int
            ),
            spearman = list(
              correlation = spearman_test$estimate,
              p_value = spearman_test$p.value,
              significant = spearman_test$p.value < 0.05
            )
          )
        }
      }
      
      return(list(
        pearson_matrix = pearson_cor,
        spearman_matrix = spearman_cor,
        kendall_matrix = kendall_cor,
        partial_correlations = partial_cor,
        correlation_tests = cor_tests,
        eigenvalues = eigen(pearson_cor)$values,
        condition_number = max(eigen(pearson_cor)$values) / min(eigen(pearson_cor)$values)
      ))
    },
    
    # Advanced statistical tests
    statistical_tests = function() {
      if (length(numeric_columns) == 0) {
        return(list(error = "No numeric data for statistical tests"))
      }
      
      test_results <- list(
        normality_tests = list(),
        homogeneity_tests = list(),
        independence_tests = list(),
        goodness_of_fit = list()
      )
      
      # Normality tests for each numeric column
      for (col in numeric_columns) {
        col_data <- data[[col]][!is.na(data[[col]])]
        if (length(col_data) >= 3) {
          
          normality_results <- list()
          
          # Shapiro-Wilk test
          if (length(col_data) <= 5000) {
            shapiro_test <- shapiro.test(col_data)
            normality_results$shapiro_wilk <- list(
              statistic = shapiro_test$statistic,
              p_value = shapiro_test$p.value,
              is_normal = shapiro_test$p.value > 0.05
            )
          }
          
          # Kolmogorov-Smirnov test
          ks_test <- ks.test(col_data, "pnorm", mean(col_data), sd(col_data))
          normality_results$kolmogorov_smirnov <- list(
            statistic = ks_test$statistic,
            p_value = ks_test$p.value,
            is_normal = ks_test$p.value > 0.05
          )
          
          # Anderson-Darling test
          if (require(nortest, quietly = TRUE)) {
            ad_test <- nortest::ad.test(col_data)
            normality_results$anderson_darling <- list(
              statistic = ad_test$statistic,
              p_value = ad_test$p.value,
              is_normal = ad_test$p.value > 0.05
            )
          }
          
          # Jarque-Bera test
          jb_test <- tryCatch({
            tseries::jarque.bera.test(col_data)
          }, error = function(e) NULL)
          
          if (!is.null(jb_test)) {
            normality_results$jarque_bera <- list(
              statistic = jb_test$statistic,
              p_value = jb_test$p.value,
              is_normal = jb_test$p.value > 0.05
            )
          }
          
          test_results$normality_tests[[col]] <- normality_results
        }
      }
      
      # Homogeneity tests if we have categorical grouping variables
      if (length(categorical_columns) > 0 && length(numeric_columns) > 0) {
        for (cat_col in categorical_columns[1:min(2, length(categorical_columns))]) {
          for (num_col in numeric_columns[1:min(3, length(numeric_columns))]) {
            
            # Create groups
            groups <- split(data[[num_col]], data[[cat_col]])
            groups <- groups[sapply(groups, function(x) length(x) >= 3)]
            
            if (length(groups) >= 2) {
              # Bartlett test for equal variances
              bartlett_test <- tryCatch({
                bartlett.test(data[[num_col]] ~ data[[cat_col]], data = data)
              }, error = function(e) NULL)
              
              if (!is.null(bartlett_test)) {
                test_results$homogeneity_tests[[paste(num_col, "by", cat_col)]] <- list(
                  test = "Bartlett",
                  statistic = bartlett_test$statistic,
                  p_value = bartlett_test$p.value,
                  equal_variances = bartlett_test$p.value > 0.05
                )
              }
            }
          }
        }
      }
      
      return(test_results)
    },
    
    # Time series analysis with R's powerful time series tools
    time_series_analysis = function(date_column, value_column) {
      if (!(date_column %in% names(data)) || !(value_column %in% names(data))) {
        return(list(error = paste("Columns", date_column, "or", value_column, "not found")))
      }
      
      tryCatch({
        # Prepare time series data
        ts_data <- data.frame(
          date = as.Date(data[[date_column]]),
          value = as.numeric(data[[value_column]])
        )
        ts_data <- ts_data[complete.cases(ts_data), ]
        ts_data <- ts_data[order(ts_data$date), ]
        
        if (nrow(ts_data) < 3) {
          return(list(error = "Insufficient data points for time series analysis"))
        }
        
        # Create time series object
        ts_values <- ts(ts_data$value, frequency = 1)
        
        # Basic time series statistics
        result <- list(
          data_points = length(ts_values),
          time_range = list(
            start = as.character(min(ts_data$date)),
            end = as.character(max(ts_data$date)),
            duration_days = as.numeric(max(ts_data$date) - min(ts_data$date))
          ),
          basic_stats = list(
            mean = mean(ts_values),
            median = median(ts_values),
            sd = sd(ts_values),
            min = min(ts_values),
            max = max(ts_values),
            trend_direction = ifelse(tail(ts_values, 1) > head(ts_values, 1), "increasing", "decreasing")
          )
        )
        
        # Decomposition if enough data points
        if (length(ts_values) >= 24) {
          # Try different frequencies for decomposition
          for (freq in c(12, 4, 7)) {
            if (length(ts_values) >= 2 * freq) {
              ts_freq <- ts(ts_values, frequency = freq)
              decomp <- tryCatch({
                decompose(ts_freq, type = "additive")
              }, error = function(e) NULL)
              
              if (!is.null(decomp)) {
                result$decomposition <- list(
                  frequency = freq,
                  seasonal_strength = var(decomp$seasonal, na.rm = TRUE) / var(ts_values, na.rm = TRUE),
                  trend_strength = var(decomp$trend, na.rm = TRUE) / var(ts_values, na.rm = TRUE),
                  remainder_strength = var(decomp$random, na.rm = TRUE) / var(ts_values, na.rm = TRUE)
                )
                break
              }
            }
          }
        }
        
        # Stationarity tests
        if (length(ts_values) >= 10) {
          # Augmented Dickey-Fuller test
          adf_test <- tryCatch({
            tseries::adf.test(ts_values)
          }, error = function(e) NULL)
          
          if (!is.null(adf_test)) {
            result$stationarity <- list(
              adf_statistic = adf_test$statistic,
              adf_p_value = adf_test$p.value,
              is_stationary = adf_test$p.value < 0.05
            )
          }
        }
        
        # Change point detection
        if (length(ts_values) >= 10 && require(changepoint, quietly = TRUE)) {
          cpt_result <- tryCatch({
            changepoint::cpt.mean(ts_values, method = "PELT")
          }, error = function(e) NULL)
          
          if (!is.null(cpt_result)) {
            changepoints <- changepoint::cpts(cpt_result)
            result$changepoints <- list(
              detected = length(changepoints) > 0,
              points = changepoints,
              count = length(changepoints)
            )
          }
        }
        
        return(result)
        
      }, error = function(e) {
        return(list(error = paste("Time series analysis failed:", e$message)))
      })
    },
    
    # Advanced clustering with R's superior clustering algorithms
    advanced_clustering = function(n_clusters = 3, method = "kmeans") {
      if (length(numeric_columns) < 2) {
        return(list(error = "Insufficient numeric data for clustering"))
      }
      
      tryCatch({
        # Prepare data
        cluster_data <- data[numeric_columns]
        cluster_data <- na.omit(cluster_data)
        
        if (nrow(cluster_data) < n_clusters) {
          return(list(error = "Not enough data points for clustering"))
        }
        
        # Scale the data
        scaled_data <- scale(cluster_data)
        
        result <- list(
          method = method,
          n_clusters = n_clusters,
          data_points = nrow(cluster_data),
          features_used = numeric_columns
        )
        
        if (method == "kmeans") {
          # K-means clustering
          kmeans_result <- kmeans(scaled_data, centers = n_clusters, nstart = 25)
          
          # Cluster statistics
          cluster_stats <- list()
          for (i in 1:n_clusters) {
            cluster_data_subset <- cluster_data[kmeans_result$cluster == i, ]
            cluster_stats[[paste("Cluster", i)]] <- list(
              size = nrow(cluster_data_subset),
              percentage = nrow(cluster_data_subset) / nrow(cluster_data) * 100,
              means = colMeans(cluster_data_subset),
              within_ss = kmeans_result$withinss[i]
            )
          }
          
          result$kmeans <- list(
            cluster_labels = kmeans_result$cluster,
            centers = kmeans_result$centers,
            withinss = kmeans_result$withinss,
            tot_withinss = kmeans_result$tot.withinss,
            betweenss = kmeans_result$betweenss,
            size = kmeans_result$size,
            cluster_statistics = cluster_stats
          )
          
          # Silhouette analysis
          if (require(cluster, quietly = TRUE)) {
            sil <- cluster::silhouette(kmeans_result$cluster, dist(scaled_data))
            result$silhouette <- list(
              average_width = mean(sil[, 3]),
              widths = sil[, 3]
            )
          }
          
        } else if (method == "hierarchical") {
          # Hierarchical clustering
          dist_matrix <- dist(scaled_data)
          hc_result <- hclust(dist_matrix, method = "ward.D2")
          clusters <- cutree(hc_result, k = n_clusters)
          
          result$hierarchical <- list(
            cluster_labels = clusters,
            method = "ward.D2",
            height = hc_result$height,
            merge_order = hc_result$merge
          )
        }
        
        # Cluster validation indices
        if (require(factoextra, quietly = TRUE)) {
          # Optimal number of clusters
          wss <- factoextra::fviz_nbclust(scaled_data, kmeans, method = "wss")
          silhouette <- factoextra::fviz_nbclust(scaled_data, kmeans, method = "silhouette")
          
          result$cluster_validation <- list(
            optimal_k_wss = which.min(diff(wss$data$y)) + 1,
            optimal_k_silhouette = which.max(silhouette$data$y)
          )
        }
        
        return(result)
        
      }, error = function(e) {
        return(list(error = paste("Clustering analysis failed:", e$message)))
      })
    },
    
    # Survival analysis (unique to R)
    survival_analysis = function(time_column, event_column, group_column = NULL) {
      if (!require(survival, quietly = TRUE)) {
        return(list(error = "Survival package not available"))
      }
      
      if (!(time_column %in% names(data)) || !(event_column %in% names(data))) {
        return(list(error = paste("Required columns not found")))
      }
      
      tryCatch({
        # Create survival object
        surv_obj <- Surv(data[[time_column]], data[[event_column]])
        
        result <- list(
          total_observations = length(surv_obj),
          events = sum(data[[event_column]]),
          censored = sum(1 - data[[event_column]]),
          event_rate = mean(data[[event_column]])
        )
        
        # Kaplan-Meier estimator
        km_fit <- survfit(surv_obj ~ 1)
        result$kaplan_meier <- list(
          median_survival = median(km_fit),
          survival_times = km_fit$time,
          survival_probs = km_fit$surv,
          conf_int_lower = km_fit$lower,
          conf_int_upper = km_fit$upper
        )
        
        # Group comparison if group column provided
        if (!is.null(group_column) && group_column %in% names(data)) {
          km_group <- survfit(surv_obj ~ data[[group_column]])
          logrank_test <- survdiff(surv_obj ~ data[[group_column]])
          
          result$group_analysis <- list(
            groups = levels(factor(data[[group_column]])),
            logrank_test = list(
              chi_square = logrank_test$chisq,
              p_value = pchisq(logrank_test$chisq, df = length(logrank_test$n) - 1, lower.tail = FALSE),
              significant = pchisq(logrank_test$chisq, df = length(logrank_test$n) - 1, lower.tail = FALSE) < 0.05
            )
          )
        }
        
        return(result)
        
      }, error = function(e) {
        return(list(error = paste("Survival analysis failed:", e$message)))
      })
    },
    
    # Advanced regression modeling
    advanced_regression = function(target_column, method = "linear") {
      if (!(target_column %in% names(data))) {
        return(list(error = paste("Target column", target_column, "not found")))
      }
      
      tryCatch({
        # Prepare data
        model_data <- data[complete.cases(data), ]
        
        if (nrow(model_data) < 10) {
          return(list(error = "Insufficient complete cases for modeling"))
        }
        
        # Create formula
        predictors <- setdiff(names(model_data), target_column)
        formula_str <- paste(target_column, "~", paste(predictors, collapse = " + "))
        model_formula <- as.formula(formula_str)
        
        result <- list(
          target = target_column,
          predictors = predictors,
          sample_size = nrow(model_data),
          method = method
        )
        
        if (method == "linear") {
          # Linear regression
          lm_model <- lm(model_formula, data = model_data)
          
          result$linear_regression <- list(
            coefficients = coef(lm_model),
            r_squared = summary(lm_model)$r.squared,
            adj_r_squared = summary(lm_model)$adj.r.squared,
            f_statistic = summary(lm_model)$fstatistic[1],
            p_value = pf(summary(lm_model)$fstatistic[1], 
                        summary(lm_model)$fstatistic[2], 
                        summary(lm_model)$fstatistic[3], lower.tail = FALSE),
            residual_se = summary(lm_model)$sigma,
            aic = AIC(lm_model),
            bic = BIC(lm_model)
          )
          
          # Diagnostic tests
          if (require(lmtest, quietly = TRUE) && require(car, quietly = TRUE)) {
            # Breusch-Pagan test for heteroscedasticity
            bp_test <- lmtest::bptest(lm_model)
            
            # Durbin-Watson test for autocorrelation
            dw_test <- car::durbinWatsonTest(lm_model)
            
            # VIF for multicollinearity
            if (length(predictors) > 1) {
              vif_values <- car::vif(lm_model)
              result$diagnostics <- list(
                heteroscedasticity = list(
                  bp_statistic = bp_test$statistic,
                  bp_p_value = bp_test$p.value,
                  homoscedastic = bp_test$p.value > 0.05
                ),
                autocorrelation = list(
                  dw_statistic = dw_test$dw,
                  dw_p_value = dw_test$p,
                  no_autocorrelation = dw_test$p > 0.05
                ),
                multicollinearity = list(
                  vif_values = vif_values,
                  max_vif = max(vif_values),
                  multicollinearity_concern = any(vif_values > 5)
                )
              )
            }
          }
          
        } else if (method == "glm") {
          # Generalized linear model
          if (is.numeric(model_data[[target_column]])) {
            family_type <- gaussian()
          } else {
            family_type <- binomial()
          }
          
          glm_model <- glm(model_formula, data = model_data, family = family_type)
          
          result$generalized_linear <- list(
            coefficients = coef(glm_model),
            deviance = glm_model$deviance,
            null_deviance = glm_model$null.deviance,
            aic = AIC(glm_model),
            bic = BIC(glm_model),
            pseudo_r_squared = 1 - (glm_model$deviance / glm_model$null.deviance)
          )
        }
        
        return(result)
        
      }, error = function(e) {
        return(list(error = paste("Regression analysis failed:", e$message)))
      })
    }
  )
)

# Command line interface
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    cat("Usage: Rscript data_analyzer.R <csv_file> <analysis_type> [additional_args]\n")
    cat("Analysis types:\n")
    cat("  basic          - Advanced statistical summary\n")
    cat("  correlation    - Advanced correlation analysis\n")
    cat("  tests          - Comprehensive statistical tests\n")
    cat("  clustering     - Advanced clustering analysis\n")
    cat("  regression <target> - Advanced regression modeling\n")
    cat("  survival <time> <event> [group] - Survival analysis\n")
    cat("  timeseries <date> <value> - Time series analysis\n")
    cat("  comprehensive  - All R analyses combined\n")
    quit(status = 1)
  }
  
  csv_file <- args[1]
  analysis_type <- tolower(args[2])
  
  # Create analyzer instance
  analyzer <- RAnalyzer$new()
  
  # Load data
  load_result <- analyzer$load_data(csv_file)
  if (!load_result$success) {
    cat(jsonlite::toJSON(load_result, pretty = TRUE, auto_unbox = TRUE))
    quit(status = 1)
  }
  
  # Perform analysis
  if (analysis_type == "basic") {
    result <- analyzer$advanced_statistics()
    
  } else if (analysis_type == "correlation") {
    result <- analyzer$advanced_correlation()
    
  } else if (analysis_type == "tests") {
    result <- analyzer$statistical_tests()
    
  } else if (analysis_type == "clustering") {
    n_clusters <- ifelse(length(args) >= 3, as.numeric(args[3]), 3)
    method <- ifelse(length(args) >= 4, args[4], "kmeans")
    result <- analyzer$advanced_clustering(n_clusters, method)
    
  } else if (analysis_type == "regression") {
    if (length(args) < 3) {
      cat("Error: Regression analysis requires target column\n")
      cat("Available columns:", paste(names(analyzer$data), collapse = ", "), "\n")
      quit(status = 1)
    }
    target <- args[3]
    method <- ifelse(length(args) >= 4, args[4], "linear")
    result <- analyzer$advanced_regression(target, method)
    
  } else if (analysis_type == "survival") {
    if (length(args) < 4) {
      cat("Error: Survival analysis requires time and event columns\n")
      cat("Available columns:", paste(names(analyzer$data), collapse = ", "), "\n")
      quit(status = 1)
    }
    time_col <- args[3]
    event_col <- args[4]
    group_col <- ifelse(length(args) >= 5, args[5], NULL)
    result <- analyzer$survival_analysis(time_col, event_col, group_col)
    
  } else if (analysis_type == "timeseries") {
    if (length(args) < 4) {
      cat("Error: Time series analysis requires date and value columns\n")
      cat("Available columns:", paste(names(analyzer$data), collapse = ", "), "\n")
      quit(status = 1)
    }
    date_col <- args[3]
    value_col <- args[4]
    result <- analyzer$time_series_analysis(date_col, value_col)
    
  } else if (analysis_type == "comprehensive") {
    # Run all analyses
    result <- list(
      load_info = load_result,
      basic_statistics = analyzer$advanced_statistics(),
      correlation_analysis = analyzer$advanced_correlation(),
      statistical_tests = analyzer$statistical_tests(),
      clustering = analyzer$advanced_clustering()
    )
    
    # Add regression if we have numeric columns
    if (length(analyzer$numeric_columns) > 0) {
      result$regression <- analyzer$advanced_regression(analyzer$numeric_columns[1])
    }
    
  } else {
    result <- list(error = paste("Unknown analysis type:", analysis_type))
  }
  
  # Output JSON result
  cat(jsonlite::toJSON(result, pretty = TRUE, auto_unbox = TRUE, na = "null"))
}
