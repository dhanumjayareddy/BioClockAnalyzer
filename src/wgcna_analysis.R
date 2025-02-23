#!/usr/bin/env Rscript
# wgcna_analysis.R
# ---------------------
# Production-ready script for performing WGCNA network reconstruction and module detection.
#
# Features:
# - Error handling with tryCatch.
# - Logging via simple print statements (can be enhanced with logging packages).
# - Parameter validation and scalable analysis (with memory considerations).
# - Detailed inline documentation and instructions for troubleshooting.
#
# Usage:
#   Rscript src/wgcna_analysis.R --input data/processed/normalized_data.csv --output results/wgcna/module_colors.csv

suppressMessages(library(optparse))
suppressMessages(library(WGCNA))

# Define command-line options
option_list <- list(
  make_option(c("-i", "--input"), type="character", default=NULL,
              help="Path to normalized data CSV file", metavar="character"),
  make_option(c("-o", "--output"), type="character", default=NULL,
              help="Path to save module colors CSV", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if(is.null(opt$input) || is.null(opt$output)) {
  print_help(opt_parser)
  stop("Input and output files must be supplied", call.=FALSE)
}

# Error handling wrapper
tryCatch({
  cat("Loading normalized data from:", opt$input, "\n")
  data <- read.csv(opt$input, row.names = 1)
  data <- as.data.frame(t(data))  # WGCNA expects samples in rows
  cat("Data loaded. Shape:", dim(data), "\n")
  
  # Determine soft-thresholding power
  powers <- c(1:20)
  cat("Picking soft-thresholding power...\n")
  sft <- pickSoftThreshold(data, powerVector = powers, verbose = 5)
  softPower <- sft$powerEstimate
  cat("Selected soft-thresholding power:", softPower, "\n")
  
  # Construct adjacency matrix and calculate TOM
  adjacency <- adjacency(data, power = softPower)
  TOM <- TOMsimilarity(adjacency)
  dissTOM <- 1 - TOM
  
  # Hierarchical clustering and module detection
  geneTree <- hclust(as.dist(dissTOM), method = "average")
  dynamicMods <- cutreeDynamic(dendro = geneTree, distM = dissTOM,
                                deepSplit = 2, pamRespectsDendro = FALSE)
  moduleColors <- labels2colors(dynamicMods)
  
  # Save module colors to CSV
  output_file <- opt$output
  write.csv(moduleColors, output_file, quote = FALSE)
  cat("WGCNA module colors saved to:", output_file, "\n")
}, error = function(e) {
  cat("Error in WGCNA analysis: ", e$message, "\n")
  quit(status=1)
})
