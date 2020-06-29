DATA_DIR = data/example
EXP_DIR = experiments/example

CV_DIR = $(EXP_DIR)/cv
FINAL_DIR = $(EXP_DIR)/final

##############
# Core rules #
##############

%/output/split-0.pt : %/cfg.json $(DATA_DIR)/train.csv
	phlanders --import src.models train --output=output $<

%/output/predictions.csv : %/output/split-0.pt
	phlanders --import src.models predictions --true useBest $(patsubst %output/split-0.pt,%cfg.json,$<) --results $(@D)

%/output/metrics.csv : %/output/split-0.pt
	phlanders --import src.models metrics --true useBest $(patsubst %output/split-0.pt,%cfg.json,$<) --results $(@D)

####################
# Cross validation #
####################

CV_DIRS = $(wildcard $(CV_DIR)/*)

CV_CONFIGS = $(foreach dir, $(CV_DIRS), $(dir)/cfg.json)
CV_OUTPUT = $(patsubst %cfg.json,%output/split-0.pt,$(CV_CONFIGS))
CV_PREDICTIONS = $(patsubst %cfg.json,%output/predictions.csv,$(CV_CONFIGS))
CV_METRICS = $(patsubst %cfg.json,%output/metrics.csv,$(CV_CONFIGS))

################
# Final models #
################

FINAL_CONFIGS = $(wildcard $(FINAL_DIR)/*/cfg.json)
FINAL_OUTPUT = $(patsubst %cfg.json,%output/split-0.pt,$(FINAL_CONFIGS))
FINAL_PREDICTIONS = $(patsubst %cfg.json,%output/predictions.csv,$(FINAL_CONFIGS))
FINAL_METRICS = $(patsubst %cfg.json,%output/metrics.csv,$(FINAL_CONFIGS))

# prevent deletion of trained models
.PRECIOUS: $(CV_OUTPUT) $(FINAL_OUTPUT)

#############
# Aggregate #
#############

experiments/cv: $(CV_PREDICTIONS) $(CV_METRICS)
experiments/final: $(FINAL_PREDICTIONS) $(FINAL_METRICS)
experiments: experiments/cv experiments/final
.PHONY: experiments experiments/cv experiments/final
