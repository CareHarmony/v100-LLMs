# JevK5 typed decisions (/v1/systemone). Defaults in .env; see 'make help'.
SERVICE := jevk5
MODEL   := alibiserikbay/JevK5
# Weights for make download: repo[@revision]. Keep the revision in step with REVISION in .env.
DOWNLOADS := alibiserikbay/JevK5@27d2d6b8d4714807f6293b0623bd7370b27e42f8
# No OpenAI API: wait on /health and test with a /v1/systemone request instead.
READY_PATH  := /health
CUSTOM_TEST := 1
SETTINGS     = GPUS=$(GPUS) PORT=$(PORT) DTYPE=$(DTYPE) REVISION=$(REVISION)

include ../common.mk

test: ## Send one typed decision to /v1/systemone
	@curl -s http://127.0.0.1:$(PORT)/v1/systemone -H 'Content-Type: application/json' \
		-d '{"state":"I was billed twice for order #4411. Please refund the duplicate charge today.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","tech":"Bugs","sales":"New purchases"}}}}' \
		| python3 -m json.tool
