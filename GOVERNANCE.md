# GOVERNANCE.md

## Intended Use

EchoSight supports a sighted-assistance use case: generating a spoken description of a
static image for someone who cannot see it. It is explicitly **not** a safety-critical
navigation, obstacle-detection, or hazard-warning system. It should not be relied on
instead of a cane, guide dog, or human assistance for any safety-relevant decision.

## Explainability Boundary

The attention map shows which of the image's 64 spatial regions the model weighted most
when generating each word of its caption — directly inspectable, not a black box. The
LLM-enriched description is explicitly labeled as elaboration on the model's own caption,
not an independently verified description of the image's actual contents; the LLM never
sees the image itself, only the caption text.

## Fairness

Flickr8k's known demographic and scene-type skew (predominantly Western/English-context
photographs) is a documented limitation of any model trained on it, not a solved problem.
Caption accuracy and the enrichment layer's assumptions may be less reliable on images
outside that distribution — this is stated here rather than left unaddressed.

## LLM Controls

The enrichment call passes `temperature=0` to the configured LLM provider, so a given
caption's enriched description is reproducible. The LLM's only input is the model's own
generated caption text — it never receives the image itself, retrieved documents, or
user-supplied free text, so the prompt-injection surface is low.

## Audit Trail

Every generated report embeds a "Run Info" block recording the checkpoint's trained
epoch count and a timestamp.

## Regulatory Framing

EchoSight is not a medical device and is not a safety-certified assistive product. It
does not currently operate under a specific regulatory framework — noted here for
completeness rather than left unaddressed, matching every other non-financial repo in
this ecosystem.
