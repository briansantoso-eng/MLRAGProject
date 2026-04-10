"""
Automatic provider detection for CloudDocs RAG.

Context
-------
Two retrieval improvement experiments (BGE embedding swap, BM25 hybrid search)
both failed to improve Recall@K beyond 0.630. Post-experiment analysis showed
the root cause: cloud services from different providers do the same thing and
are documented using the same words. Neither semantic similarity nor keyword
frequency can distinguish them.

The correct fix is provider disambiguation at query time: detect which cloud
provider the user is asking about, then scope ChromaDB retrieval to that
provider using a metadata filter. This eliminates cross-provider noise entirely
for single-provider questions.

This module implements a keyword-based classifier. It is fast (no API call),
free, interpretable, and sufficient — the provider signals in cloud queries
are almost always explicit service names or brand names.

Logic
-----
1. Score each provider by counting keyword matches in the lowercased query.
2. If one provider dominates (highest score, score > 0), return it.
3. If scores are tied or zero (cross-provider or ambiguous), return None
   so the caller falls back to searching all providers.
"""

import re

# ── Keyword lists ─────────────────────────────────────────────────────────────
# Ordered by specificity — more specific terms listed first.
# Multi-word terms are matched before single words to avoid partial matches.

AWS_KEYWORDS = [
    # Unambiguous service names
    "aws lambda", "amazon lambda",
    "amazon s3", "s3 bucket", "s3 storage", "s3 storage class",
    "amazon ec2", "ec2 instance",
    "amazon rds", "rds multi-az", "rds instance",
    "aws iam", "iam role", "iam user", "iam policy",
    "amazon vpc", "vpc subnet",
    "amazon dynamodb", "amazon cloudwatch", "aws fargate",
    "amazon cloudfront", "amazon route 53",
    # Brand signals
    "aws", "amazon web services",
    # Short unambiguous terms (checked after multi-word)
    "s3", "ec2", "rds", "iam", "vpc", "dynamodb",
    "cloudwatch", "cloudfront", "fargate", "lambda",
]

AZURE_KEYWORDS = [
    # Unambiguous service names
    "azure virtual machine", "azure vm",
    "azure blob storage", "azure blob",
    "azure sql database", "azure sql",
    "azure active directory", "azure ad",
    "azure virtual network", "azure vnet",
    "azure functions", "azure function",
    "azure storage", "azure kubernetes", "azure container",
    "azure network security group", "azure nsg",
    "azure availability set", "azure elastic pool",
    "azure conditional access",
    # Brand signals
    "azure", "microsoft azure",
]

GCP_KEYWORDS = [
    # Unambiguous service names
    "google cloud functions", "gcp cloud functions",
    "google cloud storage", "gcs",
    "google cloud sql", "cloud sql",
    "gcp iam", "google iam",
    "gcp vpc", "google vpc",
    "google kubernetes engine", "gke",
    "bigquery", "cloud run", "pub/sub", "pubsub", "dataflow",
    # Brand signals
    "gcp", "google cloud platform", "google cloud",
]

_PROVIDER_KEYWORDS = {
    "aws":   AWS_KEYWORDS,
    "azure": AZURE_KEYWORDS,
    "gcp":   GCP_KEYWORDS,
}


def detect_provider(query: str) -> str | None:
    """
    Return 'aws', 'azure', 'gcp', or None.

    None means the query is cross-provider or ambiguous — caller should
    search all providers without filtering.
    """
    q = query.lower()
    # Remove punctuation for cleaner matching
    q = re.sub(r"[^\w\s/]", " ", q)

    scores = {"aws": 0, "azure": 0, "gcp": 0}

    for provider, keywords in _PROVIDER_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                # Multi-word keywords count more than single-word
                weight = len(kw.split())
                scores[provider] += weight

    # If more than one provider has any score, the query is cross-provider
    providers_with_signal = [p for p, s in scores.items() if s > 0]

    if len(providers_with_signal) == 0:
        return None  # No provider signal found
    if len(providers_with_signal) > 1:
        return None  # Multiple providers mentioned — cross-provider query

    return providers_with_signal[0]


if __name__ == "__main__":
    # Quick sanity check
    tests = [
        ("What is AWS Lambda and what is it used for?",          "aws"),
        ("What are S3 storage classes?",                          "aws"),
        ("What is Azure Blob Storage?",                           "azure"),
        ("How does Azure AD Conditional Access work?",            "azure"),
        ("What is GCP Cloud SQL?",                                "gcp"),
        ("Compare AWS Lambda and Azure Functions",                None),
        ("How does Amazon S3 compare to Azure Blob Storage?",     None),
        ("How do I store files in the cloud?",                    None),
    ]

    print(f"{'Query':<55} {'Expected':<8} {'Got':<8} {'OK'}")
    print("-" * 80)
    all_pass = True
    for query, expected in tests:
        got = detect_provider(query)
        ok  = got == expected
        if not ok:
            all_pass = False
        print(f"{query[:54]:<55} {str(expected):<8} {str(got):<8} {'OK' if ok else 'FAIL'}")

    print(f"\n{'All tests passed' if all_pass else 'Some tests FAILED'}")
