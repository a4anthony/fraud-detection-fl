# Feature taxonomy for per-client SHAP categorisation (Task 5)

This taxonomy assigns each feature in the three FL client feature spaces to a
category. Categories are chosen to be meaningful across heterogeneous schemas
so that the per-client SHAP-by-category bar chart can answer the question
"which kinds of features matter for fraud at each client?" rather than the
narrower per-feature question.

Honest disclosure: the three datasets have **largely disjoint feature
spaces** — ULB is PCA-anonymised Time + Amount + V1–V28; BAF is mixed
demographic / behavioural / device features; Synthetic is small
domain-specific transactional features. So perfect cross-client overlap is
impossible by construction. The categorisation exists to make that
disjointness visible at the category (rather than feature) level, and to
identify the few categories where there *is* meaningful overlap.

## Categories

| Category | Definition |
|---|---|
| `velocity` | Features encoding rate-of-activity over a recent window. |
| `temporal` | Time stamps, time-since-last-event, calendar features. |
| `identity` | Customer demographics, KYC fields, account-holder attributes. |
| `device` | Device fingerprint, browser/OS, device-history counts. |
| `amount` | Transaction monetary value, amount-derived deviations. |
| `merchant` | Merchant / payment-type / channel attributes. |
| `pca_anonymised` | ULB's V1–V28 PCA-projected features (no domain meaning). |
| `account` | Account-level history (months at address, prior cards, credit limit). |
| `session` | Session duration, keep-alive, source/foreign-request flags. |
| `location` | Geographic / location features (incl. anomaly scores). |
| `risk_score` | Pre-computed risk scores produced upstream. |
| `other` | Catch-all for anything not cleanly classifiable. |

## Per-client mapping

### ULB (32 features incl. Time, Amount, V1–V28)

| Feature | Category |
|---|---|
| Time | temporal |
| Amount | amount |
| V1 – V28 | pca_anonymised (each of the 28 columns) |

### BAF (31 features)

| Feature | Category |
|---|---|
| income | identity |
| name_email_similarity | identity |
| prev_address_months_count | account |
| current_address_months_count | account |
| customer_age | identity |
| days_since_request | temporal |
| intended_balcon_amount | amount |
| payment_type | merchant |
| zip_count_4w | velocity |
| velocity_6h | velocity |
| velocity_24h | velocity |
| velocity_4w | velocity |
| bank_branch_count_8w | velocity |
| date_of_birth_distinct_emails_4w | identity |
| employment_status | identity |
| credit_risk_score | risk_score |
| email_is_free | identity |
| housing_status | identity |
| phone_home_valid | identity |
| phone_mobile_valid | identity |
| bank_months_count | account |
| has_other_cards | account |
| proposed_credit_limit | account |
| foreign_request | session |
| source | session |
| session_length_in_minutes | session |
| device_os | device |
| keep_alive_session | session |
| device_distinct_emails_8w | device |
| device_fraud_count | device |
| month | temporal |

### Synthetic (10 features)

| Feature | Category |
|---|---|
| amount | amount |
| transaction_type | merchant |
| merchant_category | merchant |
| location | location |
| device_used | device |
| time_since_last_transaction | temporal |
| spending_deviation_score | amount |
| velocity_score | velocity |
| geo_anomaly_score | location |
| payment_channel | merchant |

## Notes for interpretation

- The `pca_anonymised` category is unique to ULB and will dominate ULB's
  totals by construction (28 features sum together). When comparing across
  clients, this category should be read as "any signal" for ULB — it cannot
  be meaningfully aligned to the named-domain categories used elsewhere.
- `velocity` exists in both BAF and Synthetic (and dominates Synthetic by
  share of features), so it is the most defensible cross-client category.
- `amount` exists in all three (ULB Amount, BAF intended_balcon_amount,
  Synthetic amount + spending_deviation_score) and is the only category
  with explicit cross-client coverage.
- `temporal` exists in all three (ULB Time, BAF days_since_request + month,
  Synthetic time_since_last_transaction) — second cross-client category.