# Evaluation Benchmark Changelog

This document tracks all modifications made to benchmark queries (`eval/questions.yaml`), gold SQL definitions, and evaluation criteria, along with the precise technical rationale for each change.

---

## [Stage 4 Audit Refinements] - 2026-11-22

### Gold Query Bug Fixes

#### 1. Question `q26`
- **Question**: *"What is the average treatment cost for each treatment code, ordered from highest to lowest average cost?"*
- **Previous Gold SQL**:
  ```sql
  SELECT treatment_code, ROUND(AVG(cost), 2) AS avg_cost, COUNT(*) AS treatment_count 
  FROM treatments 
  GROUP BY treatment_code 
  ORDER BY avg_cost DESC
  ```
- **Updated Gold SQL**:
  ```sql
  SELECT treatment_code, ROUND(AVG(cost), 2) AS avg_cost 
  FROM treatments 
  GROUP BY treatment_code 
  ORDER BY avg_cost DESC
  ```
- **Rationale**: The question specifically requests the average treatment cost per treatment code and does not request `treatment_count`. Including `treatment_count` in the gold projection caused correctly predicted 2-column answers to fail the exact comparator.
- **Question Text**: Unchanged (preserves evaluation cache).

#### 2. Question `q33`
- **Question**: *"What is the payment collection efficiency ratio (total payments amount divided by total invoiced amount) per specialty as a percentage rounded to 2 decimals, ordered by collection rate descending?"*
- **Previous Gold SQL**:
  ```sql
  WITH specialty_invoices AS (
      SELECT d.specialty, SUM(i.amount) AS total_invoiced 
      FROM doctors d 
      JOIN appointments a ON d.id = a.doctor_id 
      JOIN invoices i ON a.id = i.appointment_id 
      GROUP BY d.specialty
  ), 
  specialty_payments AS (
      SELECT d.specialty, SUM(p.amount) AS total_paid 
      FROM doctors d 
      JOIN appointments a ON d.id = a.doctor_id 
      JOIN invoices i ON a.id = i.appointment_id 
      JOIN payments p ON i.id = p.invoice_id 
      GROUP BY d.specialty
  ) 
  SELECT si.specialty, ROUND(si.total_invoiced, 2) AS invoiced, ROUND(COALESCE(sp.total_paid, 0), 2) AS collected, ROUND(100.0 * COALESCE(sp.total_paid, 0) / si.total_invoiced, 2) AS collection_rate_pct 
  FROM specialty_invoices si 
  LEFT JOIN specialty_payments sp ON si.specialty = sp.specialty 
  ORDER BY collection_rate_pct DESC
  ```
- **Updated Gold SQL**:
  ```sql
  WITH specialty_invoices AS (
      SELECT d.specialty, SUM(i.amount) AS total_invoiced 
      FROM doctors d 
      JOIN appointments a ON d.id = a.doctor_id 
      JOIN invoices i ON a.id = i.appointment_id 
      GROUP BY d.specialty
  ), 
  specialty_payments AS (
      SELECT d.specialty, SUM(p.amount) AS total_paid 
      FROM doctors d 
      JOIN appointments a ON d.id = a.doctor_id 
      JOIN invoices i ON a.id = i.appointment_id 
      JOIN payments p ON i.id = p.invoice_id 
      GROUP BY d.specialty
  ) 
  SELECT si.specialty, ROUND(100.0 * COALESCE(sp.total_paid, 0) / si.total_invoiced, 2) AS collection_rate_pct 
  FROM specialty_invoices si 
  LEFT JOIN specialty_payments sp ON si.specialty = sp.specialty 
  ORDER BY collection_rate_pct DESC
  ```
- **Rationale**: The question asks for the payment collection efficiency ratio per specialty as a percentage rounded to 2 decimals. The intermediate aggregation columns `invoiced` and `collected` were not requested by the user prompt. Removing them aligns the gold output strictly with the user question.
- **Question Text**: Unchanged (preserves evaluation cache).

---

### Ambiguity Clarifications

#### 3. Question `q47`
- **Previous Question Text**: *"What is the monthly drop-off in completed appointment count from month to month in 2024, ordered by month ascending?"*
- **Updated Question Text**: *"What is the monthly drop-off in completed appointment count from month to month in 2024, ordered by month ascending, where drop-off = previous month's completed count minus the current month's?"*
- **Rationale**: Explicitly defines the non-standard term "drop-off" as `previous_month_completed - current_month_completed` rather than current minus previous or a percentage drop, removing ambiguity for the model.
