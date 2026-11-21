"""Synthetic Clinic SaaS database generator with reproducible seed (seed=42).

Generates ~5,000 patients and ~30,000 appointments over a 2-year period,
along with doctors, treatments, invoices, and payments.
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)


def build_schema(metadata: MetaData) -> dict[str, Table]:
    """Define the clinic schema with strict types, constraints, and analytics indexes."""
    patients = Table(
        "patients",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("first_name", String(50), nullable=False),
        Column("last_name", String(50), nullable=False),
        Column("gender", String(10), nullable=False),
        Column("date_of_birth", Date, nullable=False),
        Column("phone", String(30), nullable=False),  # Sensitive / Denylisted
        Column("email", String(100), nullable=False),  # Sensitive / Denylisted
        Column("address", String(200), nullable=False),
        Column("created_at", DateTime, nullable=False),
        Index("idx_patients_dob", "date_of_birth"),
        Index("idx_patients_created_at", "created_at"),
    )

    doctors = Table(
        "doctors",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("first_name", String(50), nullable=False),
        Column("last_name", String(50), nullable=False),
        Column("specialty", String(50), nullable=False),
        Column("hire_date", Date, nullable=False),
        Column("hourly_rate", Float, nullable=False),
        Index("idx_doctors_specialty", "specialty"),
    )

    appointments = Table(
        "appointments",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("patient_id", Integer, ForeignKey("patients.id"), nullable=False),
        Column("doctor_id", Integer, ForeignKey("doctors.id"), nullable=False),
        Column("scheduled_at", DateTime, nullable=False),
        Column("duration_minutes", Integer, nullable=False),
        Column("status", String(20), nullable=False),  # completed, cancelled, no_show
        Column("reason", String(150), nullable=False),
        Index("idx_appointments_patient_id", "patient_id"),
        Index("idx_appointments_doctor_id", "doctor_id"),
        Index("idx_appointments_scheduled_at", "scheduled_at"),
        Index("idx_appointments_status", "status"),
    )

    treatments = Table(
        "treatments",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("appointment_id", Integer, ForeignKey("appointments.id"), nullable=False),
        Column("treatment_code", String(20), nullable=False),
        Column("description", String(150), nullable=False),
        Column("cost", Float, nullable=False),
        Index("idx_treatments_appointment_id", "appointment_id"),
        Index("idx_treatments_code", "treatment_code"),
    )

    invoices = Table(
        "invoices",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("appointment_id", Integer, ForeignKey("appointments.id"), nullable=False),
        Column("patient_id", Integer, ForeignKey("patients.id"), nullable=False),
        Column("amount", Float, nullable=False),
        Column("status", String(20), nullable=False),  # paid, unpaid, overdue
        Column("issued_at", DateTime, nullable=False),
        Column("due_at", Date, nullable=False),
        Index("idx_invoices_appointment_id", "appointment_id"),
        Index("idx_invoices_patient_id", "patient_id"),
        Index("idx_invoices_status", "status"),
        Index("idx_invoices_issued_at", "issued_at"),
    )

    payments = Table(
        "payments",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("invoice_id", Integer, ForeignKey("invoices.id"), nullable=False),
        Column("amount", Float, nullable=False),
        Column("payment_date", DateTime, nullable=False),
        Column("payment_method", String(30), nullable=False),  # credit_card, insurance, cash, bank_transfer
        Index("idx_payments_invoice_id", "invoice_id"),
        Index("idx_payments_date", "payment_date"),
        Index("idx_payments_method", "payment_method"),
    )

    return {
        "patients": patients,
        "doctors": doctors,
        "appointments": appointments,
        "treatments": treatments,
        "invoices": invoices,
        "payments": payments,
    }


def seed_database(
    db_path: str = "clinic.db",
    num_patients: int = 5000,
    num_appointments: int = 30000,
    seed: int = 42,
) -> None:
    """Generate and insert realistic synthetic clinic data using a fixed seed."""
    print(f"Seeding database '{db_path}' (seed={seed})...")
    random.seed(seed)
    Faker.seed(seed)
    fake = Faker()

    # Clean existing database file if present
    target_path = Path(db_path)
    if target_path.exists():
        target_path.unlink()

    engine = create_engine(f"sqlite:///{db_path}")
    metadata = MetaData()
    tables = build_schema(metadata)
    metadata.create_all(engine)

    specialties = [
        "General Practice",
        "Cardiology",
        "Dermatology",
        "Pediatrics",
        "Orthopedics",
        "Neurology",
        "Oncology",
        "Psychiatry",
    ]

    doctor_rates = {
        "General Practice": 120.0,
        "Cardiology": 250.0,
        "Dermatology": 180.0,
        "Pediatrics": 140.0,
        "Orthopedics": 220.0,
        "Neurology": 260.0,
        "Oncology": 280.0,
        "Psychiatry": 190.0,
    }

    treatment_catalog = [
        ("CONS-001", "Routine Consultation", 100.0),
        ("CONS-002", "Specialist Consultation", 200.0),
        ("LAB-101", "Comprehensive Blood Panel", 75.0),
        ("LAB-102", "Lipid & Metabolic Profile", 60.0),
        ("IMG-201", "Chest X-Ray", 120.0),
        ("IMG-202", "Joint Ultrasound", 150.0),
        ("IMG-203", "Brain MRI Scan", 650.0),
        ("CARD-301", "Electrocardiogram (ECG)", 110.0),
        ("CARD-302", "Echocardiogram", 320.0),
        ("DERM-401", "Skin Biopsy & Pathology", 180.0),
        ("DERM-402", "Cryotherapy Lesion Removal", 95.0),
        ("ORTH-501", "Joint Cortisone Injection", 140.0),
        ("ORTH-502", "Splint & Immobilization", 85.0),
        ("THER-601", "Physical Therapy Session", 90.0),
        ("THER-602", "Cognitive Behavioral Therapy", 130.0),
    ]

    reasons_for_visit = [
        "Annual routine checkup",
        "Chronic lower back pain",
        "Persistent dry cough and fatigue",
        "Follow-up on hypertension management",
        "Skin rash with mild pruritus",
        "Recurrent migraine headaches",
        "Elevated fasting blood glucose follow-up",
        "Knee pain after physical activity",
        "Chest tightness during exertion",
        "Childhood developmental wellness check",
        "Seasonal allergy symptom flare",
        "Post-operative surgical follow-up",
        "Unexplained weight fluctuations",
        "Anxiety and insomnia consultation",
        "Dermatological lesion screening",
    ]

    start_date = datetime(2023, 1, 1, 8, 0, 0)
    end_date = datetime(2024, 12, 31, 18, 0, 0)
    total_seconds = int((end_date - start_date).total_seconds())

    with engine.begin() as conn:
        # 1. Doctors (24 doctors, 3 per specialty)
        doctor_rows = []
        doc_id = 1
        for specialty in specialties:
            for _ in range(3):
                first_name = fake.first_name()
                last_name = fake.last_name()
                hire_days_ago = random.randint(300, 2500)
                hire_date = (start_date - timedelta(days=hire_days_ago)).date()
                rate = doctor_rates[specialty] + random.choice([-10.0, 0.0, 10.0, 20.0])
                doctor_rows.append(
                    {
                        "id": doc_id,
                        "first_name": first_name,
                        "last_name": last_name,
                        "specialty": specialty,
                        "hire_date": hire_date,
                        "hourly_rate": round(rate, 2),
                    }
                )
                doc_id += 1

        conn.execute(tables["doctors"].insert(), doctor_rows)
        num_doctors = len(doctor_rows)
        print(f"  Inserted {num_doctors} doctors.")

        # 2. Patients (~5,000)
        patient_rows = []
        for pid in range(1, num_patients + 1):
            gender = random.choice(["Male", "Female"])
            first_name = fake.first_name_male() if gender == "Male" else fake.first_name_female()
            last_name = fake.last_name()
            dob_days = random.randint(365 * 18, 365 * 85)
            dob = (start_date - timedelta(days=dob_days)).date()
            phone = fake.phone_number()
            email = f"{first_name.lower()}.{last_name.lower()}{pid}@{fake.free_email_domain()}"
            address = fake.address().replace("\n", ", ")
            created_offset = random.randint(0, total_seconds // 2)
            created_at = start_date + timedelta(seconds=created_offset)

            patient_rows.append(
                {
                    "id": pid,
                    "first_name": first_name,
                    "last_name": last_name,
                    "gender": gender,
                    "date_of_birth": dob,
                    "phone": phone,
                    "email": email,
                    "address": address,
                    "created_at": created_at,
                }
            )

        # Batch insert patients
        batch_size = 1000
        for i in range(0, len(patient_rows), batch_size):
            conn.execute(tables["patients"].insert(), patient_rows[i : i + batch_size])
        print(f"  Inserted {num_patients} patients.")

        # 3. Appointments (~30,000)
        # Weighted status: completed (78%), cancelled (12%), no_show (10%)
        status_choices = ["completed", "cancelled", "no_show"]
        status_weights = [0.78, 0.12, 0.10]
        durations = [15, 30, 45, 60]

        appointment_rows = []
        treatment_rows = []
        invoice_rows = []
        payment_rows = []

        treatment_id = 1
        invoice_id = 1
        payment_id = 1

        for appt_id in range(1, num_appointments + 1):
            patient_id = random.randint(1, num_patients)
            doctor_id = random.randint(1, num_doctors)
            offset_sec = random.randint(0, total_seconds)
            sched_time = start_date + timedelta(seconds=offset_sec)
            # Round to 15-minute slot during clinic hours (8:00 - 17:00)
            sched_time = sched_time.replace(
                hour=random.randint(8, 16),
                minute=random.choice([0, 15, 30, 45]),
                second=0,
            )
            duration = random.choice(durations)
            status = random.choices(status_choices, weights=status_weights, k=1)[0]
            reason = random.choice(reasons_for_visit)

            appointment_rows.append(
                {
                    "id": appt_id,
                    "patient_id": patient_id,
                    "doctor_id": doctor_id,
                    "scheduled_at": sched_time,
                    "duration_minutes": duration,
                    "status": status,
                    "reason": reason,
                }
            )

            # Treatments & Invoices only for completed visits
            if status == "completed":
                # 1 to 3 treatments
                k_treat = random.choices([1, 2, 3], weights=[0.60, 0.30, 0.10], k=1)[0]
                selected_treatments = random.sample(treatment_catalog, k_treat)
                total_treatment_cost = 0.0

                for code, desc, cost in selected_treatments:
                    treatment_rows.append(
                        {
                            "id": treatment_id,
                            "appointment_id": appt_id,
                            "treatment_code": code,
                            "description": desc,
                            "cost": cost,
                        }
                    )
                    treatment_id += 1
                    total_treatment_cost += cost

                # Invoice
                issued_at = sched_time + timedelta(hours=random.randint(1, 4))
                due_at = (issued_at + timedelta(days=30)).date()
                invoice_status = random.choices(["paid", "unpaid", "overdue"], weights=[0.85, 0.10, 0.05], k=1)[0]

                invoice_rows.append(
                    {
                        "id": invoice_id,
                        "appointment_id": appt_id,
                        "patient_id": patient_id,
                        "amount": round(total_treatment_cost, 2),
                        "status": invoice_status,
                        "issued_at": issued_at,
                        "due_at": due_at,
                    }
                )

                # Payment if paid
                if invoice_status == "paid":
                    pay_offset_days = random.randint(0, 20)
                    payment_date = issued_at + timedelta(days=pay_offset_days, hours=random.randint(1, 10))
                    method = random.choices(
                        ["credit_card", "insurance", "cash", "bank_transfer"],
                        weights=[0.45, 0.35, 0.10, 0.10],
                        k=1,
                    )[0]
                    payment_rows.append(
                        {
                            "id": payment_id,
                            "invoice_id": invoice_id,
                            "amount": round(total_treatment_cost, 2),
                            "payment_date": payment_date,
                            "payment_method": method,
                        }
                    )
                    payment_id += 1

                invoice_id += 1

        # Batch insert appointments, treatments, invoices, payments
        for i in range(0, len(appointment_rows), batch_size):
            conn.execute(tables["appointments"].insert(), appointment_rows[i : i + batch_size])
        print(f"  Inserted {len(appointment_rows)} appointments.")

        for i in range(0, len(treatment_rows), batch_size):
            conn.execute(tables["treatments"].insert(), treatment_rows[i : i + batch_size])
        print(f"  Inserted {len(treatment_rows)} treatments.")

        for i in range(0, len(invoice_rows), batch_size):
            conn.execute(tables["invoices"].insert(), invoice_rows[i : i + batch_size])
        print(f"  Inserted {len(invoice_rows)} invoices.")

        for i in range(0, len(payment_rows), batch_size):
            conn.execute(tables["payments"].insert(), payment_rows[i : i + batch_size])
        print(f"  Inserted {len(payment_rows)} payments.")

    print(f"Database generation complete! Target: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the synthetic clinic database.")
    parser.add_argument("--db-path", default="clinic.db", help="Path to SQLite database file")
    parser.add_argument("--patients", type=int, default=5000, help="Number of patients to generate")
    parser.add_argument("--appointments", type=int, default=30000, help="Number of appointments to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    seed_database(
        db_path=args.db_path,
        num_patients=args.patients,
        num_appointments=args.appointments,
        seed=args.seed,
    )
