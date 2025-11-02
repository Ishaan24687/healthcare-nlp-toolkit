"""
Generate synthetic clinical notes for urgency classification.
Each note is a paragraph mixing medical phrases — urgent notes lean on critical
language while routine notes use standard follow-up vocabulary.
"""

import random
import json
import os

URGENT_PHRASES = [
    "severe chest pain radiating to left arm",
    "acute respiratory distress with O2 sat dropping to 82%",
    "patient unresponsive to verbal stimuli",
    "code blue called on floor 3",
    "massive hemorrhage from surgical site",
    "cardiac arrest witnessed by nursing staff",
    "sepsis suspected with lactate of 4.2",
    "intubation required due to airway compromise",
    "acute MI confirmed on 12-lead ECG",
    "GCS score of 6 upon arrival",
    "status epilepticus not responding to lorazepam",
    "tension pneumothorax with tracheal deviation",
    "anaphylactic shock after medication administration",
    "acute stroke symptoms with left-sided weakness",
    "BP dropping to 70/40 despite fluid resuscitation",
    "patient in ventricular fibrillation",
    "dissecting aortic aneurysm on CT",
    "acute pulmonary embolism confirmed",
    "bilateral pulmonary infiltrates with worsening hypoxia",
    "emergent dialysis required for hyperkalemia of 7.1",
    "patient found down, unknown downtime",
    "massive transfusion protocol activated",
    "peritonitis with rigid abdomen",
    "compound fracture with vascular compromise",
    "severe traumatic brain injury",
    "DKA with pH of 6.9 and altered mental status",
    "active GI bleed with hemoglobin dropping to 5.2",
    "meningitis suspected with nuchal rigidity and fever of 104F",
    "compartment syndrome right lower extremity",
    "eclampsia with seizures at 34 weeks gestation",
]

ROUTINE_PHRASES = [
    "routine follow up for hypertension management",
    "annual physical examination completed",
    "medication refill for metformin 500mg",
    "vitals within normal range BP 120/78 HR 72",
    "patient doing well post-operatively day 14",
    "wound healing normally without signs of infection",
    "routine diabetes check HbA1c 6.8",
    "well-child visit 18 month immunizations given",
    "stable chronic kidney disease stage 2",
    "routine mammogram screening ordered",
    "patient reports no new complaints",
    "continue current medication regimen",
    "routine eye exam for diabetic retinopathy screening",
    "flu vaccination administered",
    "blood pressure well controlled on current dose",
    "routine lab work ordered CBC and CMP",
    "follow up in 3 months for lipid panel",
    "physical therapy progressing well",
    "routine colonoscopy screening at age 50",
    "stable anxiety on current SSRI dose",
    "allergy shots administered per schedule",
    "prenatal visit 28 weeks uncomplicated pregnancy",
    "COPD stable on current inhaler regimen",
    "smoking cessation counseling provided",
    "BMI discussed diet and exercise plan reviewed",
    "routine skin check no suspicious lesions",
    "hearing test normal for age",
    "dental referral for routine cleaning",
    "patient satisfied with current treatment plan",
    "return to work clearance provided",
]

CONNECTORS = [
    "Patient presents with", "Assessment shows", "Noted on exam:",
    "Clinical findings include", "Evaluation reveals",
    "HPI:", "Currently experiencing", "On examination,",
    "Chart review indicates", "Nursing notes report",
]

DEMOGRAPHICS = [
    "72 yo male", "45 yo female", "58 yo male", "31 yo female",
    "67 yo female", "83 yo male", "29 yo male", "55 yo female",
    "41 yo male", "63 yo female", "78 yo male", "36 yo female",
    "50 yo male", "22 yo female", "89 yo male", "47 yo female",
]


def generate_clinical_note(label: str, min_phrases: int = 2, max_phrases: int = 5) -> str:
    phrases = URGENT_PHRASES if label == "urgent" else ROUTINE_PHRASES
    num_phrases = random.randint(min_phrases, max_phrases)
    selected = random.sample(phrases, min(num_phrases, len(phrases)))

    demographic = random.choice(DEMOGRAPHICS)
    connector = random.choice(CONNECTORS)

    # ~20% chance to sprinkle in a phrase from the other class for realism
    if random.random() < 0.2:
        noise_pool = ROUTINE_PHRASES if label == "urgent" else URGENT_PHRASES
        selected.append(random.choice(noise_pool))
        random.shuffle(selected)

    clinical_note = f"{demographic}. {connector} {'. '.join(selected)}."
    return clinical_note


def generate_dataset(n_samples: int = 500, seed: int = 42) -> list[dict]:
    random.seed(seed)

    # slight class imbalance to reflect reality — more routine than urgent
    n_urgent = int(n_samples * 0.4)
    n_routine = n_samples - n_urgent

    dataset = []
    for _ in range(n_urgent):
        clinical_note = generate_clinical_note("urgent")
        dataset.append({"text": clinical_note, "label": "urgent", "label_id": 1})

    for _ in range(n_routine):
        clinical_note = generate_clinical_note("routine")
        dataset.append({"text": clinical_note, "label": "routine", "label_id": 0})

    random.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clinical_notes.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} clinical notes to {output_path}")

    urgent_count = sum(1 for d in dataset if d["label"] == "urgent")
    routine_count = len(dataset) - urgent_count
    print(f"  Urgent: {urgent_count} ({urgent_count/len(dataset)*100:.1f}%)")
    print(f"  Routine: {routine_count} ({routine_count/len(dataset)*100:.1f}%)")

    print(f"\nSample urgent note:\n  {dataset[0]['text'][:200]}...")
    print(f"\nSample routine note:\n  {[d for d in dataset if d['label'] == 'routine'][0]['text'][:200]}...")


if __name__ == "__main__":
    dataset = generate_dataset(n_samples=500)
    save_dataset(dataset, output_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
