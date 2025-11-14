"""
Generate synthetic CPT code dataset with descriptions, categories,
regional reimbursement rates, and volume data.
Descriptions mirror real CPT language to make TF-IDF analysis realistic.
"""

import json
import os
import random
import numpy as np
import pandas as pd

CPT_CODES = [
    # E/M (Evaluation and Management)
    {"code": "99201", "description": "Office or other outpatient visit for evaluation and management of a new patient, straightforward medical decision making", "category": "E/M", "base_rate": 45},
    {"code": "99202", "description": "Office or other outpatient visit new patient low level medical decision making", "category": "E/M", "base_rate": 75},
    {"code": "99203", "description": "Office or other outpatient visit new patient moderate level medical decision making", "category": "E/M", "base_rate": 110},
    {"code": "99204", "description": "Office or other outpatient visit new patient moderate to high complexity medical decision making", "category": "E/M", "base_rate": 170},
    {"code": "99205", "description": "Office or other outpatient visit new patient high complexity medical decision making comprehensive examination", "category": "E/M", "base_rate": 210},
    {"code": "99211", "description": "Office or other outpatient visit established patient minimal presenting problem", "category": "E/M", "base_rate": 25},
    {"code": "99212", "description": "Office or other outpatient visit established patient straightforward medical decision making", "category": "E/M", "base_rate": 50},
    {"code": "99213", "description": "Office or other outpatient visit established patient low complexity medical decision making", "category": "E/M", "base_rate": 80},
    {"code": "99214", "description": "Office or other outpatient visit established patient moderate complexity medical decision making", "category": "E/M", "base_rate": 120},
    {"code": "99215", "description": "Office or other outpatient visit established patient high complexity medical decision making", "category": "E/M", "base_rate": 175},
    {"code": "99221", "description": "Initial hospital inpatient care per day low complexity", "category": "E/M", "base_rate": 150},
    {"code": "99222", "description": "Initial hospital inpatient care per day moderate complexity", "category": "E/M", "base_rate": 200},
    {"code": "99223", "description": "Initial hospital inpatient care per day high complexity", "category": "E/M", "base_rate": 275},
    {"code": "99231", "description": "Subsequent hospital inpatient care per day stable recovering condition", "category": "E/M", "base_rate": 80},
    {"code": "99232", "description": "Subsequent hospital inpatient care per day responding inadequately to therapy", "category": "E/M", "base_rate": 115},
    {"code": "99233", "description": "Subsequent hospital inpatient care per day unstable or significant complication", "category": "E/M", "base_rate": 160},
    {"code": "99281", "description": "Emergency department visit self-limited or minor problem", "category": "E/M", "base_rate": 55},
    {"code": "99282", "description": "Emergency department visit low to moderate severity problem", "category": "E/M", "base_rate": 95},
    {"code": "99283", "description": "Emergency department visit moderate severity problem", "category": "E/M", "base_rate": 145},
    {"code": "99284", "description": "Emergency department visit high severity problem requiring urgent evaluation", "category": "E/M", "base_rate": 250},
    {"code": "99285", "description": "Emergency department visit high severity problem posing immediate significant threat to life", "category": "E/M", "base_rate": 375},
    {"code": "99241", "description": "Office consultation new or established patient straightforward decision making", "category": "E/M", "base_rate": 65},
    {"code": "99242", "description": "Office consultation new or established patient low complexity", "category": "E/M", "base_rate": 100},
    {"code": "99243", "description": "Office consultation new or established patient moderate complexity", "category": "E/M", "base_rate": 150},
    {"code": "99244", "description": "Office consultation new or established patient moderate to high complexity", "category": "E/M", "base_rate": 210},
    {"code": "99245", "description": "Office consultation new or established patient high complexity", "category": "E/M", "base_rate": 275},

    # Surgery
    {"code": "10021", "description": "Fine needle aspiration biopsy without imaging guidance first lesion", "category": "Surgery", "base_rate": 120},
    {"code": "10060", "description": "Incision and drainage of abscess simple or single", "category": "Surgery", "base_rate": 180},
    {"code": "10120", "description": "Incision and removal of foreign body subcutaneous tissue simple", "category": "Surgery", "base_rate": 200},
    {"code": "11042", "description": "Debridement subcutaneous tissue including epidermis dermis and subcutaneous tissue", "category": "Surgery", "base_rate": 160},
    {"code": "11102", "description": "Tangential biopsy of skin single lesion", "category": "Surgery", "base_rate": 95},
    {"code": "11200", "description": "Removal of skin tags multiple fibrocutaneous tags any area", "category": "Surgery", "base_rate": 110},
    {"code": "17000", "description": "Destruction of premalignant lesion first lesion", "category": "Surgery", "base_rate": 75},
    {"code": "17110", "description": "Destruction of benign lesions other than skin tags up to 14 lesions", "category": "Surgery", "base_rate": 95},
    {"code": "19120", "description": "Excision of cyst or aberrant breast tissue open", "category": "Surgery", "base_rate": 850},
    {"code": "20610", "description": "Arthrocentesis aspiration or injection of major joint or bursa without ultrasound guidance", "category": "Surgery", "base_rate": 105},
    {"code": "27447", "description": "Arthroplasty knee condyle and plateau medial and lateral compartments total knee replacement", "category": "Surgery", "base_rate": 1800},
    {"code": "27130", "description": "Arthroplasty acetabular and proximal femoral prosthetic replacement total hip arthroplasty", "category": "Surgery", "base_rate": 2100},
    {"code": "29881", "description": "Arthroscopy knee surgical with meniscectomy medial or lateral including any meniscal shaving", "category": "Surgery", "base_rate": 650},
    {"code": "43239", "description": "Esophagogastroduodenoscopy with biopsy single or multiple", "category": "Surgery", "base_rate": 320},
    {"code": "43249", "description": "Esophagogastroduodenoscopy with balloon dilation of esophagus", "category": "Surgery", "base_rate": 450},
    {"code": "44388", "description": "Colonoscopy through stoma diagnostic including collection of specimens by brushing", "category": "Surgery", "base_rate": 380},
    {"code": "45378", "description": "Colonoscopy flexible diagnostic including collection of specimens by brushing or washing", "category": "Surgery", "base_rate": 350},
    {"code": "45380", "description": "Colonoscopy flexible with biopsy single or multiple", "category": "Surgery", "base_rate": 420},
    {"code": "45385", "description": "Colonoscopy flexible with removal of tumor polyp or other lesion by snare technique", "category": "Surgery", "base_rate": 520},
    {"code": "47562", "description": "Laparoscopy surgical cholecystectomy", "category": "Surgery", "base_rate": 1200},
    {"code": "49505", "description": "Repair initial inguinal hernia any age reducible", "category": "Surgery", "base_rate": 750},
    {"code": "50590", "description": "Lithotripsy extracorporeal shock wave", "category": "Surgery", "base_rate": 1100},
    {"code": "55700", "description": "Biopsy prostate needle or punch single or multiple any approach", "category": "Surgery", "base_rate": 280},
    {"code": "58661", "description": "Laparoscopy surgical with removal of adnexal structures", "category": "Surgery", "base_rate": 1400},
    {"code": "62323", "description": "Injection including imaging guidance of diagnostic or therapeutic substance epidural or subarachnoid lumbar or sacral", "category": "Surgery", "base_rate": 350},
    {"code": "64483", "description": "Injection anesthetic agent or steroid transforaminal epidural lumbar or sacral single level", "category": "Surgery", "base_rate": 400},
    {"code": "66984", "description": "Extracapsular cataract removal with insertion of intraocular lens prosthesis", "category": "Surgery", "base_rate": 950},

    # Radiology
    {"code": "70553", "description": "Magnetic resonance imaging brain without contrast then with contrast and further sequences", "category": "Radiology", "base_rate": 475},
    {"code": "71046", "description": "Radiologic examination chest two views frontal and lateral", "category": "Radiology", "base_rate": 55},
    {"code": "71250", "description": "Computed tomography thorax without contrast material", "category": "Radiology", "base_rate": 280},
    {"code": "71260", "description": "Computed tomography thorax with contrast material", "category": "Radiology", "base_rate": 350},
    {"code": "72148", "description": "Magnetic resonance imaging spinal canal and contents lumbar without contrast", "category": "Radiology", "base_rate": 420},
    {"code": "72193", "description": "Computed tomography pelvis with contrast material", "category": "Radiology", "base_rate": 310},
    {"code": "73721", "description": "Magnetic resonance imaging any joint of lower extremity without contrast", "category": "Radiology", "base_rate": 400},
    {"code": "74177", "description": "Computed tomography abdomen and pelvis with contrast material", "category": "Radiology", "base_rate": 380},
    {"code": "74178", "description": "Computed tomography abdomen and pelvis without contrast then with contrast material and further sections", "category": "Radiology", "base_rate": 450},
    {"code": "76700", "description": "Ultrasound abdominal real time with image documentation complete", "category": "Radiology", "base_rate": 180},
    {"code": "76830", "description": "Ultrasound transvaginal", "category": "Radiology", "base_rate": 200},
    {"code": "76856", "description": "Ultrasound pelvic non-obstetric real time with image documentation complete", "category": "Radiology", "base_rate": 175},
    {"code": "77067", "description": "Screening mammography bilateral including computer aided detection", "category": "Radiology", "base_rate": 145},
    {"code": "77080", "description": "Dual-energy X-ray absorptiometry bone density study one or more sites axial skeleton", "category": "Radiology", "base_rate": 110},
    {"code": "78452", "description": "Myocardial perfusion imaging tomographic SPECT multiple studies at rest or stress", "category": "Radiology", "base_rate": 550},
    {"code": "78816", "description": "Positron emission tomography PET for tumor whole body imaging", "category": "Radiology", "base_rate": 1200},

    # Pathology/Lab
    {"code": "80048", "description": "Basic metabolic panel including calcium ionized", "category": "Pathology", "base_rate": 22},
    {"code": "80053", "description": "Comprehensive metabolic panel including albumin bilirubin calcium", "category": "Pathology", "base_rate": 28},
    {"code": "80061", "description": "Lipid panel including cholesterol lipoprotein triglycerides", "category": "Pathology", "base_rate": 35},
    {"code": "80076", "description": "Hepatic function panel including albumin bilirubin total and direct", "category": "Pathology", "base_rate": 30},
    {"code": "85025", "description": "Blood count complete CBC hemogram and platelet count automated", "category": "Pathology", "base_rate": 15},
    {"code": "85610", "description": "Prothrombin time PT", "category": "Pathology", "base_rate": 12},
    {"code": "85730", "description": "Thromboplastin time partial PTT plasma or whole blood", "category": "Pathology", "base_rate": 14},
    {"code": "86140", "description": "C-reactive protein CRP", "category": "Pathology", "base_rate": 18},
    {"code": "86235", "description": "Nuclear antigen antibody ANA each antibody", "category": "Pathology", "base_rate": 25},
    {"code": "87086", "description": "Culture bacterial urine quantitative colony count", "category": "Pathology", "base_rate": 20},
    {"code": "87491", "description": "Infectious agent detection by nucleic acid chlamydia trachomatis amplified probe", "category": "Pathology", "base_rate": 42},
    {"code": "88305", "description": "Level IV surgical pathology gross and microscopic examination", "category": "Pathology", "base_rate": 95},
    {"code": "84443", "description": "Thyroid stimulating hormone TSH", "category": "Pathology", "base_rate": 28},
    {"code": "83036", "description": "Hemoglobin glycosylated A1c", "category": "Pathology", "base_rate": 22},

    # Medicine
    {"code": "90834", "description": "Psychotherapy 45 minutes with patient", "category": "Medicine", "base_rate": 100},
    {"code": "90837", "description": "Psychotherapy 60 minutes with patient", "category": "Medicine", "base_rate": 135},
    {"code": "90847", "description": "Family psychotherapy conjoint psychotherapy with patient present", "category": "Medicine", "base_rate": 120},
    {"code": "92014", "description": "Ophthalmological services medical examination and evaluation with initiation or continuation of diagnostic and treatment program comprehensive", "category": "Medicine", "base_rate": 95},
    {"code": "93000", "description": "Electrocardiogram routine ECG with at least 12 leads with interpretation and report", "category": "Medicine", "base_rate": 35},
    {"code": "93306", "description": "Echocardiography transthoracic real time with image documentation 2D with M-mode recording complete with spectral Doppler and color flow Doppler", "category": "Medicine", "base_rate": 350},
    {"code": "93458", "description": "Catheter placement in coronary artery for coronary angiography including intraprocedural injection", "category": "Medicine", "base_rate": 850},
    {"code": "95810", "description": "Polysomnography sleep study 6 or more hours recording", "category": "Medicine", "base_rate": 500},
    {"code": "96372", "description": "Therapeutic prophylactic or diagnostic injection subcutaneous or intramuscular", "category": "Medicine", "base_rate": 30},
    {"code": "96413", "description": "Chemotherapy administration intravenous infusion technique up to one hour single or initial substance or drug", "category": "Medicine", "base_rate": 180},
    {"code": "97110", "description": "Therapeutic procedure one or more areas each 15 minutes therapeutic exercises", "category": "Medicine", "base_rate": 40},
    {"code": "97140", "description": "Manual therapy techniques one or more regions each 15 minutes", "category": "Medicine", "base_rate": 38},
    {"code": "97530", "description": "Therapeutic activities direct one-on-one patient contact each 15 minutes", "category": "Medicine", "base_rate": 42},
    {"code": "99381", "description": "Initial comprehensive preventive medicine evaluation and management infant age younger than 1 year", "category": "Medicine", "base_rate": 125},
    {"code": "99396", "description": "Periodic comprehensive preventive medicine reevaluation established patient age 40-64 years", "category": "Medicine", "base_rate": 140},

    # ASC (Ambulatory Surgery Center)
    {"code": "G0105", "description": "Colorectal cancer screening colonoscopy on individual at high risk", "category": "ASC", "base_rate": 420},
    {"code": "G0121", "description": "Colorectal cancer screening colonoscopy on individual not meeting criteria for high risk", "category": "ASC", "base_rate": 380},
    {"code": "G0127", "description": "Trimming of dystrophic nails any number", "category": "ASC", "base_rate": 40},
    {"code": "G0202", "description": "Screening mammography producing direct digital image bilateral all views", "category": "ASC", "base_rate": 140},
    {"code": "G0378", "description": "Hospital observation service per hour", "category": "ASC", "base_rate": 65},
    {"code": "G0463", "description": "Hospital outpatient clinic visit for assessment and management of a patient", "category": "ASC", "base_rate": 95},
    {"code": "Q9967", "description": "Low osmolar contrast material 300-399 mg iodine per ml per ml", "category": "ASC", "base_rate": 15},

    # Anesthesia
    {"code": "00100", "description": "Anesthesia for procedures on salivary glands including biopsy", "category": "Anesthesia", "base_rate": 300},
    {"code": "00400", "description": "Anesthesia for procedures on the integumentary system on the extremities anterior trunk and perineum", "category": "Anesthesia", "base_rate": 280},
    {"code": "00520", "description": "Anesthesia for closed chest procedures including bronchoscopy not otherwise specified", "category": "Anesthesia", "base_rate": 450},
    {"code": "00630", "description": "Anesthesia for procedures in lumbar region including hip", "category": "Anesthesia", "base_rate": 500},
    {"code": "00810", "description": "Anesthesia for lower intestinal endoscopic procedures endoscope introduced distal to duodenum", "category": "Anesthesia", "base_rate": 350},
    {"code": "01402", "description": "Anesthesia for open or surgical arthroscopic procedures on knee joint total knee arthroplasty", "category": "Anesthesia", "base_rate": 600},
    {"code": "01996", "description": "Daily hospital management of epidural or subarachnoid continuous drug administration", "category": "Anesthesia", "base_rate": 150},
]

REGIONS = ["Northeast", "South", "Midwest", "West"]

# regional adjustment factors — the South systematically underpays on ASC codes
REGIONAL_FACTORS = {
    "Northeast": {"E/M": 1.12, "Surgery": 1.08, "Radiology": 1.05, "Pathology": 1.03, "Medicine": 1.10, "ASC": 1.05, "Anesthesia": 1.07},
    "South":     {"E/M": 0.92, "Surgery": 0.95, "Radiology": 0.93, "Pathology": 0.97, "Medicine": 0.94, "ASC": 0.85, "Anesthesia": 0.93},
    "Midwest":   {"E/M": 0.98, "Surgery": 1.00, "Radiology": 0.97, "Pathology": 1.00, "Medicine": 0.98, "ASC": 0.98, "Anesthesia": 0.99},
    "West":      {"E/M": 1.15, "Surgery": 1.10, "Radiology": 1.08, "Pathology": 1.02, "Medicine": 1.08, "ASC": 1.10, "Anesthesia": 1.12},
}


def generate_cpt_dataset(seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    records = []
    for cpt in CPT_CODES:
        cpt_code = cpt["code"]
        cpt_description = cpt["description"]
        cpt_category = cpt["category"]
        base_reimbursement_rate = cpt["base_rate"]

        # add per-code noise so rates aren't purely deterministic
        code_noise = np.random.normal(1.0, 0.05)
        annual_volume = int(np.random.lognormal(mean=8, sigma=1.5))

        regional_rates = {}
        for region in REGIONS:
            factor = REGIONAL_FACTORS[region][cpt_category]
            # per-region random variation on top of the systematic factor
            region_noise = np.random.normal(1.0, 0.03)
            reimbursement_rate = round(base_reimbursement_rate * factor * code_noise * region_noise, 2)
            regional_rates[region] = max(reimbursement_rate, 5.0)

        national_avg_reimbursement = round(np.mean(list(regional_rates.values())), 2)

        records.append({
            "cpt_code": cpt_code,
            "description": cpt_description,
            "category": cpt_category,
            "national_avg_reimbursement": national_avg_reimbursement,
            "reimbursement_northeast": regional_rates["Northeast"],
            "reimbursement_south": regional_rates["South"],
            "reimbursement_midwest": regional_rates["Midwest"],
            "reimbursement_west": regional_rates["West"],
            "annual_volume": annual_volume,
        })

    cpt_df = pd.DataFrame(records)
    return cpt_df


def save_cpt_dataset(output_dir: str = "data"):
    cpt_df = generate_cpt_dataset()
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "cpt_codes.csv")
    cpt_df.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, "cpt_codes.json")
    cpt_df.to_json(json_path, orient="records", indent=2)

    print(f"Generated {len(cpt_df)} CPT codes")
    print(f"Saved to {csv_path} and {json_path}")
    print(f"\nCategory distribution:")
    for cpt_category, count in cpt_df["category"].value_counts().items():
        print(f"  {cpt_category}: {count}")
    print(f"\nNational avg reimbursement stats:")
    print(cpt_df["national_avg_reimbursement"].describe().to_string())

    return cpt_df


if __name__ == "__main__":
    save_cpt_dataset(output_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
