"""
Generate synthetic clinical sentences with IOB-format NER annotations.
Each sentence contains one or more medical entities: diagnoses, medications,
procedures, lab values, and dosages.
"""

import json
import os
import random

ANNOTATED_SENTENCES = [
    {
        "text": "Patient diagnosed with type 2 diabetes mellitus and started on metformin 500mg twice daily.",
        "entities": [
            (25, 51, "DIAGNOSIS", "type 2 diabetes mellitus"),
            (67, 76, "MEDICATION", "metformin"),
            (77, 82, "DOSAGE", "500mg"),
        ]
    },
    {
        "text": "CT scan of the abdomen showed acute appendicitis, surgical consult requested.",
        "entities": [
            (0, 7, "PROCEDURE", "CT scan"),
            (30, 48, "DIAGNOSIS", "acute appendicitis"),
        ]
    },
    {
        "text": "HbA1c level is 6.8%, continue current medication regimen.",
        "entities": [
            (0, 19, "LAB_VALUE", "HbA1c level is 6.8%"),
        ]
    },
    {
        "text": "Prescribed lisinopril 10mg daily for hypertension management.",
        "entities": [
            (11, 21, "MEDICATION", "lisinopril"),
            (22, 26, "DOSAGE", "10mg"),
            (37, 49, "DIAGNOSIS", "hypertension"),
        ]
    },
    {
        "text": "Echocardiogram revealed ejection fraction of 35%, consistent with heart failure.",
        "entities": [
            (0, 16, "PROCEDURE", "Echocardiogram"),
            (26, 48, "LAB_VALUE", "ejection fraction of 35%"),
            (66, 79, "DIAGNOSIS", "heart failure"),
        ]
    },
    {
        "text": "WBC count elevated at 15.2, suspect urinary tract infection.",
        "entities": [
            (0, 26, "LAB_VALUE", "WBC count elevated at 15.2"),
            (36, 59, "DIAGNOSIS", "urinary tract infection"),
        ]
    },
    {
        "text": "Administer morphine 4mg IV push for acute pain management.",
        "entities": [
            (11, 19, "MEDICATION", "morphine"),
            (20, 23, "DOSAGE", "4mg"),
        ]
    },
    {
        "text": "Colonoscopy performed, three polyps removed and sent to pathology.",
        "entities": [
            (0, 11, "PROCEDURE", "Colonoscopy"),
        ]
    },
    {
        "text": "Creatinine 1.8 mg/dL, GFR 42, consistent with stage 3 chronic kidney disease.",
        "entities": [
            (0, 20, "LAB_VALUE", "Creatinine 1.8 mg/dL"),
            (22, 28, "LAB_VALUE", "GFR 42"),
            (47, 77, "DIAGNOSIS", "stage 3 chronic kidney disease"),
        ]
    },
    {
        "text": "Patient on warfarin 5mg, INR therapeutic at 2.4.",
        "entities": [
            (11, 19, "MEDICATION", "warfarin"),
            (20, 23, "DOSAGE", "5mg"),
            (25, 48, "LAB_VALUE", "INR therapeutic at 2.4"),
        ]
    },
    {
        "text": "MRI brain shows no acute intracranial hemorrhage or mass effect.",
        "entities": [
            (0, 9, "PROCEDURE", "MRI brain"),
            (24, 48, "DIAGNOSIS", "intracranial hemorrhage"),
        ]
    },
    {
        "text": "Started amlodipine 5mg for blood pressure control, BP 145/92.",
        "entities": [
            (8, 18, "MEDICATION", "amlodipine"),
            (19, 22, "DOSAGE", "5mg"),
            (53, 59, "LAB_VALUE", "145/92"),
        ]
    },
    {
        "text": "Chest X-ray reveals bilateral pleural effusions.",
        "entities": [
            (0, 11, "PROCEDURE", "Chest X-ray"),
            (20, 48, "DIAGNOSIS", "bilateral pleural effusions"),
        ]
    },
    {
        "text": "Hemoglobin A1c is 9.2%, increase metformin to 1000mg twice daily.",
        "entities": [
            (0, 22, "LAB_VALUE", "Hemoglobin A1c is 9.2%"),
            (33, 42, "MEDICATION", "metformin"),
            (46, 52, "DOSAGE", "1000mg"),
        ]
    },
    {
        "text": "Diagnosis of community-acquired pneumonia, started on azithromycin 500mg.",
        "entities": [
            (13, 41, "DIAGNOSIS", "community-acquired pneumonia"),
            (54, 66, "MEDICATION", "azithromycin"),
            (67, 72, "DOSAGE", "500mg"),
        ]
    },
    {
        "text": "Troponin elevated at 0.45 ng/mL, concern for acute coronary syndrome.",
        "entities": [
            (0, 31, "LAB_VALUE", "Troponin elevated at 0.45 ng/mL"),
            (46, 68, "DIAGNOSIS", "acute coronary syndrome"),
        ]
    },
    {
        "text": "Lumbar puncture performed, CSF analysis pending.",
        "entities": [
            (0, 15, "PROCEDURE", "Lumbar puncture"),
            (27, 39, "PROCEDURE", "CSF analysis"),
        ]
    },
    {
        "text": "Continue atorvastatin 40mg nightly for hyperlipidemia.",
        "entities": [
            (9, 21, "MEDICATION", "atorvastatin"),
            (22, 26, "DOSAGE", "40mg"),
            (39, 53, "DIAGNOSIS", "hyperlipidemia"),
        ]
    },
    {
        "text": "Potassium level 5.8 mEq/L, administer kayexalate and recheck in 4 hours.",
        "entities": [
            (0, 25, "LAB_VALUE", "Potassium level 5.8 mEq/L"),
            (39, 50, "MEDICATION", "kayexalate"),
        ]
    },
    {
        "text": "Thyroid ultrasound shows 2.3cm nodule on right lobe, FNA biopsy recommended.",
        "entities": [
            (0, 18, "PROCEDURE", "Thyroid ultrasound"),
            (53, 63, "PROCEDURE", "FNA biopsy"),
        ]
    },
    {
        "text": "Patient with COPD exacerbation, nebulized albuterol 2.5mg and ipratropium 0.5mg given.",
        "entities": [
            (13, 30, "DIAGNOSIS", "COPD exacerbation"),
            (42, 51, "MEDICATION", "albuterol"),
            (52, 57, "DOSAGE", "2.5mg"),
            (62, 73, "MEDICATION", "ipratropium"),
            (74, 79, "DOSAGE", "0.5mg"),
        ]
    },
    {
        "text": "Serum sodium 128 mEq/L, hyponatremia likely due to SIADH.",
        "entities": [
            (0, 22, "LAB_VALUE", "Serum sodium 128 mEq/L"),
            (24, 36, "DIAGNOSIS", "hyponatremia"),
        ]
    },
    {
        "text": "Scheduled for total knee arthroplasty next Tuesday, pre-op labs drawn.",
        "entities": [
            (14, 39, "PROCEDURE", "total knee arthroplasty"),
        ]
    },
    {
        "text": "Gabapentin 300mg three times daily for neuropathic pain.",
        "entities": [
            (0, 10, "MEDICATION", "gabapentin"),
            (11, 16, "DOSAGE", "300mg"),
            (39, 55, "DIAGNOSIS", "neuropathic pain"),
        ]
    },
    {
        "text": "BNP elevated at 1200 pg/mL, consistent with decompensated heart failure.",
        "entities": [
            (0, 26, "LAB_VALUE", "BNP elevated at 1200 pg/mL"),
            (44, 71, "DIAGNOSIS", "decompensated heart failure"),
        ]
    },
    {
        "text": "D-dimer elevated at 2.4 mg/L, CT angiogram ordered to rule out PE.",
        "entities": [
            (0, 28, "LAB_VALUE", "D-dimer elevated at 2.4 mg/L"),
            (30, 43, "PROCEDURE", "CT angiogram"),
        ]
    },
    {
        "text": "Insulin glargine 20 units at bedtime for uncontrolled diabetes.",
        "entities": [
            (0, 15, "MEDICATION", "Insulin glargine"),
            (16, 24, "DOSAGE", "20 units"),
            (41, 62, "DIAGNOSIS", "uncontrolled diabetes"),
        ]
    },
    {
        "text": "Platelet count 45,000, hold heparin, concern for HIT.",
        "entities": [
            (0, 21, "LAB_VALUE", "Platelet count 45,000"),
            (28, 35, "MEDICATION", "heparin"),
        ]
    },
    {
        "text": "Bone density scan reveals osteoporosis, start alendronate 70mg weekly.",
        "entities": [
            (0, 17, "PROCEDURE", "Bone density scan"),
            (26, 38, "DIAGNOSIS", "osteoporosis"),
            (46, 57, "MEDICATION", "alendronate"),
            (58, 62, "DOSAGE", "70mg"),
        ]
    },
    {
        "text": "Blood glucose 342 mg/dL, start insulin drip per DKA protocol.",
        "entities": [
            (0, 23, "LAB_VALUE", "Blood glucose 342 mg/dL"),
            (31, 43, "MEDICATION", "insulin drip"),
        ]
    },
    {
        "text": "Cardiac catheterization shows 90% LAD stenosis, CABG consult placed.",
        "entities": [
            (0, 25, "PROCEDURE", "Cardiac catheterization"),
            (32, 47, "DIAGNOSIS", "90% LAD stenosis"),
        ]
    },
    {
        "text": "Prednisone 60mg daily taper for acute gout flare.",
        "entities": [
            (0, 10, "MEDICATION", "Prednisone"),
            (11, 15, "DOSAGE", "60mg"),
            (32, 49, "DIAGNOSIS", "acute gout flare"),
        ]
    },
    {
        "text": "Lactic acid 4.2 mmol/L, concern for sepsis, blood cultures drawn.",
        "entities": [
            (0, 22, "LAB_VALUE", "Lactic acid 4.2 mmol/L"),
            (36, 42, "DIAGNOSIS", "sepsis"),
            (44, 58, "PROCEDURE", "blood cultures"),
        ]
    },
    {
        "text": "Spirometry shows FEV1 42% predicted, severe obstructive pattern.",
        "entities": [
            (0, 10, "PROCEDURE", "Spirometry"),
            (17, 35, "LAB_VALUE", "FEV1 42% predicted"),
        ]
    },
    {
        "text": "Omeprazole 20mg daily for gastroesophageal reflux disease.",
        "entities": [
            (0, 10, "MEDICATION", "Omeprazole"),
            (11, 15, "DOSAGE", "20mg"),
            (26, 57, "DIAGNOSIS", "gastroesophageal reflux disease"),
        ]
    },
    {
        "text": "TSH level 8.4 mIU/L, suspect hypothyroidism, start levothyroxine 50mcg.",
        "entities": [
            (0, 20, "LAB_VALUE", "TSH level 8.4 mIU/L"),
            (30, 44, "DIAGNOSIS", "hypothyroidism"),
            (52, 65, "MEDICATION", "levothyroxine"),
            (66, 71, "DOSAGE", "50mcg"),
        ]
    },
    {
        "text": "EGD performed, biopsy of gastric antrum for H. pylori testing.",
        "entities": [
            (0, 3, "PROCEDURE", "EGD"),
            (16, 42, "PROCEDURE", "biopsy of gastric antrum"),
        ]
    },
    {
        "text": "Clopidogrel 75mg daily post coronary stent placement.",
        "entities": [
            (0, 11, "MEDICATION", "Clopidogrel"),
            (12, 16, "DOSAGE", "75mg"),
            (28, 52, "PROCEDURE", "coronary stent placement"),
        ]
    },
    {
        "text": "Magnesium 1.2 mg/dL, replace with IV magnesium sulfate 2g over 2 hours.",
        "entities": [
            (0, 20, "LAB_VALUE", "Magnesium 1.2 mg/dL"),
            (38, 55, "MEDICATION", "magnesium sulfate"),
            (56, 58, "DOSAGE", "2g"),
        ]
    },
    {
        "text": "Doppler ultrasound of right leg confirms deep vein thrombosis.",
        "entities": [
            (0, 18, "PROCEDURE", "Doppler ultrasound"),
            (39, 61, "DIAGNOSIS", "deep vein thrombosis"),
        ]
    },
    {
        "text": "Dexamethasone 10mg IV given for cerebral edema.",
        "entities": [
            (0, 13, "MEDICATION", "Dexamethasone"),
            (14, 18, "DOSAGE", "10mg"),
            (32, 47, "DIAGNOSIS", "cerebral edema"),
        ]
    },
    {
        "text": "PSA level 8.2 ng/mL, prostate biopsy recommended.",
        "entities": [
            (0, 20, "LAB_VALUE", "PSA level 8.2 ng/mL"),
            (22, 37, "PROCEDURE", "prostate biopsy"),
        ]
    },
    {
        "text": "Furosemide 40mg IV for acute pulmonary edema with SpO2 88%.",
        "entities": [
            (0, 10, "MEDICATION", "Furosemide"),
            (11, 15, "DOSAGE", "40mg"),
            (23, 45, "DIAGNOSIS", "acute pulmonary edema"),
            (51, 58, "LAB_VALUE", "SpO2 88%"),
        ]
    },
    {
        "text": "Bone marrow biopsy reveals acute myeloid leukemia.",
        "entities": [
            (0, 18, "PROCEDURE", "Bone marrow biopsy"),
            (27, 50, "DIAGNOSIS", "acute myeloid leukemia"),
        ]
    },
    {
        "text": "Vancomycin 1g IV every 12 hours for MRSA bacteremia.",
        "entities": [
            (0, 10, "MEDICATION", "Vancomycin"),
            (11, 13, "DOSAGE", "1g"),
            (36, 52, "DIAGNOSIS", "MRSA bacteremia"),
        ]
    },
    {
        "text": "Albumin level 2.1 g/dL, severe malnutrition, nutrition consult ordered.",
        "entities": [
            (0, 22, "LAB_VALUE", "Albumin level 2.1 g/dL"),
            (24, 43, "DIAGNOSIS", "severe malnutrition"),
        ]
    },
    {
        "text": "Paracentesis performed, 3 liters ascitic fluid removed.",
        "entities": [
            (0, 13, "PROCEDURE", "Paracentesis"),
        ]
    },
    {
        "text": "Sertraline 50mg daily for major depressive disorder.",
        "entities": [
            (0, 10, "MEDICATION", "Sertraline"),
            (11, 15, "DOSAGE", "50mg"),
            (26, 52, "DIAGNOSIS", "major depressive disorder"),
        ]
    },
    {
        "text": "Ferritin 12 ng/mL, iron deficiency anemia, start ferrous sulfate 325mg.",
        "entities": [
            (0, 17, "LAB_VALUE", "Ferritin 12 ng/mL"),
            (19, 41, "DIAGNOSIS", "iron deficiency anemia"),
            (49, 64, "MEDICATION", "ferrous sulfate"),
            (65, 70, "DOSAGE", "325mg"),
        ]
    },
    {
        "text": "Stress test positive for ischemia, refer to cardiology for angiogram.",
        "entities": [
            (0, 11, "PROCEDURE", "Stress test"),
            (25, 33, "DIAGNOSIS", "ischemia"),
            (58, 67, "PROCEDURE", "angiogram"),
        ]
    },
    {
        "text": "Piperacillin-tazobactam 3.375g IV every 6 hours for intra-abdominal abscess.",
        "entities": [
            (0, 23, "MEDICATION", "Piperacillin-tazobactam"),
            (24, 30, "DOSAGE", "3.375g"),
            (52, 76, "DIAGNOSIS", "intra-abdominal abscess"),
        ]
    },
    {
        "text": "CRP 45 mg/L, ESR 62 mm/hr, likely rheumatoid arthritis flare.",
        "entities": [
            (0, 11, "LAB_VALUE", "CRP 45 mg/L"),
            (13, 26, "LAB_VALUE", "ESR 62 mm/hr"),
            (35, 60, "DIAGNOSIS", "rheumatoid arthritis flare"),
        ]
    },
    {
        "text": "Bronchoscopy with bronchoalveolar lavage performed for persistent infiltrate.",
        "entities": [
            (0, 12, "PROCEDURE", "Bronchoscopy"),
            (18, 40, "PROCEDURE", "bronchoalveolar lavage"),
        ]
    },
    {
        "text": "Methotrexate 15mg weekly with folic acid 1mg daily for RA.",
        "entities": [
            (0, 12, "MEDICATION", "Methotrexate"),
            (13, 17, "DOSAGE", "15mg"),
            (30, 40, "MEDICATION", "folic acid"),
            (41, 44, "DOSAGE", "1mg"),
        ]
    },
    {
        "text": "AST 245 IU/L, ALT 312 IU/L, acute hepatitis workup initiated.",
        "entities": [
            (0, 13, "LAB_VALUE", "AST 245 IU/L"),
            (15, 28, "LAB_VALUE", "ALT 312 IU/L"),
            (30, 45, "DIAGNOSIS", "acute hepatitis"),
        ]
    },
    {
        "text": "Endoscopic retrograde cholangiopancreatography for common bile duct stone.",
        "entities": [
            (0, 47, "PROCEDURE", "Endoscopic retrograde cholangiopancreatography"),
            (52, 73, "DIAGNOSIS", "common bile duct stone"),
        ]
    },
    {
        "text": "Amoxicillin 875mg twice daily for acute sinusitis.",
        "entities": [
            (0, 11, "MEDICATION", "Amoxicillin"),
            (12, 17, "DOSAGE", "875mg"),
            (34, 50, "DIAGNOSIS", "acute sinusitis"),
        ]
    },
    {
        "text": "Hemoglobin 7.2 g/dL, transfuse 2 units packed red blood cells.",
        "entities": [
            (0, 20, "LAB_VALUE", "Hemoglobin 7.2 g/dL"),
        ]
    },
    {
        "text": "EEG shows generalized slowing consistent with metabolic encephalopathy.",
        "entities": [
            (0, 3, "PROCEDURE", "EEG"),
            (44, 70, "DIAGNOSIS", "metabolic encephalopathy"),
        ]
    },
    {
        "text": "Losartan 50mg daily, potassium 4.8 mEq/L, renal function stable.",
        "entities": [
            (0, 7, "MEDICATION", "Losartan"),
            (8, 12, "DOSAGE", "50mg"),
            (20, 41, "LAB_VALUE", "potassium 4.8 mEq/L"),
        ]
    },
    {
        "text": "Punch biopsy of suspicious skin lesion reveals basal cell carcinoma.",
        "entities": [
            (0, 12, "PROCEDURE", "Punch biopsy"),
            (47, 67, "DIAGNOSIS", "basal cell carcinoma"),
        ]
    },
    {
        "text": "Enoxaparin 40mg subcutaneous daily for DVT prophylaxis.",
        "entities": [
            (0, 10, "MEDICATION", "Enoxaparin"),
            (11, 15, "DOSAGE", "40mg"),
        ]
    },
    {
        "text": "Procalcitonin 2.8 ng/mL, start empiric broad-spectrum antibiotics.",
        "entities": [
            (0, 23, "LAB_VALUE", "Procalcitonin 2.8 ng/mL"),
        ]
    },
    {
        "text": "Carotid duplex scan reveals 70% stenosis right internal carotid artery.",
        "entities": [
            (0, 18, "PROCEDURE", "Carotid duplex scan"),
            (27, 70, "DIAGNOSIS", "70% stenosis right internal carotid artery"),
        ]
    },
    {
        "text": "Duloxetine 60mg daily for diabetic peripheral neuropathy.",
        "entities": [
            (0, 10, "MEDICATION", "Duloxetine"),
            (11, 15, "DOSAGE", "60mg"),
            (26, 56, "DIAGNOSIS", "diabetic peripheral neuropathy"),
        ]
    },
    {
        "text": "Lipase 1200 U/L, CT abdomen confirms acute pancreatitis.",
        "entities": [
            (0, 15, "LAB_VALUE", "Lipase 1200 U/L"),
            (17, 27, "PROCEDURE", "CT abdomen"),
            (37, 55, "DIAGNOSIS", "acute pancreatitis"),
        ]
    },
    {
        "text": "Phenytoin level 22 mcg/mL, supratherapeutic, hold next dose.",
        "entities": [
            (0, 25, "LAB_VALUE", "Phenytoin level 22 mcg/mL"),
            (0, 9, "MEDICATION", "Phenytoin"),
        ]
    },
    {
        "text": "Arthrocentesis of right knee, synovial fluid shows uric acid crystals.",
        "entities": [
            (0, 14, "PROCEDURE", "Arthrocentesis"),
        ]
    },
    {
        "text": "Pantoprazole 40mg IV twice daily for upper GI bleed.",
        "entities": [
            (0, 13, "MEDICATION", "Pantoprazole"),
            (14, 18, "DOSAGE", "40mg"),
            (39, 52, "DIAGNOSIS", "upper GI bleed"),
        ]
    },
    {
        "text": "NT-proBNP 4500 pg/mL, worsening congestive heart failure.",
        "entities": [
            (0, 20, "LAB_VALUE", "NT-proBNP 4500 pg/mL"),
            (33, 57, "DIAGNOSIS", "congestive heart failure"),
        ]
    },
    {
        "text": "Transesophageal echocardiogram reveals mitral valve vegetation.",
        "entities": [
            (0, 30, "PROCEDURE", "Transesophageal echocardiogram"),
            (39, 62, "DIAGNOSIS", "mitral valve vegetation"),
        ]
    },
    {
        "text": "Ciprofloxacin 500mg twice daily for complicated urinary tract infection.",
        "entities": [
            (0, 13, "MEDICATION", "Ciprofloxacin"),
            (14, 19, "DOSAGE", "500mg"),
            (36, 71, "DIAGNOSIS", "complicated urinary tract infection"),
        ]
    },
    {
        "text": "Ammonia level 85 mcmol/L, lactulose titrated for hepatic encephalopathy.",
        "entities": [
            (0, 24, "LAB_VALUE", "Ammonia level 85 mcmol/L"),
            (26, 35, "MEDICATION", "lactulose"),
            (50, 72, "DIAGNOSIS", "hepatic encephalopathy"),
        ]
    },
    {
        "text": "Nerve conduction study shows carpal tunnel syndrome bilateral.",
        "entities": [
            (0, 22, "PROCEDURE", "Nerve conduction study"),
            (29, 61, "DIAGNOSIS", "carpal tunnel syndrome bilateral"),
        ]
    },
    {
        "text": "Rivaroxaban 20mg daily for nonvalvular atrial fibrillation.",
        "entities": [
            (0, 11, "MEDICATION", "Rivaroxaban"),
            (12, 16, "DOSAGE", "20mg"),
            (27, 58, "DIAGNOSIS", "nonvalvular atrial fibrillation"),
        ]
    },
    {
        "text": "Phosphorus 1.8 mg/dL, IV sodium phosphate replacement initiated.",
        "entities": [
            (0, 20, "LAB_VALUE", "Phosphorus 1.8 mg/dL"),
            (25, 41, "MEDICATION", "sodium phosphate"),
        ]
    },
    {
        "text": "Flexible sigmoidoscopy reveals internal hemorrhoids grade 2.",
        "entities": [
            (0, 23, "PROCEDURE", "Flexible sigmoidoscopy"),
            (32, 59, "DIAGNOSIS", "internal hemorrhoids grade 2"),
        ]
    },
    {
        "text": "Tramadol 50mg every 6 hours as needed for moderate post-surgical pain.",
        "entities": [
            (0, 7, "MEDICATION", "Tramadol"),
            (8, 12, "DOSAGE", "50mg"),
        ]
    },
    {
        "text": "Vitamin D level 12 ng/mL, severe deficiency, ergocalciferol 50000 units weekly.",
        "entities": [
            (0, 23, "LAB_VALUE", "Vitamin D level 12 ng/mL"),
            (44, 58, "MEDICATION", "ergocalciferol"),
            (59, 70, "DOSAGE", "50000 units"),
        ]
    },
    {
        "text": "PET scan shows increased FDG uptake in mediastinal lymph nodes.",
        "entities": [
            (0, 8, "PROCEDURE", "PET scan"),
        ]
    },
    {
        "text": "Spironolactone 25mg daily added for resistant hypertension.",
        "entities": [
            (0, 14, "MEDICATION", "Spironolactone"),
            (15, 19, "DOSAGE", "25mg"),
            (35, 58, "DIAGNOSIS", "resistant hypertension"),
        ]
    },
    {
        "text": "Direct bilirubin 3.4 mg/dL, total bilirubin 5.8 mg/dL, suspect obstructive jaundice.",
        "entities": [
            (0, 26, "LAB_VALUE", "Direct bilirubin 3.4 mg/dL"),
            (28, 52, "LAB_VALUE", "total bilirubin 5.8 mg/dL"),
            (62, 83, "DIAGNOSIS", "obstructive jaundice"),
        ]
    },
    {
        "text": "Cystoscopy performed for evaluation of gross hematuria.",
        "entities": [
            (0, 10, "PROCEDURE", "Cystoscopy"),
            (37, 54, "DIAGNOSIS", "gross hematuria"),
        ]
    },
    {
        "text": "Topiramate 100mg twice daily for migraine prophylaxis.",
        "entities": [
            (0, 10, "MEDICATION", "Topiramate"),
            (11, 16, "DOSAGE", "100mg"),
            (33, 53, "DIAGNOSIS", "migraine prophylaxis"),
        ]
    },
    {
        "text": "Fibrinogen 120 mg/dL, suspect DIC, order coagulation panel.",
        "entities": [
            (0, 20, "LAB_VALUE", "Fibrinogen 120 mg/dL"),
        ]
    },
    {
        "text": "Tilt table test positive for vasovagal syncope.",
        "entities": [
            (0, 15, "PROCEDURE", "Tilt table test"),
            (29, 47, "DIAGNOSIS", "vasovagal syncope"),
        ]
    },
    {
        "text": "Levetiracetam 500mg twice daily for new-onset seizures.",
        "entities": [
            (0, 13, "MEDICATION", "Levetiracetam"),
            (14, 19, "DOSAGE", "500mg"),
            (36, 54, "DIAGNOSIS", "new-onset seizures"),
        ]
    },
    {
        "text": "Uric acid level 9.8 mg/dL, diagnosis of gout, start allopurinol 100mg.",
        "entities": [
            (0, 25, "LAB_VALUE", "Uric acid level 9.8 mg/dL"),
            (41, 45, "DIAGNOSIS", "gout"),
            (53, 64, "MEDICATION", "allopurinol"),
            (65, 70, "DOSAGE", "100mg"),
        ]
    },
    {
        "text": "Sleep study reveals severe obstructive sleep apnea with AHI of 42.",
        "entities": [
            (0, 11, "PROCEDURE", "Sleep study"),
            (20, 48, "DIAGNOSIS", "severe obstructive sleep apnea"),
            (54, 63, "LAB_VALUE", "AHI of 42"),
        ]
    },
    {
        "text": "Meropenem 1g IV every 8 hours for multi-drug resistant infection.",
        "entities": [
            (0, 8, "MEDICATION", "Meropenem"),
            (9, 11, "DOSAGE", "1g"),
            (35, 64, "DIAGNOSIS", "multi-drug resistant infection"),
        ]
    },
    {
        "text": "Cortisol level 2.1 mcg/dL, ACTH stimulation test ordered for adrenal insufficiency.",
        "entities": [
            (0, 25, "LAB_VALUE", "Cortisol level 2.1 mcg/dL"),
            (27, 47, "PROCEDURE", "ACTH stimulation test"),
            (60, 82, "DIAGNOSIS", "adrenal insufficiency"),
        ]
    },
    {
        "text": "Quetiapine 25mg at bedtime for agitation in dementia patient.",
        "entities": [
            (0, 10, "MEDICATION", "Quetiapine"),
            (11, 15, "DOSAGE", "25mg"),
        ]
    },
    {
        "text": "HCG level 125000 mIU/mL, pelvic ultrasound confirms intrauterine pregnancy 8 weeks.",
        "entities": [
            (0, 23, "LAB_VALUE", "HCG level 125000 mIU/mL"),
            (25, 42, "PROCEDURE", "pelvic ultrasound"),
        ]
    },
    {
        "text": "Trastuzumab 6mg/kg IV every 3 weeks for HER2-positive breast cancer.",
        "entities": [
            (0, 12, "MEDICATION", "Trastuzumab"),
            (13, 19, "DOSAGE", "6mg/kg"),
            (40, 68, "DIAGNOSIS", "HER2-positive breast cancer"),
        ]
    },
    {
        "text": "ANA titer 1:640 speckled pattern, suspect systemic lupus erythematosus.",
        "entities": [
            (0, 30, "LAB_VALUE", "ANA titer 1:640 speckled pattern"),
            (40, 70, "DIAGNOSIS", "systemic lupus erythematosus"),
        ]
    },
    {
        "text": "Holter monitor shows paroxysmal atrial fibrillation with RVR.",
        "entities": [
            (0, 14, "PROCEDURE", "Holter monitor"),
            (21, 60, "DIAGNOSIS", "paroxysmal atrial fibrillation with RVR"),
        ]
    },
    {
        "text": "Methylprednisolone 125mg IV for acute multiple sclerosis relapse.",
        "entities": [
            (0, 18, "MEDICATION", "Methylprednisolone"),
            (19, 24, "DOSAGE", "125mg"),
            (38, 64, "DIAGNOSIS", "multiple sclerosis relapse"),
        ]
    },
]


def convert_to_iob(text: str, entities: list[tuple]) -> list[tuple[str, str]]:
    """Convert character-span annotations to IOB token format."""
    tokens = text.split()
    iob_tags = ["O"] * len(tokens)

    char_to_token = {}
    current_pos = 0
    for i, token in enumerate(tokens):
        start = text.index(token, current_pos)
        for c in range(start, start + len(token)):
            char_to_token[c] = i
        current_pos = start + len(token)

    for ent_start, ent_end, ent_type, _ent_text in entities:
        token_indices = set()
        for c in range(ent_start, ent_end):
            if c in char_to_token:
                token_indices.add(char_to_token[c])

        sorted_indices = sorted(token_indices)
        for j, token_idx in enumerate(sorted_indices):
            if j == 0:
                iob_tags[token_idx] = f"B-{ent_type}"
            else:
                iob_tags[token_idx] = f"I-{ent_type}"

    return list(zip(tokens, iob_tags))


def get_ner_dataset() -> list[dict]:
    dataset = []
    for entry in ANNOTATED_SENTENCES:
        iob_tokens = convert_to_iob(entry["text"], entry["entities"])
        tokens, tags = zip(*iob_tokens) if iob_tokens else ([], [])
        dataset.append({
            "text": entry["text"],
            "tokens": list(tokens),
            "tags": list(tags),
            "entities": [
                {"start": s, "end": e, "type": t, "text": txt}
                for s, e, t, txt in entry["entities"]
            ],
        })
    return dataset


def save_ner_dataset(output_dir: str = "data"):
    dataset = get_ner_dataset()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ner_annotations.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    tag_counts = {}
    for entry in dataset:
        for tag in entry["tags"]:
            if tag != "O":
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"Saved {len(dataset)} annotated sentences to {output_path}")
    print(f"\nTag distribution:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    save_ner_dataset(output_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
