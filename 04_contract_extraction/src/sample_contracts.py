"""
Generate realistic sample healthcare contract text blocks.
Each contract includes sections you'd find in a real PBM/TPA/MCO agreement:
effective dates, termination clauses, SLA percentages, pricing structures,
and compliance language.
"""

import json
import os

SAMPLE_CONTRACTS = [
    {
        "id": "CONTRACT-001",
        "title": "Pharmacy Benefit Management Services Agreement",
        "parties": ["Lantern Care Holdings, Inc.", "Northeast Health Plan"],
        "text": """PHARMACY BENEFIT MANAGEMENT SERVICES AGREEMENT

This Agreement ("Agreement") is entered into as of January 1, 2024 ("Effective Date") by and between Lantern Care Holdings, Inc., a Delaware corporation ("PBM"), and Northeast Health Plan, a New York corporation ("Client").

SECTION 1. TERM AND TERMINATION

1.1 Initial Term. This Agreement shall commence on the Effective Date and continue for a period of three (3) years, expiring on December 31, 2026 ("Initial Term").

1.2 Renewal. This Agreement shall automatically renew for successive one (1) year periods unless either party provides written notice of non-renewal at least ninety (90) days prior to the expiration of the then-current term.

1.3 Termination for Cause. Either party may terminate this Agreement upon sixty (60) days written notice if the other party materially breaches any provision of this Agreement and fails to cure such breach within thirty (30) days after receipt of written notice specifying the breach.

1.4 Termination for Convenience. Client may terminate this Agreement without cause upon one hundred eighty (180) days prior written notice to PBM, subject to payment of an early termination fee equal to $500,000.

SECTION 2. PRICING AND REIMBURSEMENT

2.1 Dispensing Fees. PBM shall charge a dispensing fee of $1.75 per prescription for retail pharmacy claims and $0.00 for mail order pharmacy claims.

2.2 Administrative Fees. Client shall pay PBM an administrative fee of $3.25 per member per month ("PMPM") for pharmacy benefit management services.

2.3 Rebate Sharing. PBM shall pass through to Client ninety percent (90%) of all manufacturer rebates received. Rebate reconciliation shall occur quarterly within forty-five (45) days of quarter end.

2.4 Specialty Drug Pricing. Specialty medications shall be priced at AWP minus fifteen percent (15%) plus a dispensing fee of $2.50 per prescription.

2.5 Generic Effective Rate. PBM guarantees a generic effective rate of AWP minus eighty percent (80%) for retail and AWP minus eighty-five percent (85%) for mail order.

SECTION 3. SERVICE LEVEL AGREEMENTS

3.1 Claims Processing. PBM shall process ninety-nine point five percent (99.5%) of clean claims within twenty-four (24) hours of receipt.

3.2 System Availability. PBM shall maintain system availability of ninety-nine point nine percent (99.9%) measured on a monthly basis, excluding scheduled maintenance windows.

3.3 Call Center Performance. PBM shall answer eighty percent (80%) of member calls within thirty (30) seconds with an abandonment rate not to exceed five percent (5%).

3.4 Prior Authorization. PBM shall complete prior authorization reviews within forty-eight (48) hours of receipt of complete clinical information.

3.5 SLA Credits. In the event PBM fails to meet any SLA metric for two (2) consecutive months, Client shall be entitled to a service credit equal to two percent (2%) of monthly administrative fees for each missed SLA.

SECTION 4. COMPLIANCE

4.1 HIPAA Compliance. PBM shall comply with all applicable provisions of the Health Insurance Portability and Accountability Act of 1996 ("HIPAA"), as amended by the Health Information Technology for Economic and Clinical Health Act ("HITECH Act"), and all regulations promulgated thereunder.

4.2 State Regulations. PBM shall comply with all applicable state pharmacy practice acts and PBM licensing requirements.

4.3 Audit Rights. Client shall have the right to audit PBM's books and records related to this Agreement upon thirty (30) days prior written notice, no more than once per calendar year.

SECTION 5. CONFIDENTIALITY

5.1 Confidential Information. Each party agrees to hold in confidence all Confidential Information received from the other party and to not disclose such information to any third party without prior written consent.

5.2 Survival. The obligations under this Section shall survive termination of this Agreement for a period of five (5) years.

SECTION 6. LIABILITY AND INDEMNIFICATION

6.1 Limitation of Liability. In no event shall either party's aggregate liability exceed the total fees paid or payable under this Agreement during the twelve (12) month period preceding the claim, except for claims arising from gross negligence or willful misconduct.

6.2 Indemnification. PBM shall indemnify, defend, and hold harmless Client from and against any third-party claims arising from PBM's negligence or breach of this Agreement, up to a maximum of $5,000,000 per occurrence.""",
    },
    {
        "id": "CONTRACT-002",
        "title": "Third Party Administrator Services Agreement",
        "parties": ["Lantern Care Holdings, Inc.", "MidAmerica Employers Trust"],
        "text": """THIRD PARTY ADMINISTRATOR SERVICES AGREEMENT

This Agreement is made effective as of March 15, 2024 by and between Lantern Care Holdings, Inc. ("Administrator") and MidAmerica Employers Trust ("Plan Sponsor").

SECTION 1. SERVICES AND TERM

1.1 Services. Administrator shall provide claims administration, utilization management, network management, and member services for Plan Sponsor's self-funded employee health benefit plan.

1.2 Term. This Agreement shall be effective from March 15, 2024 through March 14, 2027, unless terminated earlier in accordance with Section 5.

1.3 Implementation. Administrator shall complete system configuration and data migration within sixty (60) days of the Effective Date. Go-live date: May 15, 2024.

SECTION 2. FEES AND PAYMENT

2.1 Administrative Fee. Plan Sponsor shall pay Administrator a fee of $28.50 per employee per month ("PEPM") for the services described herein.

2.2 Implementation Fee. Plan Sponsor shall pay a one-time implementation fee of $75,000, payable in three installments of $25,000 each.

2.3 Stop Loss Placement. Administrator shall arrange specific stop loss coverage with a specific deductible of $250,000 per member per year. Stop loss premium estimated at $45.00 PEPM.

2.4 Network Access Fee. Plan Sponsor shall pay a network access fee of $1.50 PEPM for access to Administrator's preferred provider network.

2.5 Annual Fee Adjustment. Administrative fees shall be subject to an annual adjustment not to exceed the greater of three percent (3%) or the Consumer Price Index increase for the preceding twelve months.

SECTION 3. PERFORMANCE STANDARDS

3.1 Claims Turnaround. Administrator shall adjudicate ninety-five percent (95%) of clean claims within ten (10) business days and ninety-nine percent (99%) within fifteen (15) business days.

3.2 Claims Accuracy. Administrator shall maintain a financial claims accuracy rate of ninety-nine percent (99%) and a procedural accuracy rate of ninety-seven percent (97%).

3.3 Utilization Review. Pre-certification decisions shall be communicated within two (2) business days for non-urgent requests and within twenty-four (24) hours for urgent requests.

3.4 Member Satisfaction. Administrator shall maintain a member satisfaction score of eighty-five percent (85%) or above as measured by annual survey.

3.5 Performance Guarantees. Administrator places at risk five percent (5%) of annual administrative fees. Performance credits shall be allocated proportionally across the metrics defined above.

SECTION 4. DATA AND REPORTING

4.1 Monthly Reporting. Administrator shall provide comprehensive claims and utilization reports within fifteen (15) business days of month end.

4.2 Data Ownership. All claims data and member information shall remain the property of Plan Sponsor. Administrator shall return or destroy all data within thirty (30) days of termination.

4.3 Data Security. Administrator shall maintain SOC 2 Type II certification and comply with all applicable data protection regulations, including HIPAA and state privacy laws.

SECTION 5. TERMINATION

5.1 Termination for Cause. Either party may terminate upon ninety (90) days written notice for material breach that remains uncured for thirty (30) days.

5.2 Termination for Convenience. Plan Sponsor may terminate upon one hundred twenty (120) days written notice, subject to payment of a run-out fee of $15.00 PEPM for six (6) months.

5.3 Transition Assistance. Upon termination, Administrator shall provide transition assistance for up to ninety (90) days at a reduced rate of $15.00 PEPM.""",
    },
    {
        "id": "CONTRACT-003",
        "title": "Managed Care Organization Provider Agreement",
        "parties": ["Sunrise Health Network", "Regional Medical Center"],
        "text": """MANAGED CARE ORGANIZATION PROVIDER AGREEMENT

This Provider Agreement ("Agreement") is entered into effective July 1, 2024 by and between Sunrise Health Network, Inc. ("MCO") and Regional Medical Center ("Provider").

SECTION 1. PARTICIPATION AND TERM

1.1 Term. This Agreement shall be effective for two (2) years from July 1, 2024 through June 30, 2026, with automatic renewal for successive one-year periods.

1.2 Credentialing. Provider shall maintain current credentialing and comply with MCO's credentialing requirements including NCQA standards.

SECTION 2. COMPENSATION

2.1 Fee Schedule. MCO shall reimburse Provider according to the attached Fee Schedule (Exhibit A), which is based on one hundred twenty percent (120%) of the current Medicare Fee Schedule.

2.2 Inpatient Services. Inpatient services shall be reimbursed at a per diem rate of $2,800 for medical/surgical admissions and $4,500 for ICU admissions.

2.3 Outpatient Services. Outpatient procedures shall be reimbursed at eighty-five percent (85%) of Provider's billed charges or the Fee Schedule rate, whichever is less.

2.4 Payment Terms. MCO shall pay clean claims within thirty (30) days of receipt. Claims not paid within forty-five (45) days shall accrue interest at one percent (1%) per month.

2.5 Capitation Option. For primary care services, Provider may elect capitation at $35.00 per member per month, subject to risk corridor of plus or minus ten percent (10%).

SECTION 3. QUALITY METRICS

3.1 HEDIS Measures. Provider shall participate in MCO's quality improvement program and meet established benchmarks for applicable HEDIS measures.

3.2 Pay for Performance. Provider may earn additional compensation of up to fifteen percent (15%) above base reimbursement through achievement of quality metrics.

3.3 Readmission Rate. Provider shall maintain a thirty-day all-cause readmission rate below twelve percent (12%).

SECTION 4. REGULATORY COMPLIANCE

4.1 Licensure. Provider shall maintain all required state and federal licenses and certifications.

4.2 HIPAA. Both parties shall comply with HIPAA privacy and security requirements, including the Breach Notification Rule under 45 CFR Parts 160 and 164.

4.3 Anti-Kickback. Both parties certify compliance with the Federal Anti-Kickback Statute (42 U.S.C. § 1320a-7b) and the Stark Law (42 U.S.C. § 1395nn).

4.4 Fraud, Waste, and Abuse. Provider shall maintain a compliance program consistent with OIG guidance and shall report suspected fraud, waste, and abuse to MCO within ten (10) business days.

SECTION 5. INDEMNIFICATION

5.1 Mutual Indemnification. Each party shall indemnify the other against claims arising from its own negligent acts or omissions.

5.2 Insurance Requirements. Provider shall maintain professional liability insurance with limits of not less than $1,000,000 per occurrence and $3,000,000 in the aggregate.""",
    },
    {
        "id": "CONTRACT-004",
        "title": "Health Information Exchange Data Use Agreement",
        "parties": ["StateConnect HIE", "Community Health Partners"],
        "text": """HEALTH INFORMATION EXCHANGE DATA USE AGREEMENT

Effective Date: October 1, 2024

This Data Use Agreement ("DUA") is between StateConnect HIE ("HIE") and Community Health Partners ("Participant").

SECTION 1. PURPOSE AND SCOPE

1.1 Purpose. This DUA governs the electronic exchange of protected health information ("PHI") through the HIE for purposes of treatment, payment, and healthcare operations.

1.2 Permitted Uses. Participant may access PHI through the HIE solely for: (a) treatment of patients, (b) care coordination, (c) quality reporting, and (d) public health reporting as required by law.

SECTION 2. FEES

2.1 Participation Fee. Participant shall pay an annual participation fee of $12,000, payable quarterly in installments of $3,000.

2.2 Transaction Fees. Query-based exchange: $0.50 per transaction. Direct messaging: $0.25 per message. Bulk data exchange: $2,500 per data load.

SECTION 3. DATA GOVERNANCE

3.1 Data Quality. Participant shall submit clinical data meeting HL7 FHIR R4 standards with a data completeness rate of ninety-five percent (95%) or higher.

3.2 Timeliness. Clinical encounter data shall be submitted to the HIE within twenty-four (24) hours of the encounter.

3.3 Patient Consent. Participant shall obtain and document patient consent in accordance with applicable state law. Opt-out rate shall be tracked and reported quarterly.

SECTION 4. SECURITY AND PRIVACY

4.1 HIPAA Compliance. Participant shall comply with all provisions of HIPAA, HITECH, and 42 CFR Part 2 (Substance Abuse Confidentiality).

4.2 Minimum Necessary. Participant shall access only the minimum necessary PHI required for the permitted purpose.

4.3 Breach Notification. Participant shall notify HIE of any breach of unsecured PHI within twenty-four (24) hours of discovery, and shall cooperate in breach investigation and notification to affected individuals within sixty (60) days as required under 45 CFR § 164.408.

4.4 Encryption. All PHI transmitted through the HIE shall be encrypted using AES-256 encryption or equivalent.

SECTION 5. TERM AND TERMINATION

5.1 Term. This DUA shall remain in effect for three (3) years from the Effective Date, with automatic annual renewal.

5.2 Termination. Either party may terminate upon sixty (60) days written notice. Upon termination, Participant shall cease all access to the HIE and return or destroy all PHI within thirty (30) days.

SECTION 6. LIABILITY

6.1 Limitation. Aggregate liability under this DUA shall not exceed $500,000 per party, except for breaches of PHI obligations.""",
    },
    {
        "id": "CONTRACT-005",
        "title": "Clinical Decision Support Software License Agreement",
        "parties": ["MedLogic Systems, LLC", "Lantern Care Holdings, Inc."],
        "text": """CLINICAL DECISION SUPPORT SOFTWARE LICENSE AGREEMENT

This License Agreement ("Agreement") is effective as of February 1, 2025 between MedLogic Systems, LLC ("Licensor") and Lantern Care Holdings, Inc. ("Licensee").

SECTION 1. LICENSE GRANT

1.1 Scope. Licensor grants Licensee a non-exclusive, non-transferable license to use the MedLogic Clinical Decision Support Platform ("Software") for internal business purposes.

1.2 Term. The license term is three (3) years from February 1, 2025 through January 31, 2028.

1.3 Authorized Users. License covers up to five hundred (500) concurrent users.

SECTION 2. FEES

2.1 License Fee. Licensee shall pay an annual license fee of $450,000, payable in quarterly installments of $112,500.

2.2 Implementation. One-time implementation fee of $150,000 including configuration, training, and data integration.

2.3 Support and Maintenance. Annual support and maintenance fee of twenty percent (20%) of the license fee ($90,000), including software updates and 24/7 technical support.

2.4 Transaction Volume. Base license includes up to 5,000,000 clinical decision support transactions per year. Additional transactions billed at $0.02 per transaction.

SECTION 3. SERVICE LEVELS

3.1 Uptime. Licensor guarantees ninety-nine point nine five percent (99.95%) system uptime measured monthly, excluding planned maintenance.

3.2 Response Time. Clinical decision support queries shall return results within two (2) seconds for ninety-five percent (95%) of queries.

3.3 Support Response. Critical issues (system down): one (1) hour response, four (4) hour resolution. High priority: four (4) hour response, twenty-four (24) hour resolution. Medium: eight (8) hour response. Low: two (2) business day response.

3.4 SLA Credits. For each 0.01% below 99.95% uptime, Licensee shall receive a credit of five percent (5%) of monthly fees, up to a maximum of twenty-five percent (25%).

SECTION 4. DATA AND COMPLIANCE

4.1 HIPAA. Licensor shall execute a Business Associate Agreement in compliance with HIPAA and HITECH requirements.

4.2 Clinical Content. Clinical rules and drug interaction databases shall be updated no less than monthly using FDA-approved sources.

4.3 Audit Trail. Software shall maintain a complete audit trail of all clinical decision support interactions for a minimum of seven (7) years.

SECTION 5. CONFIDENTIALITY

5.1 Protection. Both parties shall protect Confidential Information using the same degree of care as their own confidential information, but not less than reasonable care.

5.2 Duration. Confidentiality obligations shall survive for three (3) years following termination.

SECTION 6. LIMITATION OF LIABILITY

6.1 Cap. Neither party's aggregate liability shall exceed the total fees paid during the twelve (12) months preceding the claim.

6.2 Exclusions. Neither party shall be liable for indirect, incidental, consequential, or punitive damages.

6.3 Clinical Decisions. Licensor expressly disclaims liability for clinical decisions made using the Software. The Software is intended as a decision support tool and does not replace clinical judgment.""",
    },
    {
        "id": "CONTRACT-006",
        "title": "Population Health Analytics Services Agreement",
        "parties": ["Lantern Care Holdings, Inc.", "Gulf Coast Health System"],
        "text": """POPULATION HEALTH ANALYTICS SERVICES AGREEMENT

Effective Date: April 1, 2024
Agreement Number: LCH-2024-0892

Between Lantern Care Holdings, Inc. ("Analytics Provider") and Gulf Coast Health System ("Health System").

SECTION 1. SERVICES

1.1 Analytics Platform. Analytics Provider shall provide access to its population health analytics platform including risk stratification, care gap identification, and predictive modeling capabilities.

1.2 Data Integration. Analytics Provider shall integrate data from Health System's EMR (Epic), claims feeds, HIE, and third-party social determinants of health data sources.

1.3 Reporting. Analytics Provider shall deliver monthly population health reports, quarterly trend analyses, and annual outcomes assessments.

SECTION 2. PRICING

2.1 Platform Fee. Health System shall pay $8.50 per attributed life per month ("PALPM") for the analytics platform.

2.2 Implementation. Implementation fee of $200,000 payable upon execution of this Agreement.

2.3 Custom Analytics. Ad hoc analytics projects billed at $250 per hour, not to exceed $50,000 per project without prior approval.

2.4 Minimum Commitment. Health System commits to a minimum of 100,000 attributed lives. Total minimum annual commitment: $10,200,000.

SECTION 3. PERFORMANCE METRICS

3.1 Data Refresh. Analytics platform shall reflect updated data within forty-eight (48) hours of receipt.

3.2 Report Delivery. Monthly reports delivered by the fifteenth (15th) business day of the following month. Quarterly reports within thirty (30) days of quarter end.

3.3 Model Accuracy. Predictive models shall maintain an AUC of 0.80 or greater for risk stratification and 0.75 or greater for hospital readmission prediction.

3.4 User Training. Analytics Provider shall provide up to eighty (80) hours of user training annually at no additional cost.

SECTION 4. COMPLIANCE AND SECURITY

4.1 HIPAA/HITECH. Analytics Provider shall comply with all HIPAA and HITECH requirements and shall execute a Business Associate Agreement.

4.2 De-identification. All data used for analytics development and benchmarking shall be de-identified in accordance with the Safe Harbor method under 45 CFR § 164.514(b).

4.3 Data Residency. All PHI shall be stored within the continental United States.

4.4 SOC 2. Analytics Provider shall maintain SOC 2 Type II certification and provide annual audit reports to Health System.

SECTION 5. TERM AND TERMINATION

5.1 Term. Three (3) years from April 1, 2024 through March 31, 2027.

5.2 Early Termination. Health System may terminate upon one hundred eighty (180) days notice, subject to payment of remaining minimum commitment for the current contract year.

SECTION 6. INDEMNIFICATION AND LIABILITY

6.1 Indemnification. Analytics Provider shall indemnify Health System against claims arising from Analytics Provider's breach of this Agreement or applicable law, up to $10,000,000.

6.2 Limitation. Except for indemnification obligations and breaches of confidentiality, neither party's liability shall exceed the fees paid in the preceding twelve (12) months.""",
    },
    {
        "id": "CONTRACT-007",
        "title": "Telehealth Platform Services Agreement",
        "parties": ["VirtualCare Technologies", "Pacific Northwest Medical Group"],
        "text": """TELEHEALTH PLATFORM SERVICES AGREEMENT

This Agreement is entered into as of September 1, 2024 between VirtualCare Technologies, Inc. ("Platform Provider") and Pacific Northwest Medical Group, P.C. ("Medical Group").

SECTION 1. SERVICES AND TERM

1.1 Platform Services. Platform Provider shall provide a HIPAA-compliant telehealth platform including video consultation, secure messaging, remote patient monitoring integration, and electronic prescribing.

1.2 Term. Initial term of two (2) years from September 1, 2024 through August 31, 2026.

1.3 Implementation. Platform shall be fully operational within forty-five (45) days of Agreement execution.

SECTION 2. PRICING

2.1 Per-Visit Fee. Medical Group shall pay $12.00 per completed telehealth visit.

2.2 Monthly Minimum. Minimum monthly fee of $5,000 regardless of visit volume.

2.3 Setup Fee. One-time setup fee of $25,000 including EHR integration, provider onboarding, and initial training.

2.4 Remote Monitoring. Remote patient monitoring services billed at $65.00 per patient per month for enrolled patients.

SECTION 3. SERVICE LEVELS

3.1 Platform Availability. Platform Provider guarantees ninety-nine point nine percent (99.9%) uptime during business hours (6 AM - 10 PM local time) and ninety-nine percent (99%) uptime outside business hours.

3.2 Video Quality. HD video quality (minimum 720p) for ninety-eight percent (98%) of sessions with latency below 150 milliseconds.

3.3 Technical Support. 24/7 technical support with fifteen (15) minute response for critical issues during business hours.

3.4 Downtime Credits. For each hour of unscheduled downtime during business hours, Medical Group shall receive a credit of $500, up to total monthly fees.

SECTION 4. COMPLIANCE

4.1 HIPAA. Platform Provider shall maintain full HIPAA compliance including encryption of all data in transit (TLS 1.3) and at rest (AES-256).

4.2 State Telemedicine Laws. Platform Provider shall maintain compliance with applicable state telemedicine regulations for Oregon and Washington.

4.3 DEA Compliance. Electronic prescribing module shall comply with DEA regulations for controlled substance prescribing (21 CFR Part 1311).

SECTION 5. CONFIDENTIALITY AND IP

5.1 Patient Data. All patient data remains property of Medical Group. Platform Provider acts solely as a conduit and Business Associate.

5.2 Confidentiality. Obligations survive for five (5) years after termination.

SECTION 6. LIABILITY

6.1 Cap. Total liability limited to $2,000,000 or total fees paid in the prior 24 months, whichever is greater.

6.2 Clinical Liability. Platform Provider bears no responsibility for clinical decisions or medical outcomes.""",
    },
    {
        "id": "CONTRACT-008",
        "title": "Pharmacy Network Participation Agreement",
        "parties": ["Lantern Care Holdings, Inc.", "ValueRx Pharmacy Chain"],
        "text": """PHARMACY NETWORK PARTICIPATION AGREEMENT

Effective: June 1, 2024
Between Lantern Care Holdings, Inc. ("PBM") and ValueRx Pharmacy Chain, Inc. ("Pharmacy").

SECTION 1. NETWORK PARTICIPATION

1.1 Term. Pharmacy agrees to participate in PBM's pharmacy network for a period of two (2) years commencing June 1, 2024 and ending May 31, 2026.

1.2 Locations. This Agreement covers all sixty-five (65) Pharmacy retail locations listed in Exhibit A.

1.3 Access Standards. Pharmacy shall maintain operating hours of no less than sixty (60) hours per week at each location.

SECTION 2. REIMBURSEMENT

2.1 Brand Name Drugs. Brand name prescriptions shall be reimbursed at Average Wholesale Price ("AWP") minus sixteen percent (16%) plus a dispensing fee of $1.50 per prescription.

2.2 Generic Drugs. Generic prescriptions shall be reimbursed at Maximum Allowable Cost ("MAC") plus a dispensing fee of $1.00 per prescription.

2.3 Specialty Drugs. Specialty prescriptions reimbursed at ASP plus six percent (6%) plus a dispensing fee of $3.00.

2.4 Payment. PBM shall remit payment to Pharmacy within fourteen (14) calendar days of clean claim submission.

2.5 DIR Fees. Direct and Indirect Remuneration fees shall not exceed three percent (3%) of total reimbursement and shall be disclosed at point-of-sale.

SECTION 3. PERFORMANCE REQUIREMENTS

3.1 Generic Fill Rate. Pharmacy shall maintain a generic dispensing rate of ninety percent (90%) or higher.

3.2 Medication Therapy Management. Pharmacy shall provide MTM services for eligible beneficiaries and complete comprehensive medication reviews within the required timeframes.

3.3 Star Ratings. Pharmacy shall actively participate in programs to support Medicare Part D star ratings of four (4) stars or above.

3.4 Audit Recovery. PBM reserves the right to recoup amounts identified through pharmacy audits with a look-back period of twenty-four (24) months.

SECTION 4. COMPLIANCE

4.1 DEA Registration. Pharmacy shall maintain current DEA registration and state pharmacy licenses at all locations.

4.2 HIPAA. Pharmacy shall comply with all HIPAA requirements related to the handling of PHI.

4.3 Fraud Prevention. Pharmacy shall maintain a compliance program to detect and prevent fraud, waste, and abuse, consistent with 42 CFR § 423.504.

SECTION 5. INDEMNIFICATION

5.1 Pharmacy Indemnification. Pharmacy shall indemnify PBM for claims arising from dispensing errors, up to $2,000,000 per occurrence.

5.2 Insurance. Pharmacy shall maintain professional liability insurance with limits of $1,000,000 per occurrence and $3,000,000 aggregate.""",
    },
    {
        "id": "CONTRACT-009",
        "title": "Value-Based Care Arrangement",
        "parties": ["Horizon Accountable Care Organization", "Primary Care Associates"],
        "text": """VALUE-BASED CARE ARRANGEMENT

Performance Year: January 1, 2025 through December 31, 2025
Between Horizon Accountable Care Organization ("ACO") and Primary Care Associates, P.A. ("Practice").

SECTION 1. ARRANGEMENT OVERVIEW

1.1 Model. Practice agrees to participate in ACO's shared savings/shared risk arrangement for the 2025 performance year.

1.2 Attributed Lives. Practice is attributed approximately 8,500 Medicare beneficiaries based on plurality of primary care visits.

1.3 Benchmark. Total cost of care benchmark for Practice panel: $12,400 per beneficiary per year ("PBPY"), adjusted for risk score and regional factors.

SECTION 2. FINANCIAL TERMS

2.1 Shared Savings. If total cost of care falls below benchmark by more than two percent (2%), Practice shall receive fifty percent (50%) of savings, capped at twenty percent (20%) of benchmark.

2.2 Shared Risk. If total cost of care exceeds benchmark by more than two percent (2%), Practice shall be responsible for twenty-five percent (25%) of losses, capped at ten percent (10%) of benchmark.

2.3 Quality Gate. Shared savings are contingent upon achieving a minimum quality score of seventy percent (70%) on the composite quality measure.

2.4 Infrastructure Payment. ACO shall provide a monthly care management infrastructure payment of $2.50 PMPM to support care coordination activities.

2.5 Reconciliation. Annual reconciliation shall occur within one hundred twenty (120) days of performance year end, with final settlement within one hundred fifty (150) days.

SECTION 3. QUALITY MEASURES

3.1 Composite Score. Quality composite includes: (a) HEDIS measures (40%), (b) patient experience CAHPS (20%), (c) clinical outcomes (25%), and (d) utilization metrics (15%).

3.2 Specific Targets. HbA1c control (<8%) for diabetic patients: seventy-five percent (75%). Blood pressure control: eighty percent (80%). Breast cancer screening: eighty percent (80%). Colorectal cancer screening: seventy percent (70%). ED utilization rate below 450 per 1,000 beneficiaries.

3.3 Reporting. Practice shall submit quality data monthly through the ACO's quality reporting portal.

SECTION 4. CARE MANAGEMENT

4.1 Care Coordination. Practice shall employ or contract for at least one (1) care coordinator per 2,000 attributed lives.

4.2 Annual Wellness Visits. Practice shall complete Annual Wellness Visits for at least sixty-five percent (65%) of attributed beneficiaries.

4.3 Chronic Care Management. Practice shall provide chronic care management services (CPT 99490) for high-risk patients with two or more chronic conditions.

SECTION 5. COMPLIANCE

5.1 CMS Requirements. Both parties shall comply with all CMS MSSP requirements, including 42 CFR Part 425.

5.2 Antitrust. Practice acknowledges receipt of the ACO's antitrust compliance policy.

5.3 Patient Notification. Practice shall notify attributed beneficiaries of their participation in the ACO in accordance with CMS requirements.

SECTION 6. TERM

6.1 Performance Year. This arrangement is for the 2025 performance year only. Continuation is subject to mutual agreement and ACO's CMS participation agreement renewal.""",
    },
    {
        "id": "CONTRACT-010",
        "title": "Clinical Trials Management System Agreement",
        "parties": ["TrialSync Solutions", "University Research Hospital"],
        "text": """CLINICAL TRIALS MANAGEMENT SYSTEM AGREEMENT

Effective Date: November 15, 2024

This Agreement is between TrialSync Solutions, Inc. ("Vendor") and University Research Hospital ("Institution").

SECTION 1. SYSTEM AND SERVICES

1.1 CTMS Platform. Vendor shall provide a cloud-based Clinical Trials Management System including protocol management, subject enrollment tracking, regulatory document management, and financial tracking.

1.2 Term. Five (5) years from November 15, 2024 through November 14, 2029.

1.3 Users. License for up to two hundred (200) named users with unlimited read-only access.

SECTION 2. PRICING

2.1 Annual Subscription. $350,000 per year, payable annually in advance.

2.2 Implementation. $275,000 implementation fee including data migration from legacy system, customization, validation, and training.

2.3 Integration. EMR integration (Epic): $45,000 one-time fee. CTMS-to-billing interface: $35,000 one-time fee.

2.4 Annual Escalation. Subscription fees increase by three percent (3%) annually beginning Year 2.

2.5 Additional Services. Custom report development: $200 per hour. Validation support: $225 per hour.

SECTION 3. SERVICE LEVELS

3.1 Availability. Ninety-nine point nine percent (99.9%) system availability, measured monthly.

3.2 Backup and Recovery. Daily automated backups with a recovery point objective (RPO) of four (4) hours and recovery time objective (RTO) of eight (8) hours.

3.3 Support. Dedicated account manager. Tier 1 support: 24/7. Tier 2 support: business hours (8 AM - 8 PM ET). Tier 3 escalation: four (4) hour response for critical issues.

3.4 Performance. Page load times under three (3) seconds for ninety-five percent (95%) of requests. Report generation under thirty (30) seconds for standard reports.

SECTION 4. REGULATORY COMPLIANCE

4.1 21 CFR Part 11. System shall comply with FDA 21 CFR Part 11 requirements for electronic records and electronic signatures.

4.2 HIPAA. Vendor shall execute a Business Associate Agreement and comply with all HIPAA/HITECH requirements.

4.3 GCP Compliance. System shall support Good Clinical Practice (ICH E6) requirements for clinical trial conduct.

4.4 Audit Trail. System shall maintain an immutable, time-stamped audit trail for all data modifications, accessible for a minimum of fifteen (15) years following study completion.

SECTION 5. DATA OWNERSHIP AND SECURITY

5.1 Data Ownership. All clinical trial data, study documents, and institutional data remain the exclusive property of Institution.

5.2 Encryption. AES-256 encryption at rest. TLS 1.3 for data in transit. Multi-factor authentication required for all users.

5.3 Data Residency. All data stored in US-based data centers with SOC 2 Type II and ISO 27001 certification.

SECTION 6. LIABILITY

6.1 Cap. Vendor's total aggregate liability shall not exceed $3,500,000 (equal to one year of total contract value).

6.2 Regulatory Penalties. Vendor shall indemnify Institution for regulatory penalties directly caused by system non-compliance with 21 CFR Part 11, up to $1,000,000 per occurrence.""",
    },
]


def get_sample_contracts() -> list[dict]:
    return SAMPLE_CONTRACTS


def save_contracts(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_contracts.json")
    with open(output_path, "w") as f:
        json.dump(SAMPLE_CONTRACTS, f, indent=2)
    print(f"Saved {len(SAMPLE_CONTRACTS)} sample contracts to {output_path}")

    for contract in SAMPLE_CONTRACTS:
        print(f"  {contract['id']}: {contract['title']}")
        print(f"    Parties: {', '.join(contract['parties'])}")
        print(f"    Length: {len(contract['text'])} chars")


if __name__ == "__main__":
    save_contracts(output_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
