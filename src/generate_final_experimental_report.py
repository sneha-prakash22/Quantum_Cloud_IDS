from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "Quantum_Cloud_IDS_Final_Experimental_Report.docx"

if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite existing report: {OUT}")

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.7)
sec.bottom_margin = Inches(0.7)
sec.left_margin = Inches(0.8)
sec.right_margin = Inches(0.8)

styles = doc.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"].font.size = Pt(10)
styles["Normal"].paragraph_format.space_after = Pt(6)
for name, size, color in [
    ("Title", 25, "17365D"),
    ("Heading 1", 17, "17365D"),
    ("Heading 2", 13, "2F5597"),
    ("Heading 3", 11, "2F5597"),
]:
    st = styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor.from_string(color)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)

def set_cell_text(cell, text, bold=False, color=None, size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Calibri"
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        set_cell_text(t.rows[0].cells[i], h, True, "FFFFFF", 9)
        shade(t.rows[0].cells[i], "17365D")
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, size=8.5)
    if widths:
        for row in t.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph()
    return t

def para(text="", bold_lead=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        r.bold = True
        p.add_run(text[len(bold_lead):])
    else:
        r = p.add_run(text)
        r.italic = italic
    return p

def bullet(text):
    doc.add_paragraph(text, style="List Bullet")

def numbered(text):
    doc.add_paragraph(text, style="List Number")

# Header and footer
header = sec.header.paragraphs[0]
header.text = "QUANTUM-INSPIRED MACHINE LEARNING FOR CLOUD INTRUSION DETECTION"
header.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in header.runs:
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(100, 100, 100)

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = footer.add_run("Experimental Research Report | 27 November 2026")
fr.font.size = Pt(8)
fr.font.color.rgb = RGBColor(100, 100, 100)

# Cover
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(70)
r = p.add_run("QUANTUM-INSPIRED MACHINE LEARNING\nFOR CYBER INTRUSION DETECTION\nIN ENTERPRISE CLOUD PLATFORMS")
r.bold = True
r.font.size = Pt(24)
r.font.color.rgb = RGBColor(23, 54, 93)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(20)
r = p.add_run("FINAL EXPERIMENTAL REPORT")
r.bold = True
r.font.size = Pt(16)
r.font.color.rgb = RGBColor(47, 85, 151)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(35)
p.add_run("Research project report\n27 November 2026").font.size = Pt(12)

doc.add_page_break()

doc.add_heading("Abstract", level=1)
para(
    "This project investigates a quantum-inspired feature-selection approach for "
    "multiclass cyber intrusion detection using network-flow features. The principal "
    "evaluation compares two Extra Trees classifiers under the same reported dataset "
    "split and classifier configuration: a baseline using 78 features and a proposed "
    "configuration using 20 features selected through an adaptive qubit-inspired search. "
    "The 20-feature configuration achieved 97.9656% accuracy and 86.8955% macro-F1, "
    "compared with 97.8639% accuracy and 86.7197% macro-F1 for the 78-feature baseline. "
    "The reduced model used 74.36% fewer features and had a lower observed training time "
    "in the recorded runs. A separate Mininet experiment generated controlled benign TCP "
    "traffic and produced one extracted flow that the model classified as BENIGN. This "
    "single-flow result is reported as a preliminary pipeline sanity check, not as a "
    "statistically meaningful independent validation or evidence of attack detection."
)

doc.add_heading("1. Research Scope and Objectives", level=1)
para(
    "The research examines whether a quantum-inspired search strategy can identify a "
    "smaller subset of network-flow features while retaining or modestly improving the "
    "performance of an Extra Trees intrusion-detection classifier."
)
doc.add_heading("1.1 Objectives", level=2)
for item in [
    "Prepare a multiclass network-intrusion dataset for supervised learning.",
    "Apply an adaptive qubit-inspired feature-selection procedure.",
    "Evaluate the selected 20-feature subset using Extra Trees.",
    "Compare it only with the 78-feature Extra Trees baseline under the documented setup.",
    "Exercise the trained model on controlled traffic generated in a Mininet emulation.",
    "Document limitations, provenance, and reproducibility evidence transparently.",
]:
    bullet(item)

doc.add_heading("1.2 Primary research comparison", level=2)
para(
    "The primary comparison in this report is strictly between Extra Trees with 78 "
    "features and Extra Trees with 20 quantum-inspired selected features. CatBoost, "
    "XGBoost, CNN, Random Forest, LightGBM, and QPSO-LightGBM outputs are separate "
    "experimental tracks. They are not ranked against the two primary Extra Trees "
    "configurations and are not used to claim classifier superiority."
)

doc.add_heading("2. Dataset and Preprocessing", level=1)
para(
    "The primary experiments use the project dataset at data/balanced_ids_dataset.csv. "
    "The inspected dataset contains 49,155 records, 78 feature columns, and 15 distinct "
    "class labels. The label audit found no null labels and no empty-string labels."
)
table(
    ["Dataset property", "Verified value"],
    [
        ("Records", "49,155"),
        ("Feature columns", "78"),
        ("Classes", "15"),
        ("Null labels", "0"),
        ("Empty labels", "0"),
        ("Training records", "39,324"),
        ("Testing records", "9,831"),
        ("Split", "80:20, stratified"),
        ("Random state", "42"),
    ],
)
para(
    "The Extra Trees scripts convert feature values to numeric form, replace positive "
    "and negative infinity with missing values, and remove records with invalid feature "
    "values. Labels are encoded before training. Both recorded experiments use a "
    "stratified 80:20 train-test split with random_state=42. The dataset audit found "
    "no null or empty labels, so the scripts' slightly different empty-label predicates "
    "do not change the observed label population."
)

doc.add_heading("3. Quantum-Inspired Feature Selection", level=1)
para(
    "The selected subset is produced by an adaptive qubit-inspired search. Each original "
    "feature is represented through an angle parameter theta. A feature's selection "
    "probability is derived from sin(theta) squared, with exploration smoothing. A "
    "candidate subset is sampled without replacement and evaluated; the angle parameters "
    "are then updated toward the best candidate found."
)
table(
    ["Search setting", "Recorded configuration"],
    [
        ("Candidate feature counts", "10, 15, 20, 30"),
        ("Search sample", "Up to 10,000 training records"),
        ("Candidate iterations", "30"),
        ("Learning rate", "0.15"),
        ("Exploration rate", "0.15"),
        ("Random seed", "42"),
        ("Candidate evaluator", "RandomForestClassifier, 50 estimators"),
        ("Fitness", "Mean 3-fold stratified cross-validation macro-F1"),
        ("Cross-validation", "StratifiedKFold, shuffle=True, random_state=42"),
    ],
)
para(
    "The candidate evaluator is a Random Forest model used to estimate feature-subset "
    "fitness. The final primary classifier is Extra Trees with 500 estimators. The "
    "feature-selection procedure is quantum-inspired and software-based; it does not "
    "execute quantum circuits or use quantum hardware."
)
doc.add_heading("3.1 Provenance of the 20-feature list", level=2)
para(
    "The 20-feature list consumed by src/extra_trees_20_features.py is read from "
    "results/quantum_inspired_lightgbm_feature_count_details.json. That artifact is "
    "written by the quantum-inspired LightGBM feature-count experiment script. The "
    "script contains the adaptive qubit-inspired selection procedure described above; "
    "its later LightGBM classifier evaluations form a separate experiment. A direct "
    "comparison of the stored 20-feature list with the list recorded in the Extra Trees "
    "metrics artifact confirmed that the two lists match. This shared artifact lineage "
    "is a provenance detail, not a reason to compare LightGBM and Extra Trees metrics."
)
doc.add_heading("3.2 Selected 20 features", level=2)
features = [
    "Destination Port", "Bwd Packet Length Max", "Flow Packets/s",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Std",
    "Bwd IAT Min", "Fwd PSH Flags", "PSH Flag Count", "URG Flag Count",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Subflow Bwd Packets",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "Active Max",
    "Idle Mean", "Idle Std", "Idle Max",
]
for f in features:
    numbered(f)

doc.add_heading("4. Primary Extra Trees Experimental Setup", level=1)
table(
    ["Configuration", "78-feature baseline", "20-feature proposed"],
    [
        ("Classifier", "Extra Trees", "Extra Trees"),
        ("Number of features", "78", "20"),
        ("Number of estimators", "500", "500"),
        ("Random state", "42", "42"),
        ("Test size", "20%", "20%"),
        ("Stratification", "Yes", "Yes"),
        ("Training records", "39,324", "39,324"),
        ("Testing records", "9,831", "9,831"),
        ("Classes", "15", "15"),
    ],
)
para(
    "Both scripts use the same source dataset, classifier family, estimator count, "
    "random state, and stratified split parameters. The 20-feature script selects the "
    "20 named columns before training. The recorded metric artifacts show matching "
    "training and testing record counts."
)

doc.add_heading("5. Primary Results", level=1)
table(
    ["Metric", "Extra Trees\n78 features", "Extra Trees\n20 features", "Observed change"],
    [
        ("Accuracy", "97.8639%", "97.9656%", "+0.1017 percentage points"),
        ("Macro-F1", "86.7197%", "86.8955%", "+0.1758 percentage points"),
        ("Weighted-F1", "97.8525%", "97.9062%", "+0.0536 percentage points"),
        ("Features", "78", "20", "58 fewer; 74.36% reduction"),
        ("Training time", "3.1241 s", "1.8433 s", "1.2808 s lower; about 41.0%"),
    ],
)
para(
    "The 20-feature configuration produced a modest improvement in the three reported "
    "aggregate classification metrics while using substantially fewer input features. "
    "Its recorded training time was also lower in these runs. Training time is an "
    "observed measurement from the saved experiment artifacts and should not be treated "
    "as a guarantee of the same speed difference on other hardware or repeated runs."
)
para(
    "The accuracy difference is +0.1017 percentage points, not a 0.1% relative "
    "improvement. The results support a cautious conclusion: in this experiment, the "
    "quantum-inspired selected subset retained performance and modestly improved the "
    "reported aggregate metrics while reducing feature dimensionality."
)

doc.add_heading("6. Class-Wise Results and Imbalance Considerations", level=1)
para(
    "The classification reports cover 15 classes. The test supports for some minority "
    "classes are very small: Heartbleed has 2 test records, Infiltration 7, and SQL "
    "Injection 4. XSS has 130 test records. Metrics for classes with only a handful "
    "of test examples are highly sensitive to individual predictions and should not "
    "be interpreted as stable estimates of real-world class performance."
)
para(
    "The recorded macro-F1 values are lower than weighted-F1 values, indicating that "
    "performance averaged equally across classes is less favorable than the metric "
    "weighted by class support. Both metrics are therefore reported rather than relying "
    "on accuracy alone. The minority-class support limitation should be addressed in "
    "future work through additional representative samples and class-specific analysis."
)

doc.add_heading("7. Separate Experimental Tracks", level=1)
para(
    "The project contains additional saved experiments involving CatBoost, XGBoost, "
    "1D CNN, Random Forest, LightGBM, quantum-inspired LightGBM, and QPSO-optimized "
    "LightGBM. These experiments have distinct model configurations and/or experimental "
    "purposes. Their saved metrics are retained as separate research artifacts. They "
    "are deliberately not combined into the primary Extra Trees comparison in this "
    "report. A valid cross-model ranking would require a separately designed and "
    "verified common evaluation protocol, including consistent data partitions, "
    "preprocessing, tuning budgets, and reporting."
)

doc.add_heading("8. Historical PCAP Experiment and Labeling Limitation", level=1)
para(
    "An earlier PCAP prediction workflow extracted flow records and applied the "
    "trained Extra Trees model. The Thursday labeled CSV datasets inspected for "
    "ground-truth association contained feature and label columns but did not provide "
    "the flow identity fields needed to associate their records with the separately "
    "extracted PCAP flows. Exact feature-hash matching found zero matches for both "
    "the 78-feature and selected 20-feature representations."
)
para(
    "Consequently, the historical PCAP predictions cannot be assigned trustworthy "
    "ground-truth labels through that attempted matching procedure, and no PCAP "
    "accuracy is claimed. Those files are retained as historical prediction artifacts, "
    "not as independent validation evidence."
)

doc.add_heading("9. Mininet Controlled-Traffic Experiment", level=1)
doc.add_heading("9.1 Environment and topology", level=2)
para(
    "The independent traffic-generation exercise used Mininet in WSL Ubuntu 22.04 "
    "with a two-host topology connected through an Open vSwitch bridge (OVSBridge). "
    "The source host h1 used 10.0.0.1 and the destination host h2 used 10.0.0.2. "
    "The TCP destination port was 5002. Connectivity testing reported 0% ping loss."
)
table(
    ["Component", "Recorded configuration"],
    [
        ("Environment", "WSL 2, Ubuntu 22.04"),
        ("Mininet", "2.3.0"),
        ("Open vSwitch", "2.17.12"),
        ("Topology", "Two hosts, OVSBridge, no controller"),
        ("Source / destination", "h1 10.0.0.1 / h2 10.0.0.2"),
        ("Traffic", "Controlled benign TCP socket transfer"),
        ("Destination port", "5002"),
        ("Target payload", "500,000 bytes"),
        ("Chunk size", "1,000 bytes"),
        ("Inter-chunk delay", "0.02 seconds"),
        ("Offloading", "TSO/GSO disabled and verified off on both hosts"),
    ],
)
doc.add_heading("9.2 Capture outcome", level=2)
table(
    ["Capture measure", "Observed result"],
    [
        ("Bytes sent", "500,000"),
        ("Bytes received", "500,000"),
        ("Generation duration", "10.287 seconds"),
        ("Application rate", "48,605.57 bytes/second"),
        ("Packets captured", "1,006"),
        ("Packets received by filter", "1,006"),
        ("Packets dropped by kernel", "0"),
        ("PCAP size", "582,532 bytes"),
    ],
)
para(
    "The successful paced capture was saved as "
    "data/mininet_validation/mininet_benign_tcp_paced_001.pcap. The associated "
    "traffic log records the sender, receiver, offloading checks, and tcpdump summary. "
    "Earlier unsuccessful and high-drop diagnostic captures were preserved rather "
    "than silently discarded; they are not used as the successful validation run."
)

doc.add_heading("9.3 Flow extraction and prediction", level=2)
para(
    "Offline flow extraction produced one flow record with 83 columns, including "
    "flow identity fields and extracted features. The flow identifier was "
    "10.0.0.1,55790,10.0.0.2,5002,TCP. It contained 504 forward packets and 502 "
    "backward packets, with 500,000 bytes of forward payload. The 20 required model "
    "features were available after the documented header-length compatibility mapping."
)
table(
    ["Prediction field", "Observed value"],
    [
        ("Ground truth", "BENIGN (controlled benign traffic)"),
        ("Predicted label", "BENIGN"),
        ("Prediction confidence", "0.492, or 49.2%"),
        ("Extracted flow records", "1"),
        ("Flow CSV dimensions", "1 row × 83 columns"),
    ],
)
para(
    "The model's predicted label agreed with the known benign traffic label for this "
    "single flow. The maximum class probability was 49.2%, so the prediction should "
    "not be described as highly confident. This result confirms that the controlled "
    "capture, flow extraction, feature preparation, and inference pipeline completed "
    "for one benign example."
)

doc.add_heading("9.4 Interpretation and validation boundary", level=2)
para(
    "One benign flow is not a statistically meaningful independent test set. It does "
    "not establish a generalization rate, operational detection accuracy, false-positive "
    "rate, or attack-detection capability. No attack traffic was validated in this "
    "Mininet run. The result is therefore a preliminary end-to-end sanity check only."
)

doc.add_heading("10. Reproducibility and Evidence Management", level=1)
para(
    "The successful Mininet experiment produced flow and prediction CSV files under "
    "results/mininet_validation and a SHA-256 manifest named "
    "mininet_validation_sha256_manifest.csv. The manifest records hashes and sizes "
    "for the experiment PCAPs, traffic logs, extracted flow CSV, and prediction CSV. "
    "The manifest does not include a self-hash."
)
para(
    "The primary model metrics are stored in "
    "results/extra_trees_78_features_metrics.json and "
    "results/extra_trees_20_features_metrics.json. The selected feature list is also "
    "recorded in the 20-feature metrics artifact and in the feature-count details "
    "artifact. These files, scripts, model files, and the Mininet evidence directory "
    "should be retained together to support traceability."
)

doc.add_heading("11. Limitations and Threats to Validity", level=1)
for item in [
    "The primary evaluation uses one fixed stratified train-test split; repeated seeds or nested validation were not reported here.",
    "Some classes have extremely small test support, making class-wise scores unstable.",
    "The Mininet sanity check contains only one benign flow and no attack traffic.",
    "The Mininet environment is an emulation, not a production enterprise cloud deployment.",
    "The single Mininet prediction confidence was 49.2%, which warrants cautious interpretation.",
    "The historical PCAP flow predictions could not be matched to trustworthy ground truth using the available Thursday CSV records.",
    "The observed training-time difference is specific to the recorded runs and environment.",
    "The selected-feature artifact's generation lineage passes through a script that also runs separate LightGBM experiments; this report distinguishes the feature-selection procedure from those classifier evaluations.",
]:
    bullet(item)

doc.add_heading("12. Conclusion", level=1)
para(
    "The principal experimental finding is that the Extra Trees model using 20 "
    "quantum-inspired selected features achieved slightly higher accuracy, macro-F1, "
    "and weighted-F1 than the 78-feature Extra Trees baseline in the recorded "
    "evaluation, while reducing the feature count by 74.36%. The observed training "
    "time was also lower. These results are promising within the evaluated dataset "
    "and split, but should not be overstated as proof of broad real-world superiority."
)
para(
    "The Mininet experiment adds a useful end-to-end systems check: controlled benign "
    "traffic was generated, captured without reported kernel packet drops, converted "
    "to one flow, and classified as BENIGN. Because this involved only one benign flow, "
    "it does not establish independent detection performance. Future work should "
    "expand controlled validation to multiple benign and attack scenarios, collect "
    "larger independently labeled flow sets, and evaluate across repeated splits and "
    "more realistic deployment conditions."
)

doc.add_heading("13. Key Project Artifacts", level=1)
table(
    ["Artifact", "Purpose"],
    [
        ("data/balanced_ids_dataset.csv", "Primary multiclass dataset"),
        ("src/extra_trees_78_features.py", "78-feature baseline training"),
        ("src/extra_trees_20_features.py", "20-feature Extra Trees training"),
        ("src/quantum_inspired_lightgbm_feature_count_experiment.py", "Feature-count search and separate LightGBM experiments"),
        ("results/extra_trees_78_features_metrics.json", "Baseline metrics"),
        ("results/extra_trees_20_features_metrics.json", "Proposed model metrics and selected features"),
        ("results/quantum_inspired_lightgbm_feature_count_details.json", "Feature-selection details artifact"),
        ("src/mininet_paced_benign_capture.py", "Controlled Mininet capture workflow"),
        ("data/mininet_validation/", "PCAP captures and traffic logs"),
        ("results/mininet_validation/", "Extracted flow, prediction, and SHA-256 manifest"),
    ],
)

doc.add_heading("14. Reporting Statement", level=1)
para(
    "This report intentionally separates the primary Extra Trees comparison, other "
    "classifier experiments, historical PCAP predictions, and the single-flow Mininet "
    "sanity check. No unsupported cross-model ranking, PCAP accuracy, attack-detection "
    "claim, or broad independent-validation claim is made."
)

# Keep tables from splitting across pages where possible
for t in doc.tables:
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)

doc.core_properties.title = "Quantum-Inspired Machine Learning for Cyber Intrusion Detection"
doc.core_properties.subject = "Final experimental report"
doc.core_properties.author = "Research Project"
doc.core_properties.keywords = "Quantum-inspired feature selection, Extra Trees, intrusion detection, Mininet"

doc.save(OUT)
print(f"REPORT_CREATED={OUT}")
print(f"REPORT_SIZE_BYTES={OUT.stat().st_size}")
