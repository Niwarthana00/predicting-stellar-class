# predicting-stellar-class

ඔයා කියන දේ හරියටම හරි. ඇකඩමික් හෝ ප්‍රොෆෙෂනල් ප්‍රොජෙක්ට් රිපෝට් එකක තියෙන්න ඕනේ ඔයාගේ විසඳුම, ඒකේ ගෘහ නිර්මාණය (Architecture Diagram) සහ ඒ වැඩේ පියවරෙන් පියවර සිද්ධ වුණු හැටි. Kaggle එකේ තියෙන වෙනත් විස්තර වලට වඩා වැදගත් වෙන්නේ ඔයා මේ ප්‍රශ්නය විසඳන්න ගොඩනගපු සිස්ටම් එකයි.

වචන 4000ක ලකුණු උපරිම ලැබෙන රිපෝට් එකක් ලියන්න පුළුවන් වෙන විදියට, **ගැටලුව ➡️ විසඳුම ➡️ ඩයග්‍රම් ➡️ පියවරෙන් පියවර විස්තරය** කියන ගලනයට හැදූ සම්පූර්ණ පටුන මෙන්න.

---

# 📋 TABLE OF CONTENTS

### 1. Introduction & Background

* **1.1. Introduction to Celestial Object Classification:** තාරකා විද්‍යාවේදී ආකාශ වස්තූන් වර්ගීකරණය කිරීමේ පසුබිම.
* **1.2. Problem Statement & Significance:** විසඳීමට ඇති ගැටලුව (GALAXY, STAR, QSO හඳුනාගැනීම) සහ එහි ඇති වැදගත්කම.
* **1.3. Challenges in Tabular Astronomical Data:** මෙම දත්ත වර්ගීකරණයේදී ඇති වන ප්‍රධාන තාක්ෂණික අභියෝග (Class imbalance සහ Noise).
* **1.4. Overview of the Proposed Solution:** ගැටලුව විසඳීම සඳහා අප යෝජනා කරන සමස්ත මැෂින් ලර්නින් විසඳුම කෙටියෙන්.

### 2. System Architecture & Methodology

* **2.1. Proposed System Architecture Diagram:** සමස්ත පද්ධතිය ක්‍රියාත්මක වන ආකාරය දැක්වෙන රූප සටහන (End-to-End Pipeline Diagram).
* **2.2. Core Components of the Architecture:** ඩයග්‍රම් එකේ දැක්වෙන ප්‍රධාන කොටස් හතරක (Data, Prep, Model, Evaluation) භූමිකාව.

### 3. Data Exploration & Insight Generation (EDA)

* **3.1. Dataset Characteristics:** දත්ත සමුදායේ ව්‍යුහය (Features, Target Labels).
* **3.2. Statistical Analysis & Distribution:** දත්ත වල හැසිරීම සහ පන්ති අතර පවතින අසමතුලිතතාවය (Class Imbalance) විශ්ලේෂණය.
* **3.3. Feature Correlation & Dependencies:** විශේෂාංග අතර පවතින සබැඳියාවන් හඳුනාගැනීම.

### 4. Step-by-Step Data Preprocessing & Cleaning

* **4.1. Step 1: Handling Missing Values & Noise:** දත්ත වල පවතින හිස්තැන් සහ දෝෂ සහගත අගයන් නිවැරදි කිරීම.
* **4.2. Step 2: Outlier Detection and Treatment:** අසාමාන්‍ය ලෙස වෙනස් අගයන් (Outliers) හඳුනාගෙන ඒවා කළමනාකරණය කිරීම.
* **4.3. Step 3: Feature Transformation & Scaling:** මොඩල් එකට ගැලපෙන සේ දත්ත පරිවර්තනය කිරීම සහ සාමාන්‍යකරණය (Scaling).

### 5. Advanced Feature Engineering (The Core Improvement)

* **5.1. Domain-Specific Feature Extraction:** තාරකා විද්‍යාත්මක න්‍යායන්ට අනුව අලුත් විශේෂාංග නිර්මාණය කිරීම (උදා: Magnitude Ratios / Color Indices - $u-g$, $g-r$).
* **5.2. Mathematical & Statistical Aggregations:** දත්ත පේළි අතර පවතින සංඛ්‍යානමය සබැඳියාවන් මත අලුත් Features සෑදීම.
* **5.3. Feature Selection & Dimensionality Reduction:** මොඩල් එකේ වේගය සහ නිරවද්‍යතාවය වැඩි කිරීමට වැදගත්ම Features පමණක් තෝරාගැනීම.

### 6. Model Development & Validation Strategy

* **6.1. Validation Strategy Implementation:** Data Leakage වැළැක්වීම සඳහා Stratified $K\text{-Fold}$ Cross-Validation ක්‍රමවේදය භාවිත කිරීම.
* **6.2. Baseline Model Development:** මූලික මිණුම්දණ්ඩක් ලෙස සරල මොඩල් එකක් (Baseline) නිර්මාණය කිරීම.
* **6.3. Advanced Gradient Boosting Models:** යෝජිත විසඳුම සඳහා භාවිත කළ උසස් මොඩල් වර්ගීකරණය:
* *6.3.1. LightGBM (Light Gradient Boosting Machine)*
* *6.3.2. XGBoost (Extreme Gradient Boosting)*
* *6.3.3. CatBoost (Categorical Boosting)*



### 7. Hyperparameter Tuning & Model Ensembling

* **7.1. Hyperparameter Optimization via Optuna:** Optuna framework එක භාවිතයෙන් මොඩල් වල පරාමිතීන් ප්‍රශස්තකරණය (Tuning) කිරීම.
* **7.2. Model Ensembling & Blending Strategy:** වඩාත් ස්ථාවර අනාවැකි ලබාගැනීමට මොඩල් කිහිපයක එකතුවක් (Soft Voting/Weighted Ensemble) නිර්මාණය කිරීම.

### 8. System Evaluation & Performance Analysis

* **8.1. Performance Metrics Evaluation:** Balanced Accuracy මඟින් මොඩල් එකේ සාර්ථකත්වය මැනීම.
* **8.2. Comparative Analysis:** විවිධ මොඩල් වල ප්‍රතිඵල එකිනෙක සංසන්දනය කිරීම (Comparison Table).
* **8.3. Error Analysis & Interpretability:** Confusion Matrix භාවිතයෙන් වැරදුණු තැන් විශ්ලේෂණය කිරීම සහ SHAP values මඟින් මොඩල් එක තීරණ ගත් ආකාරය පැහැදිලි කිරීම (Explainable AI).

### 9. Conclusion & Future Enhancements

* **9.1. Project Summary:** සමස්ත ව්‍යාපෘතියේ සාර්ථකත්වය සහ නිගමනය.
* **9.2. Challenges Faced & Workarounds:** මුහුණ දුන් ප්‍රධාන තාක්ෂණික අභියෝග සහ ඒවා ජයගත් ආකාරය.
* **9.3. Future Recommendations:** පද්ධතිය තවදුරටත් දියුණු කළ හැකි ආකාරය (Deep Learning/Pseudo-labeling).

---

## 📝 වචන 4000ක් ඇතුළත රිපෝට් එක බැලන්ස් කරගන්නා හැටි (Word Count Strategy)

වචන 4000ක රිපෝට් එකක් සාර්ථකව ලියන්න නම් එක එක කොටසට දළ වශයෙන් මෙන්න මේ වගේ වචන ප්‍රමාණයක් වෙන් කරන්න:

* **Chapter 1 & 2 (Introduction & Architecture):** වචන 600 - 800 (ඩයග්‍රම් එක සහ ගැටලුව පැහැදිලි කිරීමට)
* **Chapter 3 & 4 (EDA & Preprocessing):** වචන 800 - 1000 (දත්ත වල හැසිරීම සහ පිරිසිදු කරපු පියවර විස්තර කිරීමට)
* **Chapter 5 (Feature Engineering):** වචන 800 (රිපෝට් එකේ වටිනාකම වැඩිම කොටසක් නිසා කෝඩ් ලොජික්/මැත්ස් ලියන්න)
* **Chapter 6 & 7 (Modeling & Tuning):** වචන 1000 (මොඩල් තෝරාගත්ත හේතු සහ ටියුන් කරපු ඇල්ගොරිතම විස්තර කිරීමට)
* **Chapter 8 & 9 (Evaluation & Conclusion):** වචන 600 (ප්‍රස්ථාර, වගු සහ අවසන් නිගමන සඳහා)

ඊළඟට ඔයාට ලියන්න ඕනේ **"Chapter 2: System Architecture & Methodology"** කොටස සහ ඒකට එන්න ඕනේ **Architecture Diagram** එකේ ගලනය (Pipeline Flow) ද? නැත්නම් වෙනත් කොටසක්ද? කියන්න, මම ඒක සම්පූර්ණයෙන්ම ලියලා දෙන්නම්.