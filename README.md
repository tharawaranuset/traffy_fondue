# 🚦 Traffy Fondue: Multi-Label Organization Classification

This project aims to develop an AI model to categorize citizen complaints from the **Traffy Fondue** platform in Bangkok. The goal is to automatically route these issues to the responsible government agencies or organizations. Since a single complaint can involve multiple organizations, this problem is framed as a **Multi-label Text Classification** task.

> **🎓 Academic Project**
> This project is a part of the **2110446 DSDE - Data Science and Data Engineering** course.
> Semester: **2/2025** | **Engineering** | **Chulalongkorn University**
> Developed by: **Thara Waranuset**

---

## 📋 Dataset Description
The dataset contains textual descriptions of complaints and their corresponding responsible organizations (labels).
- **Train Data:** 306,419 samples
- **Test Data:** 37,406 samples
- **Labels:** 12 organizations, which are:
  1. สำนักงานตำรวจแห่งชาติ (Royal Thai Police)
  2. การรถไฟฟ้าขนส่งมวลชนแห่งประเทศไทย (Mass Rapid Transit Authority of Thailand)
  3. สภาเด็กและเยาวชนกรุงเทพมหานคร (Bangkok Children and Youth Council)
  4. กรมควบคุมมลพิษ (Pollution Control Department)
  5. กรมสรรพสามิต (Excise Department)
  6. การไฟฟ้านครหลวง (Metropolitan Electricity Authority)
  7. กรมทางหลวง (Department of Highways)
  8. สำนักงานประกันสุขภาพแห่งชาติ (National Health Security Office)
  9. การประปานครหลวง (Metropolitan Waterworks Authority)
  10. คณะกรรมการการพัฒนาเศรษฐกิจ (National Economic and Social Development Council)
  11. กระทรวงการท่องเที่ยวและกีฬา (Ministry of Tourism and Sports)
  12. สำนักงาน กสทช. ศูนย์รับแจ้งปัญหา 1200 (NBTC Call Center 1200)

---

## 🧠 Model & Techniques
This project utilizes a pre-trained Thai language model and implements several techniques to handle severe class imbalance:

- **Base Model:** [`airesearch/wangchanberta-base-att-spm-uncased`](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) (WangchanBERTa)
- **Loss Function:** **Asymmetric Loss (ASL)** to manage the imbalance between positive and negative labels (`gamma_neg=2.0`, `gamma_pos=0.0`).
- **Data Resampling & Augmentation:** 
    - **Undersampling:** Capped major labels at a maximum of 8,000 samples to prevent model bias.
    - **Oversampling & Augmentation:** Applied `simple_swap_augment` (word position swapping) to increase samples for minority labels.
- **Threshold Tuning:** Implemented label-specific threshold optimization to maximize the F1-Score for each individual label rather than using a standard 0.5 cutoff.

---

## 🛠️ Installation
To run this project locally or on Google Colab, please install the required dependencies:

```bash
# Install required Python libraries
pip install iterative-stratification datasets accelerate pythainlp
pip install -U transformers tokenizers sentencepiece huggingface_hub

# If running on Google Colab (to support Thai fonts in plots)
apt-get -y install fonts-thai-tlwg