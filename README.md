# A5: Optimization of Human Preferences

## Assignment Details

- **Name:** Min Marn Ko  
- **ID:** st125437  

This project implements **Direct Preference Optimization (DPO)** to enhance a pre-trained language model’s responses based on human preferences. Using **Hugging Face’s TRL library**, the process is divided into four key tasks:

1. **Data Preparation**  
2. **Model Fine-Tuning using DPOTrainer**  
3. **Saving & Uploading Model to Hugging Face Hub**  
4. **Developing a Web Application for Inference**  

## Assignment Overview

The goal is to fine-tune the ChatGPT Neo model for alignment with human preferences. The project involves training on a structured dataset, optimizing GPU performance with efficient memory management and hyperparameter tuning, saving the trained model, and deploying an interactive web application for inference.

---

## Task 1: Data Preparation

### Dataset
The allenai/ultrafeedback_binarized_cleaned dataset is used, containing fields such as prompt, chosen, and rejected, which are stored as dictionaries. These fields are transformed and preprocessed to ensure that each instance is correctly structured for training.
### Key Steps:
- Extract and clean **prompt**, **chosen** and **response** fields.
- Subset data for initial testing and debugging.

---

## Task 2: Fine-Tuning with DPOTrainer

### Model & Dataset
- **Pretrained Model:** `EleutherAI/gpt-neo-125M`
- **Dataset:** Preprocessed version of `allenai/ultrafeedback_binarized_cleaned`

### Training Configuration

| Parameter | Value |
|-----------|------|
| **Batch Size** | 4 (micro-batch) |
| **Epochs** | 3 |
| **Learning Rate** | 5e-5 |
| **Beta** | 0.1 |

### Training Process
1. **Load Models** – Initialize the base and reference models, ensuring they run on the GPU.  
2. **Set Training Configurations** – Configure `DPOConfig` with the necessary hyperparameters.  
3. **Train the Model** – Calculate **DPO loss**, adjust model weights, and track progress.  
4. **Optimize Hyperparameters** *(optional)* – Experiment with different learning rates, batch sizes, epochs, and beta values.


**Experiment Results**- 'learning_rate': 5e-05, 'batch_size': 4, 'epochs': 3, 'beta': 0.1, 'loss': 3.102883056271821e-05
---

## Task 3: Model Saving & Uploading

After fine-tuning, the model and tokenizer are saved and uploaded to **Hugging Face Hub** for convenient access.  

**Repository ID:** [minmarn/dpo_best_model_gpt_neo](https://huggingface.co/minmarn/dpo_best_model_gpt_neo)  

This enables seamless inference and future fine-tuning.

---

## Task 4: Web Application for Inference

#### Screenshots:
- **Result**  
  ![Home](2.png)
  ![Home](3.png)







## How to Run the Project

To run app, 
1. Load the files from this repository
2. Run
```sh
stremlit run app.py
```
3. Access the app with http://localhost:8501/
---

