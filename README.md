# PageWeaver: Execution Guide

## Execution Order

```text
0. Extract the dataset and configure model access and the Android environment
   |
   +-- PageWeaver / GPA
   |     1. page_cluster/page_cluster_byMockups.py
   |     2. page_cluster/generateSubGraphy.py
   |     3. Run the selected code-generation script
   |
   +-- E2E / E2E-prompted
         3. Run the selected code-generation script directly
   |
4. utg_generate/appium_utg.py
     Calls label_component.py internally; captures screenshots and UTG
5. Match runtime screenshots to reference mockups and rename them
6. metric/calculate_metric.py
```

The four generation methods are alternative experiments, not consecutive stages. PageWeaver and GPA require clusters and subgraphs; E2E and E2E-prompted use screenshots and UTG directly.

## 0. Prepare Data and Configuration

### Dataset

Extract the dataset into the `dataset/` directory at the project root, so that app data is located at `dataset/<category>/<app>/`.

### Environment

Use Python 3.10 or later on Windows, with Node.js/npm, a working React Native Android project, Android SDK/ADB, an emulator, and an Appium server with the UiAutomator2 driver available.

Python dependencies used across the stages include `openai`, `httpx`, `pillow`, `appium-python-client`, `selenium`, `opencv-python`, `numpy`, `scikit-image`, `torch`, OpenAI CLIP, `easyocr`, `nltk`, and `tqdm`. CLIP and OCR evaluation may require model weights on first use.

### API Configuration

Before running clustering or code generation, configure the corresponding `API_KEY`, `MODEL_NAME`, and `API_BASE` in each of the following scripts:

- `page_cluster/page_cluster_byMockups.py`
- `code_generate/code_generate_scripts/code_generate_pageWeaver.py`
- `code_generate/code_generate_scripts/baseline/GPA.py`
- `code_generate/code_generate_scripts/baseline/E2E.py`
- `code_generate/code_generate_scripts/baseline/E2E-prompted.py`

Set `API_KEY` to your API credential, `MODEL_NAME` to the model identifier, and `API_BASE` to the API endpoint URL provided by your service. Configure each script before running its corresponding step.

## 1. Generate Page Clusters

```powershell
python page_cluster/page_cluster_byMockups.py
```

## 2. Generate Page and Navigation Subgraphs

```powershell
python page_cluster/generateSubGraphy.py
```

## 3. Generate React Native Code

### PageWeaver

```powershell
python code_generate/code_generate_scripts/code_generate_pageWeaver.py
```

### GPA

```powershell
python code_generate/code_generate_scripts/baseline/GPA.py
```

### E2E

```powershell
python code_generate/code_generate_scripts/baseline/E2E.py
```

### E2E-prompted

```powershell
python code_generate/code_generate_scripts/baseline/E2E-prompted.py
```

## 4. Label Code and Capture the App with Appium for Evaluation

With Node.js/npm installed and the emulator running, install Appium and its Android driver, then start the server in a separate terminal:

```powershell
npm install -g appium
appium driver install uiautomator2
appium --address 127.0.0.1 --port 4723
```

Then run:

```powershell
python utg_generate/appium_utg.py
```

## 5. Match and Rename Runtime Screenshots

Before evaluation, compare the runtime screenshots captured by Appium with the reference mockups in `dataset/<category>/<app>/`. Match each screenshot to the corresponding reference mockup based on its page content and UI state, then rename it to use the reference mockup's filename.

## 6. Calculate Metrics

```powershell
python metric/calculate_metric.py
```
