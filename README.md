# CatalystOU

## Overview
This is the repo for our 2025 summer project: CatalystOU: A Pilot for an LLM-Powered Researcher Collaboration Network

In this project, we are trying to build a system that can find potential collaboration opportunities by digesting the researcher's publications.

This project does not develop a new LLM; instead, we are using an existing LLM as a service.

## Project Goals and Methodology
### Objectives
The primary goals of this project are:
- To develop a pipeline that utilizes the LLM to ingest researcher publications (in PDF format) and extracts key information.
- To implement the LLM to identify research collaboration potential with prompt engineering and sampling that will provide objective and consistent grading between two researchers. 
- Creating a simple user interface that allows users to input various researcher information and publication which will visualize the researcher collaborations. 

### Methodology
Our main methodology can broken down into the key steps:
#### 1. Data Collection and Ground Truth
The project began with the creation of a ground truth dataset for guiding the main development. We selected two researchers from five different disciplines at the University of Oklahoma. For each researcher, five recent and impactful publications were chosen for analysis. This manually curated datasets can be found in the [Manual Labeled Data](Manual%20Labeled%20Data) folder.

#### 2. Profile Extraction (```profile_extraction.py```)
We found that designing ```profile_extraction.py``` mainly centered around designing a system to create a researcher profile in JSON format. We decided that utilizing the method of Summarize and Synthesize each individual publication allowed for the best results in terms of time-efficiency without losing accuracy of the researcher's main interests.

#### 3. Collaboration Matching (```profile_matcher.py```)
For matching the two profiles, the primary focus was prompt engineering. The LLM was given a specific persona as an "expert researcher" and is provided a detailed rubric on its output which is also in JSON format. This rubric guides the model to score the collaboration potential based on their complementary skills, and overlapping research interests, with a main focus towards consistent and objective results. 

#### 4. Prompting Strategy
The LLM system was designed for few-shot learning in mind, where multiple examples could be provided to guide the model. In its current implementation, however, a one-shot learning approach is used for the extraction. Further testing with few-shot approach is a potential area for future work. 

#### 5. Web Interface Design (```CatalystOU_GUI.py```)
Designing the web interface mainly consisted of two pages.
- Input Page: Mainly focused on uploading the researcher publications for analysis for each researcher, as well as their names to focus which researcher. It also has the option to use the JSON format. 
- Output Page: Shows the researcher profiles for each researcher, as well as the collaboration synergy scores and detailed explanations of any potential overlap and interests.

## Contents
In this repo, we have source code and groundtruth of the project.

In the source code: we have 3 major files:
* **catalystOU\_UI.py**: The main Streamlit file that runs the web user interface.
* **profile\_extraction.py**: Contains the  functions to run a profile extraction on a single researcher and outputs a JSON format profile.
* **profile\_matcher.py**: Contains the functions to compare two researcher profiles and calculate collaboration synergies.

## Web UI Interface
![Web UI Interface](https://github.com/Aegon007/CatalystOU/blob/main/catalystOU_webUI.png)

## Getting Started
Following these steps will allow you to run this on your local machine.

### 1. Cloning the Repository
First, clone the repo to your local machine and navigate to the project root directory.
```bash
git clone <Repo URL>
cd CatalystOU
```

### 2. Installing Dependencies
This project uses many dependencies, which can be found in [requirements.txt](requirements.txt). They can easily be installed by using the respective file and doing the following.
```bash
pip install -r requirements.txt
```

### 3. Setting up Environmental Variables
This project requires API credentials to connect to the Large Language Model (LLM)
1. Within the root directory of the project, create a new file called ```.env```
2. In ```.env```, add your API credentials to the file in the following format:
```
LLM_API_KEY="your_api_key"
LLM_API_URL="your_api_url"
```
The application will automatically then load these variables when it runs.

### 4. Launching the Application
Launch the Streamlity web interface by running the command in your terminal:
```bash
streamlit run catalystOU_GUI.py
```
or if on Windows:
```ps
python -m streamlit run catalystOU_GUI.py
```

## Authors
Alexander Lee, Alexander.Lee-1@ou.edu

Chenggang Wang, chenggang.wang@ou.edu