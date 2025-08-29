# CatalystOU

## Summary
This is the repo for our 2025 summer project: CatalystOU: A Pilot for an LLM-Powered Researcher Collaboration Network

In this project, we are trying to build a system that can find potential collaboration opportunities by digest the researcher's publications.


## Contents
In this repo, we have source code and groundtruth of the project.

In the source code: we have 3 major files:
* catalystOU\_UI
* profile\_extractor
* collaboration\_matcher


## Usage
This project does not develop new LLM, we are using existing LLM as a service.

To use our code, you need to first get your LLM\_API\_KEY and your LLM\_API\_URL.

Then, you need to set the environment variable for the LLM\_API\_KEY and LLM\_API_\URL.

To run our code, you need to type the following command in a terminal:
* streamlit run catalystOU.py 


## Dependency
To run our code, you need to install the following Python Package/Libraries:
* streamlit
* pandas
* plotly
* openai
* python-dotenv


## Author
Alexander Lee, Alexander.Lee-1@ou.edu

Chenggang Wang, chenggang.wang@ou.edu
