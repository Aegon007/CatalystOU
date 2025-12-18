import os
import sys
import argparse

import json
import asyncio
import PyPDF2

from loguru import logger
from pathlib import Path
from openai import AsyncOpenAI, APITimeoutError
from dotenv import load_dotenv
from io import BytesIO # Import BytesIO to handle in-memory files


# --- 配置区域 ---
load_dotenv()
CONCURRENT_LIMIT = 5  # 限制同时并发请求数
LOG_FILE = "experiment.log"


# 配置日志：同时输出到屏幕和文件
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE, rotation="10 MB", level="DEBUG") # 文件记录更详细的 DEBUG 信息


# --- 1. Client 初始化 ---
def get_client():
    api_key = os.getenv('LLM_API_KEY')
    api_url = os.getenv('LLM_API_URL')
    if not api_key:
        logger.critical("API key not found!")
        sys.exit(1)
    return AsyncOpenAI(api_key=api_key, base_url=api_url)

client = get_client()


# --- 2. 核心功能函数 ---
# --- Part 2: PDF and Summarization Functions (Async) ---
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts all text from an in-memory PDF file object.
    """
    try:
        print(f"Reading text from {pdf_path}...")
        page_text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    page_text += extracted + "\n"
        print(f"Successfully extracted text from {pdf_path}.")
        return page_text
    except Exception as e:
        logger.warning(f"An error occurred while reading the PDF {pdf_path}: {e}")
        return ""


async def summarize_single_paper(paper_text: str, pdf_path: str, model_name: str, semaphore: asyncio.Semaphore) -> str:
    """ 总结单篇论文 (带并发限制) """
    if not paper_text:
        return ""

    async with semaphore:
        try:
            prompt = f"""
            You are an expert research analyst. Your task is to read the entire uploaded document and create a summary list for an academic audience.
            Your summary must explicitly identify and include the following verifiable information:
            The primary research objective and core topic of the paper.
            Specific Methodologies and Techniques: Go beyond broad categories. List the specific, named models, unique algorithms, formal theorems or lemmas, and proprietary frameworks that are central to the paper's argument. For example, instead of just "machine learning," specify the exact named model or algorithm used.
            Data, Platforms, and Tools: List all specific datasets, software, programming languages, libraries, and key equipment used. If none are found, state "N/A".
            The key quantitative and qualitative findings and conclusions as stated in the paper.
            Authors and Affiliations: List all authors and their corresponding affiliations as stated in the text.

            Paper Text (first 50000 characters):
            ---
            {paper_text[:50000]}
            ---
            """

            logger.info(f"Sending text from {pdf_path} to LLM for summarization...")

            resp = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant that creates detailed summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                timeout=1200.0, # 20-minute timeout
            )
            summary = resp.choices[0].message.content
            logger.info(f"Successfully generated detailed summary for {pdf_path}.")
            return summary
        except Exception as e:
            logger.error(f"An error occurred during summarization for {pdf_path}: {e}")
            return ""


# --- Part 3: Researcher Profile Creation (The "Synthesize" Step) ---
async def synthesize_profile(researcher_name: str, list_of_summaries: list[str], example_summaries: list[str], example_json_output: str, model_name: str) -> dict:
    """
    Uses the LLM to synthesize a structured profile from a list of detailed paper summaries,
    guided by a provided example and a specific researcher name.
    """
    if not list_of_summaries:
        return {}

    example_summaries_text = "\n\n---\n\n".join(f"Summary of Paper {i+1}:\n{summary}" for i, summary in enumerate(example_summaries))
    actual_summaries_text = "\n\n---\n\n".join(f"Summary of Paper {i+1}:\n{summary}" for i, summary in enumerate(list_of_summaries))

    prompt = f"""
    You are an expert academic analyst creating a profile for a formal, academic audience. Your task is to synthesize a comprehensive and highly detailed researcher profile for '{researcher_name}' from the provided summaries.
    Your output must be a single, complete JSON object and nothing else.

    ### EXAMPLE ###
    [START OF EXAMPLE INPUT SUMMARIES]
    ---
    {example_summaries_text}
    ---
    [END OF EXAMPLE INPUT SUMMARIES]

    [START OF EXAMPLE JSON OUTPUT]
    {example_json_output}
    [END OF EXAMPLE JSON OUTPUT]

    ### ACTUAL TASK ###
    Now, using the same format and level of analysis, create a profile for '{researcher_name}' based on the following summaries.
    [START OF ACTUAL INPUT SUMMARIES]
    ---
    {actual_summaries_text}
    ---
    [END OF ACTUAL INPUT SUMMARIES]
    [START OF ACTUAL JSON OUTPUT]
    """

    try:
        logger.info("\nSending summaries to LLM for final profile synthesis...")
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"You are an expert research analyst that only outputs a single, complete JSON object for the researcher '{researcher_name}'."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        response_content = resp.choices[0].message.content.replace("[END OF ACTUAL JSON OUTPUT]", "").strip()

        if response_content.startswith("```json"):
            response_content = response_content[7:-3].strip()

        profile_data = json.loads(response_content)
        logger.info("Successfully created researcher profile.")
        return profile_data
    except Exception as e:
        logger.error(f"An error occurred during profile creation: {researcher_name}: {e}")
        return {}


async def generate_profile_for_one_author(author_dir: Path, out_dir: Path, model_name: str, semaphore: asyncio.Semaphore) -> none:
    """
    An async wrapper in the frontend to orchestrate the profile extraction process
    by calling the imported backend functions.
    """
    # --- This is the example data your backend needs for its prompt ---
    example_summaries_for_prompt = [
        # Summary 1
        """• Primary Research Objective and Core Topic: The primary research objective is to introduce a pioneering approach that integrates digital twin (DT) technology with a federated learning management system (FLMS) to enhance the security and resilience of vehicular networks in the 6G era against adversarial attacks. The core topic focuses on developing novel federated unlearning (FUL) techniques to mitigate the influence of malicious or poisonous clients within already compromised networks, ensuring dependable and secure communication.
• Specific Methodologies and Techniques:
    ◦ Framework: DT-FU (Digital Twin-Driven Federated Unlearning for Resilient Vehicular Networks in the 6G Era).
    ◦ Machine Learning Approach: Federated Learning (FL).
    ◦ Unlearning Technique: Federated Unlearning (FUL), specifically client-level unlearning, to dynamically remove a client's influence from a global model.
    ◦ Unlearning Algorithms: Uses gradient ascent to maximize the loss associated with a client's data for effective unlearning and incorporates gradient descent techniques.
    ◦ Detection Mechanism: Leverages Digital Twins (DTs) to monitor client behaviors and identifies malicious clients using anomaly detection techniques that analyze unusual patterns in model performance, learning rates, and updates.
    ◦ Model Aggregation: Federated Averaging (FedAvg) is used to aggregate local updates into a global model.
    ◦ Neural Network Models (for experiments): LeNet-5 model for MNIST and Fashion-MNIST datasets, and VGG-11 model for CIFAR-10.
    ◦ Comparison Methods: Retrain (training from scratch excluding malicious clients), Federated Averaging (FedAvg), FedEraser, and FedRecovery.
• Data, Platforms, and Tools:
    ◦ Datasets: MNIST, Fashion-MNIST, and CIFAR10.
    ◦ Programming Languages/Libraries: Python and PyTorch.
    ◦ Toolkit for Adversarial Threats Simulation: Adversarial Robustness Toolbox.
    ◦ Network Technology: 6G.
    ◦ Application Domain: Vehicular networks.
• Key Quantitative and Qualitative Findings and Conclusions:
    ◦ Effectiveness against Backdoor Attacks: DT-FU reduces backdoor accuracy to baseline levels, comparable to retraining, effectively neutralizing adversarial influences. Conversely, without unlearning (FedAvg), vulnerability to malicious patterns increases.
    ◦ Efficiency in Restoring Clean Accuracy: DT-FU is more efficient than conventional retraining in restoring clean accuracy, achieving high accuracy within fewer aggregation rounds by precisely adjusting the existing global model.
    ◦ Comparative Performance: DT-FU maintains clean accuracy near the Retrain baseline and outperforms both FedEraser and FedRecovery in minimizing backdoor risks.
    ◦ Overall Strengths: Demonstrates dual strengths in excising adversarial influences and resource efficiency, ensuring model integrity and optimizing computational expenditure.
    ◦ Impact of 6G: 6G technology significantly reduces communication costs, enables quicker malicious activity detection, and accelerates FUL processes, enhancing bandwidth, lowering latency, and improving reliability for FL systems.
    ◦ Qualitative Conclusion: DT-FU is a novel, robust, and scalable framework that secures vehicular networks in the 6G era by leveraging digital twins and federated unlearning to combat data poisoning attacks, seamlessly integrating with existing FLMS.
• Authors and Affiliations:
    ◦ Wathsara Daluwatta: RMIT University, Australia.
    ◦ Shehan Edirimannage: RMIT University, Australia.
    ◦ Charitha Elvitigala: RMIT University, Australia.
    ◦ Ibrahim Khalil: RMIT University, Australia.
    ◦ Mohammed Atiquzzaman: The University of Oklahoma, USA.""",

    # Summary 2
    """Primary Research Objective and Core Topic: The primary research objective is to propose a novel approach that combines Adversarial Machine Learning (AML) with Federated Learning (FL) to address significant data privacy concerns in smart city surveillance, particularly regarding facial data captured by cameras. The core topic is achieving privacy-preserving face recognition in distributed settings by perturbing surveillance data at the source.
• Specific Methodologies and Techniques:
    ◦ Core Approach: Integration of Adversarial Machine Learning (AML) and Federated Learning (FL).
    ◦ Privacy Preservation Method: Utilizes a noise generator to perturb surveillance data directly at the source (cameras) before sharing, employing a local differential privacy (LDP) approach.
    ◦ Perturbation Algorithm: Iteratively transforms input images by introducing noise until misclassification occurs, saving the image from the last correct classification with maximum perturbation. The perturbation is updated based on the gradient of the loss function. The concept is inspired by the DeepFool adversarial attack.
    ◦ FL Algorithm: Typically uses Federated Averaging for model aggregation.
    ◦ Privacy Guarantee: Introduces a novel guarantee called γ-AdvNoise privacy.
    ◦ Neural Network Models (for experiments):
        ▪ AlexNet model used during the perturbation process.
        ▪ VGG16 model used as a baseline for centralized machine learning setup.
        ▪ ResNet18, GoogLeNet, DenseNet, MobileNet, and ResNeXt used for privacy evaluation in the FL setup.
• Data, Platforms, and Tools:
    ◦ Datasets: Pins dataset (facial images), MNIST (handwritten digits), and CIFAR-10 (color images).
    ◦ Platforms: AWS cloud computing service, specifically Super Large SageMaker Notebook instance and G4DN series N11 GPU Notebook - XXLarge instance.
    ◦ Programming Languages/Libraries: Python and PyTorch.
• Key Quantitative and Qualitative Findings and Conclusions:
    ◦ Accuracy: Achieved a testing accuracy of 99.95% in standard machine learning (centralized) settings and 96.24% in federated learning (distributed) settings.
    ◦ Privacy-Utility Trade-off: The system allows users to adjust the epsilon value in the noise generator to tailor the privacy-utility trade-off, where smaller values yield reasonable privacy with higher utility, and larger values enhance privacy at the expense of utility.
    ◦ Robustness of Perturbation: The noise introduced varies randomly per image, contributing to the robustness of machine learning models by enabling them to comprehend diverse scenarios rather than memorizing.
    ◦ Model Performance (Centralized): The VGG16 model demonstrated high robustness with a 99.95% test accuracy on surveillance images with adversarial perturbations.
    ◦ Model Performance (FL): For the perturbed dataset, models achieved varying test accuracies: ResNet18 (80.01%), GoogLeNet (83.23%), DenseNet (82.01%), MobileNet (72.33%), and ResNeXt (66.67%).
    ◦ Efficiency: The data perturbation runtime is efficient, with average times ranging from 0.0023s (MNIST) to 0.0079s (Pins) per image. The perturbation process exhibits linear time complexity based on iterations and pixel count.
    ◦ Qualitative Conclusion: The proposed framework effectively balances privacy and effectiveness in federated learning for smart city surveillance by transforming raw data into privacy-preserving data through intelligent noise generation.
• Authors and Affiliations:
    ◦ Farah Wahida: School of Computing Technologies, RMIT University, Melbourne, Australia.
    ◦ M.A.P. Chamikara: CSIRO’s Data61, Melbourne, Australia.
    ◦ Ibrahim Khalil: School of Computing Technologies, RMIT University, Melbourne, Australia.
    ◦ Mohammed Atiquzzaman: School of Computer Science, University of Oklahoma, Norman, USA.""",

    # Summary 3
    """Primary Research Objective and Core Topic: The primary research objective is to introduce "Unlearning as a Service for Safeguarding Federated Learning" (UaaS-SFL), a novel service designed to integrate seamlessly with existing FL management systems. Its core topic is to effectively remove the impact of poisoning clients and restore the integrity of the global model in Federated Learning (FL) systems within IoT networks, addressing the limitations of traditional pre-detection methods in already compromised environments.
• Specific Methodologies and Techniques:
    ◦ Service Framework: UaaS-SFL (Unlearning as a Service for Safeguarding Federated Learning).
    ◦ Core Concept: Federated Unlearning (FUL), specifically client-level federated unlearning, which enables selective erasure of specific clients' data influence from the global model.
    ◦ Unlearning Algorithms:
        ▪ Initial Model Construction via Median Aggregation: Establishes a robust baseline model by aggregating parameters from client models using median values, mitigating outlier influence.
        ▪ Client-level Unlearning via Gradient Ascent: Maximizes the loss for data targeted for unlearning locally at the client, with L1 regularization to promote sparsity and prevent overfitting.
        ▪ Early Stopping Criterion Based on Expected Calibration Error (ECE): Prevents over-unlearning by monitoring the discrepancy between predicted probabilities and actual correctness on a validation dataset against a predefined threshold.
    ◦ FL Algorithm: Federated Averaging (FedAvg), enhanced by UaaS-SFL.
    ◦ Detection Mechanism: Model Detector component within the FLMS uses anomaly detection techniques to identify malicious contributions by analyzing client updates, acting as middleware to invoke UaaS-SFL.
    ◦ Attack Simulated: Data Poisoning Attacks (e.g., backdoor triggers using pixel patterns).
    ◦ Neural Network Models (for experiments): LeNet-5 model for MNIST and Fashion-MNIST, and VGG-11 model for CIFAR-10.
    ◦ Comparison Methods: FedAvg, Retrain, FedEraser, and FedRecovery.
• Data, Platforms, and Tools:
    ◦ Datasets: MNIST, Fashion-MNIST, and CIFAR-10.
    ◦ Toolkit for Adversarial Threats Simulation: Adversarial Robustness Toolbox.
    ◦ Environment: Simulated IoT environment.
    ◦ Computing Resources: RACE (RMIT AWS Cloud Supercomputing Hub).
• Key Quantitative and Qualitative Findings and Conclusions:
    ◦ Effectiveness in Mitigating Poisoning Attacks: UaaS-SFL successfully detects and removes malicious client contributions, reducing backdoor accuracy to baseline levels comparable to retraining across all tested datasets. This effectiveness holds regardless of the stage of invocation in the FL lifecycle (early, midway, or largely trained model).
    ◦ Independence from Client Count: The service's effectiveness is independent of the number of clients or the number of malicious clients in the FL ecosystem.
    ◦ Model Accuracy Maintenance: Despite an initial temporary drop of ~5% in accuracy post-unlearning, the overall accuracy quickly recovers to baseline levels in subsequent rounds due to collaborative learning.
    ◦ Comparative Performance: UaaS-SFL consistently demonstrates superior performance over FedRecovery and FedEraser, closely matching the Retrain baseline, offering a more practical solution due to lower computational costs.
    ◦ Qualitative Conclusion: UaaS-SFL is a novel, robust, and scalable service that safeguards FL management systems in IoT networks against data poisoning attacks, ensuring model integrity and reliability post-compromise, thus providing a critical foundation for secure IoT applications.
• Authors and Affiliations:
    ◦ Wathsara Daluwatta: School of Computing Technologies, RMIT University, Melbourne, VIC, Australia.
    ◦ Ibrahim Khalil: School of Computing Technologies, RMIT University, Melbourne, Australia.
    ◦ Shehan Edirimannage: School of Computing Technologies, RMIT University, Melbourne, VIC, Australia.
    ◦ Mohammed Atiquzzaman: School of Computer Science, The University of Oklahoma, Norman, OK, USA.""",

    # Summary 4
    """Primary Research Objective and Core Topic: The primary research objective is to introduce a novel framework integrating a solar-powered High-Altitude Platform (HAP) with multiple Unmanned Aerial Vehicles (UAVs) equipped with Reconfigurable Intelligent Surfaces (RISs) to significantly enhance disaster response capabilities in 6G networks. The core topic is achieving low-latency and energy-efficient communication under compromised infrastructure by optimizing UAV energy management, RIS control, and ground device (GD) data offloading using a hybrid approach.
• Specific Methodologies and Techniques:
    ◦ Framework: An integrated system comprising a solar-powered HAP (Airship𝑏), UAVs mounted with RISs, and Ground Devices (GDs), for efficient edge computing in disaster scenarios. The multi-agent reinforcement learning part of the framework is implicitly named DIRECT.
    ◦ Overall Optimization Approach: A hybrid approach combining game theory and Multi-Agent Reinforcement Learning (MARL).
    ◦ Task Offloading Optimization: Utilizes a potential game framework to determine optimal task offloading decisions for GDs, minimizing energy consumption and latency. The Algorithm to Find Nash Equilibrium (Algorithm 1) is employed.
    ◦ UAV Movement and RIS Control Optimization: Employs a Multi-Agent Reinforcement Learning (MARL) strategy based on a Deep Reinforcement Learning (DRL) method, specifically Multi-Agent Deep Deterministic Policy Gradient (DDPG). The DIRECT Algorithm (Algorithm 2) describes this.
    ◦ Energy Management Innovation: A novel RIS ON/OFF mechanism allows UAVs to conserve energy by switching OFF RISs when not needed, enabling recharging and extending operational lifetimes.
    ◦ Energy Transfer: Wireless Energy Transfer (WET) from the HAP to UAVs and GDs.
    ◦ Communication Protocol: Non-Orthogonal Multiple Access (NOMA) is used for task offloading from GDs to HAP, with Successive Interference Cancellation (SIC) adopted at the HAP receiver.
    ◦ Channel Models: Simplified Rician fading models are used for communication between HAPs, UAVs, and GDs.
    ◦ Comparison Methods (DRL Variants): Single-agent Double DQN (S-DDQN) and Multi-agent DQN (MADQN).
    ◦ Comparison Methods (Baselines): Random and Greedy approaches.
• Data, Platforms, and Tools:
    ◦ Platforms: Simulations conducted using Python 3.8.10 and TensorFlow 2.8.0.
    ◦ Simulated Environment: A square disaster area of 10 km width (100 km²).
    ◦ Devices/Agents: 100 Ground Devices (GDs), 16 UAVs, and 1 High-Altitude Platform (HAP) (Airship𝑏) positioned at 10 km altitude.
    ◦ Neural Network Architecture: Actor and Critic networks with four fully connected hidden layers, using ReLU and Tanh activation functions.
• Key Quantitative and Qualitative Findings and Conclusions:
    ◦ Energy Efficiency: The RIS ON/OFF mechanism significantly enhances the energy efficiency and operational longevity of the UAV network. DIRECT consistently maintains higher residual energy levels for UAVs, leading to longer operational periods compared to MADQN and S-DDQN.
    ◦ Data Processing Performance & Offloading Rates: DIRECT achieves consistently higher offloading rates, indicating superior efficiency in managing and processing computational tasks from GDs.
    ◦ Network Reliability & Coverage: DIRECT maintains a consistently high density of active UAVs, ensuring robust coverage and support for GDs throughout the mission.
    ◦ Latency Minimization: The game theory-based approach for task offloading (GT-NE) results in the lowest average system cost and achieves a slower initial cost increase with varying CPU cycles and data sizes, effectively balancing local and remote processing. Overall, the integrated system reduces both energy consumption and latency, ensuring faster disaster recovery.
    ◦ Overall Superiority: Extensive simulations validate DIRECT's superior performance in energy efficiency, data processing, and overall network reliability compared to traditional methods and other DRL variants. DIRECT shows faster convergence and higher stability in reward accumulation.
    ◦ Qualitative Conclusion: The proposed framework offers a robust solution for energy-efficient, low-latency, and reliable communication in 6G disaster response scenarios, leveraging a novel integration of HAPs, RIS-equipped UAVs, game theory, and MARL.
• Authors and Affiliations:
    ◦ Jamal Alotaibi: Department of computer engineering, College of Computer, Qassim University, Buraydah, Saudi Arabia.
    ◦ Omar Sami Oubbati: LIGM, University Gustave Eiffel, Marne-la-Vallée, France.
    ◦ Mohammed Atiquzzaman: University of Oklahoma, Norman, OK, USA.
    ◦ Fares Alromithy: Electrical engineering department, University of Tabuk, Tabuk, Saudi Arabia.
    ◦ Mohammad Rashed Altimania: Electrical engineering department, University of Tabuk, Tabuk, Saudi Arabia.""",

    # Summary 5
    """Primary Research Objective and Core Topic: The primary research objective is to propose a novel cooperative framework integrating UAVs equipped with Reconfigurable Intelligent Surfaces (RIS) and Unmanned Ground Vehicles (UGVs) for real-time urban monitoring in 6G networks. The core topic is addressing the limitations of traditional urban monitoring methods, such as limited coverage, intermittent connectivity, and inefficient energy management, by leveraging AI-driven coordination, RIS-assisted communication, and real-time energy optimization.
• Specific Methodologies and Techniques:
    ◦ Framework: A novel UAV-UGV cooperative system integrated with RISs, referred to as ADVISE (implicitly, as it's the core RL framework).
    ◦ UAV Path Optimization and Recharging Schedules: Utilizes Deep Reinforcement Learning (DRL), specifically Proximal Policy Optimization (PPO) (also referred to as MAPPO for multi-agent PPO), to optimize UAV trajectory planning and recharging schedules. The ADVISE Algorithm (Algorithm 4) outlines this process.
    ◦ UGV Patrol Route Optimization: Employs a Genetic Algorithm (GA) to refine UGV patrol routes, ensuring adaptive and continuous surveillance. The Genetic Algorithm for UGV Movement Optimization (Algorithm 3) is provided.
    ◦ RIS Configuration Optimization: Incorporates Differential Evolution (DE) for RIS phase shift optimization, enhancing data transmission and mitigating urban signal degradation. The DE-based Optimization for RIS Phase-Shift Configuration (Algorithm 1) is used.
    ◦ Energy Management: Proposes a wireless UAV-based recharging system for UGVs via energy beamforming, reducing dependency on fixed charging stations. Includes AI-driven adaptive energy allocation.
    ◦ Communication Protocol: Non-Orthogonal Multiple Access (NOMA) for UGV data transmission to the central controller, with Successive Interference Cancellation (SIC) for interference mitigation.
    ◦ Risk Priority Evaluation: Implements a dynamic risk scoring method (Algorithm 2: Risk Priority Evaluation Algorithm) that continuously assesses zone priority based on real-time UV detection data and historical data.
    ◦ Channel Models: Uses Rician fading models for channel gain calculations.
    ◦ Comparison Methods (DRL Variants): Multi-Agent Deep Q-Network (MADQN) and Deep Deterministic Policy Gradient (DDPG).
    ◦ Comparison Methods (Baselines): Random and Greedy strategies.
• Data, Platforms, and Tools:
    ◦ Platforms: Simulations implemented in Python 3.8.10 with TensorFlow 2.8.0.
    ◦ Simulated Environment: A smart city with a 15 km width (225 km²).
    ◦ Devices/Agents: 20 UAVs and 50 UGVs.
    ◦ UAV Model Reference: EHang 184 (repurposed for urban monitoring).
    ◦ Equipment: Optical and Infrared Cameras for UV detection and data analysis.
    ◦ Neural Network Architecture: Actor and Critic networks with four fully connected layers, using ReLU and Tanh activation functions.
• Key Quantitative and Qualitative Findings and Conclusions:
    ◦ Communication Reliability: The framework significantly improves communication reliability by leveraging intelligent RIS beamforming to reduce signal degradation in dense urban environments. ADVISE consistently achieves superior data rates compared to other DRL methods.
    ◦ Monitoring Coverage: Achieves adaptive and continuous surveillance. The GA-based system, especially when integrated with UAVs, achieves the highest coverage of high-risk zones. ADVISE demonstrates superior high-risk zone coverage through real-time UAV trajectory optimization and dynamic RIS-assisted coordination.
    ◦ Energy Efficiency: Maximizes energy efficiency through its wireless UAV-based recharging system for UGVs and AI-driven path optimization. ADVISE maintains markedly higher residual energy for both UAVs and UGVs, supporting longer operational spans. The GA-based system with UAVs sustains higher UGV residual energy due to optimized task allocation and energy management.
    ◦ Latency Reduction: Ensures seamless data transmission and reduces latency.
    ◦ Overall Superiority: ADVISE consistently outperforms MADQN, DDPG, greedy, and random strategies across key metrics including accumulated rewards, data rates, and energy efficiency. It maintains a greater density of active UAVs and UGVs, ensuring continuous monitoring and efficient task execution.
    ◦ Qualitative Conclusion: ADVISE presents a scalable and robust solution for urban monitoring in 6G networks by effectively balancing energy, communication, and adaptability through intelligent coordination and specialized hardware integrations.
• Authors and Affiliations:
    ◦ Omar Sami Oubbati: LIGM, University Gustave Eiffel, Marne-la-Vallée, France.
    ◦ Jamal Alotaibi: Department of computer engineering, College of Computer, Qassim University, Buraydah, Saudi Arabia.
    ◦ Fares Alromithy: Electrical engineering department, University of Tabuk, Tabuk, Saudi Arabia.
    ◦ Mohammed Atiquzzaman: University of Oklahoma, Norman, OK, USA.
    ◦ Mohammad Rashed Altimania: Electrical engineering department, University of Tabuk, Tabuk, Saudi Arabia.""",
]

    example_json_for_prompt = """
{
  "Researcher Profile:": "Dr. Mohammad Atiquzzaman",
  "Affiliation:": "School of Computer Science, University of Oklahoma",
  "Research Domains": [
    "Federated Learning",
    "Machine Learning",
    "Network Security",
    "Digital Twin (DT) Technology",
    "Edge Computing",
    "Unmanned Aerial Vehicle (UAV) and Unmanned Ground Vehicle (UGV)",
    "Artificial Intelligence (AI)",
    "Data Privacy"
  ],
  "Techniques Used": [
    "Federated Unlearning which is the safe and secure removal of data from models without entire retraining",
    "Adversarial Machine Learning (AML) for noise generation",
    "Deep Neural Networks (DNN) (VGG16, Resnet18, GooLeNet, DenseNet, MobileNet, ResNeXt)",
    "Game Theory",
    "Multi-agent reinforcement learning (MARL)",
    "Deep reinforcement learning (DRL)",
    "Genetic Algorithm (GA) an algorithm for optimization inspired by natural selection",
    "Differential Evolution (DA) a population-based optimization algorithm"
  ],
  "Data & Platforms": [
    "Public Datasets: MNIST, Fashion-MNIST, CIFAR-10, Pins Face Recognition (https://www.kaggle.com/datasets/hereisburak/pins-face-recognition)",
    "Models: AlexNet Model",
    "Platforms: Python, PyTorch, AWS Cloud Computing, TensorFlow, Multi-Agent Proximal Policy Optimization (MAPPO) Framework"
  ],
  "Application Areas": [
    "Machine Learning Security",
    "Security in 6G Networks, Vehicular Security",
    "Smart City Surveillance",
    "IoT Network",
    "Healthcare",
    "Disaster Response"
  ],
  "Key Research Thinking Patterns": [
    "Adversarial Modeling: Actively considers and develops methods to anticipate counter intelligent, malicious actions or data manipulations within systems (e.g., Data poisoning attacks in Federated Learning systems, Data privacy issues in facial recognition AI).",
    "Comparative Evaluation: Assesses and contrasts various methods and systems to identify performance differences and trade-offs in metrics (e.g., Comparing Digital Twin Federated Learning Systems to already established methods, Comparing Machine Learning Facial Recognition through different Machine Learning algorithms, Benchmarked model accuracy after data poisoning).",
    "Scalability Focus: Prioritizes creating solutions that are practical and efficient, being able to be implemented within larger demands or scopes (e.g., UAV and UGV architecture efficiency expanding to smart city, Unlearning as a Service being tested for larger scales).",
    "AI/ML Utilization: Incorporates and develops artificial intelligence and machine learning systems within larger structures and scopes (e.g., developing Federated Learning for management systems, enabling AI-driven coordination for UAVs and UGVs)."
  ],
  "Summary Description": "Dr. Mohammad Atiquzzaman specializes in next-generation networking and intelligent systems. His work spans Federated Learning, network security in fields like 6G and vehicular aspects, and other applications of AI in autonomous systems like UAVs and UGVs. Notably, he introduces techniques like Unlearning as a Service for secure data removal from AI models, as well as providing more efficient solutions to UAVs and UGVs in the aspect of a smart city. Their overall contributions reflect a strong sense of comparative evaluation between current methods and their proposed solutions while also designing for larger scalability."
}
"""

    author_name = author_dir.name
    save_path = out_dir / f"{author_name.replace(' ', '_')}_profile.json"

    # if existing profile exists, skip processing
    if save_path.exists():
        logger.warning(f"Profile for {author_name} already exists at {save_path}. Skipping...")
        return

    # Step 0: Gather all PDF file paths for the author
    uploaded_files = list(author_dir.glob("*.pdf"))
    if not uploaded_files:
        logger.warning(f"No PDF files found for author {author_name} in directory {author_dir}. Skipping...")
        return

    # Step 1: Extract text from all PDFs (in parallel), using asyncio.to_thread to prevent blocking
    text_extraction_tasks = []
    for fpath in enumerate(uploaded_files):
        logger.info(f"Extracting text from file {fpath}...")
        text_extraction_tasks.append(asyncio.to_thread(extract_text_from_pdf, fpath))
    
    extracted_texts = await asyncio.gather(*text_extraction_tasks)

    # Step 2: Summarize papers in parallel
    logger.info("AI is summarizing publications...")
    summarization_tasks = []
    for text in enumerate(extracted_texts):
        file_name = uploaded_files[i]
        summarization_tasks.append(summarize_single_paper(text, file_name, model_name, semaphore))
    
    detailed_summaries = await asyncio.gather(*summarization_tasks)
    valid_summaries = [s for s in detailed_summaries if s]

    # Step 3: Synthesize the final profile
    if valid_summaries:
        logger.info("AI is synthesizing the final profile...")
        final_profile = await synthesize_profile(
            researcher_name,
            valid_summaries,
            example_summaries_for_prompt,
            example_json_for_prompt,
            model_name=model_name
        )

        if final_profile:
            with open(save_path, 'w') as f:
                json.dump(final_profile, f, indent=4)
            logger.info(f"Successfully saved profile for {researcher_name} at {save_path}.")
    else
        logger.error(label="Failed to summarize publications.", state="error")


async def main(opts) -> none:
    """ Main function to process the researcher profile extraction. """
    base_dir = Path(opts.pdf_directory)
    out_dir = Path(opts.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"The specified PDF directory does not exist or is not a directory: {base_dir}")
        return
    
    author_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(author_dirs)} author directories to process.")

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)  # Limit concurrent LLM calls to 5

    # Here we process each author directory, and within each author, process their PDFs in parallel
    for author in author_dirs:
        await generate_profile_for_one_author(author, out_dir, opts.model_name, semaphore)


def parseOpts(argv):
    parser = argparse.ArgumentParser(description="Researcher Profile Extractor")
    parser.add_argument('-p', '--pdf_directory', type=str, required=True, help='Directory containing researcher PDF files')
    parser.add_argument('-o', '--output', type=str, default='.', help='Output directory for the researcher profile JSON')
    parser.add_argument('-m', '--model_name', type=str, default='gpt-4o', help='OpenAI model name to use')
    opts = parser.parse_args(argv)
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv[1:])
    try:
        asyncio.run(main(opts))
    except Exception as e:
        logger.error("An error occurred during execution:", e)
