import streamlit as st
import plotly.graph_objects as go
import json
import asyncio
from io import BytesIO

# --- Import functions from your backend files ---
# Assumes 'profile_extraction.py' and 'profile_matcher.py' are in the same folder.
from profile_extraction import extract_text_from_pdf, summarize_single_paper, create_researcher_profile
from profile_matcher import analyze_collaboration_synergy

# --- Page Configuration ---
st.set_page_config(
    page_title="SynergyScout - Collaboration Discovery",
    page_icon="🔬",
    layout="wide"
)

# ==============================================================================
# --- Backend Orchestration Wrapper ---
# ==============================================================================

async def generate_profile_wrapper(researcher_name, uploaded_files, status_container):
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

    # Step 1: Extract text from all PDFs (in parallel)
    text_extraction_tasks = []
    for i, file in enumerate(uploaded_files):
        status_container.write(f"Extracting text from file {i+1}/{len(uploaded_files)}: {file.name}...")
        text_extraction_tasks.append(asyncio.to_thread(extract_text_from_pdf, BytesIO(file.getvalue()), file.name))
    
    extracted_texts = await asyncio.gather(*text_extraction_tasks)

    # Step 2: Summarize papers in parallel
    status_container.write("AI is summarizing publications...")
    summarization_tasks = []
    for i, text in enumerate(extracted_texts):
        if text:
            file_name = uploaded_files[i].name
            summarization_tasks.append(summarize_single_paper(text, file_name))
    
    detailed_summaries = await asyncio.gather(*summarization_tasks)
    valid_summaries = [s for s in detailed_summaries if s]

    if not valid_summaries:
        status_container.update(label="Failed to summarize publications.", state="error"); return None

    # Step 3: Synthesize the final profile
    status_container.write("AI is synthesizing the final profile...")
    final_profile = await create_researcher_profile(
        researcher_name,
        valid_summaries,
        example_summaries_for_prompt,
        example_json_for_prompt
    )
    return final_profile

# ==============================================================================
# --- UI HELPER FUNCTIONS & STYLES ---
# ==============================================================================

def create_gauge_chart(score):
    score_on_10 = score / 10.0
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score_on_10,
        number = {'suffix': "/10", 'font': {'size': 20}},
        title = {'text': "Synergy Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 10]},
            'bar': {'color': "#007AFF"},
            'steps': [
                {'range': [0, 4], 'color': "#FFD2D2"},
                {'range': [4, 7], 'color': "#FFF3C4"},
                {'range': [7, 10], 'color': "#D4EDDA"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==============================================================================
# --- UI RENDERING LOGIC ---
# ==============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'results' not in st.session_state:
    st.session_state.results = None

def render_input_page():
    st.title("SynergyScout: Discover Research Collaboration Opportunities")
    st.markdown("Provide profiles for two researchers using either existing **JSON files** or their recent **publications (PDFs)**.")
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")

    # --- INPUTS FOR RESEARCHER A (TOGGLE OUTSIDE FORM) ---
    with col1:
        st.subheader("👤 Researcher A")
        # The toggle is now outside the form, allowing for immediate script reruns
        st.toggle('Switch to Upload JSON Profile', key='use_json_a')

    # --- INPUTS FOR RESEARCHER B (TOGGLE OUTSIDE FORM) ---
    with col2:
        st.subheader("👤 Researcher B")
        st.toggle('Switch to Upload JSON Profile', key='use_json_b')

    # --- THE FORM CONTAINS ONLY THE INPUT WIDGETS AND SUBMIT BUTTON ---
    with st.form(key='input_form'):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            # Dynamically show inputs based on the toggle's state
            if st.session_state.use_json_a:
                st.file_uploader("Upload Researcher A's JSON Profile", type=['json'], key='json_a', label_visibility="collapsed")
            else:
                st.text_input("Researcher A's Name", key='name_a', placeholder="e.g., Dr. Evelyn Reed")
                st.file_uploader("Upload Researcher A's Publications (Max 5)", type=['pdf'], accept_multiple_files=True, key='pdfs_a')

        with col2:
            # Dynamically show inputs based on the toggle's state
            if st.session_state.use_json_b:
                st.file_uploader("Upload Researcher B's JSON Profile", type=['json'], key='json_b', label_visibility="collapsed")
            else:
                st.text_input("Researcher B's Name", key='name_b', placeholder="e.g., Dr. Kenji Tanaka")
                st.file_uploader("Upload Researcher B's Publications (Max 5)", type=['pdf'], accept_multiple_files=True, key='pdfs_b')

        st.markdown("")
        submitted = st.form_submit_button("🚀 Analyze Potential", use_container_width=True)

    if submitted:
        async def process_submission():
            profile_a, profile_b = None, None
            
            # Process Researcher A based on the toggle's session state
            if st.session_state.use_json_a:
                if st.session_state.json_a: profile_a = json.load(st.session_state.json_a)
                else: st.error("Please upload a JSON profile for Researcher A."); return
            else:
                if st.session_state.name_a and st.session_state.pdfs_a:
                    if len(st.session_state.pdfs_a) > 5: st.error("Max 5 publications for Researcher A."); return
                    with st.status(f"Generating profile for {st.session_state.name_a}...", expanded=True) as status:
                        profile_a = await generate_profile_wrapper(st.session_state.name_a, st.session_state.pdfs_a, status)
                        if profile_a: status.update(label="Profile for Researcher A is complete!", state="complete")
                        else: status.update(label="Profile generation failed for A!", state="error"); return
                else: st.error("Please provide a name and publications for Researcher A."); return

            # Process Researcher B based on the toggle's session state
            if st.session_state.use_json_b:
                if st.session_state.json_b: profile_b = json.load(st.session_state.json_b)
                else: st.error("Please upload a JSON profile for Researcher B."); return
            else:
                if st.session_state.name_b and st.session_state.pdfs_b:
                    if len(st.session_state.pdfs_b) > 5: st.error("Max 5 publications for Researcher B."); return
                    with st.status(f"Generating profile for {st.session_state.name_b}...", expanded=True) as status:
                        profile_b = await generate_profile_wrapper(st.session_state.name_b, st.session_state.pdfs_b, status)
                        if profile_b: status.update(label="Profile for Researcher B is complete!", state="complete")
                        else: status.update(label="Profile generation failed for B!", state="error"); return
                else: st.error("Please provide a name and publications for Researcher B."); return

            # Run Final Analysis
            if profile_a and profile_b:
                with st.status("Running final synergy analysis...", expanded=True) as status:
                    st.write("AI is comparing the two profiles...")
                    st.session_state.results = await analyze_collaboration_synergy(profile_a, profile_b)
                    if st.session_state.results:
                        status.update(label="Synergy analysis complete!", state="complete")
                        await asyncio.sleep(1)
                        st.session_state.page = 'report'
                        st.rerun()
                    else:
                        status.update(label="Analysis failed!", state="error")

        asyncio.run(process_submission())

def render_synergy_report():
    results = st.session_state.results
    
    # --- Safely extract data from the new JSON structure using .get() ---
    synergy_data = results.get('detailed_analysis_of_synergies', {})
    score_data = results.get('collaboration_synergy_score', {})
    
    profile_a = results.get('profile_a', {})
    profile_b = results.get('profile_b', {})
    
    # Get researcher names for dynamic labels
    name_a = profile_a.get('Researcher Profile:', 'Researcher A')
    name_b = profile_b.get('Researcher Profile:', 'Researcher B')
    
    st.title("Collaboration Synergy Report")
    if st.button("⬅️ Start New Analysis"):
        st.session_state.page = 'input'
        st.session_state.results = None
        st.rerun()
    st.markdown("---")

    # --- TOP ROW: Profiles and Score/Summary ---
    col1, col2, col3 = st.columns([1.2, 1.2, 1.2], gap="large")

    def display_profile_header(profile):
        st.subheader(f"👤 {profile.get('Researcher Profile:', 'N/A')}")
        st.caption(f"🏢 {profile.get('Affiliation:', 'N/A')}")
        key_map = {
            'Research Domains': 'Research Domains', 'Techniques Used': 'Methods',
            'Data & Platforms': 'Data & Platforms', 'Application Areas': 'Application Areas'
        }
        for json_key, title in key_map.items():
            st.markdown(f"**{title}:**")
            items = profile.get(json_key, [])
            if isinstance(items, list):
                for item in items: st.markdown(f"- {item}")
            elif isinstance(items, str):
                st.markdown(f"- {items}")

    with col1:
        with st.container(border=True): display_profile_header(profile_a)
    
    with col2:
        score = score_data.get('score', 0)
        st.plotly_chart(create_gauge_chart(score), use_container_width=True)
        with st.container(border=True):
            st.subheader("Executive Summary")
            st.markdown(results.get('executive_summary', 'No summary provided.'))

    with col3:
        with st.container(border=True): display_profile_header(profile_b)

    # --- BOTTOM SECTION: Detailed Analysis ---
    st.markdown("---")
    st.header("Detailed Analysis of Synergies")
    
    # Section 1: Research Interests
    interests = synergy_data.get('overlapping_and_complementary_research_interests', {})
    st.subheader("🤝 Overlapping & Complementary Research Interests")
    st.markdown("**Shared Goals:**")
    for goal in interests.get('shared_goals', []):
        st.markdown(f"- {goal}")
    st.markdown("**Complementary Areas:**")
    for area in interests.get('complementary_areas', []):
        st.markdown(f"- {area}")
    st.markdown("") # Spacer

    # Section 2: Methodological Synergy
    methods = synergy_data.get('methodological_and_technical_synergy', {})
    st.subheader("💡 Methodological & Technical Synergy")
    st.markdown(f"**How {name_a} can help {name_b}:** {methods.get('a_to_b', 'N/A')}")
    st.markdown(f"**How {name_b} can help {name_a}:** {methods.get('b_to_a', 'N/A')}")
    st.markdown("")

    # Section 3: Data & Resource Sharing
    resources = synergy_data.get('data_resources_and_model_sharing', {})
    st.subheader("📦 Data, Resources, & Model Sharing")
    st.markdown(f"**What {name_a} can provide to {name_b}:** {resources.get('a_to_b', 'N/A')}")
    st.markdown(f"**What {name_b} can provide to {name_a}:** {resources.get('b_to_a', 'N/A')}")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Start New Analysis", use_container_width=True):
            st.session_state.analysis_complete = False
            st.rerun()
    with col2:
        # Placeholder for Export functionality
        st.button("Export as PDF", use_container_width=True, type="primary")


# --- MAIN LOGIC to switch between pages ---
if st.session_state.page == 'report':
    render_synergy_report()
else:
    render_input_page()