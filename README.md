# From-Data-to-Design-Leveraging-AI-for-novel-Lipid-Nanoparticles-for-mRNA-delivery
A Python-based platform for data-driven discovery of ionizable lipids in lipid nanoparticle (LNP) formulations

# Abstract
Designing novel molecules is a complex task traditionally performed by experts in the field. One such challenge is the design of new ionizable lipids for lipid nanoparticles (LNPs) to deliver mRNA therapeutics. Machine learning offers a promising approach to assist chemists in prioritizing which molecules to synthesize, thereby reducing the time- and resource-consuming process of generating and testing large libraries of candidates. Here, we developed an explainable XGBoost classifier to predict whether ionizable lipids will be effective for mRNA delivery. By incorporating feature-importance analysis and SHAP values, our model provides insight into the molecular properties most critical for performance, guiding rational design choices. We evaluated two strategies: (1) screening expert-designed lipids and (2) generating candidates via the STONED algorithm (Aspuru-Guzik et al.) followed by ML evaluation. In both HeLa and lung epithelial cell assays, our model achieved high predictive accuracy and outperformed the FDA-approved lipid MC3. These results demonstrate that ML-assisted design of ionizable lipids can accelerate innovation in RNA therapeutic development by enabling rapid, virtual screening of promising candidates.

# Installation guide
Before starting, ensure you have Anaconda and Git installed on your system.

1. **Clone the repository**  
````
git clone https://github.com/Fabian-Link/From-Data-to-Design-Leveraging-AI-for-novel-Lipid-Nanoparticles-for-mRNA-delivery/tree/main 
````

2. **Navigate to the cloned directory**
````
cd <directory path>
````

3. **Create a virtual enviroment**
````
conda create --name <enviroment name> python=3.10
````

4. **Activate the virtual enviroment**
````
conda activate <enviroment name>
````

5. **Install dependencys**
````
pip install -r requirements.txt
````

# Usage