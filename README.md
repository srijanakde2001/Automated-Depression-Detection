# Automated-Depression-Detection
This project aims to develop a novel automated depression detection system with a rule based approach with various degrees of flexibility allowed to the clinicians for use.
We start by using scid-5-cv as the baseline structure to detect depression and then we further move on to include rules as defined by the clinicians themselves.

This repository contains the following files:
1. scid_baseline.cpp - Baseline depression detection tool based on the scid-5-cv questionnaire.
2. scid_generalisation_1.0.cpp - Various degrees of flexibility added to the baseline model while keeping scid-5-cv as the default.
3. scid-generalization_doc.docx - Detailed documentation of various usage details for scid_generalisation.
4. Other project documents - This folder contains all other relevant documents for the project.
5. scid_generalisation_1.0.cpp - BTP project - Visual Studio Code 2023-04-10 07-53-34 - This is a demo video of the working code with examples on how to use the scid_generalisation tool and with some sample outputs based on user inputted rules.

DDBERT has code for BERT based text analysis of DAIC dataset for depression detection.
![oldbs](https://github.com/srijanakde2001/Automated-Depression-Detection/assets/54339186/871725ce-4651-4797-890a-ba81a742b380)

The aim was to create a multi-modal depression detection model that incorporates all transcript, audio and video data of patients that lead to a PHQ8-binary depression score prediction. The following model pipeline was created.

Note: The project was done as part of my CSE BTech+MTech degree project at IIT Kharagpur under the guidance of Professor Partha Pratim Chakraborty, Professor Aritra Hazra and Professor Rajlakshmi Guha.
