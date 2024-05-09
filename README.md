# NLP-Scrapy: News Intelligence Platform
This repository contains the NLP-Scrapy project, an NLP-enriched News Intelligence platform designed for scraping, analyzing, and classifying news articles. The platform identifies entities, classifies news topics, analyzes sentiments, and detects potential scandals related to environmental issues.

# Getting Started

## Prerequisites
Before you start, ensure you have Python installed on your system. You will also need git for cloning the repository and pip for installing Python packages.


## Installation

1. Clone the Repository:

```` zsh
    git clone https://github.com/AntonWiklund1/nlp-scrapy
    cd nlp-scrapy
````

2. Install Required Python Packages:

Install the necessary Python packages by running:

```` zsh
    pip install -r requirements.txt
````

3. Download SpaCy Language Model:

This project uses the en_core_web_lg model for processing English text.

```` zsh
    python -m spacy download en_core_web_lg
````

## Usage

1. Scrape News Data:

Execute the script to scrape news articles from predefined news sources:

```` zsh
    python scraper_news.py
````

2. Run NLP Analysis:

To analyze the scraped news, run:

```` zsh
    python nlp_enriched_news.py
````

## Training the Topic Classifier

If you prefer to train the topic classifier from scratch:

1. Prepare Training Data:

To use a large dataset, combine the provided training parts:

````
    python make_large_traindataset.py
````

2. Train the Model:
Change the file path to the correct dataset also

Execute the training script to start the training process:
````
    python training_model.py
````


The learning curve will be saved in results/learning_curve.png to validate that the model is trained effectively without overfitting.

## Scandal Detection

### Keywords

To detect environmental scandals, the following categories and associated keywords are used:

* Pollution: Air pollution, Water contamination, Toxic waste dumping, Industrial spill, Chemical leak
* Deforestation: Illegal logging, Clear-cutting, Rainforest destruction, Habitat destruction
* Wildlife Impact: Wildlife endangerment, Poaching, Biodiversity loss
* Resource Exploitation: Overfishing, Unsustainable mining, Oil extraction, Natural gas flaring
* Climate Change Impact: Carbon emissions, Greenhouse gases, Methane release
* Land and Soil Degradation: Soil contamination, Desertification, Landfill overflow

Each category is defined with specific keywords to ensure precision in detecting relevant articles while minimizing false positives.


