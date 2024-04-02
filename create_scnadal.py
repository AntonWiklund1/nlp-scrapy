import pandas as pd

# Define the data
data = {
    "headline": [
        "Major Oil Spill Devastates Arctic Ocean Wildlife",
        "Unprecedented Deforestation Rates Recorded in the Amazon",
        "Chemical Leak Poisons Springfield River, Endangering Local Communities",
        "Coastal Waters at Risk as Industrial Waste Dumping Surfaces"
    ],
    "link": [
        "https://news.example.com/environment/oil-spill-arctic-2024",
        "https://news.example.com/environment/amazon-deforestation-2024",
        "https://news.example.com/environment/chemical-leak-springfield-2024",
        "https://news.example.com/environment/industrial-waste-coast-2024"
    ],
    "date": [
        "2024-04-01",
        "2024-04-02",
        "2024-04-03",
        "2024-04-04"
    ],
    "body": [
        "In a devastating turn of events, the Arctic Ocean has become the site of a massive oil spill, reportedly originating from operations conducted by PetroGlobal. The spill, covering several square kilometers, poses a severe threat to marine life and the ocean's fragile ecosystem. PetroGlobal has yet to release a statement, but environmentalists are calling this one of the worst ecological disasters of the decade.",
        "The Amazon Rainforest, often referred to as the Earth's lungs, is now facing an unprecedented rate of deforestation. Satellite images confirm that AgriCorp Industries is behind the clearing of thousands of hectares of forest land to make way for agricultural expansion. This deforestation not only disrupts biodiversity but also contributes significantly to climate change. Activists demand immediate action to halt AgriCorp's activities.",
        "A catastrophic chemical leak from the ChemiTech Factory has poisoned the Springfield River, a vital water source for surrounding communities and ecosystems. Early reports indicate that the leak includes hazardous substances, which have already led to massive fish die-offs and pose serious health risks to the local population. The environmental agency is investigating the incident, while ChemiTech faces widespread backlash over its safety practices.",
        "Investigations have uncovered that HeavyMetal Industries has been dumping industrial waste into coastal waters off the city's port, violating numerous environmental regulations. The waste includes heavy metals and toxic chemicals, threatening marine life and the health of beachgoers. Authorities have launched a full-scale investigation into HeavyMetal's practices, as environmental groups call for accountability and immediate cleanup efforts."
    ]
}

# Create the DataFrame
df_fixed = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path_fixed = "./data/example_scandals_fixed.csv"
df_fixed.to_csv(csv_file_path_fixed, index=False)