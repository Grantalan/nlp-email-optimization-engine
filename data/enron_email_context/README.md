# Enron Email-Reply Dataset (EER-2025)

## Overview

This dataset was created in 2023 as part of my **Master’s thesis** in Internet systems engineering at National University of Science and Technology POLITEHNICA Bucharest, with the goal of enabling research in **Automated Generation of Email Reply**.

It is derived from the Enron Email Corpus, originally released into the public domain by the U.S. Federal Energy Regulatory Commission and processed by the CALO Project (William W. Cohen, 2015). The version used was obtained from Kaggle (Will Cukierski, 2016) in CSV format containing 517,401 emails from approximately 150 individuals.

The data was parsed using Python’s email library to extract key fields (date, subject, sender, recipient, body), cleaned to remove nulls, duplicates, and non-conversational records, and filtered to retain only sender–recipient pairs with reciprocal exchanges. Emails were ordered chronologically and matched into reply pairs based on consistent subjects and participants, with further filtering for a maximum 3-day reply gap.

Subsequent cleaning removed quoted previous messages, HTML content, attachment markers, and extraneous characters. Additional processing standardized text (expanding contractions) and eliminated spacing-related duplicates. The final dataset contains 15,377 structured email–reply pairs, where each reply is paired with its preceding message as context, making it suitable for research in automatic email response generation, conversational modeling, and related natural language processing tasks.


- **Total Pairs**: 15377
- **Format**: CSV
- **Fields**:
    - `EmailSend` – text of the original email sent
    - `EmailReply` – text of the response email
    - `SubjectSend` - subject of the email sent
    - `SubjectReply` - subject of the response email
    - `From` - the person who sent the original email
    - `To` - the person who received the initial email and replied with another email
    - `DateSend` - date of the original email sent
    - `DateReply` - date of the response email
    - `Context` - the previous message of the conversation between the two people

- **Language**: English

## Source
This dataset is derived from the Enron Email Corpus, originally released into the public domain by the U.S. Federal Energy Regulatory Commission and processed by the CALO Project. The version used was obtained from Kaggle:
- **URL**: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
- **License**:  Public Domain — Original dataset released by FERC; no additional license terms stated on Kaggle.

## File Structure
- `data/EnronEmailReplyPairsWithContext.csv` – the main dataset file
- `LICENSE` – license for this dataset
- `original_sources.md` – information about the original dataset
- `citation.bib` – citation information for this dataset

## License
This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

## Citation
If you use this dataset, please cite:

@dataset{oana_ilie_2025_eer,
author       = {Oana-Mariana Ilie},
title        = {Enron Email-Reply Dataset (EER-2025)},
year         = {2025},
publisher    = {Zenodo},
doi          = {10.5281/zenodo.16853650},
url          = {}
}


Also cite the original Kaggle dataset as described in `original_sources.md`.