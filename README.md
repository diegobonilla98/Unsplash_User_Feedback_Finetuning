# Unsplash_User_Feedback_Finetuning
Enhancing CLIP-Based Image Search with User Feedback and Advanced Tag Management

## Description
This project aims to improve a CLIP-based model for image retrieval by leveraging user feedback, advanced tag management, and fine-tuning techniques. The goal is to enhance search relevance and personalization for the Unsplash dataset, making the model more responsive to user preferences and diverse query contexts.


## File Description
- **download_data.py**: Script for downloading the dataset.
- **check_fix_tags.py**: Cleans user and web keywords by removing duplicates, improving tag relevance.
- **this_utils.py**: Utilities for searching in embeddings and displaying results.
- **clip_baseline.py**: Baseline script demonstrating CLIP's image retrieval capabilities. Results for three captions are stored in "results*" folders.
- **search_by_tag.py**: Explores tag-based search using BERT embeddings. Contains placeholders for discussing results.
- **how_similar_tags_user_web.py**: Analyzes the similarity between user and web keywords for images, indicating a high degree of overlap.
- **web_tag_to_user.py**: Experiment to translate web tag embeddings to user feedback tag space. The approach was not successful.
- **clip_finetuning.py**: Script for fine-tuning CLIP with dataset-specific captions.
- **clip_finetuned_top.py**: Outcomes of CLIP fine-tuning using raw keywords.
- **tag2text.py**: Utilizes the RAM model to convert images and keywords to captions more suited for CLIP.
- **clip_finetuned_top_captions.py**: Fine-tuning results using captions derived from user tags.

