# Unsplash_User_Feedback_Finetuning
Enhancing CLIP-Based Image Search with User Feedback and Advanced Tag Management

## Description
This project aims to improve a CLIP-based model for image retrieval by leveraging user feedback, advanced tag management, and fine-tuning techniques. The goal is to enhance search relevance and personalization for the Unsplash dataset, making the model more responsive to user preferences and diverse query contexts.


## File Description
- **download_data.py**: Script for downloading the dataset.
- **merge_dataset.py**: Merges the different dataframes in the dataset using the "photo_id" value and drops rows with empty columns.
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


## Results
The project embarked on an exploratory journey to enhance image search accuracy and user experience using the CLIP model, integrated with user feedback. This endeavor unfolded through a series of methodical experiments, each contributing unique insights into the challenges and potentials of our approach.

### Initial Baseline with CLIP
The project's first milestone was establishing a CLIP baseline. This foundational step proved to be a success, with CLIP demonstrating robust capabilities in retrieving relevant images. This encouraging start validated CLIP's potential as a core component of our solution, setting a high bar for subsequent enhancements.

### Tag Embeddings Search
Venturing into tag-based searches, we anticipated a direct method to leverage the semantic richness of tags. However, this approach did not yield the expected improvements. The disconnect between tag embeddings and actual user search intent highlighted the complexity of capturing nuanced user expectations, prompting a reevaluation of our strategies.

### Web vs. User Keyword Analysis
A comparative analysis of web and user keywords revealed surprising similarities, suggesting a potential for a translation model to bridge these spaces. Despite the initial promise, this idea was eventually discarded. The attempt to train a model for converting tag embeddings from web to user spaces encountered insurmountable obstacles, primarily due to the inherent non-alignment of these semantic spaces.

### Fine-tuning CLIP with Raw Tags
Our experiment with fine-tuning CLIP using raw tags as captions encountered significant challenges. The approach, which seemed plausible in theory, faltered in practice. CLIP's architecture and training methodology were not conducive to handling raw tags effectively, leading to a one-to-many problem where an image could relate to multiple tags across different training instances, diluting the model's focus and effectiveness.

### User Feedback and CLIP-Friendly Captions
Incorporating user feedback, we ventured to create more CLIP-friendly captions. This phase involved transforming user keywords into narratives that CLIP could process more naturally. The subsequent fine-tuning phase aimed to align the model more closely with user expectations. Despite these efforts, the results did not surpass the baseline CLIP performance, underscoring the challenge of enhancing CLIP's capabilities within the constraints of our approach.

## Concluding Insights
The results of the experiments brought us to a crucial realization: none of the models significantly improved upon the original CLIP results. This outcome, while initially disheartening, provided valuable lessons on the limitations and challenges of adapting CLIP to our specific use case. It highlighted the importance of aligning model training and fine-tuning strategies more closely with the nuanced needs of image search applications and user expectations.

## Future Work
Future directions aim to address scalability, efficiency, and personalization:

- **Incorporate region and language** information to enhance personalization and relevance, potentially through detecting keyword languages or considering user-image geographical proximities.
- **Explore KISS** (Keep It Simple, Stupid) principles with a tag-match approach, potentially merging this with multimodal learning for more robust search capabilities.
- **Implement reranking** methods based on human feedback, akin to Reinforcement Learning from Human Feedback (RLHF), to further refine search results based on actual user preferences.
- **Conduct A/B testing** to gather comparative feedback on different models, guiding iterative improvements.
- **Fine-tune an image-tagging model** with user queries to better capture the nuances of user search intent.
- **Address the challenge of running powerful models** on large datasets by employing Knowledge Distillation and Quantization to improve efficiency without significant loss in performance.
- **Develop a model to assess image quality**, incorporating not just technical attributes like resolution but also aesthetic aspects, and integrate this into the search system.
- **Tackle the one-to-many and many-to-one challenges** inherent in CLIP and tagging by implementing diverse tags for images and mapping between user query keywords and image keywords in the embedding space.
- **Improve the fine-tuning of the CLIP model** by optimizing hyperparameters, ensuring correct data splits, and employing transfer learning for better generalization.

## Run it Yourself
1. Install the libraries (python = 3.9 and GPU):
```bash
pip install -r requirements.txt
```
2. Run the model training:
```bash
python clip_finetuning.py
```
4. Run the model using a text query (the first time will create the image embeddings):
```bash
python clip_finetuned_top_captions.py
```
