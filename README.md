# IBM Watson Studio Recommendation System

## Project Overview
This project analyzes user interactions with articles on the IBM Watson Studio platform to build a robust, multi-faceted recommendation engine. The system seamlessly transitions between different recommendation techniques to surface relevant articles based on user history, peer interactions, and textual content. 

The recommendation engine is built using the following methodologies:
* **Exploratory Data Analysis (EDA):** Profiling user-article interaction distributions, calculating engagement metrics, and identifying platform usage trends.
* **Rank-Based Recommendations:** Addressing the cold-start problem for new users by retrieving the most globally interacted-with articles.
* **User-User Collaborative Filtering:** Recommending articles based on the interaction history of similar users. Similarity is computed via the dot product of a binary user-item matrix.
* **Content-Based Recommendations:** Utilizing Natural Language Processing (TF-IDF) and KMeans clustering on article titles to extract latent features and recommend textually similar content.
* **Matrix Factorization:** Employing Singular Value Decomposition (SVD) on the user-item interaction matrix to predict future interactions and determine the optimal number of latent features to prevent overfitting.

## File Descriptions
* `Recommendations_with_IBM.ipynb`: The primary Jupyter Notebook containing the end-to-end implementation of the recommendation algorithms, data processing, and evaluation code.
* `Recommendations_with_IBM.html`: An exported HTML version of the notebook for easy browser viewing.
* `data/user-item-interactions.csv`: Dataset containing historical user interaction logs (mapped user IDs, article IDs, and titles).
* `data/articles_community.csv`: Dataset containing the complete article descriptions and document bodies.

## Prerequisites
To run the notebook locally, the following libraries are required:
* Python 3.x
* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`

## Results & Deployment Strategy
The SVD matrix factorization model achieved high training accuracy using approximately 200 latent features. However, because the user-item interaction matrix is highly sparse (dominated by non-interactions), offline accuracy metrics can be misleading. 

To evaluate the live effectiveness of this recommendation engine, an online A/B test is required before full deployment:
* **Control Group:** Receives standard rank-based ("New & Popular") recommendations.
* **Experimental Group:** Receives personalized recommendations driven by the Collaborative Filtering and SVD models.
* **Evaluation Metrics:** Success is defined by a statistically significant increase in the Click-Through Rate (CTR) on recommended articles and a higher average number of articles read per session.

## Acknowledgements
* **IBM Watson Studio** for providing the real-world user interaction and article datasets.
* **Udacity** for providing the project framework and automated testing suite.
