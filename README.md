# Movie Recommender System using Firefly Optimization Algorithm (FOA)

## Project Overview
This project implements a movie recommendation system by leveraging the Firefly Optimization Algorithm (FOA) to enhance the recommendation process. The following key components are included:

- **Search Method:** Firefly Algorithm
- **Dataset:** [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- **Similarity Measure:** Pearson Correlation Coefficient
- **Rating Prediction Method:** Basic K-Nearest Neighbors (KNN)

The system aims to provide accurate and relevant movie recommendations by optimizing the similarity calculations and recommendation accuracy using FOA.

## Features
- **Recommendation Techniques:**
  - Collaborative Filtering with KNN
  - Similarity computation using Pearson Correlation
- **Optimization Method:** Firefly Algorithm to improve recommendation quality
- **Evaluation Metrics:**
  - Precision
  - Accuracy
  - F-Measure (F1-Score)
  - Recall

## Prerequisites
Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- Required Python libraries:
  ```bash
  pip install pandas numpy seaborn surprise scikit-learn matplotlib
  ```

## Datasets
The project uses the [MovieLens Dataset](https://grouplens.org/datasets/movielens/), which consists of:

- **Ratings Dataset** (`rating.csv`): Contains user ratings for movies.
- **Movies Dataset** (`movie.csv`): Contains metadata about the movies.

These datasets are preprocessed and merged for analysis and recommendation.

## Getting Started
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/david11133/Movie_Recommender.git
   cd Movie_Recommender
   ```

2. **Prepare the Data:**
   - Place the `rating.csv` and `movie.csv` files in the `datasets` folder.

3. **Run the Notebook:**
   - Open the Jupyter Notebook file (`FOA_Recommendation_System.ipynb`) and execute the cells step by step.

## Implementation Details
### Libraries Used
The following Python libraries are utilized in this project:
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `seaborn`, `matplotlib`
- **Recommendation System:** `surprise`
- **Evaluation Metrics:** Custom implementation and scikit-learn

### Key Components
1. **Data Preprocessing:**
   - Load and merge the ratings and movies datasets.
   - Handle missing values and outliers (if any).

2. **Recommendation System:**
   - Calculate similarities between users/movies using Pearson Correlation.
   - Use KNN to predict ratings for movies.

3. **Optimization with FOA:**
   - Implement the Firefly Algorithm to optimize similarity computations.

4. **Evaluation:**
   - Evaluate the system's performance using metrics like Precision, Recall, and F-Measure.

## Results
The results of the recommendation system are evaluated based on the metrics mentioned above, showcasing the effectiveness of FOA in improving recommendation accuracy.

## Future Improvements
- Incorporating hybrid recommendation techniques.
- Exploring other optimization algorithms like Genetic Algorithm or Particle Swarm Optimization.
- Scaling the system to handle larger datasets.

## References
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Firefly Algorithm Overview](https://en.wikipedia.org/wiki/Firefly_algorithm)
- [Surprise Library Documentation](https://surprise.readthedocs.io/)

