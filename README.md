# CS439 Final Project: Predicting IMDB Movie Scores

## Research Question

Can movie metadata and financial information explain or predict IMDB movie scores, and what types of movies tend to perform best?

This project uses the provided `imdb_movies.csv` dataset to study how budget, revenue, profit, ROI, release year, genre, language, and country relate to movie score.

## Project Structure

- `data/imdb_movies.csv`: original dataset extracted from the zip file
- `src/final_project_analysis.py`: full project code
- `outputs/figures/`: generated EDA and PCA figures
- `outputs/model_results.csv`: regression/classification metrics
- `outputs/cluster_profiles.csv`: K-Means cluster summaries
- `outputs/cleaned_movies.csv`: cleaned feature table used for modeling

## How To Run

From this folder:

```bash
python3 -m pip install -r requirements.txt
python3 src/final_project_analysis.py
```

In this Codex workspace, the dependencies were installed locally into `.venv_pkgs`, so the script can also be run with:

```bash
PYTHONPATH=.venv_pkgs /Users/vedaparulekar/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/final_project_analysis.py
```

## Methods Included

1. **EDA and visualizations**
   - Histograms of score, budget, and revenue
   - Scatter plots of budget vs. score and revenue vs. score
   - Boxplots of score by genre, language, and country

2. **Linear regression baseline**
   - Predicts score using only budget and revenue

3. **Improved linear regression**
   - Adds engineered features: profit, ROI, release year, main genre, language, and country
   - Uses scaling and one-hot encoding

4. **Classification**
   - Converts scores into high-rated vs. not high-rated
   - Evaluates accuracy, precision, recall, and F1

5. **K-Means clustering + PCA**
   - Clusters movies using metadata and financial features
   - Visualizes clusters in two PCA dimensions
   - Summarizes which movie types distinguish each cluster

## Suggested Presentation Takeaway

Financial variables alone are usually not enough to explain movie score. The improved model tests whether metadata gives the model more signal, while clustering helps interpret different groups of movies, such as high-budget commercial films, lower-budget niche films, and high-return outliers.
