# Auto-Generated Project Summary

Cleaned dataset size: 9,967 released movies.
Score range: 10.0 to 100.0; mean score: 64.84.

## Top Genres by Average Score
```text
             movie_count  avg_score
main_genre                         
Music                 75      70.60
Animation            886      69.87
Western               72      69.06
Documentary          182      68.84
War                   77      68.74
Drama               1822      67.15
Crime                366      67.10
Family               334      66.28
```

## Model Results
```text
                                         model      rmse      mae       r2  accuracy  precision   recall       f1  positive_rate
                    Baseline linear regression 10.121755 7.547621 0.039316       NaN        NaN      NaN      NaN            NaN
                    Improved linear regression  9.198042 6.723644 0.206659       NaN        NaN      NaN      NaN            NaN
Logistic regression classification score >= 75       NaN      NaN      NaN  0.695587   0.280335 0.688356 0.398414       0.146483
```

## Cluster Profiles
```text
 cluster  movie_count  avg_score  median_budget  median_revenue  median_profit  median_roi  median_year top_genre top_language top_country
       0         4466     67.225     25000000.0      60000000.0     31324119.0       1.448       2006.0     Drama      English          AU
       1         3922     64.575    109000000.0     498161445.6    387722299.9       3.719       2018.0     Drama      English          US
       2         1385     57.487      7000000.0       2478806.0     -2787062.0      -0.636       2015.0     Drama      English          US
       3          194     67.861       450000.0     100043590.5     99549647.5     127.860       2008.5     Drama      English          AU
```

## Interpretation Starter
Use the baseline model to discuss whether budget and revenue alone predict score. Use the improved regression and classification results to evaluate whether metadata adds predictive value. Use the cluster profiles to describe the types of movies that perform best, such as genres or financial profiles associated with higher average scores.