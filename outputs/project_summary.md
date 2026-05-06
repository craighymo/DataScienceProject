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
                                                      model      rmse      mae        r2  accuracy  balanced_accuracy  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  low_rate  medium_rate  high_rate
                                        Mean score baseline 10.329464 7.717587 -0.000517       NaN                NaN              NaN           NaN       NaN                 NaN              NaN          NaN       NaN          NaN        NaN
                                 Baseline linear regression 10.121755 7.547621  0.039316       NaN                NaN              NaN           NaN       NaN                 NaN              NaN          NaN       NaN          NaN        NaN
                                 Improved linear regression  9.307344 6.762579  0.187693       NaN                NaN              NaN           NaN       NaN                 NaN              NaN          NaN       NaN          NaN        NaN
                    Majority class baseline: score category       NaN      NaN       NaN  0.367603           0.333333         0.122534      0.333333  0.179196            0.135132         0.367603     0.197619  0.367713     0.301696   0.330591
Logistic regression classification: score quantile category       NaN      NaN       NaN  0.503009           0.491641         0.481211      0.491641  0.480860            0.486530         0.503009     0.489500  0.367713     0.301696   0.330591
```

## Cluster Profiles
```text
 cluster  movie_count  avg_score  median_budget  median_revenue  median_profit  median_roi  median_year top_genre top_language
       0         2310     66.953     88450000.0     329856760.8    222875665.9       2.686       2012.0 Animation      English
       1         3575     62.738     27000000.0      61416888.0     38493064.0       1.787       2013.0    Action      English
       2          279     69.814     38000000.0      81600000.0     40100000.0       1.804       2009.0     Drama      English
       3         3803     65.171     51270000.0     180591246.6    125726374.4       2.898       2013.0     Drama      English
```

## Interpretation Starter
Use the mean score baseline to show what performance looks like without using movie features. Use the baseline linear regression to discuss whether budget and revenue alone predict score. Use the improved regression to evaluate whether metadata adds predictive value. Use the majority-class baseline and logistic regression classifier to evaluate score-category classification. Use the cluster profiles to describe the types of movies that appear together based on financial and metadata features, then compare their average scores.