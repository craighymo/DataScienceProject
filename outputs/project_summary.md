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
                                 Improved linear regression  9.219698 6.777818  0.202919       NaN                NaN              NaN           NaN       NaN                 NaN              NaN          NaN       NaN          NaN        NaN
                    Majority class baseline: score category       NaN      NaN       NaN  0.367603           0.333333         0.122534      0.333333  0.179196            0.135132         0.367603     0.197619  0.367713     0.301696   0.330591
Logistic regression classification: score quantile category       NaN      NaN       NaN  0.516048           0.510149         0.506644      0.510149  0.507964            0.512155         0.516048     0.513684  0.367713     0.301696   0.330591
```

## Cluster Profiles
```text
 cluster  movie_count  avg_score  median_budget  median_revenue  median_profit  median_roi  median_year top_genre top_language top_country
       0         4016     63.930     20000000.0      28565459.5      5270200.5       0.367       2013.0     Drama      English          AU
       1         4189     65.008    105400000.0     462439605.0    358120106.0       3.601       2017.0    Action      English          AU
       2          195     67.862       450000.0     100216295.0     99550000.0     127.860       2009.0     Drama      English          AU
       3         1567     66.356     20000000.0      63465522.0     43423795.0       2.479       1985.0     Drama      English          AU
```

## Interpretation Starter
Use the mean score baseline to show what performance looks like without using movie features. Use the baseline linear regression to discuss whether budget and revenue alone predict score. Use the improved regression to evaluate whether metadata adds predictive value. Use the majority-class baseline and logistic regression classifier to evaluate score-category classification. Use the cluster profiles to describe the types of movies that appear together based on financial and metadata features, then compare their average scores.