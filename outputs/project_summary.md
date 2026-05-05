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
                                                        model      rmse      mae        r2  accuracy  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  low_rate  medium_rate  high_rate
                                          Mean score baseline 10.329464 7.717587 -0.000517       NaN              NaN           NaN       NaN                 NaN              NaN          NaN       NaN          NaN        NaN
                                   Baseline linear regression 10.121755 7.547621  0.039316       NaN              NaN           NaN       NaN                 NaN              NaN          NaN       NaN          NaN        NaN
                                   Improved linear regression  9.174276 6.728352  0.210754       NaN              NaN           NaN       NaN                 NaN              NaN          NaN       NaN          NaN        NaN
Logistic regression classification: low / medium / high score       NaN      NaN       NaN  0.444835         0.433199      0.505778  0.430606             0.52387         0.444835     0.454278   0.28775     0.595164   0.117086
```

## Cluster Profiles
```text
 cluster  movie_count  avg_score  median_budget  median_revenue  median_profit  median_roi  median_year top_genre top_language top_country
       0         4192     65.008    105400000.0     462305284.8    357815003.5       3.601       2017.0    Action      English          AU
       1         4013     63.926     20000000.0      28542494.0      5245931.0       0.367       2013.0     Drama      English          AU
       2          195     67.862       450000.0     100216295.0     99550000.0     127.860       2009.0     Drama      English          AU
       3         1567     66.364     20000000.0      63408614.0     43151346.0       2.479       1985.0     Drama      English          AU
```

## Interpretation Starter
Use the mean score baseline to show what performance looks like without using movie features. Use the baseline linear regression to discuss whether budget and revenue alone predict score. Use the improved regression and classification results to evaluate whether metadata adds predictive value. Use the cluster profiles to describe the types of movies that appear together based on financial and metadata features, then compare their average scores.