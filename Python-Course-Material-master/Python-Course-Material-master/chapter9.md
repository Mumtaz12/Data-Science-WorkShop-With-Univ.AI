---
title: 'Case Study 7 - Movie Analysis, Part 1 - Data Preparation'
description: 'The [movie dataset on which this case study is based](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [The Movie Database (TMDb)](https://www.themoviedb.org/?language=en). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this case study, we will use this dataset to determine whether any information about a movie can predict the total revenue of a movie. We will also attempt to predict whether a movie''s revenue will exceed its budget. In Part 1, we will inspect, clean, and transform the data.'
---

## Exercise 1

```yaml
type: NormalExercise
key: 07ea54b341
lang: python
xp: 100
skills: 2
```

First, we will import several libraries. **scikit-learn** (`sklearn`) contains helpful statistical models, and we'll use the `matplotlib.pyplot` library for visualizations. Of course, we will use `numpy` and `pandas` for data manipulation throughout.

`@instructions`
- Read and execute the given code.
- Call `df.head()` to take a look at the data.

`@hint`
-  No hint on this one!

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"

```

`@sample_code`
```{python}
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
# Enter code here.


```

`@solution`
```{python}
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
df.head()

```

`@sct`
```{python}
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "It looks like `df` wasn't defined correctly.") 
test_student_typed("df.head()",
              pattern=False,
              not_typed_msg="Did you call `df.head()`?")            
success_msg("Great work!")
```

---

## Exercise 2

```yaml
type: NormalExercise
key: e2c40f651a
lang: python
xp: 100
skills: 2
```

In this exercise, we will define the regression and classification outcomes. Specifically, we will use the revenue column as the target for regression. For classification, we will construct an indicator of profitability for each movie.

`@instructions`
- Create a new column in `df` called `profitable`, defined as 1 if the movie revenue is greater than the movie budget, and 0 otherwise.
- Next, define and store the outcomes we will use for regression and classification.
    - Define `regression_target` as `'revenue'`.
    - Define `classification_target` as `'profitable'`.

`@hint`
- To create `df['profitable']`, use a simple inequality between the budget and revenue columns in `df`.  Then, we will cast this as an `int`: 1 if true, and 0 otherwise.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
```

`@sample_code`
```{python}
# Enter code here.






```

`@solution`
```{python}
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)

regression_target = 'revenue'
classification_target = 'profitable'

```

`@sct`
```{python}
test_object("df",
            undefined_msg = "Did you define `profitable`?",
            incorrect_msg = "It looks like the column `profitable` wasn't defined correctly.") 
test_object("regression_target",
            undefined_msg = "Did you define `regression_target`?",
            incorrect_msg = "It looks like `regression_target` wasn't defined correctly.") 
test_object("classification_target",
            undefined_msg = "Did you define `classification_target`?",
            incorrect_msg = "It looks like `classification_target` wasn't defined correctly.") 
success_msg("Great work!")
```

---

## Exercise 3

```yaml
type: NormalExercise
key: 177b5ae318
lang: python
xp: 100
skills: 2
```

For simplicity, we will proceed by analyzing only the rows without any missing data. In this exercise, we will remove rows with any infinite or missing values.

`@instructions`
- Use `df.replace()` to replace any cells with type `np.inf` or `-np.inf` with `np.nan`.
- Drop all rows with any `np.nan` values in that row using `df.dropna()`. Do any further arguments need to be specified in this function to remove rows with any such values?

`@hint`
- To specify the removal of rows with any missing values, add the parameter `how="any"`.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)
regression_target = 'revenue'
classification_target = 'profitable'
```

`@sample_code`
```{python}
# Enter code here.




```

`@solution`
```{python}
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")



```

`@sct`
```{python}
test_object("df",
            incorrect_msg = "It looks like not all `np.inf` and `np.nan` cells have been dropped.") 
success_msg("Great work!")
```

---

## Exercise 4

```yaml
type: NormalExercise
key: 12a3d786b3
lang: python
xp: 100
skills: 2
```

Many of the variables in our dataframe contain the names of genre, actors/actresses, and keywords. Let's add indicator columns for each genre.

`@instructions`
- Determine all the genres in the genre column. Make sure to use the `strip()` function on each genre to remove trailing characters.
- Next, include each listed genre as a new column in the dataframe. Each element of these genre columns should be 1 if the movie belongs to that particular genre, and 0 otherwise. (Keep in mind, a movie may belong to several genres at once.)
- Call `df[genres].head()` to view your results.

`@hint`
- No hint for this one.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)
regression_target = 'revenue'
classification_target = 'profitable'
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")
```

`@sample_code`
```{python}
# Enter your code here.

```

`@solution`
```{python}
list_genres = df.genres.apply(lambda x: x.split(","))
genres = []
for row in list_genres:
    row = [genre.strip() for genre in row]
    for genre in row:
        if genre not in genres:
            genres.append(genre)

for genre in genres:
    df[genre] = df['genres'].str.contains(genre).astype(int)

df[genres].head()
```

`@sct`
```{python}
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "It looks like `df` wasn't defined correctly.") 
test_student_typed("df[genres].head()",
              pattern=False,
              not_typed_msg="Did you call `df.head()`?")            
success_msg("Great work!")
```

---

## Exercise 5

```yaml
type: NormalExercise
key: 9f0ce8e050
lang: python
xp: 100
skills: 2
```

Some variables in the dataset are already numeric and perhaps useful for regression and classification. In this exercise, we will store the names of these variables for future use. We will also take a look at some of the continuous variables and outcomes by plotting each pair in a scatter plot. Finally, we will evaluate the skew of each variable.

`@instructions`
- Call `plt.show()` to observe the plot below. 
    - Consider: which of the covariates and/or outcomes are correlated with each other?
- Call `skew()` on the columns `outcomes_and_continuous_covariates` in `df`.
    - Consider: Is the skew above 1 for any of these variables?

`@hint`
- No hint for this one.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)
regression_target = 'revenue'
classification_target = 'profitable'
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")
list_genres = df.genres.apply(lambda x: x.split(","))
genres = []
for row in list_genres:
    row = [genre.strip() for genre in row]
    for genre in row:
        if genre not in genres:
            genres.append(genre)

for genre in genres:
    df[genre] = df['genres'].str.contains(genre).astype(int)
```

`@sample_code`
```{python}
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]

axes = pd.tools.plotting.scatter_matrix(df[plotting_variables], alpha = 0.15,color=(0,0,0),hist_kwds={"color":(0,0,0)},facecolor=(1,0,0))
plt.tight_layout()
# show the plot.

# determine the skew.
```

`@solution`
```{python}
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]

axes = pd.tools.plotting.scatter_matrix(df[plotting_variables], alpha = 0.15,color=(0,0,0),hist_kwds={"color":(0,0,0)},facecolor=(1,0,0))
plt.tight_layout()
plt.show()

print(df[outcomes_and_continuous_covariates].skew())

```

`@sct`
```{python}
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you call `plt.show()`?")    
              
test_student_typed(".skew()",
              pattern=False,
              not_typed_msg="Did you call `.skew()`?")   
success_msg("Great work! There is quite a bit of covariance in these pairwise plots, so our modeling strategies of regression and classification might work!")
```

---

## Exercise 6

```yaml
type: NormalExercise
key: 5caa334a5f
lang: python
xp: 100
skills: 2
```

It appears that the variables `budget`, `popularity`, `runtime`, `vote_count`, and `revenue` are all right-skewed. In this exercise, we will transform these variables to eliminate this skewness. Specifically, we will use the `np.log10()` method. Because some of these variable values are exactly 0, we will add a small positive value to each to ensure it is defined. (Note that for any base, log(0) is negative infinity!)

`@instructions`
- For each above-mentioned variable in `df`, transform value `x` into `np.log10(1+x)`.

`@hint`
- You can use the `apply()` method on a `df.Series` object. `apply()` takes a single function as its argument, and returns the `df.Series` with that function applied to each element.
- Anonymous functions can be specified using `lambda`.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)
regression_target = 'revenue'
classification_target = 'profitable'
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")
list_genres = df.genres.apply(lambda x: x.split(","))
genres = []
for row in list_genres:
    row = [genre.strip() for genre in row]
    for genre in row:
        if genre not in genres:
            genres.append(genre)

for genre in genres:
    df[genre] = df['genres'].str.contains(genre).astype(int)
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]    
```

`@sample_code`
```{python}
# Enter your code here.



```

`@solution`
```{python}
for covariate in ['budget', 'popularity', 'runtime', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))
    
    
```

`@sct`
```{python}
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "It appears you did not transform the variables correctly.") 
success_msg("Great work! This concludes the first half of this case study.  You can return to the course through this link: https://courses.edx.org/courses/course-v1:HarvardX+PH526x+1T2018")
```
