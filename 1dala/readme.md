# Files and their purposes
- analysis = Scatterplots and histograms
- correlation_data = Main correlation matrix for the data. Shows how correlated variables are to each other.
- correlation_classes = Correlation matrices for each classs ar plotted
- data_selector = produced "data_selected.csv" datukopa kas satur tikai MinorAxisLength, MajorAxisLength, Extent un Class
- duplicates = produces "out/data_duplicate.csv" only duplicate data entries saved here (should be empty)
- find_outliers = produces "out/outliers_data.csv" that contains all datapoints that fall outside the standard deviation threshhold of 4 and "out/data_with_no_outliers.csv" contains data.csv without the outliers
- find_outlying_attributes = same as find_outliers.py but with attributes. produces "out/attribute_with_no_outliers"
- missing_attributes = produces "out/missing_data.csv" which has all datapoints with empty attributes (should be empty)
---