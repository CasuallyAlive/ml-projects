The data is divided into train, dev and test splits, respectively located in ./train.csv, dev.csv and test.csv.

Data for cross-validation is located in the directory CVSplits/

The data format is as follows:
The first row in each CSV file contains the column names in comma-separated format.
<label> <feature-1> <feature-2> <feature-3> . . . . <feature-n> 
The remaining rows contain the values corresponding to each of the above columns.

If you're using pandas to load the file, it should recognize the first column as headers by default.
