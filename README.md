## Project Introduction

As part of my journey toward becoming a Bioinformatician/Data Scientist, I undertook this self-guided project to build a UK house price prediction model. My goal was to explore a real world dataset, gain hands-on experience with machine learning algorithms, and understand the process of developing, testing, and refining a models.

I used a preprocessed version of the UK Government’s Price Paid Data, made available for research via Kaggle. I chose this dataset because it was uniquely different from many others available. It focused more on categorical variables such as property type, town, new build status, and leasehold/freehold classification, rather than typical numerical features like square footage or number of bedrooms. This made the project more challenging and interesting, and pushed me to learn how to work effectively with non-numeric data.

The dataset contains 90,000 randomly sampled records spanning from 1995 to 2024.  
Dataset link: [Kaggle – UK House Price Prediction Dataset](https://www.kaggle.com/datasets/swarupsudulaganti/uk-house-price-prediction-dataset-2015-to-2024)

Throughout this project, I worked with libraries and tools such as:
- Kaggle API to download dataset
- `pandas`, `numpy` for data manipulation
- `matplotlib` for visualizations
- `scikit-learn` for model training, evaluation, and feature engineering

Despite having little prior experience with these packages, this project gave me the opportunity to build confidence and apply machine learning techniques to a real-world problem.

## Key Results 
**Before feature pruning**
- MSE: 220,864,138  
- MAE: 3,121.74  
- RMSE: 14,861.50


**After feature pruning** (removed: `Year`, `type_S`, `type_F`, `new_build`)
- MSE: 54,206,051  
- MAE: 1,129.32  
- RMSE: 7,362.48


RMSE ≈ **2.75%** of avg UK price (~£268k), indicating strong performance.

The Jupyter Notebook contains the final version of this project.
main.py was my original code, but I transitioned to the notebook format as it’s better suited for presenting and explaining my workflow.
In main.py, the dataset is downloaded using the Kaggle API. In the Jupyter Notebook, the previously downloaded dataset is simply unzipped and loaded locally.

All feedback from anyone reading this would be greatly appreciated, as this was a selfguided and taught project I understand that there is likely some errors within this.
