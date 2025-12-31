## Stock Price By Financials: Building Machine Learning Models to Determine Stock Price Based on Company Financials Statements

### Introduction

### Purpose and main idea of the project

In the securities market one can often notice stocks, the price of which changes significantly within a short period of time, usually this is due to objective and significant changes in the company's business, nevertheless, it is not uncommon when a company's stock gets excessive or insufficient popularity, timely noticed such stocks are an brilliant investment with the prospect for the time when other investors will also notice the unfair price of the stock. Finding just such (undervalued or overvalued) stocks using machine learning models is the main goal of this project.

### Model Testing
Testing and selection of the best models will take place through a “pseudo” investment in stocks (for a period of 1 year), the price of which the model considers biased, after which the program will take into account the change in stock prices for the next year and get the annual return of this model.

#### Implementation subtleties

There are several ambiguities in the implementation of the project that should be taken into account.

1. The idea of training the model is to compare a company's financial performance with its stock price, so at best the model is able to find a stock whose price is significantly different from other stocks while having a similar financial position to them. Thus, the model is unable to show the degree of market overheating and takes the stock prices in the market as an objective indicator, trying to find a stock whose price stands out.
   
2. Unfortunately, the finance.yahoo site provides only 4 latest financial statements of the company, so in the project, the accuracy of the model is determined only for short-term investments (1 year). 
   
3. The data in the project is relevant only for the year 2025 in the future it is worth updating the data and possibly determining the best models.

## Project Execution

### Data collection

1. To train and test the models, i need to get access to the prices and financial statements of companies. For this purpose, the finance.yahoo service was used, as well as the YFinance library. To begin with, we had to determine which companies should be used for training. For this purpose, tickers of all active companies and their activity sectors (All_lists) were parsed using 'macrotrends' website (data_collect.py).

2. After receiving the lists of sectors with tickers of the corresponding companies, using yfinance, new databases for the financial statements of the companies were formed (All_lists_collected). Also, during the collection of company data, several functions were written to filter out irrelevant data (main.py).

### Model Training

To accomplish this project, seven machine learning algorithms were selected: **BayesianRidge**, **ElasticNet**, **Random Forest**, **XgBoosting**, **Gradient Boosting**, **CatBoosting**, **Neural Network**, for better training efficiency Optuna was used to select the best hyperparameters (variable training parameters to achieve the best results) of the models. The efficiency of the model is determined by the percentage difference between the predicted and actual stock prices. (train_models.py)

### Data preprocessing

During data processing, i have a set of many different financial statement items. To train the model, we need the same dimensionality of data. To do this, we need to remove those companies and statements with many gaps in the data. The remaining missing data needs to be filled with "median of values from the sample based on company price" (average of the gaps (e.g., 'Working capital') based on other companies with similar share prices). For one problem, we used two coefficients (optuna hyperparameters) that were used in the model. (preprocessing.py)

At this stage, the number of financial reporting elements (input data) is usually about 200; using all of them for model training may be undesirable because of the number of iterations required to train the model, and as a consequence, retraining the model due to the relatively small number of reports in many sectors as well as the large amount of computational resources and time required for training. In order to reduce the number of input data by using the correlation between the elements of financial statements and the stock price (by sorting the results by the model), i searched for the most suitable elements for training. The number of elements of the most correlated data is determined by the hyperparameter nlarge.

Finally, i can start training the models and selecting the best hyperparameters. Once the best parameters of the models have been determined, the models will be used for pseudo-investing, and the best investment parameters will be selected for them based on the results.

### Pseudo-investing

The essence of pseudo-investing is that in a certain sector, the model predicts the prices of all companies, and if their current price is less than the predicted price by a certain percentage (hyperparameter price_discount), then the script supposedly buys n number of shares, then compares the purchase price with the price for the next year and gets the result of the investment (profit or loss) depending on the direction of the investment (long or short position) and the accuracy of the prediction. (estimate_func.py)

Once the best hyperparameters for pseudo-investing (discount percentage, investment direction) are selected, the script writes the best models and their hyperparameters to a file. 

Also, to make the results more reliable, the script is run again only with data for the penultimate year. As a result, i have the best model parameters as well as two annual investment reports for all investments. On the basis of all the data obtained, a table was compiled.

## Results
At this stage, the work is done, and it is time to summarize the results:
### Table of Results

The completed table with the results of all models is in the file **Models_Results.xlsx**.
It is worth mentioning the seven best models.
![Top7](https://github.com/Kertn/Stock_Price_by_Financials/assets/111581848/966c5ecc-c2d1-4fa1-b20d-e7273af68278)

1 Column: Investment Sector\
2 Column: Model\
3 Column: Model's annual income\
4 Column: Number of shares of different companies purchased\
5 column: Number of available shares of different companies\
6 Column: Model's annual income (year before last)\
7 Column - Number of purchased shares of different companies (Year before last)\
8 column: Number of available shares of different companies (year before last)\
9 Column: Percentage difference between predicted stock price and actual price (100% - min)\
10 column: Direction of investment\
11 column: Required discount to buy the stock\
12 columns: Model parameters

### Investing

As for using the top 7 best models for investment in 2025 (Invest), the best performing model is (Medical Multi-Sector Conglomerates, CatBoost) with a return of 36.8% as of April 19, 2025.
Next is (Full_list, Random_Forest)\ with a 1% return.
The worst performance is (Transportation, Neural Network): 26.2%\.
Overall, 5 out of 7 models have a loss rate ranging from -11% to -26%.

### Conclusions
**Because of the number of failed investments since the beginning of 2025, it is currently worth being skeptical about using the models to buy stocks; at the very least, it is worth knowing their returns in the coming year 2025, or even better in 2026. Nevertheless, model predictions are well suited for additional manual analysis of the most undervalued (in the model's opinion) companies; CatBosting is the best in comparison with others.**
