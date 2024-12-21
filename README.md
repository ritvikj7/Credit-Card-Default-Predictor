# Predicting Financial Risk: A Data Science Adventure in Credit Card Defaults

Imagine you're a bank manager, and you need to make a critical decision: Which credit card clients are likely to miss their next payment?
TADA!!

We tackled This challenge in Assignment 5 of CPSC 330, an applied machine learning course at the University of British Columbia. The task was to predict whether a credit card client would default on their payment next month and to do that we were armed with a dataset of 30,000 real clients, each characterized by a complex mix of financial behaviour and personal traits. With this, we aimed to classify clients into two groups—those who would default (1) and those who wouldn’t (0).

While it was framed as an assignment, it mirrored a real-world problem with significant implications for financial institutions—which rely on predictive models to assess risk and guide lending decisions. By predicting defaults, we could help reduce losses and improve risk management. However, the challenge for us lies in understanding how factors like age, credit limit, payment history, and many others influence the likelihood of default.

Components of the project:
- Our work was focused on three main areas to address the project’s objective:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Selection and Evaluation


## Exploratory Data Analysis (EDA):
Think of Exploratory Data Analysis (EDA) as the essential first step in preparing a dish. Just like gathering the right ingredients, equipment, and tools before baking a cake, in data science, you need to explore your data to understand its structure, patterns, and potential issues before diving into model building.
Our dataset, which included 30,000 client records and 24 features, ranged from demographic data like age and marital status to detailed payment histories. Initially, the raw data seemed like just a collection of numbers. But as we delved deeper, we uncovered patterns that offered valuable insights for predicting defaults. The dataset was rich with information, and it was these underlying patterns that told the real story.

From our exploration, several key insights emerged. First, Marital Status showed a subtle trend, where married individuals had a slightly higher likelihood of defaulting. While not a definitive predictor, this hinted at the role marital status could play in financial behaviour. The ratio of defaulters to non-defaulters was about 2,000 to 7,000 among married clients, compared to about 2,000 to 8,500 for unmarried clients as could be further seen in the Histogram Below. The blue represents the non-defaulters, and the yellow represents defaulters. 1 are married couples, and 2 are unmarried clients. 

We also noticed a significant Class Imbalance, meaning that a large majority of clients—77.7%—did not default, while only 22.3% of clients did. As illustrated in the pie chart below, the dataset is heavily skewed toward non-defaulters. This class imbalance posed a significant challenge. A naive model predicting "no default" for all clients could achieve high accuracy but fail to identify critical defaulters. Addressing this imbalance was crucial, as our priority was minimizing false negatives—i.e., the cases where defaulters are misclassified as non-defaulters. To this end, we focused on recall as our primary evaluation metric, ensuring the model identified as many potential defaulters as possible.


## Feature Engineering:
Feature engineering is like fine-tuning a recipe. After gathering your ingredients, this time for a pizza, you might need to create your special sauce or prepare it in a specific way to get the best flavour. Just as you use existing ingredients to create something new and improved in cooking, we use existing features to craft new ones that help optimize the model.

For instance, we created two new features to better capture the financial behaviour of clients:
- PAY_STATUS: An average measure of repayment behaviours across six months. Think of it as a "financial reliability score." This feature helped us assess how consistently clients had been making payments over time, providing deeper insights into their overall financial reliability.
- AVG_BILL_REMAINDER: The average unpaid balance across billing periods, offering a window into potential financial stress. Higher values here could indicate a higher risk of default, as clients with larger outstanding balances might be struggling financially.

By engineering such features, we aimed to improve the model’s ability to detect nuanced patterns in the data, ultimately helping to predict which clients were at risk of defaulting.


## Model Selection and Evaluation:
In this stage, we evaluated multiple machine learning models (Logistic Regression, Random Forest, KNeighborsClassifier, Decision Trees)from the Python sklearn library, each offering unique strengths and trade-offs. Here, we highlight the best performing we explored: Logistic Regression. 

Logistic Regression, in a nutshell, looks at each client’s record—such as their payment history, age, and credit limit—and assigns a probability of whether they will default. If the probability is above a certain threshold, the model predicts a default; otherwise, it predicts no default.

To deal with the imbalance in our data (only 22.3% of clients were defaulters), we adjusted the model to pay extra attention to the minority group by applying class-weight balancing. We also used regularization, which helps the model focus on the most important patterns while ignoring noise in the data.

The results were encouraging: the model achieved a recall of 64.8%, meaning it correctly flagged nearly two-thirds of potential defaulters. This was crucial since missing a defaulter could lead to financial losses. Additionally, the model’s straightforward nature made it easy to interpret, helping us understand which factors—like consistent late payments or high unpaid balances—were most predictive of default.

To deepen our understanding of the model's predictions, we used a SHAP (Shapley Additive exPlanations) analysis. A SHAP diagram provides a visual representation of how individual features contribute to a model's predictions. Features with higher absolute SHAP values have a stronger influence on the prediction. As shown in the SHAP diagram below, our analysis using the Logistic Regression model on test data highlights the influence of key features, such as MARRIAGE_2, PAY_6, and PAY_STATUS, on the model’s predictions. MARRIAGE_2 reflects the unmarried couple and its potential connection to financial stability or support systems, while PAY_6 and PAY_STATUS indicate recent payment behaviour. Notably, delayed payments captured by PAY_6 emerge as a significant driver in predicting defaults, underscoring the importance of early repayment habits.

## Important Caveats
While our model showed promising results, several limitations must be acknowledged:
- Data Limitations: Our dataset offers a snapshot of financial behaviour at a single point in time. Economic conditions, such as recessions or booms, significantly influence consumer behaviour, and our model might not account for these shifts. For example, a sudden economic downturn could drastically alter default rates, making our predictions less reliable in a changing financial landscape.
- Missing or Simplified Features: Although we worked with 26 features, these don’t fully capture the complexities of financial behaviour. Key factors like income, employment status, or unexpected life events—major predictors of financial risk—are missing from the dataset. This absence likely limits our model’s ability to generalize across diverse client populations or accurately assess individual risk.
- Simplistic Model Choices: We relied on relatively simple algorithms, such as Logistic Regression, to predict defaults. While this approach offers interpretability, it may not capture the subtleties and interactions within the data as effectively as more advanced methods like gradient boosting or neural networks, which are better suited for complex problems like financial risk prediction.
- Temporal Stability: Our model assumes that relationships between features and default risk remain stable over time. However, financial behaviour and its drivers can evolve due to societal, regulatory, or technological changes, meaning our model’s performance may degrade if applied to future datasets without regular updates or retraining.
These caveats underline the need for cautious application of our model in real-world settings and highlight opportunities for future refinement.

